"""Unit tests for Executive Final Review Audit, VRA Prescription Generator, and GeoTIFF Exporter."""

import unittest
import numpy as np

from app.core.config import settings
from app.pipeline.georeference import AffineGeoTransform
from app.pipeline.geotiff_exporter import PrescriptionMapGenerator, GeoTIFFExporter
from app.utils.final_review import ExecutiveFinalReviewAuditor
from app.api.endpoints import (
    get_final_review_audit,
    get_prescription_summary,
    export_prescription_geotiff,
)


class TestPrescriptionAndGeoTIFF(unittest.TestCase):
    """Test suite for VRA prescription mapping and GeoTIFF export."""

    def setUp(self):
        self.H, self.W = 32, 32
        # Create a synthetic risk map with healthy areas and a central infection hotspot
        self.risk_map = np.zeros((self.H, self.W), dtype=np.float32)
        # Hotspot in center
        self.risk_map[12:20, 12:20] = 0.65  # Zone 2 (Pre-symptomatic)
        self.risk_map[14:18, 14:18] = 0.88  # Zone 3 (Outbreak core)
        self.risk_map[10:22, 10:22] = np.maximum(self.risk_map[10:22, 10:22], 0.32)  # Zone 1 (Buffer)

    def test_prescription_generator_zones_and_savings(self):
        """Verify VRA management zones and economic savings calculations."""
        gen = PrescriptionMapGenerator(total_acres=1000.0)
        presc = gen.compute_prescription(self.risk_map)

        self.assertEqual(presc["total_monitored_acres"], 1000.0)
        self.assertGreater(presc["presymptomatic_hotspot_acres"], 0.0)
        self.assertLess(presc["presymptomatic_hotspot_acres"], 500.0)

        # 4 Zones present
        self.assertEqual(len(presc["zones"]), 4)
        for zone in presc["zones"]:
            self.assertIn("zone_id", zone)
            self.assertIn("acres", zone)
            self.assertIn("rate_l_ha", zone)

        # Significant economic savings vs blanket spraying
        impact = presc["economic_impact"]
        self.assertGreater(impact["savings_pct"], 50.0)
        self.assertGreater(impact["savings_usd"], 10000.0)
        self.assertGreater(impact["groundwater_leaching_reduction_pct"], 50.0)

    def test_geotiff_exporter_metadata_and_bytes(self):
        """Verify multi-band GeoTIFF file generation and EPSG:4326 metadata tags."""
        transform = AffineGeoTransform(height_px=self.H, width_px=self.W)
        exporter = GeoTIFFExporter(transform)

        tiff_bytes, meta = exporter.export_prescription_geotiff(self.risk_map)

        # Bytes should be generated
        self.assertIsInstance(tiff_bytes, bytes)
        self.assertGreater(len(tiff_bytes), 2000)

        # First 4 bytes are TIFF magic number (II*\x00 for little-endian or MM\x00* for big-endian)
        self.assertTrue(tiff_bytes[:2] in [b"II", b"MM"])

        # Metadata checks
        self.assertEqual(meta["format"], "GeoTIFF")
        self.assertEqual(meta["crs"], "EPSG:4326 (WGS84)")
        self.assertEqual(meta["dimensions"]["bands"], 3)
        self.assertEqual(meta["dimensions"]["height"], self.H)
        self.assertEqual(meta["dimensions"]["width"], self.W)
        self.assertEqual(len(meta["band_descriptions"]), 3)


class TestExecutiveFinalReviewAuditor(unittest.TestCase):
    """Test suite for the Executive Final Project Review audit."""

    def test_final_review_audit_execution(self):
        """Run full end-to-end final review and verify all 7 pillars pass."""
        report = ExecutiveFinalReviewAuditor.run_full_final_review()

        self.assertEqual(report["overall_status"], "PRODUCTION_READY_CERTIFIED")
        self.assertEqual(report["overall_score"], 100)
        self.assertTrue(report["all_pillars_passed"])
        self.assertEqual(len(report["pillars"]), 7)

        # Check all 7 pillars are marked passed
        for p in report["pillars"]:
            self.assertTrue(p["passed"], f"Pillar {p['name']} failed")

        # Verify sections
        sections = report["sections"]
        self.assertTrue(sections["hyperspectral_physics"]["verified"])
        self.assertTrue(sections["spectral_reduction"]["verified"])
        self.assertTrue(sections["model_vit"]["verified"])
        self.assertTrue(sections["tiling_engine"]["verified"])
        self.assertTrue(sections["gis_vra_export"]["verified"])
        self.assertTrue(sections["performance_and_memory"]["verified"])


class TestFinalReviewEndpoints(unittest.TestCase):
    """Test FastAPI route handlers for final review and export."""

    def test_api_final_review_endpoint(self):
        res = get_final_review_audit()
        self.assertEqual(res["overall_status"], "PRODUCTION_READY_CERTIFIED")
        self.assertEqual(res["overall_score"], 100)

    def test_api_prescription_summary_endpoint(self):
        res = get_prescription_summary()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_monitored_acres"], 1000.0)
        self.assertEqual(len(res["zones"]), 4)
        self.assertIn("economic_impact", res)
        self.assertIn("sample_dosage_grid", res)

    def test_api_export_prescription_geotiff_endpoint(self):
        res = export_prescription_geotiff()
        self.assertEqual(res.media_type, "image/tiff")
        self.assertGreater(len(res.body), 2000)
        self.assertIn("Content-Disposition", res.headers)
        self.assertIn("X-CRS", res.headers)
        self.assertIn("EPSG:4326", res.headers["X-CRS"])


if __name__ == "__main__":
    unittest.main()
