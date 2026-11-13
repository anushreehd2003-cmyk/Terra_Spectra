"""Mid-Project Review Audit Tests: Memory Profiling, GIS Coordinates, and Endpoints (Day 5)."""

import unittest
import numpy as np

from app.core.config import settings
from app.pipeline.georeference import AffineGeoTransform, GeoreferencedHeatmapBuilder
from app.utils.profiler import SystemMemoryProfiler, GISCoordinateAuditor
from app.api.endpoints import get_farm_heatmap, get_mid_project_review_audit


class TestGeoreferencing(unittest.TestCase):
    """Test Affine coordinate transformations and Deck.gl visual generation."""

    def setUp(self):
        self.transform = AffineGeoTransform(height_px=32, width_px=32)

    def test_coordinate_reversibility(self):
        # Convert pixel to geographic and back
        orig_y, orig_x = 10, 20
        lon, lat = self.transform.pixel_to_geo(orig_y, orig_x)
        recon_y, recon_x = self.transform.geo_to_pixel(lon, lat)

        self.assertEqual(orig_y, recon_y)
        self.assertEqual(orig_x, recon_x)

    def test_containment_and_bounds_polygon(self):
        poly = self.transform.get_pixel_bounds_polygon(0, 0)
        # Polygon must be closed (5 vertices)
        self.assertEqual(len(poly), 5)
        self.assertEqual(poly[0], poly[-1])

        # All vertices must be contained in farm bounds
        for lon, lat in poly:
            self.assertTrue(self.transform.validate_coordinate_containment(lon, lat))

    def test_deckgl_heatmap_points_and_colors(self):
        builder = GeoreferencedHeatmapBuilder(self.transform)
        risk_map = np.zeros((32, 32), dtype=np.float32)
        # Add sample risk points
        risk_map[5, 5] = 0.10   # Healthy
        risk_map[15, 15] = 0.65 # Pre-symptomatic
        risk_map[25, 25] = 0.95 # Critical

        points = builder.build_deckgl_points(risk_map)
        self.assertEqual(len(points), 1024)

        # Check color bands
        rgba_healthy = GeoreferencedHeatmapBuilder.risk_to_rgba(0.10)
        rgba_presymp = GeoreferencedHeatmapBuilder.risk_to_rgba(0.65)
        rgba_critical = GeoreferencedHeatmapBuilder.risk_to_rgba(0.95)

        self.assertEqual(rgba_healthy[0], 16)   # Emerald
        self.assertEqual(rgba_presymp[0], 245)  # Amber
        self.assertEqual(rgba_critical[0], 244) # Crimson

        geojson = builder.build_geojson_cells(risk_map)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertGreater(geojson["total_cells"], 0)


class TestMidProjectReviewAudit(unittest.TestCase):
    """Test memory profiling thresholds and spatial coordinate audit."""

    def test_system_memory_profile_audit(self):
        result = SystemMemoryProfiler.profile_pipeline_execution(height=32, width=32, pca_components=30)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["memory"]["is_memory_safe"])
        self.assertLess(result["memory"]["peak_ram_mb"], 250.0)
        self.assertIn("incremental_pca_ms", result["latency_ms"])
        self.assertIn("cnn_inference_ms", result["latency_ms"])
        self.assertEqual(result["compression_savings"]["compressed_channels"], 30)

    def test_gis_coordinate_validation_audit(self):
        result = GISCoordinateAuditor.audit_coordinates(height=32, width=32)

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["corners_contained"])
        self.assertTrue(result["sub_pixel_precision"])
        self.assertLess(result["projection_center_error_meters"], 70.0)
        self.assertEqual(result["crs"], "EPSG:4326 (WGS84)")

    def test_api_farm_heatmap(self):
        data = get_farm_heatmap()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["crs"], "EPSG:4326 (WGS84)")
        self.assertEqual(data["total_points"], 1024)
        self.assertEqual(len(data["deckgl_points"]), 1024)

    def test_api_mid_project_review_endpoint(self):
        data = get_mid_project_review_audit()
        self.assertEqual(data["overall_verdict"], "PASS")
        self.assertTrue(data["compliance"]["edge_ram_budget_under_250mb"])
        self.assertTrue(data["compliance"]["sub_pixel_geospatial_precision"])


if __name__ == "__main__":
    unittest.main()
