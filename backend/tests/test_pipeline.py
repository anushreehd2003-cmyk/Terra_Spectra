"""Unit and Integration Tests for TerraSpectra Geospatial Data Pipeline (Day 2)."""

import unittest
import numpy as np

from app.core.config import settings
from app.main import health_check
from app.api.endpoints import (
    get_system_status,
    get_wavelengths,
    get_spectral_signatures,
    get_farm_geojson,
    get_pca_summary,
    trigger_simulation,
)
from app.utils.spectral_physics import (
    generate_wavelength_grid,
    get_healthy_vegetation_signature,
    get_pre_symptomatic_fungal_signature,
    get_symptomatic_necrotic_signature,
    get_soil_signature,
)
from app.utils.cube_simulator import HyperspectralCubeSimulator
from app.pipeline.spectral_reduction import SpectralDimensionalityReducer
from app.pipeline.ingestion import HyperspectralIngestionPipeline


class TestSpectralPhysics(unittest.TestCase):
    """Test biophysical reflectance models and wavelength continuity."""

    def setUp(self):
        self.wavelengths = generate_wavelength_grid(204, 400, 2400)

    def test_wavelength_grid(self):
        self.assertEqual(len(self.wavelengths), 204)
        self.assertAlmostEqual(self.wavelengths[0], 400.0, places=1)
        self.assertAlmostEqual(self.wavelengths[-1], 2400.0, places=1)
        self.assertTrue(np.all(np.diff(self.wavelengths) > 0), "Wavelengths must be strictly monotonic")

    def test_pre_symptomatic_subvisual_nature(self):
        """Verify pre-symptomatic foliage is identical in visible bands but diverges in Red Edge/NIR."""
        healthy = get_healthy_vegetation_signature(self.wavelengths)
        pre_symp = get_pre_symptomatic_fungal_signature(self.wavelengths, severity=0.8)

        # 1. Visible spectrum (400 - 680 nm)
        vis_mask = (self.wavelengths >= 400) & (self.wavelengths <= 680)
        max_vis_diff = np.max(np.abs(healthy[vis_mask] - pre_symp[vis_mask]))
        self.assertLess(max_vis_diff, 0.02, "Visible spectrum must remain sub-visual / undetectable to RGB")

        # 2. NIR Plateau (750 - 1100 nm)
        nir_mask = (self.wavelengths >= 750) & (self.wavelengths <= 1100)
        mean_nir_drop = np.mean(healthy[nir_mask] - pre_symp[nir_mask])
        self.assertGreater(mean_nir_drop, 0.08, "NIR plateau must experience clear drop from mesophyll damage")

    def test_soil_signature(self):
        soil = get_soil_signature(self.wavelengths)
        self.assertTrue(np.all(soil >= 0.0))
        self.assertTrue(np.all(soil <= 1.0))
        # Soil reflectance at SWIR (2000nm) should be higher than visible (500nm)
        idx_500 = np.argmin(np.abs(self.wavelengths - 500))
        idx_2000 = np.argmin(np.abs(self.wavelengths - 2000))
        self.assertGreater(soil[idx_2000], soil[idx_500])


class TestCubeSimulator(unittest.TestCase):
    """Test synthetic hyperspectral cube generation."""

    def test_cube_generation(self):
        sim = HyperspectralCubeSimulator(num_bands=204, height=32, width=32, seed=123)
        data = sim.generate_scene(infection_foci_count=2)

        cube = data["cube"]
        disease_mask = data["disease_mask"]
        risk_map = data["risk_map"]
        meta = data["metadata"]

        self.assertEqual(cube.shape, (204, 32, 32))
        self.assertEqual(disease_mask.shape, (32, 32))
        self.assertEqual(risk_map.shape, (32, 32))

        # Reflectance bounds
        self.assertTrue(np.all(cube >= 0.0))
        self.assertTrue(np.all(cube <= 1.0))

        # Metadata checks
        self.assertEqual(meta["num_bands"], 204)
        self.assertIn("bbox_wgs84", meta)
        self.assertGreater(meta["infection_area_acres"], 0)


class TestIncrementalPCAReducer(unittest.TestCase):
    """Test PCA spectral dimensionality reduction."""

    def test_reduction_and_variance(self):
        sim = HyperspectralCubeSimulator(num_bands=204, height=32, width=32, seed=99)
        scene = sim.generate_scene(infection_foci_count=2)
        cube = scene["cube"]

        reducer = SpectralDimensionalityReducer(n_components=30)
        reduced = reducer.fit_transform(cube)

        self.assertEqual(reduced.shape, (30, 32, 32))
        metrics = reducer.get_metrics()
        self.assertTrue(metrics["fitted"])
        self.assertGreater(
            metrics["total_variance_explained_pct"],
            95.0,
            "30 PCA components must retain >95% spectral variance",
        )

        recon_eval = reducer.evaluate_reconstruction(cube)
        self.assertLess(recon_eval["mse"], 0.005)
        self.assertGreater(recon_eval["psnr_db"], 20.0)


class TestIngestionPipeline(unittest.TestCase):
    """Test vegetation index calculation and sliding window tile extraction."""

    def test_pipeline_indices_and_tiling(self):
        pipeline = HyperspectralIngestionPipeline(num_bands=204)
        sim = HyperspectralCubeSimulator(num_bands=204, height=32, width=32, seed=42)
        scene = sim.generate_scene()
        cube = scene["cube"]

        # Indices
        indices = pipeline.compute_vegetation_indices(cube)
        self.assertIn("ndvi", indices)
        self.assertIn("ndre", indices)
        self.assertIn("pri", indices)
        self.assertIn("pshi", indices)
        self.assertEqual(indices["ndvi"].shape, (32, 32))

        # Sliding window tiles (32x32 with tile_size=16, stride=16 gives 4 tiles)
        tiles = pipeline.extract_tiles(cube, tile_size=16, stride=16)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0]["patch"].shape, (204, 16, 16))


class TestFastAPIEndpoints(unittest.TestCase):
    """Test REST API route handlers synchronously."""

    def test_health_endpoint(self):
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertEqual(res["service"], settings.PROJECT_NAME)

    def test_status_endpoint(self):
        data = get_system_status()
        self.assertEqual(data["spectral_config"]["bands"], 204)
        self.assertEqual(data["farm"]["total_acres"], 1000.0)

    def test_spectral_signatures(self):
        data = get_spectral_signatures()
        self.assertEqual(data["bands_count"], 204)
        self.assertEqual(len(data["data_points"]), 204)
        self.assertIn("delta_pre_symp", data["data_points"][0])

    def test_farm_geojson(self):
        data = get_farm_geojson()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 4)

    def test_pca_summary(self):
        data = get_pca_summary()
        self.assertGreater(data["metrics"]["total_variance_explained_pct"], 95.0)
        self.assertEqual(data["compression"]["reduced_channels"], 30)

    def test_simulation_trigger(self):
        data = trigger_simulation(seed=12, foci=2)
        self.assertEqual(data["status"], "success")
        self.assertIn("metadata", data)


if __name__ == "__main__":
    unittest.main()
