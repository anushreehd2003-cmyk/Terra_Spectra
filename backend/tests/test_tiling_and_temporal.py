"""Unit and Integration Tests for Tiling Engine and Temporal Progression (Week 4)."""

import unittest
import numpy as np
import torch

from app.pipeline.tiling_engine import GaussianTileStitcher
from app.pipeline.temporal_progression import TemporalProgressionEngine
from app.models.spatial_spectral_cnn import SpatialSpectralCNN3D


class TestGaussianTileStitcher(unittest.TestCase):
    """Test raster chunking and seamless 2D Gaussian blending."""

    def setUp(self):
        self.stitcher = GaussianTileStitcher(tile_size=16, stride=8, sigma_ratio=3.0)

    def test_gaussian_kernel_shape_and_taper(self):
        k = self.stitcher.gaussian_kernel
        self.assertEqual(k.shape, (16, 16))
        # Center should be maximum (~1.0)
        center_val = k[7, 7]
        corner_val = k[0, 0]
        self.assertGreater(center_val, 0.95)
        self.assertLess(corner_val, 0.40)
        self.assertGreater(corner_val, 0.0)

    def test_slice_and_stitch_reconstruction(self):
        H, W = 32, 32
        dummy_raster = np.ones((H, W), dtype=np.float32) * 0.75

        tiles = self.stitcher.slice_raster(dummy_raster)
        self.assertGreater(len(tiles), 4)

        # Simulate predictions
        preds = [t["patch"] for t in tiles]
        coords = [(t["y_start"], t["x_start"]) for t in tiles]

        stitched = self.stitcher.stitch_predictions(preds, coords, (H, W))
        self.assertEqual(stitched.shape, (H, W))

        # Reconstructed flat signal must equal original without seam drops
        self.assertTrue(np.allclose(stitched, 0.75, atol=1e-3))

    def test_process_and_stitch_with_model(self):
        D, H, W = 30, 32, 32
        cube = np.random.rand(D, H, W).astype(np.float32)
        model = SpatialSpectralCNN3D(in_spectral_channels=30, num_classes=3)
        model.eval()

        def model_fn(batch_tensor):
            with torch.no_grad():
                seg_logits, _ = model(batch_tensor)
                probs = torch.softmax(seg_logits, dim=1)
                return probs[:, 1].cpu().numpy()

        res = self.stitcher.process_and_stitch_with_model(cube, model_fn, batch_size=4)

        self.assertEqual(res["mosaic"].shape, (H, W))
        self.assertTrue(res["full_coverage_verified"])
        self.assertIn("seam_artifact_suppression", res)


class TestTemporalProgressionEngine(unittest.TestCase):
    """Test 4-week timeline forecasting and early warning divergence."""

    def test_timeline_stages_continuity(self):
        stages = TemporalProgressionEngine.get_timeline_overview()
        self.assertEqual(len(stages), 4)

        # Week -3 properties
        w3 = stages[0]
        self.assertEqual(w3["week_offset"], -3)
        self.assertEqual(w3["days_before_symptoms"], 21)
        self.assertFalse(w3["rgb_visible_anomalies"], "Visual symptoms must be 0% at Week -3")
        self.assertTrue(w3["terraspectra_ai_alert"], "TerraSpectra AI alert must be ACTIVE at Week -3")

        # Week 0 properties
        w0 = stages[-1]
        self.assertEqual(w0["week_offset"], 0)
        self.assertTrue(w0["rgb_visible_anomalies"], "Week 0 exhibits visible lesions")

        # Monotonically increasing risk acreage
        risk_acres = [s["risk_acres"] for s in stages]
        self.assertTrue(all(x < y for x, y in zip(risk_acres, risk_acres[1:])))

        # Monotonically decreasing treatment savings
        cost_savings = [s["treatment_cost_savings_pct"] for s in stages]
        self.assertTrue(all(x > y for x, y in zip(cost_savings, cost_savings[1:])))

    def test_generate_temporal_sequence(self):
        seq = TemporalProgressionEngine.generate_temporal_risk_sequence(height=16, width=16)
        self.assertEqual(len(seq), 4)

        # Confirm risk spreads over time
        mean_risks = [s["mean_risk_score"] for s in seq]
        self.assertLess(mean_risks[0], mean_risks[-1])


class TestTemporalAndTilingEndpoints(unittest.TestCase):
    """Test FastAPI routes for temporal progression and Gaussian tiling."""

    def test_api_temporal_timeline(self):
        from app.api.endpoints import get_temporal_timeline
        res = get_temporal_timeline()
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["stages"]), 4)

    def test_api_temporal_sequence(self):
        from app.api.endpoints import get_temporal_sequence
        res = get_temporal_sequence()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["timeline_length"], 4)

    def test_api_tile_and_stitch(self):
        from app.api.endpoints import execute_tile_and_stitch
        res = execute_tile_and_stitch()
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["full_coverage_verified"])
        self.assertGreater(res["total_tiles_processed"], 0)


if __name__ == "__main__":
    unittest.main()
