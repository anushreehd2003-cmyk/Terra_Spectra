"""Unit and Integration Tests for PyTorch 3D-CNN and Focal Loss (Day 3/4)."""

import unittest
import numpy as np
import torch

from app.models.spatial_spectral_cnn import SpatialSpectralCNN3D
from app.models.focal_loss import FocalLoss
from app.models.dataset import HyperspectralPatchDataset, create_hyperspectral_dataloader
from app.models.trainer import DiseaseForecastingTrainer
from app.utils.cube_simulator import HyperspectralCubeSimulator
from app.pipeline.spectral_reduction import SpectralDimensionalityReducer
from app.api.endpoints import get_model_architecture, trigger_training_step, predict_scene_risk


class TestSpatialSpectralCNN3D(unittest.TestCase):
    """Test 3D-CNN architecture forward pass and shape transformations."""

    def setUp(self):
        self.model = SpatialSpectralCNN3D(in_spectral_channels=30, num_classes=3)
        self.model.eval()

    def test_forward_with_5d_tensor(self):
        # Batch: 2, Channels: 1, Spectral Depth: 30, Height: 16, Width: 16
        x = torch.randn(2, 1, 30, 16, 16)
        seg_logits, patch_logits = self.model(x)

        self.assertEqual(seg_logits.shape, (2, 3, 16, 16))
        self.assertEqual(patch_logits.shape, (2, 3))

    def test_forward_with_4d_tensor_auto_unsqueeze(self):
        # Batch: 2, Spectral Depth: 30, Height: 16, Width: 16
        x = torch.randn(2, 30, 16, 16)
        seg_logits, patch_logits = self.model(x)

        self.assertEqual(seg_logits.shape, (2, 3, 16, 16))
        self.assertEqual(patch_logits.shape, (2, 3))

    def test_predict_risk_map(self):
        x = torch.randn(2, 1, 30, 16, 16)
        risk = self.model.predict_risk_map(x)

        self.assertEqual(risk.shape, (2, 16, 16))
        self.assertTrue(torch.all(risk >= 0.0))
        self.assertTrue(torch.all(risk <= 1.0))


class TestFocalLoss(unittest.TestCase):
    """Test Focal Loss focusing mechanism and class weighting."""

    def test_focal_loss_vs_cross_entropy_focusing(self):
        loss_fn = FocalLoss(gamma=2.0, alpha=[0.2, 0.6, 0.2])

        # Confident correct prediction (easy example)
        easy_logits = torch.tensor([[10.0, -10.0, -10.0]])
        easy_target = torch.tensor([0])
        easy_loss = loss_fn(easy_logits, easy_target)

        # Ambiguous prediction (hard example)
        hard_logits = torch.tensor([[0.5, 0.6, -0.5]])
        hard_target = torch.tensor([1])  # Pre-symptomatic class
        hard_loss = loss_fn(hard_logits, hard_target)

        self.assertGreater(hard_loss.item(), easy_loss.item() * 5.0)

    def test_focal_loss_segmentation_gradient(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (2, 8, 8))

        loss = loss_fn(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


class TestDatasetAndTrainer(unittest.TestCase):
    """Test Dataset extraction, trainer optimization step, and metrics."""

    def setUp(self):
        sim = HyperspectralCubeSimulator(num_bands=204, height=32, width=32, seed=42)
        scene = sim.generate_scene(infection_foci_count=2)
        self.cube = scene["cube"]
        self.mask = scene["disease_mask"]

        self.reducer = SpectralDimensionalityReducer(n_components=30)
        self.reducer.fit(self.cube)

    def test_patch_dataset(self):
        reduced = self.reducer.transform(self.cube)
        dataset = HyperspectralPatchDataset(
            reduced_cube=reduced,
            disease_mask=self.mask,
            patch_size=16,
            stride=8,
            augment=True,
        )

        self.assertGreater(len(dataset), 0)
        patch, mask, label = dataset[0]

        self.assertEqual(patch.shape, (1, 30, 16, 16))
        self.assertEqual(mask.shape, (16, 16))
        self.assertIn(label.item(), [0, 1, 2])

    def test_trainer_optimization_and_full_inference(self):
        reduced = self.reducer.transform(self.cube)
        loader = create_hyperspectral_dataloader(
            scene_cube=self.cube,
            disease_mask=self.mask,
            reducer=self.reducer,
            patch_size=16,
            stride=8,
            batch_size=4,
            augment=False,
        )

        trainer = DiseaseForecastingTrainer(in_spectral_channels=30, num_classes=3, lr=1e-3)
        metrics = trainer.train_epoch(loader)

        self.assertIn("train_loss", metrics)
        self.assertGreater(metrics["train_loss"], 0.0)

        eval_metrics = trainer.evaluate(loader)
        self.assertIn("pre_symptomatic_f1", eval_metrics)
        self.assertGreaterEqual(eval_metrics["pixel_accuracy"], 0.0)

        # Full scene inference
        risk_map = trainer.predict_full_scene(reduced, patch_size=16, stride=8)
        self.assertEqual(risk_map.shape, (32, 32))
        self.assertTrue(np.all(risk_map >= 0.0))
        self.assertTrue(np.all(risk_map <= 1.0))


class TestModelEndpoints(unittest.TestCase):
    """Test 3D-CNN API endpoints."""

    def test_api_model_architecture(self):
        data = get_model_architecture()
        self.assertEqual(data["model_name"], "SpatialSpectralCNN3D")
        self.assertEqual(data["num_classes"], 3)
        self.assertGreater(data["total_parameters"], 10000)

    def test_api_train_step(self):
        data = trigger_training_step(epochs=1)
        self.assertEqual(data["status"], "success")
        self.assertIn("early_warning_f1", data)

    def test_api_predict_scene(self):
        data = predict_scene_risk()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["lead_time_forecast"], "3 Weeks Before Foliar Symptoms")
        self.assertIn("predicted_risk_acres", data)


if __name__ == "__main__":
    unittest.main()
