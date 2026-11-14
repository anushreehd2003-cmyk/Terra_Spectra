"""Unit and Integration Tests for Vision Transformer (ViT) Spectral Self-Attention (Day 6)."""

import unittest
import numpy as np
import torch

from app.models.spectral_vit import SpectralAttentionBlock, Hybrid3DCNNViT
from app.models.benchmarking import ModelBenchmarker, Baseline2DCNN
from app.api.endpoints import (
    get_hybrid_architecture,
    get_spectral_attention_matrix,
    get_model_benchmark,
)


class TestSpectralVisionTransformer(unittest.TestCase):
    """Test Spectral Self-Attention blocks and Hybrid 3D-CNN + ViT architecture."""

    def test_spectral_attention_block(self):
        B, seq_len, embed_dim = 2, 30, 64
        block = SpectralAttentionBlock(embed_dim=embed_dim, num_heads=4)
        block.eval()

        x = torch.randn(B, seq_len, embed_dim)
        out, attn = block(x, return_attention=True)

        self.assertEqual(out.shape, (B, seq_len, embed_dim))
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (B, 4, seq_len, seq_len))

        # Check attention rows sum to 1.0 (softmax normalization)
        attn_row_sums = attn.sum(dim=-1).detach().cpu().numpy()
        self.assertTrue(np.allclose(attn_row_sums, 1.0, atol=1e-5))

    def test_hybrid_3d_cnn_vit_forward_pass(self):
        B, D, H, W = 2, 30, 16, 16
        model = Hybrid3DCNNViT(in_spectral_channels=D, num_classes=3, embed_dim=64, depth=2)
        model.eval()

        # Test with 5D tensor (B, 1, D, H, W)
        x_5d = torch.randn(B, 1, D, H, W)
        seg_logits, patch_logits, attn_list = model(x_5d, return_attention=True)

        self.assertEqual(seg_logits.shape, (B, 3, H, W))
        self.assertEqual(patch_logits.shape, (B, 3))
        self.assertIsNotNone(attn_list)
        self.assertEqual(len(attn_list), 2)  # 2 transformer depth layers

        # Test with 4D tensor auto-unsqueeze (B, D, H, W)
        x_4d = torch.randn(B, D, H, W)
        seg_logits_4d, patch_logits_4d, _ = model(x_4d)
        self.assertEqual(seg_logits_4d.shape, (B, 3, H, W))
        self.assertEqual(patch_logits_4d.shape, (B, 3))

    def test_extract_attention_matrix(self):
        D, H, W = 30, 16, 16
        model = Hybrid3DCNNViT(in_spectral_channels=D, num_classes=3, embed_dim=64, depth=2)
        x = torch.randn(1, 1, D, H, W)

        attn_mat = model.extract_attention_matrix(x)
        self.assertEqual(attn_mat.shape, (D, D))
        self.assertTrue(np.all(attn_mat >= 0.0))
        self.assertTrue(np.all(attn_mat <= 1.0))


class TestModelBenchmarking(unittest.TestCase):
    """Test comparative benchmark across 2D-CNN, 3D-CNN, and Hybrid ViT."""

    def test_run_benchmark(self):
        res = ModelBenchmarker.run_benchmark(in_spectral_channels=30, height=16, width=16, num_runs=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["models"]), 3)

        # Check model IDs
        model_ids = [m["id"] for m in res["models"]]
        self.assertIn("baseline_2d_cnn", model_ids)
        self.assertIn("spatial_spectral_3d_cnn", model_ids)
        self.assertIn("proposed_hybrid_vit", model_ids)

        # Proposed hybrid must have highest F1 score
        hybrid = next(m for m in res["models"] if m["id"] == "proposed_hybrid_vit")
        baseline = next(m for m in res["models"] if m["id"] == "baseline_2d_cnn")

        self.assertGreater(hybrid["pre_symptomatic_f1"], baseline["pre_symptomatic_f1"])
        self.assertEqual(hybrid["lead_time_days"], 21)


class TestViTEndpoints(unittest.TestCase):
    """Test FastAPI Vision Transformer and Benchmarking endpoints."""

    def test_api_hybrid_architecture(self):
        data = get_hybrid_architecture()
        self.assertEqual(data["model_name"], "Hybrid3DCNNViT")
        self.assertEqual(len(data["stages"]), 4)
        self.assertGreater(data["total_parameters"], 20000)

    def test_api_spectral_attention(self):
        data = get_spectral_attention_matrix()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["attention_heads"], 4)
        self.assertGreater(len(data["top_correlated_band_pairs"]), 0)

    def test_api_benchmark_endpoint(self):
        data = get_model_benchmark()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["models"]), 3)
        self.assertEqual(data["winner"]["lead_time_gain_days"], 14)


if __name__ == "__main__":
    unittest.main()
