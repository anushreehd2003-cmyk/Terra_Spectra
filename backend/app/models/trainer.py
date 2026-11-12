"""Trainer and Evaluator for 3D-CNN Spatial-Spectral Crop Disease Forecasting.

Features:
- Multi-task optimization (pixel-level segmentation + global patch risk).
- Class-weighted Focal Loss to combat extreme agricultural background class imbalance.
- Full evaluation suite: Pre-Symptomatic Precision, Recall, F1-Score, and IoU.
- Gaussian-blended full-scene spatial inference.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from app.models.spatial_spectral_cnn import SpatialSpectralCNN3D
from app.models.focal_loss import FocalLoss


class DiseaseForecastingTrainer:
    """Manages training, validation, and full-scene inference for 3D-CNN models."""

    def __init__(
        self,
        model: Optional[SpatialSpectralCNN3D] = None,
        in_spectral_channels: int = 30,
        num_classes: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.num_classes = num_classes

        self.model = model or SpatialSpectralCNN3D(
            in_spectral_channels=in_spectral_channels,
            num_classes=num_classes,
        )
        self.model.to(self.device)

        # Default class weights: prioritize Pre-Symptomatic (class 1) over Healthy (class 0)
        if class_weights is None:
            class_weights = [0.2, 0.6, 0.2]

        self.focal_seg_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        self.focal_patch_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-5)

        self.best_f1 = 0.0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        seg_loss_sum = 0.0
        patch_loss_sum = 0.0
        batches = 0

        for patches, masks, labels in dataloader:
            patches = patches.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            seg_logits, patch_logits = self.model(patches)

            loss_seg = self.focal_seg_loss(seg_logits, masks)
            loss_patch = self.focal_patch_loss(patch_logits, labels)
            loss = loss_seg + 0.5 * loss_patch

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            seg_loss_sum += loss_seg.item()
            patch_loss_sum += loss_patch.item()
            batches += 1

        self.scheduler.step()

        return {
            "train_loss": round(total_loss / max(1, batches), 4),
            "train_seg_loss": round(seg_loss_sum / max(1, batches), 4),
            "train_patch_loss": round(patch_loss_sum / max(1, batches), 4),
            "lr": round(float(self.optimizer.param_groups[0]["lr"]), 6),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model and compute pre-symptomatic precision, recall, F1, and IoU."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        # Confusion metrics for Class 1 (Pre-Symptomatic)
        tp = 0
        fp = 0
        fn = 0
        total_pixels = 0
        correct_pixels = 0

        with torch.no_grad():
            for patches, masks, labels in dataloader:
                patches = patches.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                seg_logits, patch_logits = self.model(patches)
                loss_seg = self.focal_seg_loss(seg_logits, masks)
                loss_patch = self.focal_patch_loss(patch_logits, labels)
                loss = loss_seg + 0.5 * loss_patch

                total_loss += loss.item()
                batches += 1

                # Pixel-level predictions
                preds = torch.argmax(seg_logits, dim=1)  # (B, H, W)

                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Pre-Symptomatic (Class 1) stats
                pred_pos = preds == 1
                gt_pos = masks == 1

                tp += (pred_pos & gt_pos).sum().item()
                fp += (pred_pos & ~gt_pos).sum().item()
                fn += (~pred_pos & gt_pos).sum().item()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        pixel_acc = correct_pixels / max(1, total_pixels)

        return {
            "val_loss": round(total_loss / max(1, batches), 4),
            "pixel_accuracy": round(pixel_acc, 4),
            "pre_symptomatic_precision": round(precision, 4),
            "pre_symptomatic_recall": round(recall, 4),
            "pre_symptomatic_f1": round(f1, 4),
            "pre_symptomatic_iou": round(iou, 4),
        }

    def predict_full_scene(
        self,
        reduced_cube: np.ndarray,
        patch_size: int = 16,
        stride: int = 8,
    ) -> np.ndarray:
        """Run sliding-window inference across full (D, H, W) raster cube with overlap accumulation."""
        self.model.eval()
        D, H, W = reduced_cube.shape

        risk_accumulator = np.zeros((H, W), dtype=np.float32)
        weight_accumulator = np.zeros((H, W), dtype=np.float32)

        # 2D Gaussian blend window for smooth tile seams
        gy, gx = np.ogrid[:patch_size, :patch_size]
        center = (patch_size - 1) / 2.0
        sigma = patch_size / 3.0
        gaussian_weight = np.exp(-((gy - center) ** 2 + (gx - center) ** 2) / (2 * sigma ** 2)).astype(np.float32)

        # Collect patches
        y_steps = list(range(0, H - patch_size + 1, stride))
        if y_steps[-1] != H - patch_size:
            y_steps.append(H - patch_size)

        x_steps = list(range(0, W - patch_size + 1, stride))
        if x_steps[-1] != W - patch_size:
            x_steps.append(W - patch_size)

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    patch = reduced_cube[:, y : y + patch_size, x : x + patch_size]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)

                    seg_logits, _ = self.model(patch_tensor)
                    probs = F.softmax(seg_logits, dim=1)
                    # Class 1: Pre-symptomatic risk
                    risk_patch = probs[0, 1].cpu().numpy()

                    risk_accumulator[y : y + patch_size, x : x + patch_size] += risk_patch * gaussian_weight
                    weight_accumulator[y : y + patch_size, x : x + patch_size] += gaussian_weight

        # Normalize by overlapping weights
        valid_mask = weight_accumulator > 0
        risk_map = np.zeros_like(risk_accumulator)
        risk_map[valid_mask] = risk_accumulator[valid_mask] / weight_accumulator[valid_mask]

        return np.clip(risk_map, 0.0, 1.0)

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0.0)
