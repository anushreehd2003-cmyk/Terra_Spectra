"""3D-CNN Spatial-Spectral Neural Network for Hyperspectral Crop Disease Forecasting.

Captures contiguous spectral depth transitions across 30 PCA principal components
simultaneously with 2D spatial canopy textures to detect sub-visual pre-symptomatic fungal pathogens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class SpatialSpectralCNN3D(nn.Module):
    """Hybrid 3D Convolutional Network for simultaneous spatial and spectral pattern extraction."""

    def __init__(
        self,
        in_spectral_channels: int = 30,
        num_classes: int = 3,  # 0: Healthy, 1: Pre-Symptomatic, 2: Necrotic
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_spectral_channels = in_spectral_channels
        self.num_classes = num_classes

        # 3D Convolutional Stem (Input: B, 1, D, H, W)
        self.conv3d_1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(5, 3, 3),
            padding=(2, 1, 1),
            bias=False,
        )
        self.bn3d_1 = nn.BatchNorm3d(16)

        self.conv3d_2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(2, 1, 1),  # Downsamples spectral depth by 2, maintains spatial H, W
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn3d_3 = nn.BatchNorm3d(64)

        # Spectral Squeeze: Collapse spectral depth D -> 1
        self.spectral_pool = nn.AdaptiveAvgPool3d((1, None, None))

        # 2D Spatial Refinement & Segmentation Head (Output: B, num_classes, H, W)
        self.seg_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.seg_bn1 = nn.BatchNorm2d(32)
        self.seg_dropout = nn.Dropout2d(dropout)
        self.seg_classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # Global Patch Classification Head (Output: B, num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_classifier = nn.Linear(32, num_classes)

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, D, H, W) or (B, D, H, W)
               where D is spectral dimension (e.g. 30).

        Returns:
            seg_logits: (B, num_classes, H, W) pixel-wise class logits
            patch_logits: (B, num_classes) overall patch classification logits
        """
        # Ensure 5D tensor: (B, 1, D, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got {x.dim()}D with shape {x.shape}")

        B, _, D, H, W = x.shape

        # 3D Feature Extraction
        out = self.act(self.bn3d_1(self.conv3d_1(x)))
        out = self.act(self.bn3d_2(self.conv3d_2(out)))
        out = self.act(self.bn3d_3(self.conv3d_3(out)))

        # Squeeze spectral dimension: (B, 64, 1, H, W) -> (B, 64, H, W)
        squeezed = self.spectral_pool(out).squeeze(2)

        # 1. Pixel-wise Segmentation Head
        seg_feats = self.act(self.seg_bn1(self.seg_conv1(squeezed)))
        seg_feats = self.seg_dropout(seg_feats)
        seg_logits = self.seg_classifier(seg_feats)  # (B, num_classes, H, W)

        # 2. Global Patch Classification Head
        pooled = self.global_pool(squeezed).flatten(1)  # (B, 64)
        patch_feats = self.act(self.fc1(pooled))
        patch_feats = self.fc_dropout(patch_feats)
        patch_logits = self.fc_classifier(patch_feats)  # (B, num_classes)

        return seg_logits, patch_logits

    def predict_risk_map(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns pre-symptomatic disease probability map [0.0, 1.0]."""
        self.eval()
        with torch.no_grad():
            seg_logits, _ = self.forward(x)
            probs = F.softmax(seg_logits, dim=1)
            # Class index 1 is Pre-Symptomatic Fungal Infection
            pre_symptomatic_risk = probs[:, 1, :, :]
            return pre_symptomatic_risk

