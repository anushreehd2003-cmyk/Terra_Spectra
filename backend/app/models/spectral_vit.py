"""Vision Transformer (ViT) Spectral Self-Attention and Hybrid 3D-CNN + ViT Architecture.

Combines a 3D-CNN stem for localized spatial-spectral absorption gradients with a
Vision Transformer (ViT) Multi-Head Self-Attention encoder to capture non-local
spectral band interactions (e.g. Red-Edge 715nm correlations with SWIR water troughs 1450nm/1940nm).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List


class SpectralAttentionBlock(nn.Module):
    """Transformer Encoder block with Multi-Head Spectral Self-Attention (MHSA)."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (B, seq_len, embed_dim) where seq_len is spectral bands count.
            return_attention: Whether to return raw attention weight matrix.

        Returns:
            out: (B, seq_len, embed_dim)
            attn_weights: (B, num_heads, seq_len, seq_len) if return_attention else None
        """
        # Pre-LN Self-Attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class Hybrid3DCNNViT(nn.Module):
    """Hybrid Deep Learning Network combining 3D-CNN stem with Spectral Vision Transformer."""

    def __init__(
        self,
        in_spectral_channels: int = 30,
        num_classes: int = 3,  # 0: Healthy, 1: Pre-Symptomatic, 2: Necrotic
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_spectral_channels = in_spectral_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # 1. 3D-CNN Spatial-Spectral Stem (B, 1, D, H, W) -> (B, 32, D, H, W)
        self.stem_conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.stem_bn1 = nn.BatchNorm3d(16)
        self.stem_conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.stem_bn2 = nn.BatchNorm3d(32)

        # 2. Spectral Tokenizer: Project each spectral slice across spatial features into embed_dim
        # Tokenizer reduces spatial dimensions per spectral channel into a vector token
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.token_proj = nn.Linear(32, embed_dim)

        # Learnable 1D Spectral Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, in_spectral_channels, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Vision Transformer Encoder Layers (MHSA across spectral tokens)
        self.transformer_blocks = nn.ModuleList([
            SpectralAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=2.0,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Spatial-Spectral Fusion: Re-project transformer tokens to modulate spatial feature maps
        self.fusion_conv = nn.Conv2d(32 + embed_dim, 64, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(64)

        # 5. Dual Prediction Heads
        # Head A: Pixel-wise Segmentation (B, num_classes, H, W)
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        # Head B: Global Patch Risk Classification (B, num_classes)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass.
        Args:
            x: Input tensor (B, 1, D, H, W) or (B, D, H, W).
            return_attention: If True, returns self-attention weight tensors.

        Returns:
            seg_logits: (B, num_classes, H, W) pixel predictions
            patch_logits: (B, num_classes) patch classification
            attn_list: List of attention matrices per layer (or None)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {x.shape}")

        B, _, D, H, W = x.shape

        # Step 1: 3D-CNN Stem -> (B, 32, D, H, W)
        stem_out = self.act(self.stem_bn1(self.stem_conv1(x)))
        stem_out = self.act(self.stem_bn2(self.stem_conv2(stem_out)))

        # Step 2: Tokenize spectral dimension -> (B, D, embed_dim)
        # Reshape to (B * D, 32, H, W) then pool
        stem_perm = stem_out.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 32, H, W)
        pooled_spatial = self.spatial_pool(stem_perm).flatten(1)  # (B * D, 32)
        tokens = self.token_proj(pooled_spatial).view(B, D, self.embed_dim)  # (B, D, embed_dim)

        # Add positional embedding
        tokens = self.pos_drop(tokens + self.pos_embed[:, :D, :])

        # Step 3: Transformer Encoder with Self-Attention across spectral bands
        attn_list = [] if return_attention else None
        for block in self.transformer_blocks:
            tokens, attn = block(tokens, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_list.append(attn)

        tokens = self.norm(tokens)  # (B, D, embed_dim)

        # Step 4: Spatial-Spectral Fusion
        # Average transformer tokens across spectral depth -> global spectral context vector (B, embed_dim, 1, 1)
        global_spectral_ctx = tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, embed_dim, 1, 1)
        global_spectral_ctx = global_spectral_ctx.expand(-1, -1, H, W)  # Broadcast to (B, embed_dim, H, W)

        # Average 3D stem output along spectral dimension -> (B, 32, H, W)
        spatial_features = stem_out.mean(dim=2)  # (B, 32, H, W)

        # Concatenate 3D spatial stem features with global ViT spectral context
        fused = torch.cat([spatial_features, global_spectral_ctx], dim=1)  # (B, 32 + embed_dim, H, W)
        fused = self.act(self.fusion_bn(self.fusion_conv(fused)))  # (B, 64, H, W)

        # Step 5: Dual Output Heads
        seg_logits = self.seg_head(fused)  # (B, num_classes, H, W)
        patch_logits = self.cls_head(fused)  # (B, num_classes)

        return seg_logits, patch_logits, attn_list

    def extract_attention_matrix(self, x: torch.Tensor) -> np.ndarray:
        """Helper to extract normalized spectral band-to-band self-attention weights (D x D)."""
        self.eval()
        with torch.no_grad():
            _, _, attn_list = self.forward(x, return_attention=True)
            if not attn_list:
                return np.eye(self.in_spectral_channels, dtype=np.float32)

            # Take last transformer layer attention, average over batch and heads:
            # attn_list[-1] is (B, num_heads, D, D)
            last_layer_attn = attn_list[-1]
            avg_attn = last_layer_attn.mean(dim=(0, 1)).cpu().numpy()  # (D, D)
            return avg_attn.astype(np.float32)
