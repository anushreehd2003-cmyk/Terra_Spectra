"""PyTorch Dataset and Data Loader for Hyperspectral 3D Patches.

Slices 30-channel PCA cubes into overlapping 3D spatial-spectral patches (1, D, H, W)
with spatial-spectral augmentations (flips, rotations, Gaussian spectral jitter).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any

from app.core.config import settings
from app.utils.cube_simulator import HyperspectralCubeSimulator
from app.pipeline.spectral_reduction import SpectralDimensionalityReducer


class HyperspectralPatchDataset(Dataset):
    """PyTorch Dataset extracting (1, D, patch_size, patch_size) spatial-spectral patches."""

    def __init__(
        self,
        reduced_cube: np.ndarray,
        disease_mask: np.ndarray,
        patch_size: int = 16,
        stride: int = 8,
        augment: bool = True,
    ):
        """
        Args:
            reduced_cube: (D, H, W) float32 PCA-reduced cube where D = 30.
            disease_mask: (H, W) int32 ground truth labels (0=healthy, 1=pre-symptomatic, 2=necrotic, -1=soil).
            patch_size: Spatial width and height of each patch.
            stride: Step size for overlapping sliding window.
            augment: Whether to apply random spatial and spectral augmentations.
        """
        self.patch_size = patch_size
        self.augment = augment
        self.patches: List[np.ndarray] = []
        self.mask_patches: List[np.ndarray] = []
        self.patch_labels: List[int] = []

        D, H, W = reduced_cube.shape

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch_cube = reduced_cube[:, y : y + patch_size, x : x + patch_size]
                patch_mask = disease_mask[y : y + patch_size, x : x + patch_size]

                # Classify patch label based on predominant disease presence:
                # If any pre-symptomatic pixels (1) exist, label is 1 (high priority alert)
                if np.any(patch_mask == 1):
                    label = 1
                elif np.any(patch_mask == 2):
                    label = 2
                else:
                    label = 0

                self.patches.append(patch_cube)
                self.mask_patches.append(patch_mask)
                self.patch_labels.append(label)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            patch_tensor: (1, D, patch_size, patch_size) float32 tensor
            mask_tensor: (patch_size, patch_size) int64 tensor (-1 mapped to 0 or ignored)
            label_tensor: int64 patch-level class label
        """
        patch = self.patches[idx].copy()
        mask = self.mask_patches[idx].copy()
        label = self.patch_labels[idx]

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            # Random vertical flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            # Random 90 degree rotation
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                patch = np.rot90(patch, k=rot_k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=rot_k, axes=(0, 1)).copy()

            # Slight spectral Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.005, size=patch.shape).astype(np.float32)
                patch = patch + noise

        # Replace soil (-1) with ignore index (e.g. 0 or -100 for loss)
        # Here we map soil (-1) to 0 (non-diseased) for standard segmentation
        mask_clean = np.where(mask < 0, 0, mask)

        # Shape: (1, D, H, W) for 3D CNN
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_clean).long()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return patch_tensor, mask_tensor, label_tensor


def create_hyperspectral_dataloader(
    scene_cube: np.ndarray,
    disease_mask: np.ndarray,
    reducer: SpectralDimensionalityReducer,
    patch_size: int = 16,
    stride: int = 8,
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = True,
) -> DataLoader:
    """Helper to reduce a raw 204-band cube and create a PyTorch DataLoader."""
    reduced_cube = reducer.transform(scene_cube)
    dataset = HyperspectralPatchDataset(
        reduced_cube=reduced_cube,
        disease_mask=disease_mask,
        patch_size=patch_size,
        stride=stride,
        augment=augment,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

