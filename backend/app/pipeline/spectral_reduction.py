"""Incremental PCA Spectral Dimensionality Reduction Engine for Hyperspectral Cubes.

Reduces multi-band continuous hyperspectral rasters (e.g., 204 bands) down to K orthogonal principal
components (default 30 components) to eliminate spectral multicollinearity while preserving >95%
explained spectral variance for 3D-CNN and Vision Transformer architectures.
"""

import os
import joblib
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.decomposition import IncrementalPCA
from app.core.config import settings


class SpectralDimensionalityReducer:
    """Memory-efficient Incremental PCA reducer for 200+ band hyperspectral cubes."""

    def __init__(self, n_components: int = settings.PCA_COMPONENTS, batch_size: Optional[int] = 512):
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.is_fitted = False
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, cube: np.ndarray) -> "SpectralDimensionalityReducer":
        """Fit IncrementalPCA across a hyperspectral raster cube.

        Args:
            cube: Array of shape (C, H, W) or (N, C) containing reflectance values in [0, 1].
        """
        pixels = self._prepare_pixels(cube)
        self.ipca.fit(pixels)
        self.is_fitted = True
        self.cumulative_variance_ratio_ = np.cumsum(self.ipca.explained_variance_ratio_)
        return self

    def partial_fit(self, cube_batch: np.ndarray) -> "SpectralDimensionalityReducer":
        """Incrementally fit on chunks of pixels or spatial tiles without full memory residency."""
        pixels = self._prepare_pixels(cube_batch)
        self.ipca.partial_fit(pixels)
        self.is_fitted = True
        self.cumulative_variance_ratio_ = np.cumsum(self.ipca.explained_variance_ratio_)
        return self

    def transform(self, cube: np.ndarray) -> np.ndarray:
        """Transform (C, H, W) cube into (K, H, W) principal component cube.

        Args:
            cube: (C, H, W) float32 array.

        Returns:
            reduced_cube: (K, H, W) float32 array where K = n_components.
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer must be fitted before transforming cubes.")

        if cube.ndim == 3:
            C, H, W = cube.shape
            pixels = cube.reshape(C, -1).T  # (H*W, C)
            reduced_pixels = self.ipca.transform(pixels)  # (H*W, K)
            reduced_cube = reduced_pixels.T.reshape(self.n_components, H, W)
            return reduced_cube.astype(np.float32)
        elif cube.ndim == 2:
            return self.ipca.transform(cube).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D (N, C) or 3D (C, H, W) array, got shape {cube.shape}")

    def inverse_transform(self, reduced_cube: np.ndarray) -> np.ndarray:
        """Reconstruct original spectral band dimension (C, H, W) from (K, H, W)."""
        if not self.is_fitted:
            raise RuntimeError("Reducer must be fitted before inverse transform.")

        if reduced_cube.ndim == 3:
            K, H, W = reduced_cube.shape
            pixels = reduced_cube.reshape(K, -1).T  # (H*W, K)
            recon_pixels = self.ipca.inverse_transform(pixels)  # (H*W, C)
            C = self.ipca.n_features_in_
            return recon_pixels.T.reshape(C, H, W).astype(np.float32)
        else:
            return self.ipca.inverse_transform(reduced_cube).astype(np.float32)

    def fit_transform(self, cube: np.ndarray) -> np.ndarray:
        """Fit and transform in a single call."""
        self.fit(cube)
        return self.transform(cube)

    def evaluate_reconstruction(self, original_cube: np.ndarray) -> Dict[str, float]:
        """Compute Mean Squared Error (MSE) and Signal-to-Noise Ratio (PSNR) of PCA reconstruction."""
        reduced = self.transform(original_cube)
        reconstructed = self.inverse_transform(reduced)
        mse = float(np.mean((original_cube - reconstructed) ** 2))
        max_val = float(np.max(original_cube))
        psnr = float(20 * np.log10(max_val / np.sqrt(mse + 1e-10)))
        return {
            "mse": round(mse, 6),
            "psnr_db": round(psnr, 2),
            "cumulative_variance_explained": round(float(self.cumulative_variance_ratio_[-1]), 4),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return explained variance ratios, cumulative variance, and component count."""
        if not self.is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "n_components": int(self.n_components),
            "n_features_in": int(self.ipca.n_features_in_),
            "explained_variance_ratio": [round(float(v), 5) for v in self.ipca.explained_variance_ratio_],
            "cumulative_variance_ratio": [round(float(v), 5) for v in self.cumulative_variance_ratio_],
            "total_variance_explained_pct": round(float(self.cumulative_variance_ratio_[-1]) * 100.0, 2),
            "top_component_ratio": round(float(self.ipca.explained_variance_ratio_[0]), 4),
        }

    def save(self, file_path: str):
        """Save fitted model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> "SpectralDimensionalityReducer":
        """Load trained reducer from disk."""
        return joblib.load(file_path)

    def _prepare_pixels(self, data: np.ndarray) -> np.ndarray:
        """Convert input raster or batch into 2D pixel matrix (N_pixels, C_bands)."""
        if data.ndim == 3:
            C, H, W = data.shape
            return data.reshape(C, -1).T  # Shape: (H * W, C)
        elif data.ndim == 2:
            return data  # Already (N, C)
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}. Expected 2D or 3D.")

