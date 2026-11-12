"""Hyperspectral Raster Ingestion and Preprocessing Pipeline.

Provides windowed raster extraction, atmospheric band masking, spatial-spectral tiling,
and calculation of hyperspectral vegetation health indices (NDVI, NDRE, PRI, PSHI).
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from app.core.config import settings
from app.utils.spectral_physics import generate_wavelength_grid


class HyperspectralIngestionPipeline:
    """Manages ingestion, band calibration, index extraction, and windowing for hyperspectral cubes."""

    def __init__(
        self,
        num_bands: int = settings.NUM_SPECTRAL_BANDS,
        spectral_range: Tuple[int, int] = settings.SPECTRAL_RANGE_NM,
    ):
        self.num_bands = num_bands
        self.wavelengths = generate_wavelength_grid(num_bands, spectral_range[0], spectral_range[1])

    def load_cube_from_npz(self, npz_path: str, meta_path: Optional[str] = None) -> Dict[str, Any]:
        """Load compressed numpy hyperspectral cube and metadata."""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Cube file not found: {npz_path}")

        data = np.load(npz_path)
        cube = data["cube"].astype(np.float32)
        disease_mask = data["disease_mask"] if "disease_mask" in data else None
        risk_map = data["risk_map"] if "risk_map" in data else None

        metadata = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            default_meta_path = npz_path.replace(".npz", "_meta.json")
            if os.path.exists(default_meta_path):
                with open(default_meta_path, "r") as f:
                    metadata = json.load(f)

        return {
            "cube": self.calibrate_reflectance(cube),
            "disease_mask": disease_mask,
            "risk_map": risk_map,
            "metadata": metadata,
        }

    def calibrate_reflectance(self, cube: np.ndarray) -> np.ndarray:
        """Clip and clean reflectance values into physically valid [0.0, 1.0] domain."""
        clean_cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(clean_cube, 0.0, 1.0).astype(np.float32)

    def find_nearest_band_idx(self, target_wavelength_nm: float) -> int:
        """Find the index of the spectral channel closest to target wavelength in nm."""
        diffs = np.abs(self.wavelengths - target_wavelength_nm)
        return int(np.argmin(diffs))

    def extract_spectral_profile(self, cube: np.ndarray, y: int, x: int) -> Dict[str, Any]:
        """Extract continuous spectral reflectance curve at pixel (y, x)."""
        C, H, W = cube.shape
        if not (0 <= y < H and 0 <= x < W):
            raise IndexError(f"Coordinates ({y}, {x}) outside bounds ({H}, {W})")

        profile = cube[:, y, x].tolist()
        return {
            "y": y,
            "x": x,
            "wavelengths_nm": self.wavelengths.tolist(),
            "reflectance": [round(float(v), 4) for v in profile],
        }

    def compute_vegetation_indices(self, cube: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate hyperspectral narrowband physiological indices across the scene.

        Indices:
        1. NDVI (Normalized Difference Vegetation Index): (R800 - R680) / (R800 + R680)
           Measures overall green canopy biomass.
        2. NDRE (Normalized Difference Red Edge): (R790 - R720) / (R790 + R720)
           More sensitive to early chlorophyll changes than broad NDVI.
        3. PRI (Photochemical Reflectance Index): (R531 - R570) / (R531 + R570)
           Assesses photosynthetic light use efficiency and xanthophyll cycle.
        4. PSHI (Pre-Symptomatic Hyperspectral Index):
           Custom composite measuring the ratio of Red-Edge slope to NIR plateau scattering.
        """
        # Find nearest bands
        b_531 = self.find_nearest_band_idx(531)
        b_570 = self.find_nearest_band_idx(570)
        b_680 = self.find_nearest_band_idx(680)
        b_720 = self.find_nearest_band_idx(720)
        b_790 = self.find_nearest_band_idx(790)
        b_800 = self.find_nearest_band_idx(800)
        b_900 = self.find_nearest_band_idx(900)

        r_531 = cube[b_531]
        r_570 = cube[b_570]
        r_680 = cube[b_680]
        r_720 = cube[b_720]
        r_790 = cube[b_790]
        r_800 = cube[b_800]
        r_900 = cube[b_900]

        eps = 1e-6
        ndvi = (r_800 - r_680) / (r_800 + r_680 + eps)
        ndre = (r_790 - r_720) / (r_790 + r_720 + eps)
        pri = (r_531 - r_570) / (r_531 + r_570 + eps)

        # Pre-symptomatic index: healthy foliage maintains high NIR (900nm) and steep red-edge.
        # Early infection reduces mesophyll NIR and flattens red-edge slope.
        red_edge_slope = (r_790 - r_720)
        pshi = (r_900 * red_edge_slope) / (r_680 + 0.05)

        return {
            "ndvi": np.clip(ndvi, -1.0, 1.0).astype(np.float32),
            "ndre": np.clip(ndre, -1.0, 1.0).astype(np.float32),
            "pri": np.clip(pri, -1.0, 1.0).astype(np.float32),
            "pshi": np.clip(pshi, 0.0, 5.0).astype(np.float32),
        }

    def extract_tiles(
        self,
        cube: np.ndarray,
        tile_size: int = settings.TILE_SIZE,
        stride: int = settings.TILE_STRIDE,
    ) -> List[Dict[str, Any]]:
        """Slice large raster into overlapping sliding window tiles for 3D-CNN / ViT inference."""
        C, H, W = cube.shape
        tiles = []

        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                patch = cube[:, y : y + tile_size, x : x + tile_size]
                tiles.append({
                    "patch": patch,
                    "y_start": y,
                    "x_start": x,
                    "height": tile_size,
                    "width": tile_size,
                })

        return tiles

