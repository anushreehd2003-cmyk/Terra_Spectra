"""Synthetic Hyperspectral Cube Generator for TerraSpectra.

Generates realistic 204-band spatial-spectral cubes simulating a 1,000-acre agricultural terrain
with embedded pre-symptomatic fungal pathogen clusters, healthy vegetation canopy, field borders,
and ground-truth infection masks.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional
from app.core.config import settings
from app.utils.spectral_physics import (
    generate_wavelength_grid,
    get_healthy_vegetation_signature,
    get_pre_symptomatic_fungal_signature,
    get_symptomatic_necrotic_signature,
    get_soil_signature,
)


class HyperspectralCubeSimulator:
    """Generates synthetic multi-band geospatial raster data cubes for testing and model training."""

    def __init__(
        self,
        num_bands: int = settings.NUM_SPECTRAL_BANDS,
        spectral_range: Tuple[int, int] = settings.SPECTRAL_RANGE_NM,
        height: int = 128,
        width: int = 128,
        seed: Optional[int] = 42,
    ):
        self.num_bands = num_bands
        self.min_nm, self.max_nm = spectral_range
        self.height = height
        self.width = width
        self.seed = seed
        self.wavelengths = generate_wavelength_grid(num_bands, self.min_nm, self.max_nm)

        # Precompute base endmembers
        self.healthy_sig = get_healthy_vegetation_signature(self.wavelengths)
        self.infected_sig = get_pre_symptomatic_fungal_signature(self.wavelengths, severity=0.8)
        self.necrotic_sig = get_symptomatic_necrotic_signature(self.wavelengths)
        self.soil_sig = get_soil_signature(self.wavelengths)

    def generate_scene(
        self,
        infection_foci_count: int = 4,
        noise_std: float = 0.015,
    ) -> Dict[str, Any]:
        """Generate a complete hyperspectral scene with spatial layout and ground truth.

        Returns:
            dict containing:
                - 'cube': (num_bands, height, width) float32 reflectance array [0, 1]
                - 'disease_mask': (height, width) int32 ground truth labels (0=healthy, 1=pre-symptomatic, 2=necrotic, -1=soil)
                - 'risk_map': (height, width) float32 infection probability [0, 1]
                - 'metadata': geospatial and sensor metadata dictionary
        """
        rng = np.random.default_rng(self.seed)
        H, W, C = self.height, self.width, self.num_bands

        # 1. Base Agricultural Layout: Fields separated by access lanes/unplanted paths
        field_mask = np.ones((H, W), dtype=bool)
        # Perimeter borders
        field_mask[:3, :] = False
        field_mask[-3:, :] = False
        field_mask[:, :3] = False
        field_mask[:, -3:] = False
        # Cross farm dirt roads
        field_mask[H // 2 - 1 : H // 2 + 1, :] = False
        field_mask[:, W // 2 - 1 : W // 2 + 1] = False

        # 2. Initialize Risk Map (Infection Probability)
        risk_map = np.zeros((H, W), dtype=np.float32)
        disease_mask = np.zeros((H, W), dtype=np.int32)
        # Soil areas get -1
        disease_mask[~field_mask] = -1

        # 3. Seed Pre-Symptomatic Fungal Infection Foci (Elliptical Gaussian dispersal)
        foci_coords = []
        for _ in range(infection_foci_count):
            # Pick a center within healthy field regions
            cy = rng.integers(12, H - 12)
            cx = rng.integers(12, W - 12)
            sigma_y = rng.uniform(4.0, 9.0)
            sigma_x = rng.uniform(4.0, 9.0)
            peak_intensity = rng.uniform(0.65, 0.95)
            foci_coords.append((int(cy), int(cx), float(peak_intensity)))

            yy, xx = np.ogrid[:H, :W]
            gaussian = peak_intensity * np.exp(
                -(((yy - cy) ** 2) / (2 * sigma_y ** 2) + ((xx - cx) ** 2) / (2 * sigma_x ** 2))
            )
            # Accumulate risk
            risk_map = np.maximum(risk_map, gaussian.astype(np.float32))

        # Only apply risk inside active crop fields
        risk_map[~field_mask] = 0.0

        # Discretize ground truth:
        # > 0.40 probability is pre-symptomatic fungal infection
        # Small focal cores (> 0.90) may exhibit early necrotic tipping in advanced patches
        pre_symptomatic = (risk_map >= 0.35) & (risk_map < 0.88) & field_mask
        advanced_necrotic = (risk_map >= 0.88) & field_mask

        disease_mask[pre_symptomatic] = 1
        disease_mask[advanced_necrotic] = 2

        # 4. Synthesize Hyperspectral Cube (C, H, W)
        cube = np.zeros((C, H, W), dtype=np.float32)

        # Baseline: Healthy crop canopy with slight intra-field natural biomass variance
        canopy_variance = rng.normal(1.0, 0.04, size=(H, W)).astype(np.float32)

        for c in range(C):
            # Tile healthy signature modulated by biomass variance
            band_slice = self.healthy_sig[c] * canopy_variance

            # Blend Soil along dirt roads and perimeter
            band_slice[~field_mask] = self.soil_sig[c] * rng.normal(1.0, 0.03, size=(np.sum(~field_mask),))

            # Blend Pre-symptomatic infected signatures based on risk_map gradient
            infected_weight = risk_map
            # In pre-symptomatic zones, blend towards infected signature
            band_slice = (1.0 - infected_weight) * band_slice + infected_weight * self.infected_sig[c]

            # In necrotic cores, blend towards necrotic signature
            if np.any(advanced_necrotic):
                band_slice[advanced_necrotic] = self.necrotic_sig[c]

            cube[c] = band_slice

        # 5. Add Physical Sensor Noise (Simulating thermal noise & dark current)
        sensor_noise = rng.normal(0, noise_std, size=(C, H, W)).astype(np.float32)
        cube = np.clip(cube + sensor_noise, 0.005, 0.995)

        # 6. Geospatial Coordinates & Metadata
        # 1,000 acres ~= 4.047 km^2 ~= 2.01 km x 2.01 km
        # 128 pixels => ~15.7 meters per pixel
        center_lat = settings.FARM_LAT
        center_lon = settings.FARM_LON
        delta_lat = 0.018  # ~2 km latitude span
        delta_lon = 0.022  # ~2 km longitude span at 36 deg latitude

        bbox = [
            center_lon - delta_lon / 2.0,
            center_lat - delta_lat / 2.0,
            center_lon + delta_lon / 2.0,
            center_lat + delta_lat / 2.0,
        ]

        metadata = {
            "scene_name": f"{settings.FARM_NAME} - Acquisition Cycle 01",
            "farm_name": settings.FARM_NAME,
            "center": [center_lat, center_lon],
            "bbox_wgs84": bbox,
            "total_acres": settings.FARM_TOTAL_ACRES,
            "spatial_dimensions": [H, W],
            "pixel_resolution_meters": round(2011.0 / H, 2),
            "num_bands": C,
            "spectral_range_nm": [float(self.min_nm), float(self.max_nm)],
            "wavelengths_nm": self.wavelengths.tolist(),
            "foci_count": infection_foci_count,
            "foci_coordinates": foci_coords,
            "healthy_pixels": int(np.sum(disease_mask == 0)),
            "pre_symptomatic_pixels": int(np.sum(disease_mask == 1)),
            "necrotic_pixels": int(np.sum(disease_mask == 2)),
            "soil_pixels": int(np.sum(disease_mask == -1)),
            "infection_area_acres": round(float(np.sum(disease_mask == 1) / (H * W)) * settings.FARM_TOTAL_ACRES, 2),
        }

        return {
            "cube": cube,
            "disease_mask": disease_mask,
            "risk_map": risk_map,
            "metadata": metadata,
        }

    def save_scene(self, output_dir: str, prefix: str = "farm_cube_latest") -> Dict[str, str]:
        """Generate and save scene to compressed numpy npz and JSON metadata files."""
        os.makedirs(output_dir, exist_ok=True)
        data = self.generate_scene()

        npz_path = os.path.join(output_dir, f"{prefix}.npz")
        json_path = os.path.join(output_dir, f"{prefix}_meta.json")

        np.savez_compressed(
            npz_path,
            cube=data["cube"],
            disease_mask=data["disease_mask"],
            risk_map=data["risk_map"],
        )

        with open(json_path, "w") as f:
            json.dump(data["metadata"], f, indent=2)

        return {
            "npz_path": npz_path,
            "json_path": json_path,
            "metadata": data["metadata"],
        }

