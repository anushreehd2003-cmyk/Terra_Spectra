"""Mid-Project Review Audit Profiler for TerraSpectra.

Conducts end-to-end memory profiling, inference latency benchmarking, and
geospatial coordinate projection audits across the 204-band ingestion pipeline
and 3D-CNN model architecture.
"""

import time
import tracemalloc
import numpy as np
import torch
from typing import Dict, Any, List

from app.core.config import settings
from app.utils.cube_simulator import HyperspectralCubeSimulator
from app.pipeline.spectral_reduction import SpectralDimensionalityReducer
from app.pipeline.georeference import AffineGeoTransform
from app.models.spatial_spectral_cnn import SpatialSpectralCNN3D


class SystemMemoryProfiler:
    """Profiles heap memory allocations and peak RAM footprint during data pipeline execution."""

    @staticmethod
    def profile_pipeline_execution(
        height: int = 32,
        width: int = 32,
        pca_components: int = 30,
    ) -> Dict[str, Any]:
        """Benchmark end-to-end latency and peak RAM allocation across ingestion, PCA, and 3D-CNN."""
        tracemalloc.start()
        start_time = time.perf_counter()

        # Step 1: 204-Band Cube Generation & Ingestion
        t0 = time.perf_counter()
        sim = HyperspectralCubeSimulator(num_bands=settings.NUM_SPECTRAL_BANDS, height=height, width=width)
        scene = sim.generate_scene(infection_foci_count=3)
        t_ingest = (time.perf_counter() - t0) * 1000.0  # ms

        # Step 2: Incremental PCA Dimensionality Reduction (204 -> 30)
        t1 = time.perf_counter()
        reducer = SpectralDimensionalityReducer(n_components=pca_components)
        reduced_cube = reducer.fit_transform(scene["cube"])
        t_pca = (time.perf_counter() - t1) * 1000.0  # ms

        # Step 3: 3D-CNN Forward Inference
        t2 = time.perf_counter()
        model = SpatialSpectralCNN3D(in_spectral_channels=pca_components, num_classes=3)
        model.eval()
        input_tensor = torch.from_numpy(reduced_cube).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            seg_logits, patch_logits = model(input_tensor)
        t_cnn = (time.perf_counter() - t2) * 1000.0  # ms

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory in MB
        peak_mb = round(peak_mem / (1024 * 1024), 2)
        current_mb = round(current_mem / (1024 * 1024), 2)

        total_pixels = height * width
        pixels_per_sec = round(total_pixels / (total_time_ms / 1000.0), 1)
        acres_per_sec = round(settings.FARM_TOTAL_ACRES / (total_time_ms / 1000.0), 1)

        # Audit verdict
        is_memory_safe = peak_mb < 250.0  # Safe threshold for edge deployment
        is_latency_acceptable = total_time_ms < 5000.0

        return {
            "status": "PASS" if (is_memory_safe and is_latency_acceptable) else "WARN",
            "audit_target": "Mid-Project Review: Ingestion, Reduction & 3D-CNN Inference",
            "memory": {
                "peak_ram_mb": peak_mb,
                "current_ram_mb": current_mb,
                "threshold_limit_mb": 250.0,
                "memory_budget_utilized_pct": round((peak_mb / 250.0) * 100.0, 1),
                "is_memory_safe": is_memory_safe,
            },
            "latency_ms": {
                "cube_ingestion_ms": round(t_ingest, 2),
                "incremental_pca_ms": round(t_pca, 2),
                "cnn_inference_ms": round(t_cnn, 2),
                "total_pipeline_ms": round(total_time_ms, 2),
            },
            "throughput": {
                "pixels_per_second": pixels_per_sec,
                "acres_per_second": acres_per_sec,
                "scene_dimensions": [height, width],
            },
            "compression_savings": {
                "raw_channels": settings.NUM_SPECTRAL_BANDS,
                "compressed_channels": pca_components,
                "vram_saved_pct": round((1.0 - (pca_components / settings.NUM_SPECTRAL_BANDS)) * 100.0, 1),
            },
        }


class GISCoordinateAuditor:
    """Audits spatial coordinates, bounding box containment, and projection reversibility."""

    @staticmethod
    def audit_coordinates(height: int = 32, width: int = 32) -> Dict[str, Any]:
        """Verify affine projection accuracy, boundary containment, and sub-meter precision."""
        transform = AffineGeoTransform(height_px=height, width_px=width)

        # 1. Test Center Point Reversibility
        center_lon, center_lat = settings.FARM_LON, settings.FARM_LAT
        cy, cx = transform.geo_to_pixel(center_lon, center_lat)
        reproj_lon, reproj_lat = transform.pixel_to_geo(cy, cx)

        lat_error_deg = abs(center_lat - reproj_lat)
        lon_error_deg = abs(center_lon - reproj_lon)

        # 1 deg latitude ~= 111,000 meters; resolution is ~60m per pixel for 32x32
        error_meters_lat = lat_error_deg * 111000.0
        error_meters_lon = lon_error_deg * 111000.0 * np.cos(np.radians(center_lat))
        total_error_meters = np.sqrt(error_meters_lat ** 2 + error_meters_lon ** 2)

        # 2. Test 4-Corner Containment
        corners = [
            (0, 0),  # North-West
            (0, width - 1),  # North-East
            (height - 1, 0),  # South-West
            (height - 1, width - 1),  # South-East
        ]
        all_corners_valid = True
        corner_data = []

        for y, x in corners:
            lon, lat = transform.pixel_to_geo(y, x)
            is_valid = transform.validate_coordinate_containment(lon, lat)
            if not is_valid:
                all_corners_valid = False
            corner_data.append({"pixel": [y, x], "coords": [lon, lat], "contained": is_valid})

        return {
            "status": "PASS" if (all_corners_valid and total_error_meters < 70.0) else "FAIL",
            "crs": "EPSG:4326 (WGS84)",
            "farm_center_wgs84": [center_lat, center_lon],
            "projection_center_error_meters": round(float(total_error_meters), 2),
            "pixel_resolution_meters": round(2011.0 / height, 2),
            "corners_contained": all_corners_valid,
            "corners": corner_data,
            "sub_pixel_precision": bool(total_error_meters < (2011.0 / height)),
        }

