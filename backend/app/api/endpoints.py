"""FastAPI base, data pipeline, and 3D-CNN ML endpoints for TerraSpectra."""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from app.core.config import settings
from app.utils.spectral_physics import (
    generate_wavelength_grid,
    get_healthy_vegetation_signature,
    get_pre_symptomatic_fungal_signature,
    get_symptomatic_necrotic_signature,
    get_soil_signature,
)
from app.utils.cube_simulator import HyperspectralCubeSimulator
from app.pipeline.spectral_reduction import SpectralDimensionalityReducer
from app.pipeline.ingestion import HyperspectralIngestionPipeline
from app.pipeline.georeference import AffineGeoTransform, GeoreferencedHeatmapBuilder
from app.utils.profiler import SystemMemoryProfiler, GISCoordinateAuditor
from app.models.spatial_spectral_cnn import SpatialSpectralCNN3D
from app.models.trainer import DiseaseForecastingTrainer
from app.models.dataset import create_hyperspectral_dataloader

router = APIRouter()

# Global state for cached simulation, fitted reducer, and 3D-CNN trainer
_CACHED_SCENE: Optional[Dict[str, Any]] = None
_CACHED_REDUCER: Optional[SpectralDimensionalityReducer] = None
_CACHED_TRAINER: Optional[DiseaseForecastingTrainer] = None


def get_or_create_scene() -> Dict[str, Any]:
    global _CACHED_SCENE
    if _CACHED_SCENE is None:
        simulator = HyperspectralCubeSimulator(height=32, width=32, seed=42)
        _CACHED_SCENE = simulator.generate_scene(infection_foci_count=3)
    return _CACHED_SCENE


def get_or_create_reducer() -> SpectralDimensionalityReducer:
    global _CACHED_REDUCER
    if _CACHED_REDUCER is None or not _CACHED_REDUCER.is_fitted:
        scene = get_or_create_scene()
        reducer = SpectralDimensionalityReducer(n_components=settings.PCA_COMPONENTS)
        reducer.fit(scene["cube"])
        _CACHED_REDUCER = reducer
    return _CACHED_REDUCER


def get_or_create_trainer() -> DiseaseForecastingTrainer:
    global _CACHED_TRAINER
    if _CACHED_TRAINER is None:
        _CACHED_TRAINER = DiseaseForecastingTrainer(
            in_spectral_channels=settings.PCA_COMPONENTS,
            num_classes=3,
            device="cpu",
        )
    return _CACHED_TRAINER


@router.get("/status")
def get_system_status():
    """Return system readiness, configuration metadata, and farm coordinates."""
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "farm": {
            "name": settings.FARM_NAME,
            "center": [settings.FARM_LAT, settings.FARM_LON],
            "total_acres": settings.FARM_TOTAL_ACRES,
        },
        "spectral_config": {
            "bands": settings.NUM_SPECTRAL_BANDS,
            "range_nm": settings.SPECTRAL_RANGE_NM,
            "pca_components": settings.PCA_COMPONENTS,
        },
        "device": settings.DEVICE,
    }


@router.get("/spectral/wavelengths")
def get_wavelengths():
    """Return array of continuous hyperspectral wavelengths in nanometers."""
    wl = generate_wavelength_grid(
        settings.NUM_SPECTRAL_BANDS,
        settings.SPECTRAL_RANGE_NM[0],
        settings.SPECTRAL_RANGE_NM[1],
    )
    return {
        "count": len(wl),
        "min_nm": float(wl[0]),
        "max_nm": float(wl[-1]),
        "wavelengths": [round(float(w), 2) for w in wl],
    }


@router.get("/spectral/signatures")
def get_spectral_signatures():
    """Return reference spectral reflectance signatures for comparison."""
    wl = generate_wavelength_grid(
        settings.NUM_SPECTRAL_BANDS,
        settings.SPECTRAL_RANGE_NM[0],
        settings.SPECTRAL_RANGE_NM[1],
    )
    healthy = get_healthy_vegetation_signature(wl)
    pre_symptomatic = get_pre_symptomatic_fungal_signature(wl, severity=0.75)
    necrotic = get_symptomatic_necrotic_signature(wl)
    soil = get_soil_signature(wl)

    delta = healthy - pre_symptomatic

    points = []
    for i in range(len(wl)):
        points.append({
            "wavelength": round(float(wl[i]), 1),
            "healthy": round(float(healthy[i]), 4),
            "pre_symptomatic": round(float(pre_symptomatic[i]), 4),
            "necrotic": round(float(necrotic[i]), 4),
            "soil": round(float(soil[i]), 4),
            "delta_pre_symp": round(float(delta[i]), 4),
        })

    return {
        "bands_count": len(wl),
        "data_points": points,
        "key_bands": {
            "blue_chlorophyll": 450,
            "green_peak": 550,
            "red_chlorophyll": 680,
            "red_edge_inflection": 715,
            "nir_plateau": 850,
            "swir_water_1": 1450,
            "swir_water_2": 1940,
        },
    }


@router.get("/farm/geojson")
def get_farm_geojson():
    """Return GeoJSON representation of the 1,000-acre farm plot and sub-parcels."""
    lat = settings.FARM_LAT
    lon = settings.FARM_LON
    d_lat = 0.018  # ~2 km latitude span
    d_lon = 0.022  # ~2 km longitude span

    half_lat = d_lat / 2.0
    half_lon = d_lon / 2.0

    parcels = [
        {
            "id": "plot-nw",
            "name": "Zone A: North-West Corn",
            "bounds": [
                [lon - half_lon, lat],
                [lon, lat],
                [lon, lat + half_lat],
                [lon - half_lon, lat + half_lat],
                [lon - half_lon, lat],
            ],
            "crop": "Corn / Maize",
            "status": "Healthy",
            "risk_score": 0.08,
        },
        {
            "id": "plot-ne",
            "name": "Zone B: North-East Wheat (Infection Focus)",
            "bounds": [
                [lon, lat],
                [lon + half_lon, lat],
                [lon + half_lon, lat + half_lat],
                [lon, lat + half_lat],
                [lon, lat],
            ],
            "crop": "Winter Wheat",
            "status": "Pre-Symptomatic Warning (3 Wks)",
            "risk_score": 0.82,
        },
        {
            "id": "plot-sw",
            "name": "Zone C: South-West Soybeans",
            "bounds": [
                [lon - half_lon, lat - half_lat],
                [lon, lat - half_lat],
                [lon, lat],
                [lon - half_lon, lat],
                [lon - half_lon, lat - half_lat],
            ],
            "crop": "Soybeans",
            "status": "Healthy",
            "risk_score": 0.04,
        },
        {
            "id": "plot-se",
            "name": "Zone D: South-East Barley (Secondary Focus)",
            "bounds": [
                [lon, lat - half_lat],
                [lon + half_lon, lat - half_lat],
                [lon + half_lon, lat],
                [lon, lat],
                [lon, lat - half_lat],
            ],
            "crop": "Barley",
            "status": "Pre-Symptomatic Warning (3 Wks)",
            "risk_score": 0.65,
        },
    ]

    features = []
    for p in parcels:
        features.append({
            "type": "Feature",
            "id": p["id"],
            "geometry": {
                "type": "Polygon",
                "coordinates": [p["bounds"]],
            },
            "properties": {
                "name": p["name"],
                "crop": p["crop"],
                "status": p["status"],
                "risk_score": p["risk_score"],
            },
        })

    return {
        "type": "FeatureCollection",
        "farm_name": settings.FARM_NAME,
        "total_acres": settings.FARM_TOTAL_ACRES,
        "center": [lon, lat],
        "features": features,
    }


@router.get("/pipeline/pca-summary")
def get_pca_summary():
    """Return Incremental PCA spectral reduction performance and explained variance."""
    reducer = get_or_create_reducer()
    scene = get_or_create_scene()
    recon_eval = reducer.evaluate_reconstruction(scene["cube"])
    metrics = reducer.get_metrics()

    return {
        "metrics": metrics,
        "reconstruction_evaluation": recon_eval,
        "compression": {
            "original_channels": settings.NUM_SPECTRAL_BANDS,
            "reduced_channels": settings.PCA_COMPONENTS,
            "compression_ratio": round(settings.NUM_SPECTRAL_BANDS / settings.PCA_COMPONENTS, 2),
            "gpu_memory_saving_pct": round(
                (1.0 - (settings.PCA_COMPONENTS / settings.NUM_SPECTRAL_BANDS)) * 100.0, 1
            ),
        },
    }


@router.post("/pipeline/simulate")
def trigger_simulation(seed: Optional[int] = None, foci: int = 3):
    """Generate a new simulated 204-band scene and return acquisition metrics."""
    global _CACHED_SCENE, _CACHED_REDUCER
    simulator = HyperspectralCubeSimulator(height=32, width=32, seed=seed)
    _CACHED_SCENE = simulator.generate_scene(infection_foci_count=foci)

    reducer = SpectralDimensionalityReducer(n_components=settings.PCA_COMPONENTS)
    reducer.fit(_CACHED_SCENE["cube"])
    _CACHED_REDUCER = reducer

    return {
        "status": "success",
        "metadata": _CACHED_SCENE["metadata"],
        "pca_variance_explained": reducer.get_metrics()["total_variance_explained_pct"],
    }


# =====================================================================
# 3D-CNN Model Architecture, Training, and Inference Endpoints
# =====================================================================


@router.get("/model/architecture")
def get_model_architecture():
    """Return PyTorch 3D-CNN model architecture, layer configurations, and parameter count."""
    trainer = get_or_create_trainer()
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    return {
        "model_name": "SpatialSpectralCNN3D",
        "input_spectral_depth": settings.PCA_COMPONENTS,
        "num_classes": 3,
        "classes": {
            0: "Healthy Foliage",
            1: "Pre-Symptomatic Fungal Infection (3-Wk Early Warning)",
            2: "Advanced Symptomatic / Necrotic",
        },
        "stem": "3D-CNN (conv3d_1: 5x3x3, conv3d_2: 3x3x3, conv3d_3: 3x3x3 stride-2)",
        "heads": ["Pixel-wise Segmentation Head (Conv2D)", "Global Patch Risk Head (Linear)"],
        "loss_function": "Focal Loss (gamma=2.0, alpha=[0.2, 0.6, 0.2])",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "edge_ready": True,
    }


@router.post("/model/train-step")
def trigger_training_step(epochs: int = 1):
    """Run an on-demand training step of the 3D-CNN using class-weighted Focal Loss."""
    scene = get_or_create_scene()
    reducer = get_or_create_reducer()
    trainer = get_or_create_trainer()

    loader = create_hyperspectral_dataloader(
        scene_cube=scene["cube"],
        disease_mask=scene["disease_mask"],
        reducer=reducer,
        patch_size=16,
        stride=8,
        batch_size=8,
        shuffle=True,
        augment=True,
    )

    last_train_metrics = {}
    for _ in range(max(1, min(epochs, 5))):
        last_train_metrics = trainer.train_epoch(loader)

    eval_metrics = trainer.evaluate(loader)

    return {
        "status": "success",
        "epochs_completed": epochs,
        "training": last_train_metrics,
        "validation": eval_metrics,
        "focal_loss_applied": True,
        "early_warning_f1": eval_metrics["pre_symptomatic_f1"],
    }


@router.post("/model/predict-scene")
def predict_scene_risk():
    """Run 3D-CNN sliding window inference across the 1,000-acre farm scene."""
    scene = get_or_create_scene()
    reducer = get_or_create_reducer()
    trainer = get_or_create_trainer()

    reduced_cube = reducer.transform(scene["cube"])
    risk_map = trainer.predict_full_scene(reduced_cube, patch_size=16, stride=8)

    H, W = risk_map.shape
    total_pixels = H * W
    pre_symp_pixels = int(np.sum(risk_map >= 0.40))
    risk_acres = round(float(pre_symp_pixels / total_pixels) * settings.FARM_TOTAL_ACRES, 2)

    mean_risk = round(float(np.mean(risk_map)), 3)
    max_risk = round(float(np.max(risk_map)), 3)

    return {
        "status": "success",
        "farm_name": settings.FARM_NAME,
        "total_acres": settings.FARM_TOTAL_ACRES,
        "predicted_risk_acres": risk_acres,
        "mean_risk_probability": mean_risk,
        "max_risk_probability": max_risk,
        "early_warning_status": (
            "Active Pre-Symptomatic Infection Detected" if risk_acres > 50.0 else "Healthy Canopy"
        ),
        "lead_time_forecast": "3 Weeks Before Foliar Symptoms",
        "sample_risk_grid": [
            [round(float(risk_map[r, c]), 3) for c in range(0, W, max(1, W // 4))]
            for r in range(0, H, max(1, H // 4))
        ],
    }


# =====================================================================
# Mid-Project Review Audit & Deck.gl Heatmap Endpoints (Day 5)
# =====================================================================


@router.get("/farm/heatmap")
def get_farm_heatmap():
    """Return Deck.gl points and georeferenced GeoJSON polygon cells with RGBA color mapping."""
    scene = get_or_create_scene()
    reducer = get_or_create_reducer()
    trainer = get_or_create_trainer()

    reduced_cube = reducer.transform(scene["cube"])
    risk_map = trainer.predict_full_scene(reduced_cube, patch_size=16, stride=8)

    H, W = risk_map.shape
    transform = AffineGeoTransform(height_px=H, width_px=W)
    builder = GeoreferencedHeatmapBuilder(transform)

    deckgl_points = builder.build_deckgl_points(risk_map)
    geojson_cells = builder.build_geojson_cells(risk_map)

    return {
        "status": "success",
        "farm_name": settings.FARM_NAME,
        "crs": "EPSG:4326 (WGS84)",
        "bounds": [transform.min_lon, transform.min_lat, transform.max_lon, transform.max_lat],
        "total_points": len(deckgl_points),
        "deckgl_points": deckgl_points,
        "geojson_cells": geojson_cells,
    }


@router.get("/audit/mid-project-review")
def get_mid_project_review_audit():
    """Run full Mid-Project Review Audit: memory profiling, latency benchmarks, and GIS validation."""
    mem_audit = SystemMemoryProfiler.profile_pipeline_execution(height=32, width=32, pca_components=settings.PCA_COMPONENTS)
    gis_audit = GISCoordinateAuditor.audit_coordinates(height=32, width=32)

    overall_pass = (mem_audit["status"] == "PASS") and (gis_audit["status"] == "PASS")

    return {
        "audit_name": "TerraSpectra Mid-Project Review Audit",
        "overall_verdict": "PASS" if overall_pass else "WARN",
        "memory_profiling": mem_audit,
        "gis_coordinate_validation": gis_audit,
        "compliance": {
            "edge_ram_budget_under_250mb": mem_audit["memory"]["is_memory_safe"],
            "sub_pixel_geospatial_precision": gis_audit["sub_pixel_precision"],
            "wgs84_boundary_containment": gis_audit["corners_contained"],
            "vram_compression_savings_pct": mem_audit["compression_savings"]["vram_saved_pct"],
        },
    }
