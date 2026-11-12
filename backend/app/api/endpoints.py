"""FastAPI base and data pipeline API endpoints for TerraSpectra."""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, List, Optional
import numpy as np

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

router = APIRouter()

# Global state for cached simulation / fitted reducer
_CACHED_SCENE: Optional[Dict[str, Any]] = None
_CACHED_REDUCER: Optional[SpectralDimensionalityReducer] = None


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
    """Return reference spectral reflectance signatures for comparison.

    Demonstrates how pre-symptomatic fungal infection diverges significantly in the
    Red-Edge (700-740nm) and NIR Plateau (750-1300nm) while remaining identical to healthy
    vegetation in the visible spectrum (400-680nm).
    """
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
    
    # Re-fit reducer
    reducer = SpectralDimensionalityReducer(n_components=settings.PCA_COMPONENTS)
    reducer.fit(_CACHED_SCENE["cube"])
    _CACHED_REDUCER = reducer

    return {
        "status": "success",
        "metadata": _CACHED_SCENE["metadata"],
        "pca_variance_explained": reducer.get_metrics()["total_variance_explained_pct"],
    }
