"""FastAPI base API endpoints for TerraSpectra."""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/status")
async def get_system_status():
    """Return system readiness, configuration metadata, and farm coordinates."""
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "farm": {
            "name": settings.FARM_NAME,
            "center": [settings.FARM_LAT, settings.FARM_LON],
            "total_acres": settings.FARM_TOTAL_ACRES
        },
        "spectral_config": {
            "bands": settings.NUM_SPECTRAL_BANDS,
            "range_nm": settings.SPECTRAL_RANGE_NM,
            "pca_components": settings.PCA_COMPONENTS
        },
        "device": settings.DEVICE
    }
