"""Configuration settings for TerraSpectra backend."""

import os
from typing import Tuple

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_NAME: str = "TerraSpectra"
        API_V1_STR: str = "/api/v1"
        DESCRIPTION: str = "Hyperspectral Crop Disease Forecasting Engine"
        VERSION: str = "0.1.0"
        
        # Target Farm Coordinates (Default: 1,000-acre agricultural test plot)
        FARM_NAME: str = "TerraSpectra Valley Farm"
        FARM_LAT: float = 36.3650
        FARM_LON: float = -120.4300
        FARM_TOTAL_ACRES: float = 1000.0
        
        # Hyperspectral Parameters
        NUM_SPECTRAL_BANDS: int = 204
        SPECTRAL_RANGE_NM: Tuple[int, int] = (400, 2400)
        PCA_COMPONENTS: int = 30
        
        # Model & Inference Defaults
        TILE_SIZE: int = 64
        TILE_STRIDE: int = 48
        DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

        class Config:
            env_file = ".env"
            case_sensitive = True

except ImportError:
    # Graceful fallback when running in lightweight environments before pip install
    class Settings:  # type: ignore
        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "TerraSpectra")
        API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
        DESCRIPTION: str = "Hyperspectral Crop Disease Forecasting Engine"
        VERSION: str = "0.1.0"
        
        FARM_NAME: str = os.getenv("FARM_NAME", "TerraSpectra Valley Farm")
        FARM_LAT: float = float(os.getenv("FARM_LAT", "36.3650"))
        FARM_LON: float = float(os.getenv("FARM_LON", "-120.4300"))
        FARM_TOTAL_ACRES: float = float(os.getenv("FARM_TOTAL_ACRES", "1000.0"))
        
        NUM_SPECTRAL_BANDS: int = int(os.getenv("NUM_SPECTRAL_BANDS", "204"))
        SPECTRAL_RANGE_NM: Tuple[int, int] = (400, 2400)
        PCA_COMPONENTS: int = int(os.getenv("PCA_COMPONENTS", "30"))
        
        TILE_SIZE: int = int(os.getenv("TILE_SIZE", "64"))
        TILE_STRIDE: int = int(os.getenv("TILE_STRIDE", "48"))
        DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


settings = Settings()
