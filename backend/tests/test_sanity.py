"""Sanity tests for TerraSpectra project structure and configuration."""

from app.core.config import settings


def test_settings_loaded():
    assert settings.PROJECT_NAME == "TerraSpectra"
    assert settings.NUM_SPECTRAL_BANDS == 204
    assert settings.FARM_TOTAL_ACRES == 1000.0
    assert settings.PCA_COMPONENTS == 30
