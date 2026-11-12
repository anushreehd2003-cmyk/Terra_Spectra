"""Sanity tests for TerraSpectra project structure and configuration."""

import unittest
from app.core.config import settings


class TestSanityConfig(unittest.TestCase):
    def test_settings_loaded(self):
        self.assertEqual(settings.PROJECT_NAME, "TerraSpectra")
        self.assertEqual(settings.NUM_SPECTRAL_BANDS, 204)
        self.assertEqual(settings.FARM_TOTAL_ACRES, 1000.0)
        self.assertEqual(settings.PCA_COMPONENTS, 30)


if __name__ == "__main__":
    unittest.main()
