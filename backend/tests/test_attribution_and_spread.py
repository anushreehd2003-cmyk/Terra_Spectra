"""Unit tests for the newly added attribution and disease spread API endpoints.
These tests call the endpoint functions directly (no HTTP server) and verify
that a valid response structure is returned.
"""

import sys
import unittest

# Ensure the backend project path is on sys.path
sys.path.insert(0, "backend")

from app.api.endpoints import get_spectral_band_attribution, get_disease_spread_simulation

class TestAttributionAndSpread(unittest.TestCase):
    def test_spectral_attribution_basic(self):
        # Default target_class=1 (pre‑symptomatic)
        resp = get_spectral_band_attribution()
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "success")
        # Verify that a sampled per‑band list is present
        self.assertIn("per_band_scores_sampled", resp)
        self.assertGreater(len(resp["per_band_scores_sampled"]), 0)
        # Verify top diagnostic structure exists
        self.assertIn("top_diagnostic_bands", resp)
        self.assertGreater(len(resp["top_diagnostic_bands"]), 0)

    def test_spectral_attribution_target_range(self):
        # Edge cases: target_class below and above valid range
        for tc in [-5, 0, 2, 5]:
            resp = get_spectral_band_attribution(target_class=tc)
            self.assertEqual(resp.get("status"), "success")
            # target_class is clamped to [0,2]
            self.assertIn("target_class", resp)
            self.assertIn(resp["target_class"], [0,1,2])

    def test_disease_spread_default(self):
        resp = get_disease_spread_simulation()
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "success")
        # Check presence of key sections
        for key in ["final_state", "economic_impact", "daily_progression"]:
            self.assertIn(key, resp)
        # Daily progression should have entries for each simulated day
        total_days = resp.get("total_days_simulated")
        self.assertEqual(len(resp["daily_progression"]), total_days)
        # Verify that the first and last snapshots are present
        self.assertIn("first_day_snapshot", resp)
        self.assertIn("last_day_snapshot", resp)

    def test_disease_spread_custom_days(self):
        # Request a shorter simulation (10 days)
        resp = get_disease_spread_simulation(total_days=10)
        self.assertEqual(resp.get("total_days_simulated"), 10)
        self.assertEqual(len(resp["daily_progression"]), 10)

if __name__ == "__main__":
    unittest.main()
