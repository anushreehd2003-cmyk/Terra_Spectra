"""4-Week Temporal Disease Progression and Forecasting Engine.

Simulates the biophysical progression of fungal pathogen spread across a 4-week timeline,
demonstrating how TerraSpectra detects infections 3 weeks before standard satellites
and human visual scouts notice chlorotic/necrotic foliar lesions.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from app.core.config import settings
from app.utils.spectral_physics import (
    generate_wavelength_grid,
    get_healthy_vegetation_signature,
    get_pre_symptomatic_fungal_signature,
    get_symptomatic_necrotic_signature,
)


class TemporalProgressionEngine:
    """Models multi-week fungal pathogen growth and spectral divergence over time."""

    TIMELINE_STAGES = [
        {
            "week_offset": -3,
            "label": "Week -3 (Early Detection Window)",
            "days_before_symptoms": 21,
            "visual_status": "Completely Vibrant Green (0% Visible Symptoms)",
            "rgb_visible_anomalies": False,
            "terraspectra_ai_alert": True,
            "chlorophyll_intact": True,
            "mesophyll_nir_drop_pct": 20.5,
            "recommended_action": "Apply Targeted Low-Dose Bio-fungicide & Adjust Irrigation",
            "treatment_cost_savings_pct": 78.0,
            "risk_acres": 265.4,
            "primary_mechanism": "Pathogen hyphae colonize spongy mesophyll air cavities; thylakoids intact.",
        },
        {
            "week_offset": -2,
            "label": "Week -2 (Pathogen Spread)",
            "days_before_symptoms": 14,
            "visual_status": "Vibrant Green (No Foliar Spots)",
            "rgb_visible_anomalies": False,
            "terraspectra_ai_alert": True,
            "chlorophyll_intact": True,
            "mesophyll_nir_drop_pct": 28.0,
            "recommended_action": "Isolate Affected Field Zones; Targeted Biocontrol Spray",
            "treatment_cost_savings_pct": 55.0,
            "risk_acres": 448.2,
            "primary_mechanism": "Fungal spore dispersal along moisture corridors; localized transpiration impairment.",
        },
        {
            "week_offset": -1,
            "label": "Week -1 (Pre-Symptomatic Crisis)",
            "days_before_symptoms": 7,
            "visual_status": "Micro-chlorosis (Faint, Invisible to Satellites)",
            "rgb_visible_anomalies": False,
            "terraspectra_ai_alert": True,
            "chlorophyll_intact": False,
            "mesophyll_nir_drop_pct": 36.5,
            "recommended_action": "Emergency Full-Canopy Treatment Needed",
            "treatment_cost_savings_pct": 28.0,
            "risk_acres": 682.0,
            "primary_mechanism": "Extensive cellular wall breakdown and initial chlorophyll breakdown.",
        },
        {
            "week_offset": 0,
            "label": "Week 0 (Visible Outbreak)",
            "days_before_symptoms": 0,
            "visual_status": "Widespread Chlorosis & Necrotic Brown Lesions",
            "rgb_visible_anomalies": True,
            "terraspectra_ai_alert": True,
            "chlorophyll_intact": False,
            "mesophyll_nir_drop_pct": 52.0,
            "recommended_action": "Damage Control / Harvest Rescue (Irreversible Foliar Loss)",
            "treatment_cost_savings_pct": 0.0,
            "risk_acres": 894.5,
            "primary_mechanism": "Full leaf tissue necrosis and chloroplast destruction; broad RGB NDVI drops.",
        },
    ]

    @classmethod
    def get_timeline_overview(cls) -> List[Dict[str, Any]]:
        """Return metadata summary for the 4-week timeline progression."""
        return cls.TIMELINE_STAGES

    @classmethod
    def generate_temporal_risk_sequence(
        cls,
        height: int = 32,
        width: int = 32,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate 2D spatial risk grids across each of the 4 timeline stages."""
        rng = np.random.default_rng(seed)
        sequence = []

        # 3 initial infection seed foci (cy, cx)
        foci = [(10, 12), (22, 20), (14, 25)]

        yy, xx = np.ogrid[:height, :width]

        for stage in cls.TIMELINE_STAGES:
            week = stage["week_offset"]
            # Growth dynamics: spread radius grows over time
            # Week -3: radius ~3.5; Week 0: radius ~9.0
            radius_growth = 3.5 + (week + 3) * 1.8
            intensity_growth = min(0.98, 0.65 + (week + 3) * 0.10)

            grid = np.zeros((height, width), dtype=np.float32)

            for cy, cx in foci:
                # Add slight random drift per focus
                dy = rng.uniform(-0.5, 0.5)
                dx = rng.uniform(-0.5, 0.5)
                dist_sq = (yy - (cy + dy)) ** 2 + (xx - (cx + dx)) ** 2
                focus_risk = intensity_growth * np.exp(-dist_sq / (2.0 * radius_growth ** 2))
                grid = np.maximum(grid, focus_risk.astype(np.float32))

            # Add micro-variation
            grid = np.clip(grid + rng.normal(0, 0.015, (height, width)).astype(np.float32), 0.0, 1.0)

            # Sample 4x4 matrix for fast JSON payload
            sample_grid = [
                [round(float(grid[r, c]), 3) for c in range(0, width, max(1, width // 4))]
                for r in range(0, height, max(1, height // 4))
            ]

            sequence.append({
                "stage": stage,
                "week_offset": week,
                "label": stage["label"],
                "days_before_symptoms": stage["days_before_symptoms"],
                "risk_acres": stage["risk_acres"],
                "rgb_visible": stage["rgb_visible_anomalies"],
                "mean_risk_score": round(float(np.mean(grid)), 3),
                "max_risk_score": round(float(np.max(grid)), 3),
                "grid_sample": sample_grid,
            })

        return sequence

