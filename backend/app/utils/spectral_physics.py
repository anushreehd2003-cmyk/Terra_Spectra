"""Spectral Physics and Endmember Reflectance Models for Hyperspectral Sensing.

Models realistic foliar reflectance profiles across 400 nm - 2400 nm:
- Healthy Vegetation (chlorophyll a/b absorption, steep red-edge, high NIR plateau, SWIR water absorption)
- Pre-symptomatic Fungal Infection (sub-visual cellular damage: mesophyll cavity collapse dropping NIR by 15-25%,
  subtle red-edge blue shift, while VISIBLE spectrum 400-680nm remains virtually indistinguishable from healthy)
- Advanced Symptomatic / Necrotic (chlorosis, breakdown of chlorophyll, browned tissue)
- Agricultural Soil Background (gradual monotonic slope from VIS to SWIR)
"""

import numpy as np
from typing import Tuple, Dict


def generate_wavelength_grid(num_bands: int = 204, min_nm: int = 400, max_nm: int = 2400) -> np.ndarray:
    """Generate monotonically increasing wavelength array in nanometers."""
    return np.linspace(min_nm, max_nm, num_bands, dtype=np.float32)


def get_healthy_vegetation_signature(wavelengths: np.ndarray) -> np.ndarray:
    """Simulate canonical healthy green canopy reflectance curve (PROSPECT/SAIL physics model approximation).

    Physics characteristics:
    - 400-500 nm (Blue): Strong chlorophyll a/b and carotenoid absorption (reflectance 0.04 - 0.08)
    - 550 nm (Green Peak): Slight reflection bump around 0.12 - 0.15
    - 650-680 nm (Red): Strong chlorophyll a absorption trough (reflectance 0.05 - 0.07)
    - 690-740 nm (Red Edge): Extremely steep inflection slope as light ceases to be absorbed by pigments
    - 750-1300 nm (NIR Plateau): Multiple scattering in spongy mesophyll air cavities (reflectance 0.50 - 0.60)
    - 1450 nm & 1940 nm (SWIR): Deep liquid water absorption bands
    """
    wl = wavelengths
    refl = np.zeros_like(wl, dtype=np.float32)

    for i, w in enumerate(wl):
        if w < 500:
            # Blue: Strong chlorophyll absorption
            r = 0.05 + 0.03 * np.exp(-((w - 450) ** 2) / (2 * 30 ** 2))
        elif w < 600:
            # Green peak (~550nm)
            r = 0.06 + 0.08 * np.exp(-((w - 550) ** 2) / (2 * 25 ** 2))
        elif w < 685:
            # Red trough (strong Chl absorption at 660-680nm)
            r = 0.05 + 0.02 * np.exp(-((w - 620) ** 2) / (2 * 40 ** 2))
        elif w < 750:
            # Red Edge: Sigmoidal jump from 0.06 to 0.55
            # Inflection point ~715 nm
            t = (w - 715) / 15.0
            sigmoid = 1.0 / (1.0 + np.exp(-t))
            r = 0.06 + (0.54 - 0.06) * sigmoid
        elif w <= 1300:
            # NIR plateau: High reflectance due to spongy mesophyll scattering
            r = 0.54 - 0.04 * ((w - 750) / 550.0)
            # Minor water/oxygen absorption around 970nm & 1200nm
            r -= 0.04 * np.exp(-((w - 970) ** 2) / (2 * 25 ** 2))
            r -= 0.05 * np.exp(-((w - 1200) ** 2) / (2 * 35 ** 2))
        elif w <= 1800:
            # SWIR-1: Water absorption at 1450nm
            base = 0.38 - 0.10 * ((w - 1300) / 500.0)
            water_1450 = 0.22 * np.exp(-((w - 1450) ** 2) / (2 * 45 ** 2))
            r = max(0.04, base - water_1450)
        else:
            # SWIR-2: Strong water absorption at 1940nm, recovery up to 2200nm
            base = 0.25 - 0.08 * ((w - 1800) / 600.0)
            water_1940 = 0.18 * np.exp(-((w - 1940) ** 2) / (2 * 50 ** 2))
            cellulose_2100 = 0.05 * np.exp(-((w - 2100) ** 2) / (2 * 40 ** 2))
            r = max(0.03, base - water_1940 + cellulose_2100)

        refl[i] = np.clip(r, 0.01, 0.99)

    return refl


def get_pre_symptomatic_fungal_signature(wavelengths: np.ndarray, severity: float = 0.7) -> np.ndarray:
    """Simulate pre-symptomatic fungal infection reflectance profile.

    Crucial Biophysical Phenotype (3 Weeks Before Foliar Lesions):
    1. VISIBLE (400-680 nm): UNCHANGED (< 2% variance from healthy). Chlorophyll has not degraded yet;
       to human eyes and standard RGB / simple NDVI cameras, the leaves appear perfectly vibrant green.
    2. RED EDGE (695-730 nm): Slight blue-shift (shift of inflection point toward shorter wavelengths
       by ~5-8 nm) and slope flattening as early pathogen hyphae impair electron transport.
    3. NIR PLATEAU (750-1300 nm): Distinct 15-25% drop in reflectance caused by microscopic pathogen
       colonization collapsing the intercellular spongy mesophyll air cavities.
    4. SWIR WATER BANDS (1450 nm & 1940 nm): Moderate weakening of absorption troughs due to localized
       water stress and transpiration dysfunction.
    """
    severity = np.clip(severity, 0.1, 1.0)
    healthy = get_healthy_vegetation_signature(wavelengths)
    infected = healthy.copy()

    wl = wavelengths
    for i, w in enumerate(wl):
        if w < 685:
            # Visible spectrum remains identical to healthy to model sub-visual stage!
            # Micro-noise < 0.005 only
            infected[i] = healthy[i]
        elif w < 750:
            # Red Edge blue-shift & slight damping
            # Inflection shifts leftwards (e.g. from 715nm to 708nm)
            t_healthy = (w - 715) / 15.0
            t_shifted = (w - (715 - 7.0 * severity)) / (15.0 + 3.0 * severity)
            sig_healthy = 1.0 / (1.0 + np.exp(-t_healthy))
            sig_shifted = 1.0 / (1.0 + np.exp(-t_shifted))
            edge_healthy = 0.06 + (0.54 - 0.06) * sig_healthy
            edge_shifted = 0.06 + ((0.54 - 0.12 * severity) - 0.06) * sig_shifted
            infected[i] = edge_shifted
        elif w <= 1300:
            # NIR plateau drop: loss of cell wall reflectance
            nir_drop_factor = 1.0 - (0.24 * severity)
            infected[i] = healthy[i] * nir_drop_factor
        else:
            # SWIR water stress perturbation
            swir_shift = 0.03 * severity * np.exp(-((w - 1450) ** 2) / (2 * 60 ** 2))
            infected[i] = min(0.95, healthy[i] + swir_shift)

    return np.clip(infected, 0.01, 0.99)


def get_symptomatic_necrotic_signature(wavelengths: np.ndarray) -> np.ndarray:
    """Simulate advanced symptomatic necrotic / chlorotic canopy (post-visual outbreak)."""
    wl = wavelengths
    refl = np.zeros_like(wl, dtype=np.float32)

    for i, w in enumerate(wl):
        if w < 500:
            r = 0.10 + 0.02 * ((w - 400) / 100.0)
        elif w < 700:
            # High visible reflectance (yellow/brown leaf necrosis)
            r = 0.14 + 0.10 * ((w - 500) / 200.0)
        elif w <= 1300:
            # Collapsed NIR plateau: only 0.28 - 0.32 reflectance
            r = 0.30 - 0.04 * ((w - 700) / 600.0)
        else:
            # Desiccated tissue
            r = 0.26 + 0.05 * ((w - 1300) / 1100.0)

        refl[i] = np.clip(r, 0.01, 0.99)

    return refl


def get_soil_signature(wavelengths: np.ndarray) -> np.ndarray:
    """Simulate bare agricultural soil reflectance (smooth monotonic increase from visible to SWIR)."""
    wl = wavelengths
    # Soil curve steadily increases from ~0.08 at 400nm to ~0.35 at 2400nm
    refl = 0.08 + 0.27 * ((wl - 400.0) / 2000.0)
    # Mineral absorption around 2200nm (clay hydroxyl absorption)
    refl -= 0.04 * np.exp(-((wl - 2200) ** 2) / (2 * 60 ** 2))
    return np.clip(refl, 0.02, 0.95)

