"""Geospatial Affine Georeferencing and Deck.gl Heatmap Engine.

Handles sub-meter affine coordinate transformations between raster pixel grid space (y, x)
and EPSG:4326 (WGS84) latitude/longitude space for the 1,000-acre TerraSpectra Valley Farm,
with Deck.gl RGBA color mapped tensor outputs.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from app.core.config import settings


class AffineGeoTransform:
    """Affine coordinate projection between pixel indices and WGS84 coordinates."""

    def __init__(
        self,
        center_lat: float = settings.FARM_LAT,
        center_lon: float = settings.FARM_LON,
        delta_lat: float = 0.018,  # ~2.0 km latitude span
        delta_lon: float = 0.022,  # ~2.0 km longitude span at 36 deg latitude
        height_px: int = 32,
        width_px: int = 32,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.delta_lat = delta_lat
        self.delta_lon = delta_lon
        self.height_px = height_px
        self.width_px = width_px

        self.min_lon = center_lon - delta_lon / 2.0
        self.max_lon = center_lon + delta_lon / 2.0
        self.min_lat = center_lat - delta_lat / 2.0
        self.max_lat = center_lat + delta_lat / 2.0

        # Pixel resolution in degrees
        self.res_x = delta_lon / width_px
        self.res_y = delta_lat / height_px

    def pixel_to_geo(self, y: float, x: float) -> Tuple[float, float]:
        """Convert pixel (row y, col x) to (longitude, latitude) WGS84."""
        # y=0 is north (max_lat), y=height is south (min_lat)
        lon = self.min_lon + (x + 0.5) * self.res_x
        lat = self.max_lat - (y + 0.5) * self.res_y
        return round(float(lon), 6), round(float(lat), 6)

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert geographic (longitude, latitude) to nearest integer pixel (y, x)."""
        x = int(np.clip(np.floor((lon - self.min_lon) / self.res_x), 0, self.width_px - 1))
        y = int(np.clip(np.floor((self.max_lat - lat) / self.res_y), 0, self.height_px - 1))
        return y, x

    def get_pixel_bounds_polygon(self, y: int, x: int) -> List[List[float]]:
        """Return 4-corner bounding polygon [[lon, lat], ...] for pixel (y, x)."""
        w = self.min_lon + x * self.res_x
        e = self.min_lon + (x + 1) * self.res_x
        n = self.max_lat - y * self.res_y
        s = self.max_lat - (y + 1) * self.res_y
        return [
            [round(w, 6), round(n, 6)],
            [round(e, 6), round(n, 6)],
            [round(e, 6), round(s, 6)],
            [round(w, 6), round(s, 6)],
            [round(w, 6), round(n, 6)],
        ]

    def validate_coordinate_containment(self, lon: float, lat: float) -> bool:
        """Check whether a coordinate pair lies inside the 1,000-acre farm boundary."""
        return (self.min_lon <= lon <= self.max_lon) and (self.min_lat <= lat <= self.max_lat)


class GeoreferencedHeatmapBuilder:
    """Converts 2D disease risk matrices into georeferenced Deck.gl visualization payloads."""

    def __init__(self, transform: Optional[AffineGeoTransform] = None):
        self.transform = transform or AffineGeoTransform()

    @staticmethod
    def risk_to_rgba(risk: float) -> List[int]:
        """Color mapping function for disease risk scores in [0.0, 1.0]."""
        if risk < 0.25:
            # Healthy vegetation: Emerald Green
            return [16, 185, 129, 90]
        elif risk < 0.40:
            # Low Caution: Cyan/Teal
            return [6, 182, 212, 130]
        elif risk < 0.80:
            # Pre-Symptomatic Fungal Warning (3 Weeks Early): Amber
            return [245, 158, 11, 210]
        else:
            # Critical / Advanced Necrotic Core: Crimson
            return [244, 63, 94, 255]

    def build_deckgl_points(self, risk_map: np.ndarray) -> List[Dict[str, Any]]:
        """Generate point array for Deck.gl HeatmapLayer / ScatterplotLayer."""
        H, W = risk_map.shape
        points = []

        for y in range(H):
            for x in range(W):
                risk = float(risk_map[y, x])
                lon, lat = self.transform.pixel_to_geo(y, x)
                rgba = self.risk_to_rgba(risk)

                points.append({
                    "coordinates": [lon, lat],
                    "risk": round(risk, 4),
                    "color": rgba,
                    "is_pre_symptomatic": bool(0.40 <= risk < 0.80),
                    "is_critical": bool(risk >= 0.80),
                    "grid_coord": [y, x],
                })

        return points

    def build_geojson_cells(self, risk_map: np.ndarray) -> Dict[str, Any]:
        """Generate GeoJSON Polygon features for parcel grid overlay."""
        H, W = risk_map.shape
        features = []

        for y in range(H):
            for x in range(W):
                risk = float(risk_map[y, x])
                # Only output features above threshold to keep payload lean
                if risk >= 0.25:
                    poly = self.transform.get_pixel_bounds_polygon(y, x)
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [poly],
                        },
                        "properties": {
                            "risk": round(risk, 4),
                            "color": self.risk_to_rgba(risk),
                            "severity": "pre_symptomatic" if risk < 0.80 else "critical",
                        },
                    })

        return {
            "type": "FeatureCollection",
            "total_cells": len(features),
            "features": features,
        }

