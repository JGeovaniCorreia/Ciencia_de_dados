"""Geographic region assignment shared by the fairness and truncation reports.

Each block group is assigned to one of the three largest metropolitan areas by
haversine distance to the city centre, or to one of two remaining regions split
by latitude:

  - bay_area     : <= 80 km from San Francisco
  - grande_la    : <= 80 km from Los Angeles
  - san_diego    : <= 50 km from San Diego
  - demais_norte : remaining samples with Latitude >= 35.5
  - demais_sul   : remaining samples with Latitude <  35.5

Rules are applied in that order (the three radii do not overlap). The two remaining
regions are named "Demais" on purpose: each mixes coast and inland, rural areas and
smaller metros (north: Sacramento, Fresno, Central Valley, Monterey, north coast,
Sierra; south: Bakersfield, San Luis Obispo, Santa Barbara, Ventura, Inland Empire,
deserts). City centres are the same used by the model's dist_* features
(src.transformers.CaliforniaHousingTransformer), so the regions follow the
structure the model already uses — heterogeneity inside each metro is not segmented.

Why not median quadrants: Latitude and Longitude have correlation -0.92 in this
dataset (the state runs NW -> SE), so cutting both at the median yields two large
diagonal quadrants and two small off-diagonal slivers that do not match the
regions their names suggest (e.g. downtown LA and San Diego fell outside the
"south coast" quadrant).

The 80/50 km radii were the first configuration tried, with results in view; the
sensitivity to the radii is reported by reports/fairness_geo.py
(fairness_geo_sensitivity.csv).
"""

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.transformers import CaliforniaHousingTransformer  # noqa: E402

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
METRO_RADIUS_KM = 80.0
SAN_DIEGO_RADIUS_KM = 50.0
NORTH_SOUTH_LATITUDE = 35.5

_CITIES = CaliforniaHousingTransformer._CITIES

# Ordered: first matching rule wins — (key, (lat, lon))
METRO_CENTRES: list[tuple[str, tuple[float, float]]] = [
    ("bay_area", _CITIES["sf"]),
    ("grande_la", _CITIES["la"]),
    ("san_diego", _CITIES["sd"]),
]

REGION_LABELS: dict[str, str] = {
    "bay_area": "Bay Area",
    "grande_la": "Grande LA",
    "san_diego": "San Diego",
    "demais_norte": "Demais — Norte",
    "demais_sul": "Demais — Sul",
}

REGION_ORDER: list[str] = list(REGION_LABELS)


def metro_areas(
    metro_radius_km: float = METRO_RADIUS_KM,
    san_diego_radius_km: float = SAN_DIEGO_RADIUS_KM,
) -> list[tuple[str, tuple[float, float], float]]:
    """Return (key, (lat, lon), radius_km) for each metro area, in rule order.

    Args:
        metro_radius_km: Radius for the Bay Area and Greater LA.
        san_diego_radius_km: Radius for San Diego.

    Returns:
        List of (region key, city centre, radius in km).
    """
    return [
        (key, centre, san_diego_radius_km if key == "san_diego" else metro_radius_km)
        for key, centre in METRO_CENTRES
    ]


def haversine_km(
    lat: np.ndarray | pd.Series,
    lon: np.ndarray | pd.Series,
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """Great-circle distance in km from each point to a reference point.

    Args:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        ref_lat: Reference latitude in degrees.
        ref_lon: Reference longitude in degrees.

    Returns:
        Array of distances in km.
    """
    lat1 = np.radians(np.asarray(lat, dtype=float))
    lat2 = np.radians(ref_lat)
    dlat = lat2 - lat1
    dlon = np.radians(ref_lon - np.asarray(lon, dtype=float))
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def assign_regions(
    df: pd.DataFrame,
    metro_radius_km: float = METRO_RADIUS_KM,
    san_diego_radius_km: float = SAN_DIEGO_RADIUS_KM,
) -> pd.Series:
    """Assign each sample to a metro area or to one of the remaining regions.

    Args:
        df: DataFrame with 'Latitude' and 'Longitude' columns.
        metro_radius_km: Radius for the Bay Area and Greater LA (sensitivity runs).
        san_diego_radius_km: Radius for San Diego (sensitivity runs).

    Returns:
        pd.Series of region keys (see REGION_LABELS) aligned to df index.
    """
    areas = metro_areas(metro_radius_km, san_diego_radius_km)
    conditions = [
        haversine_km(df["Latitude"], df["Longitude"], lat, lon) <= radius
        for _, (lat, lon), radius in areas
    ]
    choices = [key for key, _, _ in areas]
    conditions.append((df["Latitude"] >= NORTH_SOUTH_LATITUDE).to_numpy())
    choices.append("demais_norte")

    return pd.Series(
        np.select(conditions, choices, default="demais_sul"),
        index=df.index,
        name="regiao",
    )
