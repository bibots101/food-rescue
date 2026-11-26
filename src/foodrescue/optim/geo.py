"""Distances and travel times at city scale.

No map API: straight-line (haversine) distance times a road factor, at an average speed.
This keeps the project offline and deterministic, which keeps simulated days reproducible.
"""

from __future__ import annotations

import math
import random

from foodrescue.models import Location

EARTH_RADIUS_KM = 6371.0

BoundingBox = tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)


def haversine_km(a: Location, b: Location) -> float:
    """Great-circle distance between two points, in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def road_km(a: Location, b: Location, road_factor: float = 1.3) -> float:
    """Approximate driving distance: roads are never a straight line."""
    return haversine_km(a, b) * road_factor


def travel_minutes(a: Location, b: Location, *, road_factor: float = 1.3,
                   speed_kmh: float = 25.0) -> int:
    """Driving time in whole minutes (rounded up; 0 for the same place)."""
    km = road_km(a, b, road_factor)
    return math.ceil(km / speed_kmh * 60) if km > 0 else 0


def random_location(rng: random.Random, bbox: BoundingBox) -> Location:
    min_lat, min_lon, max_lat, max_lon = bbox
    return Location(lat=round(rng.uniform(min_lat, max_lat), 5),
                    lon=round(rng.uniform(min_lon, max_lon), 5))
