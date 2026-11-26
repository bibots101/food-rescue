"""Street-following lines for the map: each leg of a driver's trip is drawn along real streets.

Each leg asks two public OSRM routing services built on OpenStreetMap for a street path: the car one, and the
bike one, which may use the small streets of the Medina and does not drive around one-way systems. The map
draws the shorter of the two. In the centre the car path is often a detour (the Medina is closed to cars,
one-way streets, no U-turns): from Dar Slah to Association El Fourat, 240 m apart, the car path is 1.8 km
around the old town, the bike path 330 m. On long trips the car path along the expressways is the shorter.

Shapes are stored in `data/streets_cache/` (one small JSON file per leg and service), so the demo draws real
streets offline. Without the internet and without a stored shape, a leg is a straight line.

This only changes the drawing: it shows a plausible street path, not turn-by-turn car navigation. The
planning still estimates travel times from straight-line distance x 1.3 (see `optim/geo.py`).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from foodrescue.config import PROJECT_ROOT

CACHE = PROJECT_ROOT / "data" / "streets_cache"
SERVICES = {
    "car": "https://router.project-osrm.org/route/v1/driving",
    "bike": "https://routing.openstreetmap.de/routed-bike/route/v1/driving",
}
HEADERS = {"User-Agent": "foodrescue-academic-demo"}
RETRY_AFTER_S = 60  # after a failed request, stay offline for a minute instead of waiting on every leg
DETOUR_NEAR_M = 35  # back within this distance of an earlier point (covers both sides of a divided avenue)...
DETOUR_MIN_M = 120  # ...after driving at least this far: a detour, not a real part of the route

Point = tuple[float, float]  # (lat, lon)
_offline_until = 0.0


def _key(a: Point, b: Point, service: str = "car") -> str:
    text = f"{a[0]:.5f},{a[1]:.5f}->{b[0]:.5f},{b[1]:.5f}" + ("" if service == "car" else f"|{service}")
    return hashlib.sha1(text.encode()).hexdigest()[:20]


def _path(a: Point, b: Point, service: str) -> Path:
    return CACHE / f"{_key(a, b, service)}.json"


def cached(a: Point, b: Point) -> bool:
    """Both services' shapes are stored for this leg."""
    return all(_path(a, b, service).exists() for service in SERVICES)


def _metres(a: Point, b: Point) -> float:
    dy = (a[0] - b[0]) * 111_320
    dx = (a[1] - b[1]) * 111_320 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def length_m(points: list[Point]) -> float:
    return sum(_metres(p, q) for p, q in zip(points, points[1:], strict=False))


def without_detours(points: list[Point], near_m: float = DETOUR_NEAR_M, min_detour_m: float = DETOUR_MIN_M
                    ) -> list[Point]:
    """Remove the detours where a route goes away and comes back to (almost) the same spot.

    A routing service snaps each place to the nearest road and avoids U-turns in the middle of a street, so
    some legs first go the wrong way to the next turning point, or along one side of a divided avenue and back
    on the other. A shortest route never returns within `near_m` metres of a point it has passed after going
    more than `min_detour_m`, so that part is cut from the drawing."""
    if len(points) < 3:
        return list(points)
    along = [0.0]
    for p, q in zip(points, points[1:], strict=False):
        along.append(along[-1] + _metres(p, q))
    kept, i, last = [], 0, len(points) - 1
    while i <= last:
        kept.append(points[i])
        back = next((k for k in range(last, i + 1, -1)
                     if along[k] - along[i] > min_detour_m and _metres(points[i], points[k]) < near_m), None)
        i = (back if back is not None else i) + 1
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


@functools.lru_cache(maxsize=8192)
def _read_stored(path: Path) -> tuple[Point, ...]:
    raw = [tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))["points"]]
    return tuple(without_detours(raw))


def _shape(a: Point, b: Point, service: str, timeout: float) -> list[Point] | None:
    """One service's street path, stored or fetched; None when it is not available."""
    global _offline_until
    path = _path(a, b, service)
    if path.exists():
        return list(_read_stored(path))
    if os.environ.get("FOODRESCUE_OFFLINE_STREETS") or time.monotonic() < _offline_until:
        return None
    url = f"{SERVICES[service]}/{a[1]},{a[0]};{b[1]},{b[0]}?overview=full&geometries=geojson"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as response:
            answer = json.load(response)
        coordinates = answer["routes"][0]["geometry"]["coordinates"]  # [lon, lat] pairs
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError):
        _offline_until = time.monotonic() + RETRY_AFTER_S
        return None
    # The service starts and ends on the nearest street: join the places themselves to it.
    points = [a] + [(round(lat, 5), round(lon, 5)) for lon, lat in coordinates] + [b]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"from": a, "to": b, "service": service, "points": points}), encoding="utf-8")
    return without_detours(points)


def leg(a: Point, b: Point, timeout: float = 4.0) -> list[Point]:
    """Points along the streets from `a` to `b` (both included): the shorter of the car and bike paths."""
    shapes = [shape for service in SERVICES if (shape := _shape(a, b, service, timeout))]
    return min(shapes, key=length_m) if shapes else [a, b]


def along_streets(points: list[Point]) -> list[Point]:
    """A whole trip (home, stop, stop, ...) drawn along the streets."""
    if len(points) < 2:
        return list(points)
    line = [points[0]]
    for a, b in zip(points, points[1:], strict=False):
        if a != b:
            line += leg(a, b)[1:]
    return line
