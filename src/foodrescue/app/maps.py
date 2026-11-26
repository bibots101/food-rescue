"""The route map (folium), shared by the overview and the driver app.

Places are told apart by their icon, not only by color: merchants show a fork and spoon, charities a
heart in a hand, drivers a truck with their name next to it. Routes use at most three calm colors
(validated together for color-blind readers); the truck's name label says whose route it is.
"""

from __future__ import annotations

import hashlib
import math

import folium
from streamlit_folium import st_folium

from foodrescue.app import icons, streets, ui
from foodrescue.core.clock import format_hhmm
from foodrescue.models import Stop, StopKind
from foodrescue.sim.runner import Simulation

ROUTE_COLORS = ["#1d4ed8", "#be185d", "#0891b2"]  # deep blue, raspberry, cyan: none of them is a road color
OFF_DUTY = "#71717a"
MERCHANT = icons.DARK


def driver_color(sim: Simulation, driver_id: str) -> str:
    ids = [d.id for d in sim.scenario.drivers]
    return ROUTE_COLORS[ids.index(driver_id) % len(ROUTE_COLORS)]


def _icon(markup: str, size: tuple[int, int], anchor: tuple[int, int]) -> folium.DivIcon:
    return folium.DivIcon(html=markup, icon_size=size, icon_anchor=anchor)


def _badge(inner: str, background: str, size: int, radius: str, border: str = "#fff") -> str:
    return (f'<div style="width:{size}px;height:{size}px;border-radius:{radius};background:{background};'
            f'border:2px solid {border};box-shadow:0 1px 4px rgba(0,0,0,.3);box-sizing:border-box;display:flex;'
            f'align-items:center;justify-content:center">{inner}</div>')


def merchant_badge(size: int = 26) -> str:
    return _badge(icons.fork_and_spoon(size - 10), MERCHANT, size, "50%")


def charity_badge(size: int = 28) -> str:
    return _badge(icons.heart_in_hand(size - 8), "#ffffff", size, "50%", border=MERCHANT)


def truck_badge(color: str, size: int = 30) -> str:
    return _badge(icons.truck(size - 10, cut=color), color, size, "9px")


def _stops(sim: Simulation, driver_id: str) -> list[Stop]:
    """The driver's stops, without deliveries they have not accepted yet."""
    jobs = sim.logistics.jobs
    return [s for s in sim.logistics.routes[driver_id].stops
            if not sim.logistics.confirmation or jobs[s.donation_id].confirmed]


def last_position(sim: Simulation, driver_id: str) -> tuple[float, float]:
    """Where the driver is now: their last stop done, a paused car, or home."""
    driver = next(d for d in sim.scenario.drivers if d.id == driver_id)
    if driver_id in sim.logistics.paused:
        car = sim.logistics.paused[driver_id][1]
        return (car.lat, car.lon)
    position = (driver.home.lat, driver.home.lon)
    for stop in _stops(sim, driver_id):
        if stop.eta is not None and stop.eta <= sim.now:
            position = (stop.location.lat, stop.location.lon)
    return position


def route_map(sim: Simulation, show_driver: str | None = None, only_places_on_route: bool = False,
              height: int = 440, part: str = "all") -> None:
    """All routes, or only `show_driver`'s (its stops are then numbered).

    `part`: "all" (the whole evening), "current" (from where the driver is, through the stops still to do) or
    "history" (the legs already driven).

    Two parts, so the map does not reload when something moves: the base map (tiles and the view) is drawn once
    and kept, and the places, routes and drivers are one layer that the map updates in place
    (`feature_group_to_add`). A driver's map follows the route by panning and zooming (`center`, `zoom`), so the
    base map is never redrawn while the evening runs."""
    scenario, now = sim.scenario, sim.now
    places = [*scenario.merchants, *scenario.charities]
    center = (sum(p.location.lat for p in places) / len(places), sum(p.location.lon for p in places) / len(places))
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", scroll_wheel_zoom=False)
    layer = folium.FeatureGroup(name="live")

    drivers = [d for d in scenario.drivers if show_driver is None or d.id == show_driver]
    trips = {}
    for driver in drivers:
        stops = _stops(sim, driver.id)
        done = [s for s in stops if s.eta is not None and s.eta <= now]
        todo = [s for s in stops if s not in done]
        position = last_position(sim, driver.id)
        home = (driver.home.lat, driver.home.lon)
        start, path = {"current": (position, todo), "history": (home, done), "all": (home, stops)}[part]
        trips[driver.id] = (start, path, position)
    on_route = {s.place_id for start, stops, _ in trips.values() for s in stops}
    # Zoom to what is shown: every place, or only the chosen driver's trip (its whole street path).
    shown: list[tuple[float, float]] = [] if show_driver else [(p.location.lat, p.location.lon) for p in places]
    for merchant in scenario.merchants:
        if not only_places_on_route or merchant.id in on_route:
            folium.Marker((merchant.location.lat, merchant.location.lon),
                          icon=_icon(merchant_badge(), (26, 26), (13, 13)),
                          tooltip=f"Merchant: {merchant.name}").add_to(layer)
    for charity in scenario.charities:
        if not only_places_on_route or charity.id in on_route:
            folium.Marker((charity.location.lat, charity.location.lon),
                          icon=_icon(charity_badge(), (28, 28), (14, 14)),
                          tooltip=f"Charity: {charity.name} (open {format_hhmm(charity.opening.start)}"
                                  f"-{format_hhmm(charity.opening.end)})").add_to(layer)

    for driver in drivers:
        active = driver.id in sim.logistics.active
        color = driver_color(sim, driver.id) if active else OFF_DUTY
        start, stops, position = trips[driver.id]
        points = [start] + [(s.location.lat, s.location.lon) for s in stops]
        if len(points) > 1:
            weight = 7 if show_driver else 5
            line = streets.along_streets(points)  # real roads (stored, so it also works offline)
            if show_driver:
                shown += line
            past = part == "history"
            folium.PolyLine(line, color="#ffffff", weight=weight + 4, opacity=0.9).add_to(layer)  # white casing
            folium.PolyLine(line, color=color, weight=weight, opacity=0.55 if past else 1.0,
                            dash_array=None if active else "8 10",
                            tooltip=f"{driver.name}{'' if active else ' (off duty)'}").add_to(layer)
        if show_driver:
            for number, stop in enumerate(stops, start=1):
                if part == "history":
                    verb = "Picked up at" if stop.kind is StopKind.PICKUP else "Delivered to"
                else:
                    verb = "Pick up at" if stop.kind is StopKind.PICKUP else "Deliver to"
                folium.Marker((stop.location.lat, stop.location.lon), icon=_icon(
                    f'<div style="width:18px;height:18px;border-radius:50%;background:{color};color:#fff;'
                    'font:700 10px/18px Inter,Segoe UI,sans-serif;text-align:center;border:2px solid #fff;'
                    f'box-sizing:content-box;transform:translate(12px,-12px)">{number}</div>', (22, 22), (11, 11)),
                    tooltip=f"{number}. {format_hhmm(stop.eta) if stop.eta is not None else ''} {verb} "
                            f"{stop.place_id}").add_to(layer)
        if part != "history":
            label = (f'<div style="display:flex;align-items:center;gap:4px">{truck_badge(color)}'
                     '<span style="background:#ffffff;color:#18181b;border-radius:6px;padding:1px 6px;'
                     'font:600 11px/16px Inter,Segoe UI,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.25);'
                     f'white-space:nowrap">{driver.name}</span></div>')
            folium.Marker(position, icon=_icon(label, (110, 30), (15, 15)), z_index_offset=1000,
                          tooltip=f"{driver.name} ({driver.id}): last known position").add_to(layer)
            if show_driver:
                shown.append(position)
    # The base map never changes for a given map (same key, same tiles and start view), so it is never reloaded:
    # the layer is swapped inside it, and a driver's map pans and zooms to the route (`center`, `zoom`).
    center_now, zoom_now = None, None
    if len(shown) > 1:
        south, west = min(p[0] for p in shown), min(p[1] for p in shown)
        north, east = max(p[0] for p in shown), max(p[1] for p in shown)
        if show_driver:
            center_now, zoom_now = ((south + north) / 2, (west + east) / 2), _fit_zoom(south, west, north, east,
                                                                                         height)
        else:  # every place: the same for the whole evening
            fmap.fit_bounds([[south, west], [north, east]], padding=(30, 30), max_zoom=16)
    view = repr((show_driver, only_places_on_route, part, height))
    st_folium(fmap, height=height, use_container_width=True, returned_objects=[], feature_group_to_add=layer,
              center=center_now, zoom=zoom_now, key="map-" + hashlib.md5(view.encode()).hexdigest()[:12])

    legend = [f"<span>{merchant_badge(20)}Merchant</span>", f"<span>{charity_badge(20)}Charity</span>"]
    legend += [f'<span><i class="fr-swatch" style="background:{driver_color(sim, d.id)}"></i>{ui.e(d.name)}</span>'
               for d in drivers]
    legend.append('<span><i class="fr-swatch dashed"></i>Off duty</span>')
    ui.html(f'<div class="fr-legend">{"".join(legend)}</div>')


def _fit_zoom(south: float, west: float, north: float, east: float, height: int, width: int = 560,
              padding: int = 40) -> int:
    """The web-map zoom level that shows the box (like Leaflet's fitBounds), for a map about `width` pixels wide."""
    lon_span = max(east - west, 1e-6)
    y = [math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) for lat in (south, north)]
    lat_span = max(y[1] - y[0], 1e-9)
    zoom_x = math.log2((width - 2 * padding) * 360 / (256 * lon_span))
    zoom_y = math.log2((height - 2 * padding) * 2 * math.pi / (256 * lat_span))
    return max(10, min(16, math.floor(min(zoom_x, zoom_y))))
