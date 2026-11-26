"""All the tools agents can call, registered in one place."""

from __future__ import annotations

from pydantic import BaseModel

from foodrescue.config import Config
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import Location
from foodrescue.optim.geo import road_km, travel_minutes


class TravelInput(BaseModel):
    origin: Location
    destination: Location


def build_toolbox(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    geo = config.geo

    @registry.register("travel_time", "Driving time in minutes between two points.", TravelInput)
    def travel_time(args: TravelInput) -> int:
        return travel_minutes(args.origin, args.destination, road_factor=geo.road_factor,
                              speed_kmh=geo.speed_kmh)

    @registry.register("road_distance", "Approximate driving distance in km between two points.",
                       TravelInput)
    def road_distance(args: TravelInput) -> float:
        return round(road_km(args.origin, args.destination, geo.road_factor), 2)

    return registry
