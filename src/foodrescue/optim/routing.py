"""Vehicle routing with OR-Tools: the pickup-and-delivery problem with time windows (PDPTW).

Given donations to carry (pickup at a merchant, delivery to the winning charity) and available
drivers, find routes that:
  * pick up each donation before delivering it, with the same driver,
  * respect time windows: the merchant's pickup window and the charity's opening hours,
  * never exceed a vehicle's capacity (kg on board),
  * end within each driver's shift,
  * minimize total driving time.

When not everything fits, the solver may drop a donation, at a penalty that grows with its weight,
so it always prefers to rescue more kg. Dropped donations are reported, never silently lost.

Model details (for the report):
  * one node per vehicle start (its current position), one dummy end node per vehicle (routes end
    after the last delivery; the trip home is not counted), two nodes per donation;
  * "time" dimension: travel minutes + service minutes, with waiting allowed (slack);
  * "load" dimension: +kg at a pickup, -kg at the delivery (in tenths of kg, OR-Tools needs integers);
  * search: parallel cheapest insertion, then greedy-descent local search. No time-based
    metaheuristic, so the result is deterministic (the same day always gives the same plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from foodrescue.models import Location, Stop, StopKind
from foodrescue.optim.geo import road_km, travel_minutes

KG_SCALE = 10  # load is counted in tenths of kg


@dataclass(frozen=True)
class Task:
    """One donation to carry from a merchant to a charity."""

    donation_id: str
    pickup_place: str
    pickup: Location
    delivery_place: str
    delivery: Location
    kg: float
    pickup_window: tuple[int, int]
    delivery_window: tuple[int, int]
    keep_driver: str | None = None  # if set, only this driver may take it (stable plans)
    excluded_drivers: tuple[str, ...] = ()  # drivers who declined it or let the request expire


@dataclass(frozen=True)
class Vehicle:
    """A driver as the solver sees them: where and when they become available."""

    driver_id: str
    start: Location
    available_from: int
    shift_end: int
    capacity_kg: float


@dataclass
class RoutingSolution:
    routes: dict[str, list[Stop]] = field(default_factory=dict)
    distance_km: dict[str, float] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)  # donation ids no driver can serve in time


def solve_routes(tasks: list[Task], vehicles: list[Vehicle], *, now: int, road_factor: float = 1.3,
                 speed_kmh: float = 25.0, service_minutes: int = 5, time_limit_s: float = 2.0,
                 pickup_delay_cost: int = 1) -> RoutingSolution:
    solution = RoutingSolution(routes={v.driver_id: [] for v in vehicles},
                               distance_km={v.driver_id: 0.0 for v in vehicles})
    usable = [v for v in vehicles if max(v.available_from, now) < v.shift_end]

    # Donations whose windows are already impossible are dropped before solving.
    feasible: list[Task] = []
    for task in tasks:
        earliest_pickup = max(task.pickup_window[0], now)
        if not usable or earliest_pickup > task.pickup_window[1] or task.delivery_window[1] < earliest_pickup:
            solution.dropped.append(task.donation_id)
        else:
            feasible.append(task)
    if not feasible:
        return solution

    n_vehicles = len(usable)
    locations = [v.start for v in usable] + [v.start for v in usable]  # starts, then dummy ends
    for task in feasible:
        locations += [task.pickup, task.delivery]
    first_task_node = 2 * n_vehicles
    end_nodes = set(range(n_vehicles, first_task_node))

    def travel(a: int, b: int) -> int:
        if b in end_nodes or a in end_nodes:
            return 0  # the route ends after the last delivery
        return travel_minutes(locations[a], locations[b], road_factor=road_factor, speed_kmh=speed_kmh)

    travel_matrix = [[travel(a, b) for b in range(len(locations))] for a in range(len(locations))]
    service = [0] * first_task_node + [service_minutes] * (2 * len(feasible))
    load = [0] * first_task_node
    for task in feasible:
        kg = round(task.kg * KG_SCALE)
        load += [kg, -kg]

    manager = pywrapcp.RoutingIndexManager(len(locations), n_vehicles, list(range(n_vehicles)),
                                           list(range(n_vehicles, first_task_node)))
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        a, b = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return travel_matrix[a][b] + service[a]

    time_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_cb)
    horizon = max(v.shift_end for v in usable) + 1
    routing.AddDimension(time_cb, horizon, horizon, False, "time")  # slack = waiting is allowed
    time_dim = routing.GetDimensionOrDie("time")

    load_cb = routing.RegisterUnaryTransitCallback(lambda index: load[manager.IndexToNode(index)])
    routing.AddDimensionWithVehicleCapacity(load_cb, 0, [round(v.capacity_kg * KG_SCALE) for v in usable],
                                            True, "load")

    for v, vehicle in enumerate(usable):
        start = max(vehicle.available_from, now)
        time_dim.CumulVar(routing.Start(v)).SetRange(start, vehicle.shift_end)
        time_dim.CumulVar(routing.End(v)).SetRange(start, vehicle.shift_end)

    solver = routing.solver()
    vehicle_index = {vehicle.driver_id: v for v, vehicle in enumerate(usable)}
    for i, task in enumerate(feasible):
        pickup = manager.NodeToIndex(first_task_node + 2 * i)
        delivery = manager.NodeToIndex(first_task_node + 2 * i + 1)
        routing.AddPickupAndDelivery(pickup, delivery)
        solver.Add(routing.VehicleVar(pickup) == routing.VehicleVar(delivery))
        solver.Add(time_dim.CumulVar(pickup) <= time_dim.CumulVar(delivery))
        earliest = max(task.pickup_window[0], now)
        time_dim.CumulVar(pickup).SetRange(earliest, task.pickup_window[1])
        # Each minute of pickup delay costs as much as a minute of driving: early pickups keep food
        # fresh and keep drivers free for donations that have not been offered yet.
        time_dim.SetCumulVarSoftUpperBound(pickup, earliest, pickup_delay_cost)
        time_dim.CumulVar(delivery).SetRange(task.delivery_window[0], task.delivery_window[1])
        penalty = 100_000 + round(task.kg * 1_000)  # always worth serving; heavier donations first
        routing.AddDisjunction([pickup, delivery], penalty, 2)
        if task.keep_driver in vehicle_index:
            routing.VehicleVar(pickup).SetValues([-1, vehicle_index[task.keep_driver]])  # -1 = dropped
        excluded = [vehicle_index[d] for d in task.excluded_drivers if d in vehicle_index]
        if excluded:
            routing.VehicleVar(pickup).RemoveValues(excluded)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    assignment = routing.SolveWithParameters(params)
    if assignment is None:
        solution.dropped += [task.donation_id for task in feasible]
        return solution

    served: set[str] = set()
    for v, vehicle in enumerate(usable):
        index = assignment.Value(routing.NextVar(routing.Start(v)))
        position, km = vehicle.start, 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            task = feasible[(node - first_task_node) // 2]
            is_pickup = (node - first_task_node) % 2 == 0
            location = task.pickup if is_pickup else task.delivery
            km += road_km(position, location, road_factor)
            position = location
            solution.routes[vehicle.driver_id].append(Stop(
                kind=StopKind.PICKUP if is_pickup else StopKind.DELIVERY,
                donation_id=task.donation_id,
                place_id=task.pickup_place if is_pickup else task.delivery_place,
                location=location,
                eta=assignment.Min(time_dim.CumulVar(index)),
            ))
            served.add(task.donation_id)
            index = assignment.Value(routing.NextVar(index))
        solution.distance_km[vehicle.driver_id] = round(km, 2)
    solution.dropped += [task.donation_id for task in feasible if task.donation_id not in served]
    return solution
