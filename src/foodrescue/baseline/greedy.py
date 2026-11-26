"""Greedy baseline: "phone calls, first come, first served".

What a merchant would do today without our system:
  * Charity: call the nearest charities one by one (at most `max_calls`). On the phone a charity
    says no if it has no room, no fridge for chilled food, or will be closed when the food arrives.
    Nobody asks what each charity needs most, and nobody spreads the food fairly.
  * Driver: ask the driver who can be at the shop first. Drivers do one donation at a time
    (shop -> charity), never combining several pickups in one trip.
  * Disruption: no replanning. If a driver cancels, what they had not delivered yet is lost. If a merchant
    is late, the driver already sent waits; the food is lost if it then misses the charity's closing time.

To keep the comparison fair, the baseline uses the same safety rules and the same structured
donations as our system. The difference comes only from matching, routing and replanning.
"""

from __future__ import annotations

from dataclasses import dataclass

from foodrescue.config import Config, load_config
from foodrescue.models import Charity, DisruptionKind, Driver, Location, Scenario, Storage, TimeWindow
from foodrescue.optim.geo import road_km, travel_minutes
from foodrescue.safety.rules import Decision, RuleBook
from foodrescue.sim.metrics import Outcome, Status


@dataclass
class DriverState:
    driver: Driver
    position: Location
    free_at: int
    active: bool = True


@dataclass
class BaselineResult:
    outcomes: list[Outcome]
    km: float


def run_baseline(scenario: Scenario, config: Config | None = None, rulebook: RuleBook | None = None,
                 max_calls: int = 3) -> BaselineResult:
    config = config or load_config()
    rulebook = rulebook or RuleBook.load()
    geo, service = config.geo, config.routing.service_minutes

    def drive(a: Location, b: Location) -> int:
        return travel_minutes(a, b, road_factor=geo.road_factor, speed_kmh=geo.speed_kmh)

    merchants = {m.id: m for m in scenario.merchants}
    received = {c.id: 0.0 for c in scenario.charities}
    drivers = {d.id: DriverState(d, d.home, d.shift.start) for d in scenario.drivers}
    outcomes: dict[str, Outcome] = {}
    km = 0.0

    charities = {c.id: c for c in scenario.charities}
    pending_delays: dict[str, int] = {}
    events = [(o.time, 1, "offer", o) for o in scenario.offers]
    for d in scenario.disruptions:  # cancellations first at equal times, delays last
        events.append((d.time, 0, "cancel", d) if d.kind is DisruptionKind.DRIVER_CANCEL else (d.time, 2, "delay", d))
    for _, _, kind, event in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "cancel":
            state = drivers[event.target_id]
            state.active = False
            for outcome in outcomes.values():  # no replanning: undelivered food of this driver is lost
                if outcome.driver_id == event.target_id and outcome.delivered_at > event.time:
                    outcome.status, outcome.delivered_at, outcome.pickup_at = Status.DRIVER_CANCELLED, None, None
            continue
        if kind == "delay":
            outcome = outcomes.get(event.target_id)
            if outcome is None:  # not offered yet: the food will simply be ready later
                pending_delays[event.target_id] = pending_delays.get(event.target_id, 0) + event.minutes
            elif outcome.status is Status.DELIVERED and outcome.pickup_at > event.time:
                # The driver already sent waits at the shop. (Simplification in the baseline's favour: the
                # knock-on delay on that driver's later trips is ignored.)
                state, charity = drivers[outcome.driver_id], charities[outcome.charity_id]
                ready = event.time + event.minutes
                shift = max(0, ready - outcome.pickup_at)
                outcome.pickup_at += shift
                outcome.delivered_at += shift
                if outcome.delivered_at > min(charity.opening.end, state.driver.shift.end):
                    outcome.status, outcome.delivered_at, outcome.pickup_at = Status.NO_DRIVER, None, None
            continue

        offer, now = event, event.time
        donation = offer.donation
        if offer.id in pending_delays:
            window = donation.pickup_window
            start = max(window.start, now) + pending_delays.pop(offer.id)
            donation = donation.model_copy(update={"pickup_window": TimeWindow(start=start,
                                                                               end=max(window.end, start + 30))})
        verdict = rulebook.check(donation, now)
        kg = round(sum(i.kg for i in verdict.approved_items), 2)
        if verdict.decision is not Decision.APPROVE:
            status = Status.UNSAFE if verdict.decision is Decision.REJECT else Status.NEEDS_INFO
            outcomes[offer.id] = Outcome(donation_id=offer.id, kg=donation.total_kg if status is Status.UNSAFE else kg,
                                         status=status, offered_at=now)
            continue
        shop = merchants[offer.merchant_id].location

        # 1. Choose the driver who can be at the shop first.
        candidates = []
        for state in drivers.values():
            if state.active and state.driver.capacity_kg >= kg:
                arrival = max(state.free_at, now) + drive(state.position, shop)
                candidates.append((arrival, state.driver.id, state))
        if not candidates:
            outcomes[offer.id] = Outcome(donation_id=offer.id, kg=kg, status=Status.NO_DRIVER, offered_at=now)
            continue
        arrival, _, state = min(candidates, key=lambda c: (c[0], c[1]))
        pickup = max(arrival, donation.pickup_window.start)

        # 2. Phone the nearest charities until one says yes.
        charity = _first_charity_that_says_yes(scenario.charities, shop, donation.storage, kg, pickup + service,
                                               received, drive, road_factor=geo.road_factor, max_calls=max_calls)
        if charity is None:
            outcomes[offer.id] = Outcome(donation_id=offer.id, kg=kg, status=Status.NO_CHARITY, offered_at=now)
            continue
        delivery = pickup + service + drive(shop, charity.location)

        # 3. Is it still in time?
        if pickup > donation.pickup_window.end or delivery > min(charity.opening.end, state.driver.shift.end):
            outcomes[offer.id] = Outcome(donation_id=offer.id, kg=kg, status=Status.NO_DRIVER, offered_at=now,
                                         charity_id=charity.id)
            continue

        km += road_km(state.position, shop, geo.road_factor) + road_km(shop, charity.location, geo.road_factor)
        received[charity.id] += kg
        state.position, state.free_at = charity.location, delivery + service
        outcomes[offer.id] = Outcome(donation_id=offer.id, kg=kg, status=Status.DELIVERED, offered_at=now,
                                     pickup_at=pickup, delivered_at=delivery, charity_id=charity.id,
                                     driver_id=state.driver.id)
    return BaselineResult(outcomes=list(outcomes.values()), km=round(km, 2))


def _first_charity_that_says_yes(charities: list[Charity], shop: Location, storage: Storage, kg: float,
                                 earliest_departure: int, received: dict[str, float], drive, *,
                                 road_factor: float, max_calls: int) -> Charity | None:
    nearest = sorted(charities, key=lambda c: (road_km(shop, c.location, road_factor), c.id))
    for charity in nearest[:max_calls]:
        has_room = received[charity.id] + kg <= charity.capacity_kg
        has_fridge = charity.has_fridge or storage not in (Storage.CHILLED, Storage.FROZEN)
        open_on_arrival = earliest_departure + drive(shop, charity.location) <= charity.opening.end
        if has_room and has_fridge and open_on_arrival:
            return charity
    return None
