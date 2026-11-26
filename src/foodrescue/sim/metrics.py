"""Measuring a simulated day, the same way for our system and for the baseline.

Each donation ends with one `Outcome`. The day's metrics are computed from the outcomes only, so
both systems are judged by the same code.

Main metric: % of safe food rescued = kg delivered in time / kg that was safe to donate.
Food rejected as unsafe is reported separately, not counted as waste.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from statistics import mean

from pydantic import BaseModel

from foodrescue.models import DonationStatus, Performative, StopKind


class Status(StrEnum):
    DELIVERED = "delivered"
    UNSAFE = "unsafe"  # rejected by the safety check
    NEEDS_INFO = "needs_info"  # waiting for the merchant's answer
    NO_CHARITY = "no_charity"  # nobody could take it
    NO_DRIVER = "no_driver"  # nobody could carry it in time
    DRIVER_CANCELLED = "driver_cancelled"  # lost because a driver cancelled
    ON_THE_ROAD = "on_the_road"  # planned or picked up, not delivered yet (only before the evening is over)


class Outcome(BaseModel):
    donation_id: str
    kg: float
    status: Status
    offered_at: int
    pickup_at: int | None = None
    delivered_at: int | None = None
    charity_id: str | None = None
    driver_id: str | None = None


class DayMetrics(BaseModel):
    system: str
    offers: int
    safe_kg: float
    delivered_kg: float
    rescued_pct: float
    delivered: int
    lost_kg: dict[str, float]  # by status
    km: float
    km_per_kg: float | None
    mean_wait_min: float | None  # from the offer to the pickup


def compute_metrics(system: str, outcomes: list[Outcome], km: float) -> DayMetrics:
    safe = [o for o in outcomes if o.status is not Status.UNSAFE]
    delivered = [o for o in outcomes if o.status is Status.DELIVERED]
    safe_kg = round(sum(o.kg for o in safe), 2)
    delivered_kg = round(sum(o.kg for o in delivered), 2)
    lost: Counter[str] = Counter()
    for outcome in outcomes:
        if outcome.status is not Status.DELIVERED:
            lost[outcome.status.value] += outcome.kg
    return DayMetrics(
        system=system,
        offers=len(outcomes),
        safe_kg=safe_kg,
        delivered_kg=delivered_kg,
        rescued_pct=round(100 * delivered_kg / safe_kg, 1) if safe_kg else 0.0,
        delivered=len(delivered),
        lost_kg={status: round(kg, 2) for status, kg in sorted(lost.items())},
        km=round(km, 2),
        km_per_kg=round(km / delivered_kg, 3) if delivered_kg else None,
        mean_wait_min=round(mean(o.pickup_at - o.offered_at for o in delivered), 1) if delivered else None,
    )


def outcomes_from_simulation(result) -> list[Outcome]:  # result: SimulationResult (avoids an import cycle)
    """Read each donation's fate from the final routes and the message log.

    A donation counts as delivered only once the logistics agent has recorded the delivery (its planned time
    has passed). Read in the middle of the evening, planned or picked-up food is "on the road"."""
    offered = {offer.id: offer for offer in result.scenario.offers}
    stops = {(s.donation_id, s.kind): (driver_id, s) for driver_id, route in result.routes.items() for s in route.stops}
    status_from_messages: dict[str, Status] = {}
    for message in result.messages:
        from_safety = message.protocol == "safety-check" and message.sender == "safety"
        if from_safety and message.performative is Performative.REFUSE:
            status_from_messages[message.conversation_id] = Status.UNSAFE
        elif from_safety and message.performative is Performative.REQUEST:  # a question to the merchant
            status_from_messages[message.conversation_id] = Status.NEEDS_INFO
        elif message.protocol == "follow-up" and message.sender == "intake":  # quantity unknown
            status_from_messages[message.conversation_id] = Status.NEEDS_INFO
        elif message.protocol == "contract-net" and message.performative is Performative.FAILURE:
            status_from_messages[message.conversation_id] = Status.NO_CHARITY

    outcomes = []
    for donation_id, offer in offered.items():
        donation = result.donations.get(donation_id, offer.donation)
        kg = donation.total_kg if donation else 0.0
        delivery = stops.get((donation_id, StopKind.DELIVERY))
        if delivery is not None:
            driver_id, stop = delivery
            pickup = stops[(donation_id, StopKind.PICKUP)][1]
            delivered = donation is not None and donation.status is DonationStatus.DELIVERED
            outcomes.append(Outcome(donation_id=donation_id, kg=kg,
                                    status=Status.DELIVERED if delivered else Status.ON_THE_ROAD,
                                    offered_at=offer.time, pickup_at=pickup.eta,
                                    delivered_at=stop.eta if delivered else None, charity_id=stop.place_id,
                                    driver_id=driver_id))
        elif donation_id in result.lost_to_cancellation:
            outcomes.append(Outcome(donation_id=donation_id, kg=kg, status=Status.DRIVER_CANCELLED,
                                    offered_at=offer.time, charity_id=result.awards.get(donation_id)))
        else:
            status = status_from_messages.get(donation_id, Status.NO_DRIVER)
            outcomes.append(Outcome(donation_id=donation_id, kg=kg, status=status, offered_at=offer.time,
                                    charity_id=result.awards.get(donation_id)))
    return outcomes
