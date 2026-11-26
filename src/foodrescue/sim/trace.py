"""Readable text output of a simulation: one line per message, then a summary."""

from __future__ import annotations

from collections import Counter

from foodrescue.core.clock import format_hhmm
from foodrescue.models import Message, StopKind
from foodrescue.sim.runner import SimulationResult


def format_message(message: Message) -> str:
    performative = message.performative.value.upper()
    return (f"{format_hhmm(message.time)}  {message.sender:>13} → {message.receiver:<13} "
            f"{performative:<16} [{message.conversation_id}] {message.summary}")


def format_trace(messages: list[Message], protocol: str | None = None) -> str:
    return "\n".join(format_message(m) for m in messages if protocol is None or m.protocol == protocol)


def format_summary(result: SimulationResult) -> str:
    scenario = result.scenario
    performatives = Counter(m.performative.value for m in result.messages)
    lines = [
        f"Scenario {scenario.name}: {len(scenario.offers)} offers, {len(scenario.charities)} charities, "
        f"{len(scenario.drivers)} drivers",
        f"Messages: {len(result.messages)}  "
        + "  ".join(f"{name}={count}" for name, count in sorted(performatives.items())),
        f"Donations awarded to a charity: {len(result.awards)}/{len(scenario.offers)}",
    ]
    routed = {s.donation_id for route in result.routes.values() for s in route.stops}
    dropped = sorted(set(result.awards) - routed)
    total_km = round(sum(route.distance_km for route in result.routes.values()), 2)
    lines.append(f"Donations on a driver's route: {len(routed)}/{len(result.awards)}"
                 + (f"  (no driver in time: {', '.join(dropped)})" if dropped else "")
                 + f"  total distance {total_km} km")
    lines.append("Routes:")
    for driver_id, route in result.routes.items():
        if not route.stops:
            lines.append(f"  {driver_id}: no stops")
            continue
        stops = " → ".join(
            f"{'↑' if s.kind is StopKind.PICKUP else '↓'}{s.place_id}@{format_hhmm(s.eta)}" for s in route.stops
        )
        lines.append(f"  {driver_id} ({route.distance_km} km): {stops}")
    lines.append("(↑ = pickup at a merchant, ↓ = delivery to a charity)")
    return "\n".join(lines)
