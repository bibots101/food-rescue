"""Run our system and the baseline on the same day(s) and compare them with the same metrics."""

from __future__ import annotations

from statistics import mean

from foodrescue.baseline.greedy import run_baseline
from foodrescue.config import Config, load_config
from foodrescue.models import Scenario
from foodrescue.sim.metrics import DayMetrics, compute_metrics, outcomes_from_simulation
from foodrescue.sim.runner import Simulation


def compare_day(scenario: Scenario, config: Config | None = None) -> tuple[DayMetrics, DayMetrics]:
    """Both systems on the same scenario (same offers, same disruptions): a paired comparison."""
    config = config or load_config()
    ours = Simulation(scenario, config).run()
    ours_km = sum(route.distance_km for route in ours.routes.values())
    baseline = run_baseline(scenario, config)
    return (compute_metrics("ours", outcomes_from_simulation(ours), ours_km),
            compute_metrics("baseline", baseline.outcomes, baseline.km))


ROWS = [
    ("Food rescued (% of safe kg)", lambda m: f"{m.rescued_pct:.1f} %"),
    ("Delivered (kg)", lambda m: f"{m.delivered_kg:.1f}"),
    ("Donations delivered", lambda m: f"{m.delivered}/{m.offers}"),
    ("Distance driven (km)", lambda m: f"{m.km:.1f}"),
    ("Km per kg rescued", lambda m: "-" if m.km_per_kg is None else f"{m.km_per_kg:.2f}"),
    ("Mean wait offer -> pickup (min)", lambda m: "-" if m.mean_wait_min is None else f"{m.mean_wait_min:.0f}"),
    ("Lost (kg, by cause)", lambda m: ", ".join(f"{k} {v:g}" for k, v in m.lost_kg.items()) or "-"),
]


def format_comparison(ours: DayMetrics, baseline: DayMetrics) -> str:
    width = max(len(label) for label, _ in ROWS) + 2
    lines = [f"{'':<{width}}{'Our system':<26}Baseline (phone calls)"]
    lines += [f"{label:<{width}}{fmt(ours):<25} {fmt(baseline)}" for label, fmt in ROWS]
    return "\n".join(lines)


def compare_many(scenarios: list[Scenario], config: Config | None = None) -> str:
    """Mean over several days, and on how many days our system rescued more."""
    pairs = [compare_day(s, config) for s in scenarios]
    lines = [f"{len(pairs)} days", f"{'':<30}{'Our system':<14}Baseline"]
    for label, attr in [("Food rescued (%)", "rescued_pct"), ("Delivered (kg)", "delivered_kg"),
                        ("Distance (km)", "km")]:
        ours = mean(getattr(o, attr) for o, _ in pairs)
        base = mean(getattr(b, attr) for _, b in pairs)
        lines.append(f"{label:<30}{ours:<14.1f}{base:.1f}")
    wins = sum(o.rescued_pct > b.rescued_pct for o, b in pairs)
    ties = sum(o.rescued_pct == b.rescued_pct for o, b in pairs)
    lines.append(f"Days where our system rescued more: {wins}/{len(pairs)} (ties: {ties})")
    return "\n".join(lines)
