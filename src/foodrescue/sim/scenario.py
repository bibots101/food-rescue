"""Loading scenarios: a YAML file, the demo, or a randomly generated day."""

from __future__ import annotations

from pathlib import Path

import yaml

from foodrescue.config import PROJECT_ROOT
from foodrescue.models import Scenario
from foodrescue.sim.generator import generate_scenario

SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"
ALIASES = {"demo": "demo_day.yaml"}


PLATFORM_ACCOUNTS = "demo_day.yaml"  # the real Tunis merchants, charities and drivers
PLATFORM_END = 23 * 60 + 59


def platform_scenario(start: int = 18 * 60) -> Scenario:
    """The platform as people use it: the registered accounts and nothing scripted. Food, incidents and new
    accounts come from what people do in the app."""
    accounts = load_scenario(str(SCENARIO_DIR / PLATFORM_ACCOUNTS))
    return accounts.model_copy(update={"name": "platform", "offers": [], "disruptions": [], "start": start,
                                       "end": PLATFORM_END})


def load_scenario(spec: str) -> Scenario:
    """`spec` is "demo", "random:<seed>[:<cancellations>[:<late merchants>]]", or a path to a YAML file."""
    if spec.startswith("random:"):
        parts = [int(p) for p in spec.removeprefix("random:").split(":")] + [0, 0]
        return generate_scenario(seed=parts[0], cancellations=parts[1], delays=parts[2])
    path = SCENARIO_DIR / ALIASES[spec] if spec in ALIASES else Path(spec)
    with open(path, encoding="utf-8") as fh:
        return Scenario.model_validate(yaml.safe_load(fh))
