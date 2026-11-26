"""Impact of rescued food in words people understand: meal equivalents.

The factor and its source are in `data/impact.yaml` (WRAP's 420 g meal). Only food really delivered is counted.
"""

from __future__ import annotations

from functools import lru_cache

import yaml

from foodrescue.config import PROJECT_ROOT

FACTORS_FILE = PROJECT_ROOT / "data" / "impact.yaml"


@lru_cache(maxsize=1)
def factors() -> dict:
    return yaml.safe_load(FACTORS_FILE.read_text(encoding="utf-8"))


def meals(kg: float) -> int:
    """Meal equivalents for this weight of food."""
    return round(kg / factors()["meals"]["kg_per_meal"])


def source(name: str) -> str:
    """The source sentence, for hover help."""
    return " ".join(factors()[name]["source"].split())
