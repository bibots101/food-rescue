"""Rule engine for food safety.

The rules themselves (thresholds, explanations, sources) live in data/safety/rules.yaml so that
they can be read and checked without reading code. This module only knows how to evaluate the
few kinds of conditions the rules use.

Each item of a donation is checked separately, so one unsafe item (say, rice left out since noon)
does not block the safe items (the bread next to it):
  * every item rejected      -> REJECT the donation
  * any item needs more info -> NEEDS_HUMAN (ask the merchant before going further)
  * otherwise                -> APPROVE the items that passed (with warnings if any)
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from foodrescue.config import PROJECT_ROOT
from foodrescue.models import Donation, FoodCategory, FoodItem, Storage

RULES_FILE = PROJECT_ROOT / "data" / "safety" / "rules.yaml"


class Decision(StrEnum):
    APPROVE = "approve"
    WARN = "warn"
    NEEDS_HUMAN = "needs_human"
    REJECT = "reject"


SEVERITY = {Decision.APPROVE: 0, Decision.WARN: 1, Decision.NEEDS_HUMAN: 2, Decision.REJECT: 3}


class Rule(BaseModel):
    id: str
    title: str
    condition: str
    decision: Decision
    explanation: str
    source: str
    categories: list[FoodCategory] | None = None  # None = every category
    storage: list[Storage] | None = None
    name_contains: list[str] | None = None
    params: dict[str, float] = Field(default_factory=dict)

    def applies_to(self, item: FoodItem, donation: Donation) -> bool:
        if self.categories is not None and item.category not in self.categories:
            return False
        if self.storage is not None and donation.storage_of(item) not in self.storage:
            return False
        if self.name_contains is not None:
            name = item.name.lower()
            return any(keyword.lower() in name for keyword in self.name_contains)
        return True


class Finding(BaseModel):
    rule_id: str
    item: str
    decision: Decision
    title: str
    explanation: str
    source: str


class SafetyVerdict(BaseModel):
    decision: Decision  # APPROVE, NEEDS_HUMAN or REJECT for the whole donation
    approved_items: list[FoodItem]
    rejected_items: list[FoodItem]
    findings: list[Finding]

    @property
    def rule_ids(self) -> list[str]:
        return sorted({f.rule_id for f in self.findings})

    def explain(self) -> str:
        """One line per finding, e.g. "cooked rice: R03 Cooked food left at room temperature too long"."""
        return "; ".join(f"{f.item}: {f.rule_id} {f.title}" for f in self.findings) or "no rule triggered"


# Condition types. Each gets (item, donation, time, params) and says whether the rule fires.
Condition = Callable[[FoodItem, Donation, int, dict[str, float]], bool]

def _clock_start(item: FoodItem, donation: Donation) -> int | None:
    """When the time limit starts: since the food has been at room temperature if the merchant said so (details
    message), otherwise since it was cooked."""
    if donation.ambient_since is not None and donation.storage_of(item) is Storage.AMBIENT:
        return donation.ambient_since
    return donation.prepared_at


CONDITIONS: dict[str, Condition] = {
    "always": lambda item, donation, time, params: True,
    "expired": lambda item, donation, time, params: donation.expires_at is not None and donation.expires_at < time,
    "preparation_time_unknown": lambda item, donation, time, params: _clock_start(item, donation) is None,
    "time_since_preparation_over": lambda item, donation, time, params: (
        _clock_start(item, donation) is not None and time - _clock_start(item, donation) > params["max_minutes"]
    ),
    "reported_problem": lambda item, donation, time, params: bool(donation.problems)
    and not donation.confirmed_by_merchant,
    # The merchant said food that should be cold has been at room temperature since a given time (details).
    "out_of_fridge_over": lambda item, donation, time, params: (
        donation.ambient_since is not None and donation.storage_of(item) is Storage.AMBIENT
        and time - donation.ambient_since > params["max_minutes"] and not donation.confirmed_by_merchant
    ),
}


class RuleBook:
    def __init__(self, rules: list[Rule], sources: dict[str, str]) -> None:
        unknown = {rule.condition for rule in rules} - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown condition types in rules: {sorted(unknown)}")
        missing = {rule.source for rule in rules} - set(sources)
        if missing:
            raise ValueError(f"rules cite unknown sources: {sorted(missing)}")
        self.rules = rules
        self.sources = sources

    @classmethod
    def load(cls, path: Path = RULES_FILE) -> RuleBook:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls([Rule.model_validate(rule) for rule in data["rules"]], data["sources"])

    def check(self, donation: Donation, now: int) -> SafetyVerdict:
        """Evaluate every rule on every item, at the earliest time the food can be picked up."""
        time = max(now, donation.pickup_window.start)
        findings: list[Finding] = []
        approved, rejected = [], []
        needs_human = False
        for item in donation.items:
            worst = Decision.APPROVE
            for rule in self.rules:
                if rule.applies_to(item, donation) and CONDITIONS[rule.condition](item, donation, time, rule.params):
                    findings.append(Finding(rule_id=rule.id, item=item.name, decision=rule.decision,
                                            title=rule.title, explanation=rule.explanation,
                                            source=self.sources[rule.source]))
                    worst = max(worst, rule.decision, key=SEVERITY.__getitem__)
            if worst is Decision.REJECT:
                rejected.append(item)
            else:
                approved.append(item)
                needs_human = needs_human or worst is Decision.NEEDS_HUMAN

        if not approved:
            decision = Decision.REJECT
        elif needs_human:
            decision = Decision.NEEDS_HUMAN
        else:
            decision = Decision.APPROVE
        return SafetyVerdict(decision=decision, approved_items=approved, rejected_items=rejected, findings=findings)
