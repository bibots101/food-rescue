"""Thompson sampling of charity acceptance (a Bernoulli multi-armed bandit).

For every (charity, food category) pair the matching agent keeps a Beta(alpha, beta) belief about
the probability that the charity confirms such a donation after winning it:
  * start: Beta(1, 1), i.e. "no idea" (uniform between 0 and 1);
  * confirmed: alpha += 1; declined: beta += 1.

To choose, it does not use the mean: it draws a random sample from each belief (Thompson sampling).
Charities it knows little about get wide beliefs and sometimes a high sample, so they keep being
tried (exploration); charities that often decline end up with low samples (exploitation). The
learner lives across days, so the system improves over time.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from foodrescue.models import FoodCategory


class AcceptanceLearner:
    def __init__(self, seed: int = 0, prior: tuple[float, float] = (1.0, 1.0)) -> None:
        self.rng = random.Random(f"learner-{seed}")
        self.prior = prior
        self.beliefs: dict[tuple[str, FoodCategory], list[float]] = {}  # (charity, category) -> [alpha, beta]
        self.history: list[tuple[str, bool]] = []  # (charity, confirmed), in order

    def _belief(self, charity_id: str, category: FoodCategory) -> list[float]:
        return self.beliefs.setdefault((charity_id, category), list(self.prior))

    def sample(self, charity_id: str, categories: Iterable[FoodCategory]) -> float:
        """A plausible acceptance probability for this charity and this mix of food (Thompson draw)."""
        probability = 1.0
        for category in sorted(set(categories)):
            alpha, beta = self._belief(charity_id, category)
            probability *= self.rng.betavariate(alpha, beta)
        return probability

    def mean(self, charity_id: str, category: FoodCategory) -> float:
        alpha, beta = self._belief(charity_id, category)
        return alpha / (alpha + beta)

    def update(self, charity_id: str, categories: Iterable[FoodCategory], confirmed: bool) -> None:
        for category in set(categories):
            self._belief(charity_id, category)[0 if confirmed else 1] += 1
        self.history.append((charity_id, confirmed))
