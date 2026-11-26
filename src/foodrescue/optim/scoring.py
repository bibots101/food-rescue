"""Scoring used in the Contract Net.

Two sides, two kinds of knowledge:

* The charity knows its private state, so it computes `need_fit`: how much of the donation it
  really needs today (0 to 1). It refuses outright what it cannot accept (see `CharityAgent`).
* The matching agent (manager) only sees public information plus the bids. It combines the
  charity's need fit with distance (less driving, fresher food) and fairness (spread the food
  across charities) into one award score.
"""

from __future__ import annotations

from foodrescue.config import MatchingConfig
from foodrescue.models import Donation, FoodCategory


def need_fit(donation: Donation, remaining_needs: dict[FoodCategory, float]) -> float:
    """Share of the donation's weight that matches what the charity still needs (0 to 1)."""
    if donation.total_kg <= 0:
        return 0.0
    left = dict(remaining_needs)
    useful = 0.0
    for item in donation.items:
        take = min(item.kg, max(left.get(item.category, 0.0), 0.0))
        useful += take
        left[item.category] = left.get(item.category, 0.0) - take
    return round(useful / donation.total_kg, 3)


def award_score(need: float, distance_km: float, share_received: float, weights: MatchingConfig) -> float:
    """Manager's score for one bid. All three terms are between 0 and 1; higher is better.

    need:           the charity's own need fit (from its bid)
    distance_km:    road distance from the merchant to the charity
    share_received: share of all kg awarded so far that went to this charity
    """
    closeness = 1.0 - min(distance_km / weights.max_distance_km, 1.0)
    fairness = 1.0 - share_received
    score = weights.w_need * need + weights.w_distance * closeness + weights.w_fairness * fairness
    return round(score, 4)
