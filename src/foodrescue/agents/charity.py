"""Charity agent: a participant in the Contract Net.

It answers each call for proposals (CFP) using private information that the matching agent does
not have: how much room is left today, whether it has a fridge, and what it still needs.

Hard constraints give a REFUSE with the reason:
  * a kind of food it does not accept (e.g. a food bank handing out parcels and cooked meals),
  * not enough room left for the donation,
  * chilled food but no fridge, frozen food but no freezer,
  * the food cannot arrive before the charity closes.

The charity can update its profile during the evening: its needs and room left are then recomputed
from what it has already received.
Otherwise it PROPOSEs with its need fit (`optim.scoring.need_fit`) as its score, and the reasons it wants the
food in words (what it still needs, its fridge or freezer, its room left), which the charity's screen shows.

When it wins (ACCEPT_PROPOSAL), it confirms (INFORM) or declines after all (FAILURE). Two policies:
  * "auto": the agent decides; the probability of confirming depends on hidden preferences
    (`Charity.acceptance`) that the matching agent learns;
  * "manual": a person accepts or declines in the Charity screen (`Simulation.answer_award`), with a reason.
"""

from __future__ import annotations

import random

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import (
    CATEGORY_LABELS,
    Bid,
    Charity,
    Directory,
    Donation,
    FoodCategory,
    Message,
    Performative,
    Storage,
)
from foodrescue.optim.scoring import need_fit


class CharityAgent(Agent):
    groups = ("charities",)

    def __init__(self, charity: Charity, bus: MessageBus, tools: ToolRegistry | None = None,
                 directory: Directory | None = None, seed: int = 0, policy: str = "auto") -> None:
        super().__init__(charity.id, bus, tools)
        self.charity = charity
        self.policy = policy
        self.directory = directory
        self.received_kg = 0.0
        self.received: dict[FoodCategory, float] = {}  # kg received today, per category
        self.remaining_needs = dict(charity.needs)
        self.rng = random.Random(f"{seed}-{charity.id}-confirm")  # deterministic for a given scenario

    @property
    def remaining_kg(self) -> float:
        return round(self.charity.capacity_kg - self.received_kg, 2)

    def update_profile(self, charity: Charity) -> None:
        """New account details: what is still needed is the new need minus what was already received."""
        self.charity = charity
        self.remaining_needs = {category: max(0.0, kg - self.received.get(category, 0.0))
                                for category, kg in charity.needs.items()}

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions = []
        for message in percepts:
            if message.performative is Performative.CFP:
                actions.append(self._answer_call(now, message))
            elif message.performative is Performative.ACCEPT_PROPOSAL and self.policy == "auto":
                actions.append(self._confirm(now, message))
            # "manual": the award waits for the person's answer (`answer_award`)
            elif message.performative is Performative.REQUEST and message.protocol == "assignment":
                actions.append(self._answer_assignment(now, message))
            # REJECT_PROPOSAL and INFORM notifications need no answer
        return actions

    def _answer_call(self, now: int, cfp: Message) -> Message:
        donation = Donation.model_validate(cfp.content["donation"])
        reason = self._refusal_reason(donation, now)
        if reason:
            return self.reply(now, cfp, Performative.REFUSE, summary=reason, reason=reason)
        fit = need_fit(donation, self.remaining_needs)
        bid = Bid(donation_id=donation.id, charity_id=self.name, accept_kg=donation.total_kg, score=fit)
        reasons = self.reasons(donation)
        return self.reply(now, cfp, Performative.PROPOSE, summary="; ".join(reasons),
                          bid=bid.model_dump(mode="json"), reasons=reasons)

    def reasons(self, donation: Donation) -> list[str]:
        """Why this charity should get the donation, in words, from its private details."""
        reasons = []
        for category in sorted(donation.categories):
            label = CATEGORY_LABELS[category].lower()
            still = self.remaining_needs.get(category, 0.0)
            if still > 0:
                reasons.append(f"we still need {still:g} kg of {label} today")
            else:
                reasons.append(f"{label} is not on our list of needs today, but we can use it")
        if Storage.CHILLED in donation.storages:
            reasons.append("we have a fridge to keep it cold")
        if Storage.FROZEN in donation.storages:
            reasons.append("we have a freezer to keep it frozen")
        reasons.append(f"we have room for it ({self.remaining_kg:g} kg free today)")
        return reasons

    def _confirm(self, now: int, award: Message) -> Message:
        """Last step of the Contract Net: confirm the award (INFORM), or decline it after all (FAILURE),
        according to the charity's hidden acceptance for this kind of food."""
        donation = Donation.model_validate(award.content["donation"])
        accept = self.rng.random() < self.charity.acceptance_of(donation.categories)
        return self.answer_award(now, award, accept)

    def answer_award(self, now: int, award: Message, accept: bool, reason: str = "") -> Message:
        """The charity's answer to an award (from its agent, or from the person in the Charity screen)."""
        if accept:
            self._receive(Donation.model_validate(award.content["donation"]))
            return self.reply(now, award, Performative.INFORM, summary="confirmed, we will receive it")
        return self.reply(now, award, Performative.FAILURE, reason=reason,
                          summary=f"sorry, we cannot receive it: {reason}" if reason
                          else "sorry, we cannot receive it after all")

    def _answer_assignment(self, now: int, request: Message) -> Message:
        """Ablation without Contract Net: the charity is told to take a donation it never bid for."""
        donation = Donation.model_validate(request.content["donation"])
        reason = self._refusal_reason(donation, now)
        if reason:
            return self.reply(now, request, Performative.REFUSE, summary=reason, reason=reason)
        self._receive(donation)
        return self.reply(now, request, Performative.INFORM, summary="accepted")

    def _refusal_reason(self, donation: Donation, now: int) -> str | None:
        refused = [CATEGORY_LABELS[c].lower() for c in sorted(donation.categories) if c not in self.charity.accepted]
        if refused:
            return f"we do not accept {', '.join(refused)}"
        if donation.total_kg > self.remaining_kg:
            return f"no room left ({self.remaining_kg} kg free)"
        if Storage.CHILLED in donation.storages and not self.charity.has_fridge:
            return "no fridge for chilled food"
        if Storage.FROZEN in donation.storages and not self.charity.has_freezer:
            return "no freezer for frozen food"
        if self.directory is not None:
            merchant = self.directory.get(donation.merchant_id).location
            drive = self.tools.call("travel_time", {"origin": merchant, "destination": self.charity.location})
            earliest_arrival = max(donation.pickup_window.start, now) + drive
            if earliest_arrival > self.charity.opening.end:
                return "closed before the food can arrive"
        return None

    def _receive(self, donation: Donation) -> None:
        self.received_kg += donation.total_kg
        for item in donation.items:
            self.received[item.category] = self.received.get(item.category, 0.0) + item.kg
            if item.category in self.remaining_needs:
                self.remaining_needs[item.category] = max(0.0, self.remaining_needs[item.category] - item.kg)
