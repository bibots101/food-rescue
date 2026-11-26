"""Matching agent: the manager of the Contract Net protocol.

For each approved donation:
  1. announce it to every charity (CFP),
  2. collect their proposals (PROPOSE) and refusals (REFUSE) until all answered or the deadline,
  3. award it to the best bid (ACCEPT_PROPOSAL) and wait for the winner's confirmation;
     if the winner declines after all (FAILURE), or does not answer within `confirm_timeout` minutes,
     offer it to the next-best bidder,
  4. once confirmed, turn down the other bidders (REJECT_PROPOSAL) and ask logistics to deliver it.

The award uses `optim.scoring.award_score`: the charity's own need fit, the distance from the
merchant, and fairness (charities that received less so far get a boost). With learning on
(`features.learning`), the score is multiplied by a Thompson sample of the probability that this
charity confirms this kind of food (`learning.bandit`), learned from every confirmation and decline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from foodrescue.config import MatchingConfig
from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.tools import ToolRegistry
from foodrescue.learning.bandit import AcceptanceLearner
from foodrescue.models import Bid, Directory, Donation, DonationStatus, Message, Performative
from foodrescue.optim.scoring import award_score

PROTOCOL = "contract-net"


@dataclass
class OpenCall:
    """A CFP waiting for answers."""

    donation: Donation
    deadline: int
    expected: set[str]
    bids: list[Bid] = field(default_factory=list)
    refused: set[str] = field(default_factory=set)

    def complete(self) -> bool:
        return {b.charity_id for b in self.bids} | self.refused >= self.expected


@dataclass
class Award:
    """An award waiting for the winner's confirmation; `ranked` holds the bidders still to try."""

    donation: Donation
    ranked: list[Bid]
    scores: dict[str, float]
    losers: list[str] = field(default_factory=list)
    offered_at: int = 0  # when the current bidder was asked to confirm


class MatchingAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry, directory: Directory,
                 weights: MatchingConfig | None = None, cfp_timeout: int = 3,
                 learner: AcceptanceLearner | None = None, confirm_timeout: int = 10) -> None:
        super().__init__(name, bus, tools)
        self.learner = learner  # None = no learning
        self.pending: dict[str, Award] = {}
        self.declines = 0
        self.directory = directory
        self.weights = weights or MatchingConfig()
        self.cfp_timeout = cfp_timeout
        self.confirm_timeout = confirm_timeout
        self.open_calls: dict[str, OpenCall] = {}
        self.awards: dict[str, str] = {}  # donation id -> charity id
        self.kg_awarded: dict[str, float] = defaultdict(float)  # charity id -> kg
        self.contract_net = True  # False only for the ablation study (see `_assign_nearest`)
        self.waiting: dict[str, Donation] = {}  # ablation: donations waiting for the charity's answer

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions: list[Message] = []
        for message in percepts:
            if message.performative is Performative.INFORM and message.protocol == "safety-check":
                if self.contract_net:
                    actions.append(self._announce(now, message))
                else:
                    actions.append(self._assign_nearest(now, Donation.model_validate(message.content["donation"])))
            elif message.protocol == "assignment" and message.conversation_id in self.waiting:
                actions += self._assignment_answer(now, message)
            elif message.performative is Performative.PROPOSE and message.conversation_id in self.open_calls:
                self.open_calls[message.conversation_id].bids.append(Bid.model_validate(message.content["bid"]))
            elif message.performative is Performative.REFUSE and message.conversation_id in self.open_calls:
                self.open_calls[message.conversation_id].refused.add(message.sender)
            elif (message.protocol == PROTOCOL and message.conversation_id in self.pending
                  and message.performative in (Performative.INFORM, Performative.FAILURE)):
                actions += self._confirmation(now, message)

        for donation_id, call in list(self.open_calls.items()):
            if call.complete() or now >= call.deadline:
                actions += self._award(now, call)
                del self.open_calls[donation_id]
        for donation_id, award in list(self.pending.items()):  # a winner who never answers counts as a decline
            if now >= award.offered_at + self.confirm_timeout:
                actions.append(self.message(now, award.ranked[0].charity_id, Performative.CANCEL, donation_id,
                                            protocol=PROTOCOL,
                                            summary="no answer in time, offered to the next charity"))
                actions += self._declined(now, award, f"did not answer within {self.confirm_timeout} minutes")
        return actions

    def _announce(self, now: int, message: Message) -> Message:
        donation = Donation.model_validate(message.content["donation"])
        self.open_calls[donation.id] = OpenCall(
            donation=donation, deadline=now + self.cfp_timeout,
            expected=set(self.bus.members("charities")),
        )
        return self.message(now, "@charities", Performative.CFP, donation.id, protocol=PROTOCOL,
                            summary=f"who can take {donation.describe()} ({donation.total_kg} kg)?",
                            donation=donation.model_dump(mode="json"))

    def _award(self, now: int, call: OpenCall) -> list[Message]:
        donation = call.donation
        if not call.bids:
            donation.status = DonationStatus.UNMATCHED
            return [self.message(now, "communication", Performative.FAILURE, donation.id,
                                 protocol=PROTOCOL, summary="no charity can take it",
                                 donation=donation.model_dump(mode="json"))]

        scores = {bid.charity_id: self._score(donation, bid) for bid in call.bids}
        ranked = sorted(call.bids, key=lambda bid: (-scores[bid.charity_id], bid.charity_id))
        award = Award(donation, ranked, scores)
        self.pending[donation.id] = award
        return [self._offer_to_next(now, award)]

    def _offer_to_next(self, now: int, award: Award) -> Message:
        bid = award.ranked[0]
        award.offered_at = now
        return self.message(now, bid.charity_id, Performative.ACCEPT_PROPOSAL, award.donation.id, protocol=PROTOCOL,
                            summary=f"you won (award score {award.scores[bid.charity_id]:.2f}), please confirm",
                            kg=bid.accept_kg, donation=award.donation.model_dump(mode="json"), scores=award.scores)

    def _confirmation(self, now: int, answer: Message) -> list[Message]:
        award = self.pending[answer.conversation_id]
        donation, winner = award.donation, award.ranked[0]
        confirmed = answer.performative is Performative.INFORM
        if self.learner is not None:
            self.learner.update(winner.charity_id, donation.categories, confirmed)
        if not confirmed:
            return self._declined(now, award, answer.content.get("reason") or "could not receive it after all")

        del self.pending[donation.id]
        self.awards[donation.id] = winner.charity_id
        self.kg_awarded[winner.charity_id] += winner.accept_kg
        donation.status = DonationStatus.ASSIGNED
        actions = [self.message(now, bid.charity_id, Performative.REJECT_PROPOSAL, donation.id,
                                protocol=PROTOCOL, summary=f"another charity won ({winner.charity_id})")
                   for bid in award.ranked[1:]]
        actions.append(self.message(now, "logistics", Performative.REQUEST, donation.id, protocol="delivery",
                                    summary=f"deliver {donation.describe()} to {winner.charity_id}",
                                    donation=donation.model_dump(mode="json"), charity_id=winner.charity_id))
        return actions

    def _declined(self, now: int, award: Award, reason: str) -> list[Message]:
        """The winner declined after all (or did not answer): try the next-best bidder, and say so (the merchant is
        told who declined, why, and who is asked next)."""
        donation = award.donation
        self.declines += 1
        charity_id = award.ranked[0].charity_id
        award.losers.append(charity_id)
        award.ranked.pop(0)
        following = award.ranked[0].charity_id if award.ranked else None
        notice = self.message(now, "communication", Performative.INFORM, donation.id, protocol="award",
                              summary=f"{charity_id} declined ({reason})" + (f", next: {following}" if following
                                                                              else ", no bidder left"),
                              charity_id=charity_id, reason=reason, next_charity=following,
                              merchant_id=donation.merchant_id)
        if award.ranked:
            return [notice, self._offer_to_next(now, award)]
        del self.pending[donation.id]
        donation.status = DonationStatus.UNMATCHED
        return [notice, self.message(now, "communication", Performative.FAILURE, donation.id, protocol=PROTOCOL,
                                     summary="every bidder declined", donation=donation.model_dump(mode="json"))]

    # --- ablation: no Contract Net ---------------------------------------------------------

    def _assign_nearest(self, now: int, donation: Donation) -> Message:
        """Pick the nearest charity whose public opening hours fit, and tell it to take the donation.

        No call for proposals, no private information, no second chance: this measures what the
        Contract Net adds.
        """
        merchant = self.directory.get(donation.merchant_id).location
        candidates = []
        for charity_id in self.bus.members("charities"):
            place = self.directory.get(charity_id)
            drive = self.tools.call("travel_time", {"origin": merchant, "destination": place.location})
            if place.opening is None or max(donation.pickup_window.start, now) + drive <= place.opening.end:
                distance = self.tools.call("road_distance", {"origin": merchant, "destination": place.location})
                candidates.append((distance, charity_id))
        if not candidates:
            return self.message(now, "communication", Performative.FAILURE, donation.id, protocol=PROTOCOL,
                                summary="no charity open in time", donation=donation.model_dump(mode="json"))
        _, charity_id = min(candidates)
        self.waiting[donation.id] = donation
        return self.message(now, charity_id, Performative.REQUEST, donation.id, protocol="assignment",
                            summary=f"take {donation.describe()} (nearest charity, no bidding)",
                            donation=donation.model_dump(mode="json"))

    def _assignment_answer(self, now: int, answer: Message) -> list[Message]:
        donation = self.waiting.pop(answer.conversation_id)
        if answer.performative is not Performative.INFORM:
            return [self.message(now, "communication", Performative.FAILURE, donation.id, protocol=PROTOCOL,
                                 summary=f"{answer.sender} refused: {answer.summary}",
                                 donation=donation.model_dump(mode="json"))]
        self.awards[donation.id] = answer.sender
        self.kg_awarded[answer.sender] += donation.total_kg
        return [self.message(now, "logistics", Performative.REQUEST, donation.id, protocol="delivery",
                             summary=f"deliver {donation.describe()} to {answer.sender}",
                             donation=donation.model_dump(mode="json"), charity_id=answer.sender)]

    def _score(self, donation: Donation, bid: Bid) -> float:
        merchant = self.directory.get(donation.merchant_id).location
        charity = self.directory.get(bid.charity_id).location
        distance = self.tools.call("road_distance", {"origin": merchant, "destination": charity})
        total = sum(self.kg_awarded.values())
        share = self.kg_awarded[bid.charity_id] / total if total else 0.0
        score = award_score(bid.score, distance, share, self.weights)
        if self.learner is not None:  # expected value: utility x (sampled) chance the charity confirms
            score *= self.learner.sample(bid.charity_id, donation.categories)
        return round(score, 4)
