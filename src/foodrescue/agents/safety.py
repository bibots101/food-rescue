"""Safety agent: decides whether a donation is safe to give away, and says why.

A rule engine (`safety.rules`, rules in data/safety/rules.yaml) decides. Every verdict cites the
rule IDs and sources it used.
  * APPROVE     -> the safe items go to the matching agent (unsafe items are removed)
  * REJECT      -> the merchant is told why (REFUSE)
  * NEEDS_HUMAN -> the merchant is asked for the missing information (REQUEST)
Retrieval over the source documents (RAG) writes richer explanations. Hard rules
always win: the LLM can never approve something a rule rejects.
"""

from __future__ import annotations

from collections.abc import Callable

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import Donation, DonationStatus, Message, Performative
from foodrescue.rag.explain import GroundedExplanation
from foodrescue.safety.rules import Decision, RuleBook, SafetyVerdict

PROTOCOL = "safety-check"

Explainer = Callable[[SafetyVerdict, Donation], GroundedExplanation | None]


class SafetyAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry | None = None,
                 rulebook: RuleBook | None = None, explainer: Explainer | None = None) -> None:
        super().__init__(name, bus, tools)
        self.rulebook = rulebook or RuleBook.load()
        self.explainer = explainer  # RAG (features.rag_safety): explains, never changes the decision

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions = []
        for message in percepts:
            if message.performative is Performative.REQUEST and message.protocol == PROTOCOL:
                actions.append(self._check(now, Donation.model_validate(message.content["donation"])))
        return actions

    def _check(self, now: int, donation: Donation) -> Message:
        verdict = self.rulebook.check(donation, now)
        rules = ", ".join(verdict.rule_ids)
        explanation = self.explainer(verdict, donation) if self.explainer else None
        extra = {"explanation": explanation.model_dump(mode="json")} if explanation else {}
        cited = f" [{explanation.source}]" if explanation and explanation.grounded else ""
        if verdict.decision is Decision.REJECT:
            donation.status = DonationStatus.REJECTED
            return self.message(now, "communication", Performative.REFUSE, donation.id, protocol=PROTOCOL,
                                summary=f"rejected: {verdict.explain()}{cited}",
                                donation=donation.model_dump(mode="json"), verdict=verdict.model_dump(mode="json"),
                                **extra)
        if verdict.decision is Decision.NEEDS_HUMAN:
            return self.message(now, "communication", Performative.REQUEST, donation.id, protocol=PROTOCOL,
                                summary=f"need more information ({rules}){cited}",
                                donation=donation.model_dump(mode="json"), verdict=verdict.model_dump(mode="json"),
                                **extra)

        safe = donation.model_copy(update={"items": verdict.approved_items, "status": DonationStatus.APPROVED})
        summary = f"approved {safe.describe()}"
        if verdict.rejected_items:
            summary += f", removed {', '.join(i.name for i in verdict.rejected_items)}"
        if rules:
            summary += f" ({rules})"
        return self.message(now, "matching", Performative.INFORM, donation.id, protocol=PROTOCOL,
                            summary=summary + cited, donation=safe.model_dump(mode="json"),
                            verdict=verdict.model_dump(mode="json"), **extra)
