"""Intake agent: turns a merchant's message into a structured `Donation`.

With the `llm_intake` feature flag on, the LLM reads the raw text (`llm.extraction`) and our code
normalizes the result (`llm.normalize`: kg, times, storage assumptions). If the LLM fails (not
running, no recorded answer, invalid output), the agent falls back to the structured version
attached to the offer, so the demo never breaks. With the flag off, which is what the simulated
days use, it uses the structured version directly.

Human in the loop: when the LLM finds no quantity for an item, the weight cannot be known, so the
agent asks the merchant instead of guessing (protocol "follow-up"). The merchant's answer comes back
as the same offer with `answer` set, and the LLM reads the message and the answer together.
"""

from __future__ import annotations

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.clock import format_hhmm
from foodrescue.core.tools import ToolRegistry
from foodrescue.llm.client import LLMClient, LLMError
from foodrescue.llm.details import apply_details, read_details
from foodrescue.llm.extraction import ExtractedDonation, extract_donation
from foodrescue.llm.normalize import to_donation
from foodrescue.models import Donation, DonationOffer, Message, Performative


class IntakeAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry | None = None,
                 llm: LLMClient | None = None) -> None:
        super().__init__(name, bus, tools)
        self.llm = llm  # None = LLM intake switched off

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions = []
        for message in percepts:
            if message.performative is not Performative.REQUEST or message.protocol != "intake":
                continue
            offer = DonationOffer.model_validate(message.content["offer"])
            donation, how, missing = self._structure(offer, now)
            if missing:
                question = (f"How much {' and '.join(missing)} do you have? "
                            "Please give a number of kg or of pieces.")
                actions.append(self.message(now, "communication", Performative.REQUEST, offer.id,
                                            protocol="follow-up", summary=f"ask the merchant: {question}",
                                            merchant_id=offer.merchant_id, question=question))
                continue
            if donation is None:
                actions.append(self.reply(now, message, Performative.FAILURE,
                                          summary=f"could not understand the message ({how})"))
                continue
            window = donation.pickup_window
            actions.append(self.message(
                now, "safety", Performative.REQUEST, donation.id, protocol="safety-check",
                summary=f"[{how}] {donation.describe()} = {donation.total_kg} kg, {donation.storage_text()}, "
                        f"pickup {format_hhmm(window.start)}-{format_hhmm(window.end)}",
                donation=donation.model_dump(mode="json"), read_by=how,
            ))
        return actions

    def _from_form(self, offer: DonationOffer, now: int) -> tuple[Donation, str, list[str]]:
        """A form offer is already structured. If the merchant wrote details, the LLM reads them for safety facts
        the form could not hold; if it cannot (offline), the unread details become a reported problem, so a person
        checks the food instead of it going out unread."""
        donation = offer.donation.model_copy(update={"raw_text": offer.text, "details": offer.details})
        if not offer.details.strip():
            return donation, "form", []
        if self.llm is None:
            return donation.model_copy(update={"problems": ["details not read (the AI is switched off)"]}), "form", []
        try:
            facts = read_details(self.llm, offer.details)
        except LLMError as exc:
            return donation.model_copy(update={"problems": [f"details not read (AI unavailable: {exc})"]}), "form", []
        return apply_details(donation, facts, now), "form + AI read the details", []

    def _structure(self, offer: DonationOffer, now: int) -> tuple[Donation | None, str, list[str]]:
        """Returns the donation, how it was obtained ("llm", "form", or why it failed), and the names of
        items whose quantity is missing (then the merchant is asked)."""
        if offer.from_form and offer.donation is not None:  # the merchant filled in the form
            return self._from_form(offer, now)
        if self.llm is not None:
            try:
                extracted = extract_donation(self.llm, offer.full_text(), now)
                missing = missing_quantities(extracted)
                if missing:
                    return None, "llm", missing
                return to_donation(extracted, donation_id=offer.id, merchant_id=offer.merchant_id, now=now,
                                   raw_text=offer.full_text()), "llm", []
            except LLMError as exc:
                if offer.donation is None:
                    return None, f"LLM unavailable: {exc}", []
        if offer.donation is None:
            return None, "no structured form and LLM intake is off", []
        return offer.donation.model_copy(update={"raw_text": offer.text}), "form", []


def missing_quantities(extracted: ExtractedDonation) -> list[str]:
    return [item.name for item in extracted.items if item.quantity is None]
