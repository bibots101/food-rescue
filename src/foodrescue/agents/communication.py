"""Communication agent: tells merchants, charities and drivers what was decided, in plain English sentences."""

from __future__ import annotations

from typing import Any

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.clock import format_hhmm
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import Directory, Donation, Message, Performative

PROTOCOL = "notification"


def _extra(plan: dict[str, Any]) -> str:
    """The merchant's note and allergens, read from their details message, for the charity."""
    extra = f" Note from the merchant: {plan['note']}" if plan.get("note") else ""
    if plan.get("allergens"):
        extra += f" Allergens: {', '.join(plan['allergens'])}."
    return extra


class CommunicationAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry | None,
                 directory: Directory) -> None:
        super().__init__(name, bus, tools)
        self.directory = directory

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions: list[Message] = []
        for message in percepts:
            if message.performative is Performative.INFORM and message.protocol == "delivery":
                actions += self._new_plan(now, message.content["plan"], changed=message.content.get("changed", False))
            elif message.performative is Performative.INFORM and message.protocol == "replan":
                for plan in message.content["moves"]:
                    if plan.get("confirmed", True) or plan["driver_id"] is None:  # told once a driver accepted
                        actions += self._new_plan(now, plan, changed=True)
            elif message.performative is Performative.INFORM and message.protocol == "availability":
                actions.append(self._duty_notice(now, message))
            elif message.performative is Performative.INFORM and message.protocol == "progress":
                actions += self._progress(now, message.content["event"], message.content["at"],
                                          message.content["plan"])
            elif message.protocol == "safety-check" and message.performative in (Performative.REFUSE,
                                                                                 Performative.REQUEST):
                actions.append(self._safety_notice(now, message))
            elif message.protocol == "follow-up" and message.performative is Performative.REQUEST:
                actions.append(self._notify(now, message.content["merchant_id"], message.conversation_id,
                                            f"Before we send a driver: {message.content['question']}",
                                            question=True))
            elif message.performative is Performative.INFORM and message.protocol == "award":
                actions.append(self._decline_notice(now, message))
            elif message.performative is Performative.FAILURE and message.protocol == "contract-net":
                donation = Donation.model_validate(message.content["donation"])
                actions.append(self._notify(now, donation.merchant_id, donation.id,
                                            "Sorry, no charity can take your donation tonight."))
        return actions

    def _new_plan(self, now: int, plan: dict[str, Any], changed: bool = False) -> list[Message]:
        donation_id = plan["donation_id"]
        if plan["driver_id"] is None:
            return [self._notify(now, plan["merchant_id"], donation_id,
                                 "Sorry, no driver is available for your donation right now.")]
        prefix = "Change of plan: " if changed else ""
        pickup, delivery = format_hhmm(plan["pickup_eta"]), format_hhmm(plan["delivery_eta"])
        merchant = self.directory.get(plan["merchant_id"]).name
        charity = self.directory.get(plan["charity_id"]).name
        pickup_place = plan.get("pickup_name", merchant)  # a merchant, or a broken-down car (transfer)
        transfer = pickup_place != merchant
        notices = []
        if not transfer:  # after a transfer, the merchant's part is already done
            notices.append(self._notify(now, plan["merchant_id"], donation_id,
                                        f"{prefix}{plan['driver_name']} will pick up your donation at {pickup}. "
                                        "Thank you!"))
        notices += [
            self._notify(now, plan["charity_id"], donation_id,
                         f"{prefix}{plan['description']} from {merchant} arrives around {delivery}." + _extra(plan)),
            self._notify(now, plan["driver_id"], donation_id,
                         f"{prefix}pick up at {pickup_place} at {pickup}, deliver to {charity} at {delivery}."),
        ]
        return notices

    def _progress(self, now: int, event: str, at: int, plan: dict[str, Any]) -> list[Message]:
        """What really happened: the food was picked up, then delivered."""
        donation_id, time = plan["donation_id"], format_hhmm(at)
        merchant = self.directory.get(plan["merchant_id"]).name
        charity = self.directory.get(plan["charity_id"]).name
        if event == "picked_up":
            return [
                self._notify(now, plan["merchant_id"], donation_id,
                             f"{plan['driver_name']} picked up your donation at {time}. Thank you!"),
                self._notify(now, plan["charity_id"], donation_id,
                             f"On its way: {plan['description']} from {merchant}, picked up at {time}, "
                             f"arrives around {format_hhmm(plan['delivery_eta'])}."),
            ]
        return [
            self._notify(now, plan["charity_id"], donation_id,
                         f"Delivered at {time}: {plan['description']} from {merchant}."),
            self._notify(now, plan["merchant_id"], donation_id,
                         f"Your donation reached {charity} at {time}. Thank you for saving this food!"),
        ]

    def _decline_notice(self, now: int, message: Message) -> Message:
        """A charity that won the donation declined it (or did not answer): tell the merchant why, and what next."""
        content = message.content
        charity = self.directory.get(content["charity_id"]).name
        following = content.get("next_charity")
        then = (f" It now goes to {self.directory.get(following).name}." if following
                else "")  # no one left: the "no charity" notice follows
        return self._notify(now, content["merchant_id"], message.conversation_id,
                            f"{charity} could not take your donation: {content['reason']}.{then}")

    def _duty_notice(self, now: int, message: Message) -> Message:
        content = message.content
        if content.get("refused"):
            text = ("You still have food on board: deliver it first, then you can go off duty. "
                    f"({', '.join(content['on_board'])})")
        elif content["available"]:
            text = "You are on duty: new deliveries can be sent to you."
        else:
            released = content.get("released", 0)
            deliveries = "1 planned delivery goes" if released == 1 else f"{released} planned deliveries go"
            text = "You are off duty." + (f" {deliveries} to other drivers." if released else "")
        return self._notify(now, content["driver_id"], message.conversation_id, text)

    def _safety_notice(self, now: int, message: Message) -> Message:
        donation = Donation.model_validate(message.content["donation"])
        findings = message.content["verdict"]["findings"]
        cited = message.content.get("explanation")
        if message.performative is Performative.REFUSE:
            reasons = " ".join(f"{f['item']}: {f['explanation']} (rule {f['rule_id']})" for f in findings
                               if f["decision"] == "reject")
            if cited and cited["grounded"]:  # RAG: plain-words explanation quoting the source document
                reasons = f"{cited['text']} Source: {cited['source']}: \"{cited['quote']}\"."
            text = f"Sorry, we cannot accept this donation for safety reasons. {reasons}"
        else:
            asked: dict[str, list[str]] = {}  # one question per rule, naming the foods it is about
            for f in findings:
                if f["decision"] == "needs_human" and f["item"] not in asked.setdefault(f["explanation"], []):
                    asked[f["explanation"]].append(f["item"])
            questions = " ".join(f"({', '.join(items)}) {text}" for text, items in asked.items())
            if donation.problems:  # the details message reported something (rule R11)
                questions += f" You told us: {'; '.join(donation.problems)}. Is the food still safe to eat?"
            text = f"Before we send a driver: {questions} Please reply with this information."
            return self._notify(now, donation.merchant_id, donation.id, text, question=True)
        return self._notify(now, donation.merchant_id, donation.id, text)

    def _notify(self, now: int, receiver: str, donation_id: str, text: str, question: bool = False) -> Message:
        """`question` marks notifications that wait for the merchant's answer (reply box on the dashboard)."""
        return self.message(now, receiver, Performative.INFORM, donation_id, protocol=PROTOCOL,
                            summary=text, text=text, question=question)
