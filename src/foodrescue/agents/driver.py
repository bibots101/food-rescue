"""Driver agent: the driver's phone app.

The logistics agent proposes each delivery to a driver (PROPOSE, protocol "job-offer"); the driver accepts
(ACCEPT_PROPOSAL) or declines (REJECT_PROPOSAL). A request nobody answers expires and goes to another driver.

Two policies:
  * "auto": a simulated driver, who accepts after 1 to 2 minutes (simulated days, demo check, and the presenter's
    "Drivers answer by themselves" switch);
  * "manual": a person answers in the dashboard's Driver screen; the agent itself does nothing and the
    simulation sends the person's answer (`Simulation.answer_offer`).
"""

from __future__ import annotations

import random

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.models import Message, Performative

PROTOCOL = "job-offer"


class DriverAgent(Agent):
    def __init__(self, driver_id: str, bus: MessageBus, policy: str = "auto", seed: int = 0,
                 accept_probability: float = 1.0) -> None:
        super().__init__(driver_id, bus)
        self.policy = policy
        self.accept_probability = accept_probability
        self.rng = random.Random(f"{seed}-{driver_id}-offers")
        self._answers: list[tuple[int, Message]] = []  # (when to send, answer)

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        if self.policy == "auto":
            for message in percepts:
                if message.performative is Performative.PROPOSE and message.protocol == PROTOCOL:
                    accept = self.rng.random() < self.accept_probability
                    answer = self.reply(now, message, Performative.ACCEPT_PROPOSAL if accept
                                        else Performative.REJECT_PROPOSAL,
                                        summary="I'll take it" if accept else "I can't take it")
                    self._answers.append((now + self.rng.randint(0, 1), answer))
        due = [answer for when, answer in self._answers if when <= now]
        self._answers = [(when, answer) for when, answer in self._answers if when > now]
        return [answer.model_copy(update={"time": now}) for answer in due]
