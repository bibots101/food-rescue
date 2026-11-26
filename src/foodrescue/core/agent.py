"""Base agent.

Every agent runs the same cycle at each clock tick:

    perceive  ->  decide  ->  act
    (read the inbox)  (choose messages to send)  (send them through the bus)

Subclasses only implement `decide`, the agent's policy. Keeping `decide` free of side effects
makes agents easy to test: give it messages, check the messages it returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from foodrescue.core.bus import MessageBus
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import Message, Performative


class Agent(ABC):
    groups: tuple[str, ...] = ()  # bus groups this kind of agent joins, e.g. ("charities",)

    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry | None = None) -> None:
        self.name = name
        self.bus = bus
        self.tools = tools if tools is not None else ToolRegistry()
        bus.register(name, self.groups)

    def step(self, now: int) -> None:
        """One tick of the agent's life."""
        percepts = self.perceive(now)
        actions = self.decide(now, percepts)
        self.act(actions)

    def perceive(self, now: int) -> list[Message]:
        """What the agent observes this tick. By default, the messages in its inbox."""
        return self.bus.receive(self.name)

    @abstractmethod
    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        """The agent's policy: from what it perceived, choose the messages to send."""

    def act(self, actions: list[Message]) -> None:
        for message in actions:
            self.bus.send(message)

    # --- helpers for building messages ---------------------------------------------------

    def message(
        self,
        now: int,
        receiver: str,
        performative: Performative,
        conversation_id: str,
        *,
        protocol: str = "",
        summary: str = "",
        **content: Any,
    ) -> Message:
        return Message(
            time=now,
            sender=self.name,
            receiver=receiver,
            performative=performative,
            conversation_id=conversation_id,
            protocol=protocol,
            content=content,
            summary=summary,
        )

    def reply(
        self, now: int, to: Message, performative: Performative, *, summary: str = "", **content: Any
    ) -> Message:
        """Answer `to` in the same conversation and protocol."""
        return self.message(
            now,
            to.sender,
            performative,
            to.conversation_id,
            protocol=to.protocol,
            summary=summary,
            **content,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
