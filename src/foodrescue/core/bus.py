"""Message bus: how agents talk to each other.

Agents never call each other directly. They send `Message`s through the bus, using FIPA-style
performatives (CFP, PROPOSE, INFORM, ...). Two rules keep the simulation deterministic:

* A message sent during tick t is delivered at the start of tick t+1, so the order in which
  agents run inside a tick never changes the result.
* Every message is kept in `bus.log`, which the trace and the dashboard replay.

A receiver is either a registered name ("matching") or a group ("@charities").
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from foodrescue.models import Message

GROUP_PREFIX = "@"


class UnknownReceiverError(KeyError):
    """The message is addressed to a name or group nobody registered."""


class MessageBus:
    def __init__(self) -> None:
        self._inboxes: dict[str, deque[Message]] = {}
        self._groups: dict[str, list[str]] = defaultdict(list)
        self._pending: list[Message] = []
        self._next_id = 1
        self.log: list[Message] = []

    # --- registration --------------------------------------------------------------------

    def register(self, name: str, groups: Iterable[str] = ()) -> None:
        """Give `name` an inbox. Agents and outside people (merchants, drivers) both register."""
        if name.startswith(GROUP_PREFIX):
            raise ValueError(f"names cannot start with {GROUP_PREFIX!r}: {name!r}")
        if name in self._inboxes:
            raise ValueError(f"{name!r} is already registered")
        self._inboxes[name] = deque()
        for group in groups:
            self._groups[group].append(name)

    def members(self, group: str) -> list[str]:
        return list(self._groups.get(group.removeprefix(GROUP_PREFIX), []))

    # --- messaging -----------------------------------------------------------------------

    def send(self, message: Message) -> Message:
        """Queue a message for the next tick. Returns the message with its id set."""
        self._resolve(message.receiver)  # fail now, not at delivery, if nobody can receive it
        stamped = message.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self._pending.append(stamped)
        self.log.append(stamped)
        return stamped

    def deliver(self) -> int:
        """Move queued messages into inboxes. The simulation calls this once per tick."""
        pending, self._pending = self._pending, []
        for message in pending:
            for name in self._resolve(message.receiver):
                if name != message.sender:  # a group message never comes back to its sender
                    self._inboxes[name].append(message)
        return len(pending)

    def receive(self, name: str) -> list[Message]:
        """Empty `name`'s inbox and return what was in it."""
        inbox = self._inboxes[name]
        messages = list(inbox)
        inbox.clear()
        return messages

    def conversation(self, conversation_id: str) -> list[Message]:
        return [m for m in self.log if m.conversation_id == conversation_id]

    def _resolve(self, receiver: str) -> list[str]:
        if receiver.startswith(GROUP_PREFIX):
            group = receiver.removeprefix(GROUP_PREFIX)
            if group not in self._groups:
                raise UnknownReceiverError(receiver)
            return self._groups[group]
        if receiver not in self._inboxes:
            raise UnknownReceiverError(receiver)
        return [receiver]
