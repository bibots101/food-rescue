"""Dispatcher agent: decides what a driver's problem report means for the plan.

A driver writes in their own words ("pneu crevé, je repars dans 20 min", "accident, I'm done"). The
dispatcher chooses one of three tools from its tool registry:
  * pause_driver(driver_id, minutes): the driver continues after the pause, food stays in their car;
  * cancel_driver(driver_id): the driver is out; everything is replanned, food on board is transferred;
  * ask_driver(driver_id, question): unclear, so ask instead of guessing.

Policies (`simulation.dispatcher_policy`, or `features.llm_dispatcher` for "llm"):
  * "cautious": every incident is a cancellation (no language understanding; the safe default);
  * "llm": the LLM reads the message and picks a tool; our code validates the call with the registry
    and falls back to "cautious" if the model fails or proposes an invalid call;
  * "oracle": uses the true duration recorded in the simulation (upper bound: what perfect reading would give).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.tools import ToolError, ToolRegistry
from foodrescue.llm.client import LLMClient, LLMError
from foodrescue.llm.dispatch import choose_tool
from foodrescue.models import Message, Performative


class PauseInput(BaseModel):
    driver_id: str
    minutes: int = Field(ge=5, le=180, description="minutes until the driver can continue")


class CancelInput(BaseModel):
    driver_id: str


class AskInput(BaseModel):
    driver_id: str
    question: str = Field(min_length=5, description="a short question in the driver's language")


class DispatcherAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, policy: str = "cautious", llm: LLMClient | None = None) -> None:
        super().__init__(name, bus, ToolRegistry())  # its own tools: the actions it may take
        self.policy = policy
        self.llm = llm
        self.decisions: list[dict] = []  # what was decided for each incident (dashboard, evaluation)
        self._outbox: list[Message] = []
        self._now = 0
        self._register_tools()

    def _register_tools(self) -> None:
        @self.tools.register("pause_driver", "The driver can continue after a pause; food stays in the car.",
                             PauseInput)
        def pause_driver(args: PauseInput) -> str:
            self._send("logistics", Performative.REQUEST, f"pause {args.driver_id} for {args.minutes} min",
                       kind="driver_pause", driver_id=args.driver_id, minutes=args.minutes)
            return f"paused {args.driver_id} for {args.minutes} min"

        @self.tools.register("cancel_driver", "The driver cannot continue tonight; replan everything.", CancelInput)
        def cancel_driver(args: CancelInput) -> str:
            self._send("logistics", Performative.CANCEL, f"{args.driver_id} out for the night",
                       driver_id=args.driver_id, reason="reported a problem")
            return f"cancelled {args.driver_id}"

        @self.tools.register("ask_driver", "Unclear report: ask the driver a question.", AskInput)
        def ask_driver(args: AskInput) -> str:
            self._outbox.append(self.message(
                self._now, args.driver_id, Performative.INFORM, f"incident-{args.driver_id}", protocol="notification",
                summary=args.question, text=args.question, question=True))
            return f"asked {args.driver_id}"

    def _send(self, receiver: str, performative: Performative, summary: str, **content) -> None:
        self._outbox.append(self.message(self._now, receiver, performative, f"incident-{content['driver_id']}",
                                         protocol="disruption", summary=summary, **content))

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        self._now, self._outbox = now, []
        for message in percepts:
            if message.performative is Performative.INFORM and message.protocol == "incident":
                self._handle(message)
        return self._outbox

    def _handle(self, incident: Message) -> None:
        self._incident = incident
        driver_id, text = incident.content["driver_id"], incident.content["text"]
        decision, how = self._decide(driver_id, text, incident.content.get("truth_minutes"))
        try:
            outcome = self.tools.call(decision["tool"], decision["args"])
        except ToolError as exc:  # the model proposed an invalid call: be cautious
            how = f"fallback (invalid call: {exc})"
            decision = {"tool": "cancel_driver", "args": {"driver_id": driver_id}, "reason": decision["reason"]}
            outcome = self.tools.call("cancel_driver", decision["args"])
        self.decisions.append({"time": self._now, "driver_id": driver_id, "text": text, "how": how, **decision,
                               "outcome": outcome})

    def _decide(self, driver_id: str, text: str, truth_minutes: int | None) -> tuple[dict, str]:
        cautious = {"tool": "cancel_driver", "args": {"driver_id": driver_id}, "reason": "treated as a cancellation"}
        if form := self._incident.content.get("form"):  # the app's form: the driver said what to do
            if form["action"] == "delay":
                return {"tool": "pause_driver", "args": {"driver_id": driver_id, "minutes": form["minutes"]},
                        "reason": "the driver will be late"}, "driver form"
            return {"tool": "cancel_driver", "args": {"driver_id": driver_id},
                    "reason": "the driver cannot continue"}, "driver form"
        if self.policy == "oracle" and truth_minutes is not None:
            if truth_minutes > 0:
                return {"tool": "pause_driver", "args": {"driver_id": driver_id, "minutes": truth_minutes},
                        "reason": "true duration"}, "oracle"
            return cautious, "oracle"
        if self.policy == "llm" and self.llm is not None:
            try:
                choice = choose_tool(self.llm, text, self.tools)
            except LLMError:
                return cautious, "fallback (LLM unavailable)"
            args: dict = {"driver_id": driver_id}
            if choice.tool == "pause_driver":
                args["minutes"] = choice.minutes
            elif choice.tool == "ask_driver":
                args["question"] = choice.question
            return {"tool": choice.tool, "args": args, "reason": choice.reason}, "llm"
        return cautious, "cautious"
