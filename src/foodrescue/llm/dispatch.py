"""The dispatcher's language task: read a driver's problem report and choose one tool.

This is single-step tool use, implemented by us: the tools' descriptions and JSON schemas come from
our `ToolRegistry`, the model must answer with JSON naming one tool and its arguments (constrained by
Ollama's JSON-schema mode), and our code validates the call through the registry before running it.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from foodrescue.core.tools import ToolRegistry
from foodrescue.llm.client import LLMClient

ToolName = Literal["pause_driver", "cancel_driver", "ask_driver"]


class DispatchDecision(BaseModel):
    """What the model must return."""

    reason: str = Field(description="one short sentence: what the driver's message means")
    tool: ToolName
    minutes: int | None = Field(default=None, description="pause_driver only: minutes until the driver can continue")
    question: str | None = Field(default=None, description="ask_driver only: a short question in the driver's language")


RULES = """\
You are the dispatcher of a food rescue service. A volunteer driver reports a problem in their own words
(French, English, Arabic, Tunisian dialect in Arabic or Latin letters). Choose exactly ONE tool:
- pause_driver: the driver says they can continue and roughly when. Give minutes: "half an hour", "nos sa3a",
  "نص ساعة" = 30; "an hour", "une heure", "sa3a", "ساعة" = 60; "20 min" = 20.
- cancel_driver: the driver cannot continue tonight, or the problem is serious (accident, car needs a tow,
  sick, police). Food in the car will be collected by another driver.
- ask_driver: it is unclear whether or when the driver can continue. Ask one short question in the
  driver's language.
Never guess a duration that is not said: if a pause has no duration, use ask_driver.
Tools available (JSON schemas):
"""

EXAMPLES = [
    ("Pneu crevé, je change la roue, je repars dans 20 minutes", DispatchDecision(
        reason="flat tire, back in 20 minutes", tool="pause_driver", minutes=20)),
    ("Had an accident, nobody hurt but the car must be towed. Sorry, I'm done for tonight.", DispatchDecision(
        reason="accident, the car must be towed, cannot continue", tool="cancel_driver")),
    ("3andi mochkla fel karhba", DispatchDecision(
        reason="a problem with the car, no details", tool="ask_driver",
        question="Chnowa el mochkla? Tnajjem tkammel, w waktech?")),
]


def build_messages(text: str, registry: ToolRegistry) -> list[dict[str, str]]:
    tools = json.dumps([t for t in registry.describe() if t["name"] in ToolName.__args__], indent=1)
    messages = [{"role": "system", "content": RULES + tools}]
    for example, answer in EXAMPLES:
        messages += [{"role": "user", "content": f"Driver's message: {example}"},
                     {"role": "assistant", "content": answer.model_dump_json(exclude_none=True)}]
    messages.append({"role": "user", "content": f"Driver's message: {text}"})
    return messages


def choose_tool(llm: LLMClient, text: str, registry: ToolRegistry) -> DispatchDecision:
    return llm.chat_json(build_messages(text, registry), DispatchDecision)
