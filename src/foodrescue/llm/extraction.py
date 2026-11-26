"""Reading a merchant's message with the LLM.

Design rules for a small (4B) model:
* one narrow task: copy what the message says into fixed fields, nothing else;
* enums instead of free text wherever possible;
* a few worked examples, including Tunisian dialect;
* the model writes times as said ("9" stays 9); our code turns them into real times with
  `resolve_time`, and later turns quantities into kg. Arithmetic is not the model's job.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from foodrescue.core.clock import format_hhmm
from foodrescue.llm.client import LLMClient
from foodrescue.models import FoodCategory, Storage


class Unit(StrEnum):
    KG = "kg"
    G = "g"
    PIECE = "piece"
    PORTION = "portion"
    TRAY = "tray"
    CRATE = "crate"
    BOX = "box"
    BAG = "bag"
    LITER = "liter"
    LOAF = "loaf"
    CARTON = "carton"
    OTHER = "other"


class ExtractedItem(BaseModel):
    name: str = Field(description="short English name of the food")
    category: FoodCategory
    quantity: float | None = Field(description="number as written; null if not said")
    unit: Unit


class ClockTime(BaseModel):
    hour: int = Field(ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    day_offset: int = Field(default=0, ge=0, le=7, description="0 = today, 1 = tomorrow")


class ExtractedDonation(BaseModel):
    """What the model must return. Every field it cannot find in the message stays null.

    Field order matters: the model writes the fields in this order, so the deadline (almost
    always present) comes before the rarer start time. With the start first, a 4B model put
    "before 21h" into the start field in 7 of 10 test messages.
    """

    items: list[ExtractedItem] = Field(min_length=1)
    available_until: ClockTime | None = Field(
        default=None, description="latest time to collect: before / until / avant / jusqu'à / 9bal / قبل")
    available_from: ClockTime | None = Field(
        default=None, description="only if a start time is said, e.g. 'between 19h and 21h' gives 19:00")
    prepared_at: ClockTime | None = None
    expiry_days: int | None = Field(default=None, ge=0,
                                    description="days until the date on the product: 0 today, 1 tomorrow")
    storage: Storage | None = Field(
        default=None, description="only if the message mentions fridge, frozen, kept warm or room temperature")
    language: Literal["fr", "en", "ar", "mixed"] = "mixed"


SYSTEM_PROMPT = """\
You read messages from shops and restaurants that want to donate leftover food.
Messages can be in French, English, Arabic, Tunisian dialect (in Arabic letters, or in Latin letters
with digits such as 3, 5, 7, 9), or a mix of them.
Copy ONLY what the message says into the JSON fields. Never guess: if something is not said, use null.

Rules:
- items: one entry per distinct food. name = short English name.
- category: bakery (bread, pastries), produce (raw fruits and vegetables), dairy (milk, yogurt,
  cheese), meat_fish (RAW meat or fish), prepared_meal (cooked or ready-to-eat dishes, sandwiches,
  salads; cooked fish or meat too), dry_goods (packaged, canned, dry food), other.
- quantity and unit exactly as written. "about 8 kg" gives quantity 8, unit kg. Plates, dishes, parts
  (assiettes, parts, صحن, s7an) are unit portion.
- Times: use the 24-hour clock when am/pm, "evening" or "soir" makes it clear; otherwise copy
  the number as written. "et demi", "w nos", "و نص", "half past" add 30 minutes.
  day_offset is 0 (today) unless the message says tomorrow (demain, ghodwa, غدوة, غدا): then 1.
- available_until: the deadline to collect the food ("before 21h", "jusqu'à 22h", "9bal 9" all go here).
- available_from: only when the message gives a start ("between 19h and 21h": from 19:00, until 21:00).
  A single time with "before" or "until" is NEVER a start time.
- prepared_at: when cooked food was prepared, only if said. Words count as times: noon / midi / lunch
  = 12:00, "since noon" or "depuis ce midi" gives prepared_at 12:00, "this morning" = 08:00.
- "5odra", "khodra", "خضرة" mean raw vegetables (produce), not a cooked dish.
- expiry_days: days until the date on the product, only if said (tomorrow = 1). "Made today" is not an expiry.
- storage: ambient (room temperature), chilled (fridge), frozen, hot (kept warm), only if the message
  says it. Do not deduce it from the type of food.
- language: main language of the message (fr, en, ar, or mixed)."""


def _example(now: str, text: str, answer: ExtractedDonation) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Current time: {now}\nMessage: {text}"},
            {"role": "assistant", "content": answer.model_dump_json()}]


FEW_SHOT: list[dict[str, str]] = [
    *_example(
        "17:00",
        "Salut, on a 12 sandwichs thon et 3 kg de pâtes cuites au frigo, à prendre entre 20h et 21h30",
        ExtractedDonation(
            items=[ExtractedItem(name="tuna sandwiches", category=FoodCategory.PREPARED_MEAL,
                                 quantity=12, unit=Unit.PIECE),
                   ExtractedItem(name="cooked pasta", category=FoodCategory.PREPARED_MEAL,
                                 quantity=3, unit=Unit.KG)],
            available_until=ClockTime(hour=21, minute=30), available_from=ClockTime(hour=20),
            storage=Storage.CHILLED, language="fr"),
    ),
    *_example(
        "16:00",
        "3andi 2 cageots tmatem w 40 khobza, ta3dou 9bal 7",
        ExtractedDonation(
            items=[ExtractedItem(name="tomatoes", category=FoodCategory.PRODUCE, quantity=2, unit=Unit.CRATE),
                   ExtractedItem(name="bread", category=FoodCategory.BAKERY, quantity=40, unit=Unit.PIECE)],
            available_until=ClockTime(hour=7), language="ar"),
    ),
    *_example(
        "15:00",
        "لدينا 6 كيلو دجاج نيء صالح حتى بعد غد، الاستلام من الساعة 5 إلى 7 مساء",
        ExtractedDonation(
            items=[ExtractedItem(name="raw chicken", category=FoodCategory.MEAT_FISH, quantity=6, unit=Unit.KG)],
            available_until=ClockTime(hour=19), available_from=ClockTime(hour=17), expiry_days=2, language="ar"),
    ),
]


def build_messages(text: str, now: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT,
            {"role": "user", "content": f"Current time: {format_hhmm(now)}\nMessage: {text}"}]


def extract_donation(llm: LLMClient, text: str, now: int) -> ExtractedDonation:
    """Ask the model to structure one merchant message. `now` is in minutes since midnight."""
    return llm.chat_json(build_messages(text, now), ExtractedDonation)


def resolve_time(t: ClockTime | None, now: int, *, future: bool = True) -> int | None:
    """Turn a time as said ("before 9") into minutes since midnight, using the current time.

    future=True (pickup times): "9" said at 18:30 means 21:00, the next occurrence.
    future=False (when food was cooked): "6" said at 19:00 means 18:00, but "9" means 9:00,
    the most recent occurrence in the past.
    """
    if t is None:
        return None
    minutes = t.day_offset * 1440 + t.hour * 60 + t.minute
    if t.day_offset == 0 and t.hour < 12:
        afternoon = minutes + 12 * 60
        if future and minutes < now:
            return afternoon
        if not future and afternoon <= now:
            return afternoon
    return minutes
