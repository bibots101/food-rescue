"""Reading the merchant's optional details message (the "anything else?" box of the food form).

The form already gives the food, how it is kept, when it was cooked and the pickup times. The details message is
where a merchant writes what the form could not say. The model copies only the facts that matter for safety
(how the food was really kept, since when it has been out of the fridge, when it was cooked, the date on it,
allergens, anything wrong with it) and a one-sentence summary for the charity. Our code then decides what they
change (`apply_details`), and the safety rules judge the result. The model never approves or rejects food.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from foodrescue.core.clock import format_hhmm
from foodrescue.llm.client import LLMClient
from foodrescue.llm.extraction import ClockTime, resolve_time
from foodrescue.models import Donation, Storage

KEPT = {"room temperature": Storage.AMBIENT, "fridge": Storage.CHILLED, "hot": Storage.HOT,
        "frozen": Storage.FROZEN}


class DetailFacts(BaseModel):
    """What the model must return. Anything the message does not say stays null or empty."""

    kept: Literal["room temperature", "fridge", "hot", "frozen"] | None = Field(
        default=None, description="how the food is actually kept now, only if the message says it")
    out_of_fridge_since: ClockTime | None = Field(
        default=None, description="since when food has been at room temperature (left out, on the counter, the "
                                  "fridge broke...)")
    hours_out_of_fridge: float | None = Field(
        default=None, description="how long it has been at room temperature, if said as a duration")
    cooked_at: ClockTime | None = Field(default=None, description="when it was cooked, only if said")
    expiry_days: int | None = Field(default=None, ge=0, description="date on the product: 0 today, 1 tomorrow")
    allergens: list[str] = Field(default_factory=list, description="allergens mentioned: nuts, gluten, milk...")
    problems: list[str] = Field(default_factory=list,
                                description="anything wrong with the food's condition: smell, mould, damaged or "
                                            "opened packaging, partly eaten, cold chain broken. Not allergens.")
    confirmed_safe: bool = Field(default=False, description="true only if the merchant, answering our question, "
                                                            "explicitly confirms the food is still safe to eat")
    summary: str = Field(description="one short English sentence for the charity; empty if nothing useful")


SYSTEM_PROMPT = """\
A shop or restaurant donating leftover food filled in a form and added a free message with extra details.
The message can be in French or English.
Copy ONLY the facts the message states into the JSON fields. Never guess: if something is not said, use null
or an empty list. Times: 24-hour clock; "midi" or "noon" is 12:00, "1 pm" is 13:00.
- kept: how the food is kept now, whenever the message says it: "in the fridge", "kept cold", "au frais" ->
  fridge; "kept hot", "gardé au chaud" -> hot; "frozen", "congelé" -> frozen; left out, on the counter ->
  room temperature.
- out_of_fridge_since: the clock time since when food has been at room temperature ("since 2 pm", "depuis
  16h30", "the fridge broke at 16h"). hours_out_of_fridge: the same said as a duration, in hours ("for 3
  hours" -> 3, "90 minutes" -> 1.5). Never put a duration in out_of_fridge_since.
- cooked_at: the clock time the food was cooked or prepared ("cooked at 1 pm", "préparé à midi", "made around
  4" in the afternoon -> 16:00). Null if no time is given ("this morning" is not a time).
- expiry_days: the date on the product, in days from today: "today", "aujourd'hui" -> 0; "tomorrow",
  "demain" -> 1; "the day after tomorrow" -> 2; "in 3 days" -> 3.
- problems: anything wrong with the food itself (bad smell, mould, damaged, dented, crushed or opened
  packaging, partly eaten). Write them in short English. Do not put allergens here.
- allergens: every allergen the message names, directly or as an ingredient, written as its group: almonds,
  hazelnuts, walnuts or pistachios are "nuts"; peanuts, peanut butter are "peanuts"; wheat flour is "gluten";
  milk, cream, butter or cheese are "milk"; shrimp or prawns are "shellfish"; also "eggs", "fish", "sesame",
  "soy", "celery", "mustard". A food that is itself an allergen counts (fish fillets -> fish). "Gluten-free",
  "nut-free", "sans gluten" mean the allergen is NOT there: do not list it.
- confirmed_safe: true only if the message contains our question's answer ("Answer to your question: ...")
  and the merchant clearly confirms the food is still safe to eat. Otherwise false.
- summary: one short English sentence a charity would want to know; empty if the message adds nothing.
"""

EXAMPLES: list[tuple[str, DetailFacts]] = [
    ("Le riz est resté sur le comptoir depuis 14h, et il y a des noix dans le gâteau.",
     DetailFacts(kept="room temperature", out_of_fridge_since=ClockTime(hour=14), allergens=["nuts"],
                 summary="The rice has been on the counter since 14:00; the cake contains nuts.")),
    ("Two of the yogurt boxes are dented but sealed. Best before tomorrow.",
     DetailFacts(expiry_days=1, problems=["two boxes dented (sealed)"],
                 summary="Two yogurt boxes are dented but still sealed; best before tomorrow.")),
    ("Les croissants sont aux amandes. The custard tarts have been out of the fridge for 3 hours.",
     DetailFacts(hours_out_of_fridge=3, allergens=["nuts"],
                 summary="Almond croissants; the custard tarts have been out of the fridge for 3 hours.")),
    ("Tajine préparé à 17h, gardé au chaud, sans gluten.",
     DetailFacts(kept="hot", cooked_at=ClockTime(hour=17), summary="Tajine cooked at 17:00, kept hot, gluten-free.")),
    ("Answer to your question: yes, it stayed in the fridge, it is safe to eat.",
     DetailFacts(kept="fridge", confirmed_safe=True, summary="The merchant confirms it stayed in the fridge.")),
    ("Merci, à récupérer à l'arrière.",
     DetailFacts(summary="")),
]


def build_messages(text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, answer in EXAMPLES:
        messages += [{"role": "user", "content": f"Message: {example}"},
                     {"role": "assistant", "content": answer.model_dump_json(exclude_none=True)}]
    messages.append({"role": "user", "content": f"Message: {text}"})
    return messages


# Safety net: a small model sometimes names an allergen in its summary but leaves the list empty. Words for the
# common allergens (EU Regulation 1169/2011, Annex II) found in the merchant's own text are always kept, unless
# the text says the food is free of them ("gluten-free", "sans gluten", "no nuts").
ALLERGEN_WORDS = {
    "nuts": ("almond", "amande", "hazelnut", "noisette", "walnut", "noix", "pistachio", "pistache", "cashew",
             "cajou", "pecan", "nut"),
    "peanuts": ("peanut", "cacahuète", "cacahouète", "arachide"),
    "gluten": ("gluten",),
    "eggs": ("egg", "oeuf", "œuf"),
    "milk": ("milk", "lait", "lactose", "cream", "crème", "butter", "beurre", "cheese", "fromage"),
    "sesame": ("sesame", "sésame"),
    "fish": ("fish", "poisson"),
    "shellfish": ("shrimp", "prawn", "crevette", "shellfish", "crustacé", "crustacean", "fruits de mer",
                  "seafood"),
    "soy": ("soy", "soja"),
    "celery": ("celery", "céleri"),
    "mustard": ("mustard", "moutarde"),
}
NOT_ALLERGENS = ("laitue", "peanut butter")  # lettuce is not milk; peanut butter is peanuts, not milk
_NEGATION = re.compile(r"\b(?:sans|without|no|aucun|aucune|pas de|free of)\s+(?:\w+\s+){0,2}$")
_ANSWER_MARKERS = ("answer to your question", "réponse à votre question")


def allergens_named(text: str) -> list[str]:
    """Allergen groups named in the text as words (so "laitue" is not milk), unless negated ("gluten-free")."""
    lowered = text.lower()
    found = []
    for group, words in ALLERGEN_WORDS.items():
        for match in re.finditer(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es|x)?\b", lowered):
            if any(lowered.startswith(other, match.start()) for other in NOT_ALLERGENS if group == "milk") \
                    or (group == "milk" and lowered[max(0, match.start() - 7):match.start()] == "peanut "):
                continue
            if lowered[match.end():].startswith(("-free", " free")) or _NEGATION.search(lowered[:match.start()]):
                continue
            found.append(group)
            break
    return found


def read_details(llm: LLMClient, text: str) -> DetailFacts:
    return checked(llm.chat_json(build_messages(text), DetailFacts), text)


def checked(facts: DetailFacts, text: str) -> DetailFacts:
    """Our code's checks on the model's answer."""
    update: dict = {}
    missing = [a for a in allergens_named(text) if a not in facts.allergens]
    if missing:
        update["allergens"] = [*facts.allergens, *missing]
    # Only a person answering our question can confirm the food is safe (the app writes "Answer to your
    # question: ..." before the merchant's reply), never a description of the food.
    if facts.confirmed_safe and not any(marker in text.lower() for marker in _ANSWER_MARKERS):
        update["confirmed_safe"] = False
    return facts.model_copy(update=update) if update else facts


def apply_details(donation: Donation, facts: DetailFacts, now: int) -> Donation:
    """What the facts change in the donation, written down so the merchant can see why."""
    update: dict = {"allergens": facts.allergens, "problems": facts.problems, "details_summary": facts.summary}
    changes = []
    if facts.kept is not None and KEPT[facts.kept] is not donation.storage:
        update["storage"] = KEPT[facts.kept]
        changes.append(f"kept: {facts.kept}")
    cooked = resolve_time(facts.cooked_at, now, future=False)
    if cooked is not None and cooked != donation.prepared_at:
        update["prepared_at"] = cooked
        changes.append(f"cooked at {format_hhmm(cooked)}")
    since = resolve_time(facts.out_of_fridge_since, now, future=False)
    if since is None and facts.hours_out_of_fridge is not None:
        since = now - round(facts.hours_out_of_fridge * 60)
    if since is not None:
        update["storage"] = Storage.AMBIENT
        update["ambient_since"] = since
        changes.append(f"at room temperature since {format_hhmm(since)}")
    if facts.expiry_days is not None:
        update["expires_at"] = facts.expiry_days * 1440 + 1439
        changes.append("date on the product: " + ("today" if facts.expiry_days == 0 else
                                                  "tomorrow" if facts.expiry_days == 1 else
                                                  f"in {facts.expiry_days} days"))
    if facts.allergens:
        changes.append(f"allergens: {', '.join(facts.allergens)}")
    if facts.problems:
        changes.append(f"reported: {', '.join(facts.problems)}")
    if "storage" in update:  # said for the whole donation: it replaces what the form said per food
        update["items"] = [item.model_copy(update={"storage": None}) for item in donation.items]
    if facts.confirmed_safe:  # the merchant (a person) decided, after our question
        update["confirmed_by_merchant"] = True
        changes.append("confirmed safe by the merchant")
    update["details_changes"] = changes
    return donation.model_copy(update=update)
