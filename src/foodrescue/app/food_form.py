"""The merchant's "post food" form.

The form holds what the safety rules and the planning need (each food and how it is kept, when it was cooked, the
pickup times, the date on it). Each food line has its own storage: a shop gives bananas at room temperature and
milk from the fridge in the same donation, and the safety rules judge each food by how it is really kept.
An optional details message holds anything else; the intake agent's LLM reads it
for safety facts (`llm.details`). Food lines are added one by one with "Add food".
"""

from __future__ import annotations

import streamlit as st

from foodrescue.app import forms
from foodrescue.core.clock import format_hhmm
from foodrescue.llm.extraction import ExtractedItem, Unit
from foodrescue.llm.normalize import item_kg
from foodrescue.models import CATEGORY_LABELS, Donation, DonationOffer, FoodCategory, FoodItem, Storage, TimeWindow

UNITS = {"portions": Unit.PORTION, "kg": Unit.KG, "pieces": Unit.PIECE, "loaves": Unit.LOAF, "trays": Unit.TRAY,
         "crates": Unit.CRATE, "boxes": Unit.BOX, "bags": Unit.BAG, "litres": Unit.LITER}
STORAGE = {"Room temperature": Storage.AMBIENT, "In the fridge": Storage.CHILLED, "Kept hot": Storage.HOT,
           "Frozen": Storage.FROZEN}
USE_BY = {"No date on it": None, "Today": 1439, "Tomorrow": 2879, "Later": 7 * 1440 + 1439}  # end of that day
KINDS = {label: category for category, label in CATEGORY_LABELS.items()}
FIELDS = ("food", "kind", "qty", "unit", "kept")
KIND_HELP = ("Decides the safety rules and which charities can take it. Fresh dairy, meat and fish must stay in the "
             "fridge. Sealed food that keeps without a fridge (cans, pasta, UHT milk, juice cartons) is \"Packaged "
             "and dry food\".")


def _lines_key(merchant_id: str) -> str:
    return f"_food_lines_{merchant_id}"


def _lines(merchant_id: str) -> list[int]:
    """The ids of the food lines on screen (one at first)."""
    key = _lines_key(merchant_id)
    if key not in st.session_state:
        st.session_state[key] = [0]
        st.session_state[f"_food_next_{merchant_id}"] = 1
    return st.session_state[key]


def _add_line(merchant_id: str) -> None:
    line = st.session_state[f"_food_next_{merchant_id}"]
    st.session_state[f"_food_next_{merchant_id}"] = line + 1
    st.session_state[_lines_key(merchant_id)].append(line)


def _remove_line(merchant_id: str, line: int) -> None:
    st.session_state[_lines_key(merchant_id)].remove(line)
    for field in FIELDS:
        st.session_state.pop(f"{field}-{merchant_id}-{line}", None)


def _generation(merchant_id: str) -> int:
    """Changes once an offer is posted, so the other fields come back empty as new widgets (a text box keeps its
    text in the browser even when its value is deleted from the session)."""
    return st.session_state.get(f"_food_form_{merchant_id}", 0)


def post_food_form(merchant_id: str, now: int, next_id: str, disabled: bool) -> DonationOffer | None:
    """Draw the form; returns the offer to post when "Post this food" is pressed and everything is valid."""
    st.markdown("**What food?**")
    lines = _lines(merchant_id)
    gen = _generation(merchant_id)
    values = []
    for position, line in enumerate(lines):
        row = st.columns([2.6, 2.4, 1.2, 1.3, 2, 0.5], vertical_alignment="bottom")
        show = "visible" if position == 0 else "collapsed"
        values.append({
            "food": row[0].text_input("Food", key=f"food-{merchant_id}-{line}", label_visibility=show,
                                      placeholder="e.g. spaghetti bolognese"),
            "kind": row[1].selectbox("Kind", list(KINDS), key=f"kind-{merchant_id}-{line}", label_visibility=show,
                                     help=KIND_HELP),
            "qty": row[2].number_input("Quantity", min_value=0.0, step=1.0, value=0.0,
                                       key=f"qty-{merchant_id}-{line}", label_visibility=show),
            "unit": row[3].selectbox("Unit", list(UNITS), key=f"unit-{merchant_id}-{line}", label_visibility=show),
            "kept": row[4].selectbox("Kept", list(STORAGE), key=f"kept-{merchant_id}-{line}", label_visibility=show,
                                     help="How this food is kept right now"),
        })
        if len(lines) > 1:
            row[5].button("", key=f"remove-{merchant_id}-{line}", icon=":material/close:", help="Remove this line",
                          on_click=_remove_line, args=(merchant_id, line))
    st.button("Add food", key=f"add-food-{merchant_id}", icon=":material/add:", on_click=_add_line,
              args=(merchant_id,))

    times = st.columns(3)
    cooked = times[0].time_input("Cooked at (cooked meals)", value=None, step=900,
                                 key=f"cooked-{merchant_id}-{gen}",
                                 help="Needed for cooked meals: the 2-hour rule depends on it")
    pickup_from = times[1].time_input("Pickup from", forms.to_time(now), step=900,
                                      key=f"from-{merchant_id}-{gen}")
    pickup_until = times[2].time_input("Pickup until", forms.to_time(min(now + 180, 23 * 60 + 45)), step=900,
                                       key=f"until-{merchant_id}-{gen}")
    use_by = st.radio("Use-by date on the packaging", list(USE_BY), horizontal=True,
                      key=f"useby-{merchant_id}-{gen}")
    details = st.text_area(
        "Anything else about this food? (optional)", key=f"details-{merchant_id}-{gen}", height=80,
        placeholder="e.g. the rice was left on the counter since 14:00 · the cake contains nuts · one box is dented",
        help="Read by the AI: anything that matters for safety (left out of the fridge, allergens, damaged "
             "packaging...) is taken into account by the safety check, and the charity sees a summary.")
    if not st.button("Post this food", type="primary", icon=":material/send:", disabled=disabled,
                     key=f"post-food-{merchant_id}"):
        return None

    items, problems = [], []
    for value in values:
        name, quantity = value["food"].strip(), float(value["qty"] or 0)
        if not name:
            continue
        if quantity <= 0:
            problems.append(f"Give a quantity for {name}.")
            continue
        category, unit = KINDS[value["kind"]], UNITS[value["unit"]]
        kg = item_kg(ExtractedItem(name=name, category=category, quantity=quantity, unit=unit))
        items.append(FoodItem(name=name, category=category, quantity=quantity, unit=unit.value, kg=kg,
                              storage=STORAGE[value["kept"]]))
    start, end = max(forms.to_minutes(pickup_from), now), forms.to_minutes(pickup_until)
    if not items and not problems:
        problems.append("Add at least one food (its name and quantity).")
    if end <= start:
        problems.append("The pickup must end after it starts, and after now.")
    if any(i.category is FoodCategory.PREPARED_MEAL for i in items) and cooked is None:
        problems.append("Say when the cooked meals were made.")
    for problem in problems:
        st.error(problem)
    if problems:
        return None

    donation = Donation(id=next_id, merchant_id=merchant_id, items=items, storage=items[0].storage,
                        pickup_window=TimeWindow(start=start, end=end),
                        prepared_at=forms.to_minutes(cooked) if cooked is not None else None,
                        expires_at=USE_BY[use_by], created_at=now)
    said = {storage: label.lower() for label, storage in STORAGE.items()}
    if len(donation.storages) == 1:
        text = f"{donation.describe()} ({donation.total_kg:g} kg), {said[items[0].storage]}"
    else:
        text = ", ".join(f"{i.describe()} ({said[i.storage]})" for i in items) + f" ({donation.total_kg:g} kg)"
    text += f", pickup {format_hhmm(start)}-{format_hhmm(end)}"
    return DonationOffer(id=next_id, time=now, merchant_id=merchant_id, text=text, donation=donation,
                         from_form=True, details=details.strip())


def clear(merchant_id: str) -> None:
    """Back to one empty line once the offer is posted."""
    for line in list(_lines(merchant_id)):
        for field in FIELDS:
            st.session_state.pop(f"{field}-{merchant_id}-{line}", None)
    st.session_state[_lines_key(merchant_id)] = [st.session_state[f"_food_next_{merchant_id}"]]
    st.session_state[f"_food_next_{merchant_id}"] += 1
    st.session_state[f"_food_form_{merchant_id}"] = _generation(merchant_id) + 1
