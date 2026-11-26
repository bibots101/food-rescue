"""From what the model read to a `Donation` the system can use.

The model only copies what the message says. This module does the arithmetic and the
assumptions, in plain code that can be tested and explained:
  * quantities to kg, with typical weights per unit (and per item for "pieces"),
  * times as said ("before 9") to minutes since midnight (`resolve_time`),
  * a pickup window when the message gives no start or no end,
  * a storage assumption when the message does not say (chosen so the safety rules stay cautious).
"""

from __future__ import annotations

from foodrescue.llm.extraction import ExtractedDonation, ExtractedItem, Unit, resolve_time
from foodrescue.models import Donation, FoodCategory, FoodItem, Storage, TimeWindow

# Typical weight of one unit, in kg.
UNIT_KG: dict[Unit, float] = {
    Unit.KG: 1.0, Unit.G: 0.001, Unit.LITER: 1.0, Unit.PORTION: 0.4, Unit.TRAY: 3.0, Unit.CRATE: 10.0,
    Unit.BOX: 2.0, Unit.BAG: 2.0, Unit.LOAF: 0.5, Unit.CARTON: 1.0, Unit.OTHER: 1.0,
}

# Weight of one piece, found by keyword in the item name (several languages).
PIECE_KG: list[tuple[tuple[str, ...], float]] = [
    (("croissant",), 0.07),
    (("baguette",), 0.25),
    (("tabouna", "bread", "pain", "khobz", "khobza", "loaf"), 0.2),
    (("yogurt", "yoghurt", "yaourt", "ياغورت"), 0.125),
    (("sandwich",), 0.2),
    (("cake", "gâteau", "gateau"), 0.5),
    (("can", "conserve"), 0.4),
]
DEFAULT_PIECE_KG = 0.2

# When the message does not say how the food is stored. Shops keep dairy, meat and fish in the
# fridge. For cooked food we assume room temperature: the cautious choice, so the 2-hour rule
# applies (and the merchant is asked when it was cooked).
DEFAULT_STORAGE: dict[FoodCategory, Storage] = {
    FoodCategory.DAIRY: Storage.CHILLED,
    FoodCategory.MEAT_FISH: Storage.CHILLED,
}

DEFAULT_PICKUP_HOURS = 3  # pickup window when the message gives no deadline


def item_kg(item: ExtractedItem) -> float:
    quantity = item.quantity if item.quantity is not None else 1.0
    if item.unit is Unit.PIECE:
        name = item.name.lower()
        per_piece = next((kg for keywords, kg in PIECE_KG if any(k in name for k in keywords)), DEFAULT_PIECE_KG)
        return round(quantity * per_piece, 2)
    return round(quantity * UNIT_KG[item.unit], 2)


def to_donation(extracted: ExtractedDonation, *, donation_id: str, merchant_id: str, now: int,
                raw_text: str = "") -> Donation:
    items = [FoodItem(name=i.name, category=i.category, quantity=i.quantity if i.quantity is not None else 1.0,
                      unit=i.unit.value, kg=item_kg(i)) for i in extracted.items]

    start = resolve_time(extracted.available_from, now) or now
    start = max(start, now)
    end = resolve_time(extracted.available_until, now)
    if end is None or end <= start:
        end = start + DEFAULT_PICKUP_HOURS * 60

    storage = extracted.storage
    if storage is None:
        categories = {i.category for i in items}
        storage = next((DEFAULT_STORAGE[c] for c in DEFAULT_STORAGE if c in categories), Storage.AMBIENT)

    expires_at = extracted.expiry_days * 1440 + 1439 if extracted.expiry_days is not None else None
    return Donation(
        id=donation_id, merchant_id=merchant_id, items=items, storage=storage,
        pickup_window=TimeWindow(start=start, end=end),
        prepared_at=resolve_time(extracted.prepared_at, now, future=False),
        expires_at=expires_at, created_at=now, raw_text=raw_text,
    )
