"""Random city generator: merchants, charities, drivers and an evening of donation offers.

Everything comes from one seed, so the same seed always gives the same day. That is what
lets us compare our system and the baseline on exactly the same days (common random numbers).
"""

from __future__ import annotations

import random

from foodrescue.core.clock import MINUTES_PER_DAY, format_hhmm
from foodrescue.models import (
    Charity,
    Disruption,
    DisruptionKind,
    Donation,
    DonationOffer,
    Driver,
    FoodCategory,
    FoodItem,
    Merchant,
    Scenario,
    Storage,
    TimeWindow,
)
from foodrescue.optim.geo import BoundingBox, random_location

TUNIS_BBOX: BoundingBox = (36.78, 10.12, 36.87, 10.25)

# What each kind of merchant typically gives away: (name, category, unit, kg per unit, max quantity)
CATALOG: dict[str, list[tuple[str, FoodCategory, str, float, int]]] = {
    "bakery": [
        ("bread", FoodCategory.BAKERY, "piece", 0.25, 60),
        ("croissants", FoodCategory.BAKERY, "piece", 0.07, 40),
        ("cakes", FoodCategory.BAKERY, "piece", 0.5, 10),
    ],
    "restaurant": [
        ("cooked meals", FoodCategory.PREPARED_MEAL, "portion", 0.4, 30),
        ("salads", FoodCategory.PREPARED_MEAL, "portion", 0.3, 15),
    ],
    "supermarket": [
        ("yogurts", FoodCategory.DAIRY, "piece", 0.125, 48),
        ("vegetables", FoodCategory.PRODUCE, "kg", 1.0, 20),
        ("fruit", FoodCategory.PRODUCE, "kg", 1.0, 20),
        ("canned food", FoodCategory.DRY_GOODS, "piece", 0.4, 30),
    ],
    "greengrocer": [
        ("vegetables", FoodCategory.PRODUCE, "kg", 1.0, 25),
        ("fruit", FoodCategory.PRODUCE, "kg", 1.0, 25),
    ],
}

FIRST_NAMES = ["Sami", "Leila", "Youssef", "Amira", "Karim", "Nour", "Hedi", "Ines"]


def generate_scenario(
    seed: int = 0,
    *,
    n_merchants: int = 10,
    n_charities: int = 5,
    n_drivers: int = 3,
    n_offers: int = 15,
    cancellations: int = 0,
    delays: int = 0,
    incidents: int = 0,
    day: int | None = None,
    preferences: bool = False,
    start: int = 18 * 60,
    end: int = 23 * 60,
    bbox: BoundingBox = TUNIS_BBOX,
) -> Scenario:
    """`cancellations` drivers cancel at random times. They come from a separate random stream, so the
    same seed gives the same day (merchants, offers, ...) whatever the number of cancellations.

    `day`: with the same seed, every day has the same city (merchants, charities, drivers) and new offers,
    which is what learning over several days needs. `preferences`: each charity often declines two kinds
    of food after winning them (hidden `Charity.acceptance`), the signal the learner learns from."""
    rng = random.Random(seed)
    kinds = list(CATALOG)

    merchants = [
        Merchant(id=f"M{i:02d}", name=f"{kind.title()} {i:02d}", kind=kind,
                 location=random_location(rng, bbox))
        for i, kind in ((i, rng.choice(kinds)) for i in range(1, n_merchants + 1))
    ]

    charities = []
    for i in range(1, n_charities + 1):
        wanted = rng.sample(list(FoodCategory)[:6], k=rng.randint(2, 4))
        charities.append(Charity(
            id=f"C{i:02d}", name=f"Charity {i:02d}", location=random_location(rng, bbox),
            opening=TimeWindow(start=start - rng.choice([0, 60, 120]),
                               end=end - rng.choice([0, 30, 60, 90])),
            capacity_kg=float(rng.choice([30, 50, 80, 120])),
            has_fridge=rng.random() < 0.6,
            needs={category: float(rng.choice([10, 20, 30])) for category in wanted},
        ))

    drivers = [
        Driver(id=f"D{i}", name=FIRST_NAMES[(i - 1) % len(FIRST_NAMES)],
               home=random_location(rng, bbox), capacity_kg=float(rng.choice([60, 80, 100])),
               shift=TimeWindow(start=start, end=end))
        for i in range(1, n_drivers + 1)
    ]

    if preferences:
        preference_rng = random.Random(f"{seed}-preferences")
        food = list(FoodCategory)[:6]
        for charity in charities:
            disliked = preference_rng.sample(food, k=2)
            charity.acceptance = {c: (round(preference_rng.uniform(0.1, 0.4), 2) if c in disliked else 0.95)
                                  for c in food}

    offer_rng = rng if day is None else random.Random(f"{seed}-day{day}")
    offers = []
    for i in range(1, n_offers + 1):
        merchant = offer_rng.choice(merchants)
        posted = offer_rng.randint(start, max(start, end - 90))
        donation = _random_donation(offer_rng, f"O{i:02d}", merchant, posted, end)
        offers.append(DonationOffer(
            id=donation.id, time=posted, merchant_id=merchant.id, donation=donation,
            text=f"{merchant.name}: {donation.describe()}, pickup before "
                 f"{format_hhmm(donation.pickup_window.end)}",
        ))
    offers.sort(key=lambda offer: (offer.time, offer.id))

    disruption_rng = random.Random(f"{seed}-disruptions" + (f"-day{day}" if day is not None else ""))
    cancelled = disruption_rng.sample(drivers, k=min(cancellations, len(drivers)))
    disruptions = [Disruption(time=disruption_rng.randint(start + 60, end - 60), kind=DisruptionKind.DRIVER_CANCEL,
                              target_id=driver.id, reason="car broke down") for driver in cancelled]
    # Late merchants: a few minutes after posting, the merchant says the food will be ready 30-60 min later.
    for offer in disruption_rng.sample(offers, k=min(delays, len(offers))):
        disruptions.append(Disruption(
            time=offer.time + disruption_rng.randint(5, 25), kind=DisruptionKind.MERCHANT_DELAY,
            target_id=offer.id, minutes=disruption_rng.choice([30, 45, 60]), reason="food not ready yet"))
    # Driver incidents: half are short stops (the driver gives a duration), half end the driver's night.
    incident_rng = random.Random(f"{seed}-incidents" + (f"-day{day}" if day is not None else ""))
    for driver in incident_rng.sample(drivers, k=min(incidents, len(drivers))):
        minutes = incident_rng.choice([15, 30, 45, 60]) if incident_rng.random() < 0.5 else 0
        text = (f"Flat tire, I can continue in {minutes} minutes" if minutes
                else "The car broke down and needs a tow, I cannot continue tonight")
        disruptions.append(Disruption(time=incident_rng.randint(start + 60, end - 60),
                                      kind=DisruptionKind.DRIVER_INCIDENT, target_id=driver.id, reason=text,
                                      minutes=minutes))
    disruptions.sort(key=lambda d: (d.time, d.target_id))

    name = (f"random-{seed}" + (f"-day{day}" if day is not None else "")
            + (f"-cancel{cancellations}" if cancellations else "") + (f"-late{delays}" if delays else "")
            + (f"-incident{incidents}" if incidents else ""))
    return Scenario(name=name, seed=seed if day is None else seed * 1000 + day, city="Tunis", start=start, end=end,
                    merchants=merchants, charities=charities, drivers=drivers, offers=offers, disruptions=disruptions)


def _random_donation(rng: random.Random, donation_id: str, merchant: Merchant, posted: int,
                     day_end: int) -> Donation:
    choices = rng.sample(CATALOG[merchant.kind], k=rng.randint(1, min(2, len(CATALOG[merchant.kind]))))
    items = []
    for name, category, unit, kg_per_unit, max_quantity in choices:
        quantity = rng.randint(max(1, max_quantity // 4), max_quantity)
        items.append(FoodItem(name=name, category=category, quantity=quantity, unit=unit,
                              kg=round(quantity * kg_per_unit, 2)))

    categories = {item.category for item in items}
    if FoodCategory.PREPARED_MEAL in categories:
        storage = Storage.HOT if rng.random() < 0.3 else Storage.CHILLED
    elif categories & {FoodCategory.DAIRY, FoodCategory.MEAT_FISH}:
        storage = Storage.CHILLED
    else:
        storage = Storage.AMBIENT

    window_end = min(day_end, posted + rng.randint(90, 180))
    return Donation(
        id=donation_id, merchant_id=merchant.id, items=items, storage=storage,
        pickup_window=TimeWindow(start=posted + 10, end=window_end),
        prepared_at=posted - rng.randint(30, 240) if FoodCategory.PREPARED_MEAL in categories else None,
        expires_at=MINUTES_PER_DAY + 23 * 60 + 59 if FoodCategory.DAIRY in categories else None,
        created_at=posted,
    )
