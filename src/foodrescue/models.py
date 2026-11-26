"""Shared data model.

Agents exchange these objects inside messages as JSON-compatible dictionaries, so they never
share Python objects directly (like real independent agents). Times are integers: minutes
since midnight of the simulated day (see `core.clock`). In YAML files they can be written
as "HH:MM".
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from foodrescue.core.clock import parse_hhmm

Minutes = Annotated[int, BeforeValidator(parse_hhmm)]
"""Minutes since midnight; accepts an int or an "HH:MM" string."""


# --- Food ------------------------------------------------------------------------------------


class FoodCategory(StrEnum):
    BAKERY = "bakery"  # bread, pastries
    PRODUCE = "produce"  # raw fruits and vegetables
    DAIRY = "dairy"  # milk, yogurt, cheese
    MEAT_FISH = "meat_fish"  # raw meat or fish
    PREPARED_MEAL = "prepared_meal"  # cooked or ready-to-eat dishes
    DRY_GOODS = "dry_goods"  # packaged, canned, dry food
    OTHER = "other"


# Plain names shown to people. Cooked meals (ready to eat) are the riskiest food (2-hour rule, kept hot or
# cold, served quickly), so a charity needs a kitchen or a place to serve them; bread and pastries are low
# risk and keep a day or two. The categories drive the safety rules, the bidding and charity eligibility.
CATEGORY_LABELS: dict[FoodCategory, str] = {
    FoodCategory.PREPARED_MEAL: "Cooked meals (ready to eat)",
    FoodCategory.BAKERY: "Bread and pastries",
    FoodCategory.PRODUCE: "Fruit and vegetables",
    FoodCategory.DAIRY: "Dairy (fresh)",
    FoodCategory.MEAT_FISH: "Raw meat and fish",
    FoodCategory.DRY_GOODS: "Packaged and dry food",
    FoodCategory.OTHER: "Other",
}


class Storage(StrEnum):
    AMBIENT = "ambient"  # room temperature
    CHILLED = "chilled"  # fridge
    FROZEN = "frozen"
    HOT = "hot"  # kept warm


class FoodItem(BaseModel):
    name: str
    category: FoodCategory
    quantity: float
    unit: str
    kg: float
    # How this food is kept, when the merchant said it per food (the form); None = the donation's storage.
    storage: Storage | None = None

    def describe(self) -> str:
        quantity = f"{self.quantity:g}"
        return f"{quantity} {self.name}" if self.unit == "piece" else f"{quantity} {self.unit} {self.name}"


# --- Places and people -----------------------------------------------------------------------


class Location(BaseModel):
    lat: float
    lon: float
    address: str = ""


class TimeWindow(BaseModel):
    start: Minutes
    end: Minutes

    def contains(self, t: int) -> bool:
        return self.start <= t <= self.end


class Merchant(BaseModel):
    id: str
    name: str
    kind: str  # bakery, restaurant, supermarket, ...
    location: Location
    language: str = "fr"
    phone: str = ""
    pickup_notes: str = ""  # shown to the driver, e.g. "back door, ring twice"


class Charity(BaseModel):
    """A charity. Capacity, storage, accepted food and needs are private: only its own agent knows them."""

    id: str
    name: str
    location: Location
    opening: TimeWindow
    capacity_kg: float  # how much food it can still receive today
    has_fridge: bool  # needed for chilled food
    has_freezer: bool = False  # needed for frozen food
    # The kinds of food it can use at all (a food bank handing out parcels may not take cooked meals).
    accepted: set[FoodCategory] = Field(default_factory=lambda: set(FoodCategory))
    needs: dict[FoodCategory, float] = Field(default_factory=dict)  # kg wanted today, per category
    phone: str = ""
    notes: str = ""
    # Probability that the charity confirms a donation of this category after winning it (missing = 1.0).
    # Hidden from every other agent: the matching agent can only learn it from experience.
    acceptance: dict[FoodCategory, float] = Field(default_factory=dict)

    def acceptance_of(self, categories: set[FoodCategory]) -> float:
        probability = 1.0
        for category in categories:
            probability *= self.acceptance.get(category, 1.0)
        return probability


VEHICLES = {"bike": "Bike", "car": "Car", "truck": "Truck"}
VEHICLE_CAPACITY = {"bike": 15.0, "car": 60.0, "truck": 300.0}  # suggested kg when a driver picks a vehicle


class Driver(BaseModel):
    id: str
    name: str
    home: Location
    capacity_kg: float  # what the driver says their vehicle carries
    shift: TimeWindow
    phone: str = ""
    vehicle: Literal["bike", "car", "truck"] = "car"


class Place(BaseModel):
    """Public information about a merchant or a charity: what every agent may know."""

    id: str
    name: str
    location: Location
    opening: TimeWindow | None = None  # charities' opening hours are public


class Directory(BaseModel):
    """Public address book of all merchants and charities."""

    places: dict[str, Place]

    def get(self, place_id: str) -> Place:
        return self.places[place_id]

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> Directory:
        places = [Place(id=m.id, name=m.name, location=m.location) for m in scenario.merchants]
        places += [Place(id=c.id, name=c.name, location=c.location, opening=c.opening) for c in scenario.charities]
        return cls(places={p.id: p for p in places})


# --- Donations -------------------------------------------------------------------------------


class DonationStatus(StrEnum):
    RECEIVED = "received"
    APPROVED = "approved"  # passed the safety check
    REJECTED = "rejected"  # unsafe
    ASSIGNED = "assigned"  # a charity won the Contract Net
    SCHEDULED = "scheduled"  # on a driver's route
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    UNMATCHED = "unmatched"  # no charity or no driver could take it
    EXPIRED = "expired"


class Donation(BaseModel):
    id: str
    merchant_id: str
    items: list[FoodItem]
    storage: Storage
    pickup_window: TimeWindow
    prepared_at: Minutes | None = None  # when cooked food was prepared
    expires_at: Minutes | None = None  # date on the product
    created_at: Minutes = 0
    raw_text: str = ""
    status: DonationStatus = DonationStatus.RECEIVED
    # From the merchant's details message, read by the LLM (`llm.details`)
    details: str = ""  # the merchant's own words
    details_summary: str = ""  # one sentence for the charity
    details_changes: list[str] = Field(default_factory=list)  # what the details changed, said plainly
    allergens: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)  # anything wrong with the food's condition
    ambient_since: Minutes | None = None  # at room temperature since (the 2-hour rule counts from here)
    # The merchant answered our question and confirmed the food is still safe: the questions that only a person
    # can settle (R11, R12) are not asked again. Missing facts (when was it cooked?) are still asked.
    confirmed_by_merchant: bool = False

    @property
    def total_kg(self) -> float:
        return round(sum(item.kg for item in self.items), 2)

    def storage_of(self, item: FoodItem) -> Storage:
        """How this food is kept: its own storage if the merchant gave one, otherwise the donation's."""
        return item.storage or self.storage

    @property
    def storages(self) -> set[Storage]:
        return {self.storage_of(item) for item in self.items} or {self.storage}

    def storage_text(self) -> str:
        """"chilled", or "mixed: Banane ambient, Milk chilled" when the foods are not kept the same way."""
        if len(self.storages) <= 1:
            return next(iter(self.storages)).value
        return "mixed: " + ", ".join(f"{i.name} {self.storage_of(i).value}" for i in self.items)

    @property
    def categories(self) -> set[FoodCategory]:
        return {item.category for item in self.items}

    def describe(self) -> str:
        """Short summary such as "30 croissants, 5 kg cooked rice"."""
        return ", ".join(item.describe() for item in self.items)


class Bid(BaseModel):
    """A charity's answer to a call for proposals."""

    donation_id: str
    charity_id: str
    accept_kg: float
    score: float  # the charity's own estimate of how useful the donation is to it (0 to 1)


# --- Routes ----------------------------------------------------------------------------------


class StopKind(StrEnum):
    PICKUP = "pickup"
    DELIVERY = "delivery"


class Stop(BaseModel):
    kind: StopKind
    donation_id: str
    place_id: str  # merchant (pickup) or charity (delivery)
    location: Location
    eta: Minutes | None = None


class Route(BaseModel):
    driver_id: str
    stops: list[Stop] = Field(default_factory=list)
    distance_km: float = 0.0


# --- Messages (FIPA-ACL inspired) ------------------------------------------------------------


class Performative(StrEnum):
    """The intention behind a message, from the FIPA agent communication language."""

    CFP = "cfp"  # call for proposals
    PROPOSE = "propose"
    REFUSE = "refuse"
    ACCEPT_PROPOSAL = "accept_proposal"
    REJECT_PROPOSAL = "reject_proposal"
    INFORM = "inform"
    REQUEST = "request"
    CANCEL = "cancel"
    FAILURE = "failure"


class Message(BaseModel):
    id: int = 0  # set by the bus when the message is sent
    time: Minutes
    sender: str
    receiver: str  # an agent name, or "@group" for every member of a group
    performative: Performative
    conversation_id: str  # usually the donation id
    protocol: str = ""  # e.g. "intake", "safety-check", "contract-net", "delivery"
    content: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""  # one readable line for the trace and the dashboard


# --- Scenario (the environment of one simulated day) ------------------------------------------


class DonationOffer(BaseModel):
    """Environment event: a merchant posts surplus food at `time`."""

    id: str
    time: Minutes
    merchant_id: str
    text: str
    # The structured version of the text. Used when LLM intake is switched off, and by the
    # simulator, which never calls the LLM.
    donation: Donation | None = None
    # The merchant's answer to a follow-up question about this same donation (same id). The LLM
    # then reads the original text and the answer together.
    answer: str | None = None
    # Posted with the app's form (every field filled in): the intake agent uses `donation` as it is, without
    # asking the LLM to read anything.
    from_form: bool = False
    details: str = ""  # the form's optional "anything else?" message, read by the LLM

    def full_text(self) -> str:
        return self.text if not self.answer else f"{self.text}\n(Answer to our question: {self.answer})"


class DisruptionKind(StrEnum):
    DRIVER_CANCEL = "driver_cancel"  # target: a driver id
    MERCHANT_DELAY = "merchant_delay"  # target: a donation id; the food is ready `minutes` later
    # target: a driver id; `reason` is the driver's own message (any language). `minutes` is the truth, used
    # only to evaluate: >0 the driver is back after that many minutes, 0 the driver is out for the night.
    DRIVER_INCIDENT = "driver_incident"


class Disruption(BaseModel):
    """Environment event: something goes wrong at `time`."""

    time: Minutes
    kind: DisruptionKind
    target_id: str
    reason: str = ""
    minutes: int = 0
    # Driver incidents reported through the app's form: "delay" (late by `minutes`) or "cancel"
    # (cannot continue). None: a free-text report, read by the LLM dispatcher.
    action: str | None = None


class Scenario(BaseModel):
    name: str
    seed: int = 0
    city: str = ""
    start: Minutes
    end: Minutes
    merchants: list[Merchant]
    charities: list[Charity]
    drivers: list[Driver]
    offers: list[DonationOffer] = Field(default_factory=list)
    disruptions: list[Disruption] = Field(default_factory=list)
