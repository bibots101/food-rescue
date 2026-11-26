"""Logistics agent: decides which driver carries each donation, in which order, and when.

It keeps a plan for every driver and re-solves it with OR-Tools (`optim.routing`) whenever
something changes. This is rolling-horizon replanning:

* What has already happened is frozen. The agent assumes drivers follow the plan, so stops whose
  time has passed are done. A donation that is on board stays with its driver until it is
  delivered, so no donation is ever left half-done.
* Everything not yet started is planned again from each driver's current position and time.
* Stable plans: when a new donation arrives, donations already given to a driver stay with that
  driver (only the order and times may change). After a disruption, everything can move.

Progress: drivers follow the plan, so when a stop's planned time passes, the pickup or the delivery has
happened. The agent records it once (the donation becomes "picked up", then "delivered") and tells the
communication agent, which tells the merchant and the charity. A donation counts as delivered only then.

Drivers accept each delivery (`features.driver_confirmation`): a delivery given to a driver is first
proposed to them (PROPOSE, protocol "job-offer"). Until they accept, nothing of it happens and the merchant is not
told; if they decline or do not answer within `offer_timeout` minutes, it is planned again without them. A
decline is final; a driver who only let the request expire is asked again once every other working driver has had
their chance, so a delivery is never left without anyone to ask. Once a
driver has a delivery (accepted, or proposed and waiting), it stays theirs: another driver coming on duty or being
closer never takes it. Only a problem of that driver moves it (cancellation, off duty, pause).

Disruptions handled:
* a driver cancels: their remaining pickups go to other drivers, and food already **on board** is
  collected from the broken-down car (a transfer) and delivered by another driver;
* a merchant is late: the pickup window moves and the plan is re-solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foodrescue.config import GeoConfig, RoutingConfig
from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.clock import format_hhmm
from foodrescue.core.tools import ToolRegistry
from foodrescue.models import (
    Directory,
    Donation,
    DonationStatus,
    Driver,
    Location,
    Message,
    Performative,
    Route,
    Stop,
    StopKind,
    TimeWindow,
)
from foodrescue.optim.geo import road_km
from foodrescue.optim.routing import Task, Vehicle, solve_routes

TRANSFER_MINUTES = 90  # food taken out of a broken-down car must be collected within this time


@dataclass
class Transfer:
    """Food on board a cancelled driver's car: collect it there instead of at the merchant."""

    place_id: str  # e.g. "D2-car"
    name: str  # e.g. "Leila's car"
    location: Location
    window: tuple[int, int]


@dataclass
class Job:
    donation: Donation
    charity_id: str
    driver_id: str | None = None
    pickup_eta: int | None = None
    delivery_eta: int | None = None
    transfer: Transfer | None = None
    pinned_driver: str | None = None  # food on board a paused driver's car stays with that driver
    history: list[str] = field(default_factory=list)  # what happened to it, for notifications
    # Where to pick up and deliver, fixed when the job is created: a merchant or charity who edits their
    # profile later does not move food already planned.
    pickup_location: Location | None = None
    delivery_location: Location | None = None
    picked_up_at: int | None = None  # when the food was really picked up
    delivered_at: int | None = None  # when it was really delivered
    confirmed: bool = False  # its driver accepted it
    confirmed_by: str | None = None
    offered_to: str | None = None  # waiting for this driver's answer...
    offered_at: int | None = None  # ...since then
    declined_by: set[str] = field(default_factory=set)  # said no: never asked again for this delivery
    expired_for: set[str] = field(default_factory=set)  # did not answer in time: asked again when nobody else is left

    def excluded(self) -> set[str]:
        return self.declined_by | self.expired_for


class LogisticsAgent(Agent):
    def __init__(self, name: str, bus: MessageBus, tools: ToolRegistry | None, directory: Directory,
                 drivers: list[Driver], geo: GeoConfig | None = None, routing: RoutingConfig | None = None,
                 replanning: bool = True, confirmation: bool = False, offer_timeout: int = 5) -> None:
        super().__init__(name, bus, tools)
        self.replanning = replanning  # False only for the ablation study
        self.confirmation = confirmation  # drivers accept each delivery
        self.offer_timeout = offer_timeout
        self.lost: set[str] = set()  # donations lost to a cancellation (only without replanning)
        self.directory = directory
        self.geo = geo or GeoConfig()
        self.routing = routing or RoutingConfig()
        self.drivers = {driver.id: driver for driver in drivers}
        self.active = set(self.drivers)
        self.routes = {driver.id: Route(driver_id=driver.id) for driver in drivers}
        self.jobs: dict[str, Job] = {}
        self.replans = 0
        self.transfers = 0
        self.delays: dict[str, int] = {}  # merchant delays (minutes) announced before the donation reached us
        self._unfrozen: set[str] = set()  # delayed donations: a driver on the way is not committed to them
        self.paused: dict[str, tuple[int, Location]] = {}  # driver -> (available again at, where the car is)

    def decide(self, now: int, percepts: list[Message]) -> list[Message]:
        actions: list[Message] = []
        for message in percepts:
            if message.performative is Performative.REQUEST and message.protocol == "delivery":
                job = Job(Donation.model_validate(message.content["donation"]), message.content["charity_id"])
                job.pickup_location = self.directory.get(job.donation.merchant_id).location
                job.delivery_location = self.directory.get(job.charity_id).location
                self.jobs[job.donation.id] = job
                if job.donation.id in self.delays:
                    self._apply_delay(job, now, self.delays.pop(job.donation.id))
                moved = self._replan(now, keep_assignments=True)
                if not self.confirmation or job.driver_id is None:  # with confirmation: told once accepted
                    actions.append(self.message(now, "communication", Performative.INFORM, job.donation.id,
                                                protocol="delivery", summary=self._describe(job),
                                                plan=self._plan(job)))
                moved.discard(job.donation.id)
                if moved:
                    actions.append(self._replan_message(now, job.donation.id, moved, "new donation"))
            elif message.performative is Performative.CANCEL and message.protocol == "disruption":
                driver_id = message.content["driver_id"]
                if driver_id not in self.active:
                    continue
                if self.replanning:
                    self._plan_transfers(driver_id, now)
                    self.active.discard(driver_id)
                    self.paused.pop(driver_id, None)
                    moved = self._replan(now, keep_assignments=False)
                else:
                    self.active.discard(driver_id)
                    moved = self._drop(driver_id, now)
                reason = f"{driver_id} cancelled ({message.content.get('reason', '')})"
                actions.append(self._replan_message(now, message.conversation_id, moved, reason,
                                                    cancelled_driver=driver_id))
            elif (message.performative is Performative.INFORM and message.protocol == "disruption"
                  and message.content.get("kind") == "merchant_delay"):
                actions += self._merchant_delay(now, message)
            elif (message.performative is Performative.REQUEST and message.protocol == "disruption"
                  and message.content.get("kind") == "driver_pause"):
                driver_id, minutes = message.content["driver_id"], int(message.content["minutes"])
                if driver_id in self.active and self.replanning:
                    self._pause_driver(driver_id, now, minutes)
                    moved = self._replan(now, keep_assignments=False)
                    actions.append(self._replan_message(now, message.conversation_id, moved,
                                                        f"{driver_id} paused for {minutes} min"))
            elif message.performative is Performative.REQUEST and message.protocol == "availability":
                actions += self._availability(now, message.content["driver_id"], bool(message.content["available"]))
            elif message.protocol == "job-offer" and message.performative in (Performative.ACCEPT_PROPOSAL,
                                                                            Performative.REJECT_PROPOSAL):
                actions += self._answer(now, message)
        if self.confirmation:
            actions += self._expire(now)
            actions += self._offers(now)
        return actions + self._progress(now)

    # --- driver confirmation -------------------------------------------------------------

    def pending_offers(self, driver_id: str) -> list[tuple[Job, int]]:
        """Deliveries waiting for this driver's answer, with the minute the request expires."""
        return [(job, job.offered_at + self.offer_timeout) for _, job in sorted(self.jobs.items())
                if job.offered_to == driver_id and not job.confirmed and job.offered_at is not None]

    def _answer(self, now: int, message: Message) -> list[Message]:
        job = self.jobs.get(message.conversation_id)
        driver_id = message.sender
        if job is None or job.offered_to != driver_id or job.confirmed:
            return [self._to_driver(now, driver_id, message.conversation_id, Performative.CANCEL,
                                    "Too late: this delivery was given to another driver.")]
        if message.performative is Performative.ACCEPT_PROPOSAL:
            changed = job.confirmed_by is not None and job.confirmed_by != driver_id
            self._replan(now, keep_assignments=True)  # the driver sets off now: plan the times from here
            job.confirmed, job.confirmed_by = True, driver_id
            job.history.append(f"accepted by {self.drivers[driver_id].name}")
            return [self.message(now, "communication", Performative.INFORM, job.donation.id, protocol="delivery",
                                 summary=f"{driver_id} accepted: " + self._describe(job), plan=self._plan(job),
                                 changed=changed)]
        job.declined_by.add(driver_id)
        job.offered_to = None
        self._second_chance(job)
        moved = self._replan(now, keep_assignments=True)
        moved.add(job.donation.id)
        return [self._replan_message(now, job.donation.id, moved, f"{driver_id} declined {job.donation.id}")]

    def _expire(self, now: int) -> list[Message]:
        """Requests nobody answered in time go to another driver."""
        expired = [job for job in self.jobs.values() if job.offered_to is not None and not job.confirmed
                   and job.offered_at is not None and now - job.offered_at >= self.offer_timeout]
        if not expired:
            return []
        actions = []
        for job in expired:
            actions.append(self._to_driver(now, job.offered_to, job.donation.id, Performative.CANCEL,
                                           f"Your request for {job.donation.id} expired: it goes to another driver."))
            driver_id, job.offered_to = job.offered_to, None
            job.expired_for.add(driver_id)
            self._second_chance(job, just_expired=driver_id)
        moved = self._replan(now, keep_assignments=True) | {job.donation.id for job in expired}
        actions.append(self._replan_message(now, expired[0].donation.id, moved, "delivery requests expired"))
        return actions

    def _second_chance(self, job: Job, just_expired: str | None = None) -> None:
        """Every working driver has said no or let the request expire: those who only let it expire are asked again
        (a decline stays final), starting with the others than the one whose request has just expired."""
        working = self.active - set(self.paused)
        if working and working <= job.excluded() and working - job.declined_by:
            others = working - job.declined_by - {just_expired}
            job.expired_for = {just_expired} if others and just_expired else set()

    def _offers(self, now: int) -> list[Message]:
        """Propose each delivery to its newly planned driver; tell drivers who lose one."""
        actions = []
        for job_id, job in sorted(self.jobs.items()):
            if job_id in self.lost or job.delivered_at is not None:
                continue
            driver_id = job.driver_id
            if job.confirmed and driver_id == job.confirmed_by:
                continue  # accepted and still theirs
            if driver_id is not None and job.offered_to == driver_id:
                continue  # waiting for their answer
            if job.offered_to is not None:  # proposed to someone else, now withdrawn
                actions.append(self._to_driver(now, job.offered_to, job_id, Performative.CANCEL,
                                               f"Request for {job_id} withdrawn: plans changed."))
                job.offered_to = None
            if job.confirmed and job.confirmed_by in self.active:
                actions.append(self._to_driver(now, job.confirmed_by, job_id, Performative.CANCEL,
                                               f"{job_id} was given to another driver."))
            job.confirmed = False
            if driver_id is None:
                continue
            job.offered_to, job.offered_at = driver_id, now
            plan = self._plan(job)
            text = (f"New delivery: pick up {plan['description']} ({job.donation.total_kg:g} kg) at "
                    f"{plan['pickup_name']} at {format_hhmm(job.pickup_eta)}, deliver to "
                    f"{self.directory.get(job.charity_id).name} at {format_hhmm(job.delivery_eta)}. "
                    f"Please answer within {self.offer_timeout} minutes.")
            actions.append(self.message(now, driver_id, Performative.PROPOSE, job_id, protocol="job-offer",
                                        summary=f"can you take {job_id}?", text=text, plan=plan,
                                        deadline=now + self.offer_timeout))
        return actions

    def _to_driver(self, now: int, driver_id: str, donation_id: str, performative: Performative,
                   text: str) -> Message:
        return self.message(now, driver_id, performative, donation_id, protocol="job-offer", summary=text, text=text)

    def add_driver(self, driver: Driver) -> None:
        """A driver signs up: known from now on, off duty until they switch on."""
        self.drivers[driver.id] = driver
        self.routes[driver.id] = Route(driver_id=driver.id)

    def _availability(self, now: int, driver_id: str, available: bool) -> list[Message]:
        """The driver's on/off duty switch. Deliveries already given to a driver who keeps working
        never move: a driver coming on duty only gets what nobody has, even if closer."""
        if available == (driver_id in self.active):
            return []
        conversation = f"duty-{driver_id}-{now}"
        if available:
            self.active.add(driver_id)
            moved = self._replan(now, keep_assignments=True)
            reason = f"{driver_id} is on duty"
        else:
            on_board = sorted(job_id for job_id, job in self.jobs.items() if job.driver_id == driver_id
                              and job.picked_up_at is not None and job.delivered_at is None)
            if on_board:
                return [self.message(now, "communication", Performative.INFORM, conversation, protocol="availability",
                                     summary=f"{driver_id} cannot go off duty: food on board",
                                     driver_id=driver_id, available=True, refused=True, on_board=on_board)]
            self.active.discard(driver_id)
            self.paused.pop(driver_id, None)
            moved = self._replan(now, keep_assignments=True)  # only this driver's deliveries move
            reason = f"{driver_id} went off duty"
        actions = [self.message(now, "communication", Performative.INFORM, conversation, protocol="availability",
                                summary=reason, driver_id=driver_id, available=available, refused=False,
                                released=len(moved))]
        if moved:
            actions.append(self._replan_message(now, conversation, moved, reason))
        return actions

    def _progress(self, now: int) -> list[Message]:
        """Pickups and deliveries whose planned time has passed have happened: record them once."""
        events = []
        for job_id, job in sorted(self.jobs.items()):
            if job.driver_id is None or job_id in self.lost or (self.confirmation and not job.confirmed):
                continue
            if job.picked_up_at is None and job.pickup_eta is not None and job.pickup_eta <= now:
                job.picked_up_at = job.pickup_eta
                job.donation.status = DonationStatus.PICKED_UP
                events.append(self._progress_message(now, job, "picked_up"))
            if (job.delivered_at is None and job.picked_up_at is not None and job.delivery_eta is not None
                    and job.delivery_eta <= now):
                job.delivered_at = job.delivery_eta
                job.donation.status = DonationStatus.DELIVERED
                events.append(self._progress_message(now, job, "delivered"))
        return events

    def _progress_message(self, now: int, job: Job, event: str) -> Message:
        at = job.picked_up_at if event == "picked_up" else job.delivered_at
        driver = self.drivers[job.driver_id].name
        what = "picked up" if event == "picked_up" else f"delivered to {job.charity_id}"
        return self.message(now, "communication", Performative.INFORM, job.donation.id, protocol="progress",
                            summary=f"{driver} {what} at {format_hhmm(at)}", event=event, at=at,
                            plan=self._plan(job))

    def _pause_driver(self, driver_id: str, now: int, minutes: int) -> None:
        """The driver stops for a while (flat tire...): food on board stays in their car and is delivered
        by them after the pause; their other stops are re-solved (someone else may take them)."""
        until = now + minutes
        stops = self.routes[driver_id].stops
        done = [s for s in stops if s.eta <= now]
        on_board = ({s.donation_id for s in done if s.kind is StopKind.PICKUP}
                    - {s.donation_id for s in done if s.kind is StopKind.DELIVERY})
        car = done[-1].location if done else self.drivers[driver_id].home
        for donation_id in on_board:
            job = self.jobs[donation_id]
            job.transfer = Transfer(place_id=f"{driver_id}-car", name=f"{self.drivers[driver_id].name}'s car",
                                    location=car, window=(until, until + TRANSFER_MINUTES))
            job.pinned_driver = driver_id
            job.history.append(f"waits in {job.transfer.name} for {minutes} min")
        self.paused[driver_id] = (until, car)

    # --- disruptions ---------------------------------------------------------------------

    def _plan_transfers(self, driver_id: str, now: int) -> None:
        """Food already picked up but not delivered is in the broken-down car: collect it there."""
        stops = self.routes[driver_id].stops
        done = [s for s in stops if s.eta <= now]
        on_board = ({s.donation_id for s in done if s.kind is StopKind.PICKUP}
                    - {s.donation_id for s in done if s.kind is StopKind.DELIVERY})
        car = done[-1].location if done else self.drivers[driver_id].home
        for donation_id in on_board:
            job = self.jobs[donation_id]
            job.transfer = Transfer(place_id=f"{driver_id}-car", name=f"{self.drivers[driver_id].name}'s car",
                                    location=car, window=(now, now + TRANSFER_MINUTES))
            job.pinned_driver = None  # anyone can collect it now
            job.history.append(f"collected from {job.transfer.name}")
            self.transfers += 1

    def _merchant_delay(self, now: int, message: Message) -> list[Message]:
        donation_id, minutes = message.content["donation_id"], int(message.content["minutes"])
        job = self.jobs.get(donation_id)
        if job is None:  # still being matched: apply it when the donation reaches us
            self.delays[donation_id] = self.delays.get(donation_id, 0) + minutes
            return []
        if job.pickup_eta is not None and job.pickup_eta <= now and job.driver_id is not None:
            return []  # already picked up: nothing to change
        self._apply_delay(job, now, minutes)
        moved = self._replan(now, keep_assignments=False)
        moved.add(donation_id)  # its times changed even if its driver did not
        return [self._replan_message(now, message.conversation_id, moved,
                                     f"{job.donation.merchant_id} is {minutes} min late")]

    def _apply_delay(self, job: Job, now: int, minutes: int) -> None:
        window = job.donation.pickup_window
        start = max(window.start, now) + minutes
        job.donation = job.donation.model_copy(
            update={"pickup_window": TimeWindow(start=start, end=max(window.end, start + 30))})
        job.history.append(f"merchant {minutes} min late")
        self._unfrozen.add(job.donation.id)

    # --- planning ------------------------------------------------------------------------

    def _replan(self, now: int, keep_assignments: bool) -> set[str]:
        """Freeze the past, re-solve the future. Returns the donations whose driver changed."""
        self.replans += 1
        self.paused = {d: pause for d, pause in self.paused.items() if pause[0] > now}  # pauses that are over
        before = {job_id: job.driver_id for job_id, job in self.jobs.items()}
        frozen = {driver_id: self._frozen_prefix(driver_id, now) for driver_id in self.drivers}
        locked = set()  # donations whose plan can no longer change
        for driver_id, stops in frozen.items():
            if driver_id in self.active and driver_id not in self.paused:
                locked |= {stop.donation_id for stop in stops if not self._stale(stop)}
            else:  # cancelled or paused: only delivered food is final; food on board is re-planned from the car
                locked |= {stop.donation_id for stop in stops if stop.kind is StopKind.DELIVERY}

        vehicles = []
        for driver_id in sorted(self.active):
            driver, prefix = self.drivers[driver_id], frozen[driver_id]
            if driver_id in self.paused:
                available_from, start = self.paused[driver_id]
            else:
                start = prefix[-1].location if prefix else driver.home
                available_from = prefix[-1].eta + self.routing.service_minutes if prefix else driver.shift.start
            vehicles.append(Vehicle(driver_id=driver_id, start=start, available_from=available_from,
                                    shift_end=driver.shift.end, capacity_kg=driver.capacity_kg))
        tasks = [self._task(job, keep_assignments) for job_id, job in sorted(self.jobs.items())
                 if job_id not in locked and job_id not in self.lost]

        solution = solve_routes(tasks, vehicles, now=now, road_factor=self.geo.road_factor,
                                speed_kmh=self.geo.speed_kmh, service_minutes=self.routing.service_minutes,
                                time_limit_s=self.routing.time_limit_s,
                                pickup_delay_cost=self.routing.pickup_delay_cost)

        for job_id, job in self.jobs.items():
            if job_id not in locked:
                job.driver_id = job.pickup_eta = job.delivery_eta = None
                if job.donation.status is DonationStatus.SCHEDULED:
                    job.donation.status = DonationStatus.APPROVED  # scheduled again below if a driver takes it
        self._unfrozen.clear()
        # Cancelled drivers first, so a transfer's new driver overwrites the old pickup.
        for driver_id in sorted(self.drivers, key=lambda d: (d in self.active, d)):
            stops = frozen[driver_id] + solution.routes.get(driver_id, [])
            self.routes[driver_id] = Route(driver_id=driver_id, stops=stops,
                                           distance_km=self._distance(driver_id, stops))
            for stop in stops:
                if self._stale(stop) or (stop in frozen[driver_id] and stop.donation_id not in locked):
                    continue  # an old pickup of food now collected from a car (cancelled or paused driver)
                job = self.jobs[stop.donation_id]
                job.driver_id = driver_id
                if job.donation.status not in (DonationStatus.PICKED_UP, DonationStatus.DELIVERED):
                    job.donation.status = DonationStatus.SCHEDULED
                if stop.kind is StopKind.PICKUP:
                    job.pickup_eta = stop.eta
                else:
                    job.delivery_eta = stop.eta
        return {job_id for job_id, job in self.jobs.items() if job.driver_id != before.get(job_id)}

    def _drop(self, driver_id: str, now: int) -> set[str]:
        """Ablation without replanning: what the cancelled driver had not delivered is lost."""
        kept = self._frozen_prefix(driver_id, now)  # the driver is inactive: delivered donations only
        lost = {s.donation_id for s in self.routes[driver_id].stops} - {s.donation_id for s in kept}
        self.routes[driver_id] = Route(driver_id=driver_id, stops=kept, distance_km=self._distance(driver_id, kept))
        for donation_id in lost:
            job = self.jobs[donation_id]
            job.driver_id = job.pickup_eta = job.delivery_eta = None
        self.lost |= lost
        return lost

    def _frozen_prefix(self, driver_id: str, now: int) -> list[Stop]:
        """The stops this driver has done (or must finish) by `now`."""
        stops = self.routes[driver_id].stops
        if self.confirmation:  # a delivery not accepted yet has not started
            stops = [s for s in stops if self.jobs[s.donation_id].confirmed]
        if driver_id in self.paused:  # stopped where they are: only what is already done is frozen
            return [s for s in stops if s.eta <= now]
        if driver_id not in self.active:
            # The driver is gone: they keep what they delivered, and the pickups of food that was on
            # board (it is collected from their car by someone else).
            delivered = {s.donation_id for s in stops if s.kind is StopKind.DELIVERY and s.eta <= now}
            transferred = {s.donation_id for s in stops if s.kind is StopKind.PICKUP and s.eta <= now
                           and self.jobs[s.donation_id].transfer is not None}
            return [s for s in stops if s.donation_id in delivered
                    or (s.donation_id in transferred and s.kind is StopKind.PICKUP and s.eta <= now)]
        cut = max((i for i, s in enumerate(stops) if s.eta <= now), default=-1)
        # A driver already driving to the next stop is committed to it (unless that pickup was delayed).
        if cut + 1 < len(stops):
            previous = stops[cut].location if cut >= 0 else self.drivers[driver_id].home
            upcoming = stops[cut + 1]
            drive = self.tools.call("travel_time", {"origin": previous, "destination": upcoming.location})
            delayed = upcoming.kind is StopKind.PICKUP and upcoming.donation_id in self._unfrozen
            if now >= upcoming.eta - drive and not delayed:
                cut += 1
        # Food on board stays with its driver until it is delivered.
        while True:
            prefix = stops[: cut + 1]
            on_board = {s.donation_id for s in prefix if s.kind is StopKind.PICKUP and not self._stale(s)}
            on_board -= {s.donation_id for s in prefix if s.kind is StopKind.DELIVERY}
            last_delivery = max((i for i, s in enumerate(stops)
                                 if s.kind is StopKind.DELIVERY and s.donation_id in on_board), default=cut)
            if last_delivery == cut:
                return prefix
            cut = last_delivery

    def _stale(self, stop: Stop) -> bool:
        """The old pickup at the merchant of food that must now be collected from a car (kept as history)."""
        transfer = self.jobs[stop.donation_id].transfer
        return stop.kind is StopKind.PICKUP and transfer is not None and stop.place_id != transfer.place_id

    def _task(self, job: Job, keep_assignments: bool) -> Task:
        charity = self.directory.get(job.charity_id)
        opening = charity.opening or job.donation.pickup_window
        if job.transfer is not None:
            place, location, window = job.transfer.place_id, job.transfer.location, job.transfer.window
        else:
            merchant = self.directory.get(job.donation.merchant_id)
            place, location = merchant.id, job.pickup_location or merchant.location
            window = (job.donation.pickup_window.start, job.donation.pickup_window.end)
        return Task(donation_id=job.donation.id, pickup_place=place, pickup=location,
                    delivery_place=charity.id, delivery=job.delivery_location or charity.location,
                    kg=job.donation.total_kg,
                    pickup_window=window, delivery_window=(opening.start, opening.end),
                    keep_driver=job.pinned_driver or self._keep(job, keep_assignments),
                    excluded_drivers=tuple(sorted(job.excluded())))

    def _keep(self, job: Job, keep_assignments: bool) -> str | None:
        """With driver confirmation, a delivery stays with a driver who is still working (accepted or
        waiting for their answer), whatever the reason for re-planning; it moves only for their own problem."""
        if not self.confirmation:
            return job.driver_id if keep_assignments else None
        driver = job.driver_id
        working = driver in self.active and driver not in self.paused and driver not in job.excluded()
        return driver if driver is not None and working else None

    def _distance(self, driver_id: str, stops: list[Stop]) -> float:
        position, km = self.drivers[driver_id].home, 0.0
        for stop in stops:
            km += road_km(position, stop.location, self.geo.road_factor)
            position = stop.location
        return round(km, 2)

    # --- messages ------------------------------------------------------------------------

    def _replan_message(self, now: int, conversation_id: str, moved: set[str], reason: str,
                        cancelled_driver: str | None = None) -> Message:
        moves = [self._plan(self.jobs[job_id]) for job_id in sorted(moved)]
        lost = sum(1 for move in moves if move["driver_id"] is None)
        summary = f"{reason}: {len(moves) - lost} donations reassigned" + (f", {lost} without a driver" if lost else "")
        return self.message(now, "communication", Performative.INFORM, conversation_id, protocol="replan",
                            summary=summary, cancelled_driver=cancelled_driver, moves=moves)

    def _plan(self, job: Job) -> dict[str, Any]:
        return {
            "donation_id": job.donation.id,
            "description": job.donation.describe(),
            "merchant_id": job.donation.merchant_id,
            "charity_id": job.charity_id,
            "driver_id": job.driver_id,
            "driver_name": self.drivers[job.driver_id].name if job.driver_id else None,
            "pickup_eta": job.pickup_eta,
            "delivery_eta": job.delivery_eta,
            "pickup_name": job.transfer.name if job.transfer else self.directory.get(job.donation.merchant_id).name,
            "history": list(job.history),
            "confirmed": job.confirmed or not self.confirmation,
            "note": job.donation.details_summary,
            "allergens": list(job.donation.allergens),
        }

    def _describe(self, job: Job) -> str:
        if job.driver_id is None:
            return f"no driver can reach {job.donation.id} in time (dropped by the solver)"
        return (f"{job.driver_id} picks up at {format_hhmm(job.pickup_eta)}, "
                f"delivers to {job.charity_id} at {format_hhmm(job.delivery_eta)}")
