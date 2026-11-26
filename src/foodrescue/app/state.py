"""The running evening, shared by every browser tab and device, and what the screens read from it.

One `Platform` per server holds the simulation: a merchant on a laptop, a driver on a phone and a charity on a
tablet all see and change the same evening, and a page refresh keeps it. Each page run holds the platform's lock,
so a tab never reads the evening while another one changes it. Per-person things (who is signed in, which
notifications were already shown, a half-filled form) stay in each tab's `st.session_state`.

The clock moves when the presenter clicks a time button or turns on live mode. People's actions (a merchant's
post, a driver's report) are injected as ordinary environment events, so the agents react exactly as in the
command-line runs.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import dataclass, field

import streamlit as st

from foodrescue.config import Config, load_config
from foodrescue.core.clock import format_hhmm
from foodrescue.models import Donation, DonationOffer, Message, Performative
from foodrescue.sim.runner import Simulation
from foodrescue.sim.scenario import platform_scenario

STEPS = ["Posted", "Checked", "Charity found", "Pickup planned", "Picked up", "Delivered"]


# --- the shared platform -----------------------------------------------------------------------


@dataclass
class Platform:
    sim: Simulation | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_tick: float = 0.0  # live mode: when the clock last moved by itself (any tab)


@st.cache_resource
def platform() -> Platform:
    """The one evening of this server (kept until the server stops or "Start a new evening")."""
    return Platform()


def stamp() -> tuple:
    """Changes whenever the shared evening changes, the clock included (live mode redraws its own tab)."""
    sim = platform().sim
    return (id(sim), sim.now, len(sim.bus.log)) if sim is not None else ()


def news() -> tuple:
    """Changes when something happens in the evening (a new message, a new evening), not when only the clock
    moves: other tabs redraw on news only, so a person clicking there is not interrupted every second."""
    sim = platform().sim
    return (id(sim), len(sim.bus.log)) if sim is not None else ()


def platform_config() -> Config:
    """The platform as a real service: the AI reads every text people write (live calls to the local model, no
    recorded answers), explains safety decisions with sources, and drivers accept each delivery."""
    config = load_config()
    config.llm.mode = "live"
    config.features.llm_intake = True  # merchants' details messages
    config.features.llm_dispatcher = True  # drivers' "something else" reports
    config.features.rag_safety = True  # safety explanations quoting WHO and EU documents
    config.features.driver_confirmation = True  # drivers accept each delivery...
    config.simulation.driver_policy = "manual"  # ...in the Driver screen
    config.simulation.charity_policy = "manual"  # a charity accepts each donation it won, in the Charity screen
    return config


def start(start_minutes: int = 18 * 60) -> None:
    """A new evening on the platform, for everyone: the registered accounts, nothing scripted."""
    shared = platform()
    with shared.lock, st.spinner("Starting the agents..."):
        shared.sim = Simulation(platform_scenario(start_minutes), platform_config())


def next_donation_id(sim: Simulation) -> str:
    """N1, N2... for donations posted on the platform (read under the lock, so two devices never share one)."""
    return f"N{len({o.id for o in sim.scenario.offers if o.id.startswith('N')}) + 1}"


@st.cache_data(ttl=10, show_spinner=False)
def ai_status() -> tuple[bool, str]:
    """Is the local model answering? Checked at most every 10 seconds for the whole server (not per tab).

    Called before a page takes the platform lock (`dashboard.py`): a slow answer from Ollama must never hold
    the evening, or every open screen greys out while it waits."""
    config = load_config()
    try:
        with urllib.request.urlopen(f"{config.llm.host}/api/tags", timeout=1.5) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
        ok = config.llm.model in names
        detail = config.llm.model if ok else f"{config.llm.model} is not installed in Ollama"
    except (OSError, ValueError):
        ok, detail = False, "Ollama is not running: start it with `ollama serve`"
    return ok, detail


def advance(minutes: int, spinner: bool = True) -> None:
    shared = platform()
    with shared.lock:
        sim = shared.sim
        if not spinner:  # live mode: no spinner flashing every second
            sim.run_until(sim.now + minutes - 1)
            return
        with st.spinner("The agents are working..."):
            sim.run_until(sim.now + minutes - 1)


def announce(sim: Simulation, person_id: str) -> None:
    """Pop up the notifications this person received since their screen was last shown."""
    messages = inbox(sim, person_id)
    seen_key = f"_seen_{id(sim)}_{person_id}"
    last = st.session_state.get(seen_key)
    if last is not None:
        for message in [m for m in messages if m.id > last][-3:]:
            st.toast(message.content["text"], icon=":material/notifications:")
    st.session_state[seen_key] = messages[-1].id if messages else 0


def announce_requests(sim: Simulation) -> None:
    """Delivery requests pop up on every screen, so the presenter never misses one."""
    shown = st.session_state.setdefault(f"_requests_{id(sim)}", set())
    for driver in sim.scenario.drivers:
        for job, deadline in requests_for(sim, driver.id):
            key = (job.donation.id, driver.id, job.offered_at)
            if key not in shown:
                shown.add(key)
                st.toast(f"New delivery request for {driver.name}: {job.donation.id}, answer by "
                         f"{format_hhmm(deadline)}", icon=":material/local_shipping:")
    for charity in sim.scenario.charities:
        for award in sim.awards_waiting(charity.id):
            key = (award.conversation_id, charity.id, award.time)
            if key not in shown:
                shown.add(key)
                st.toast(f"{charity.name} won {award.conversation_id}: accept or decline it in the Charity screen",
                         icon=":material/volunteer_activism:")


def current() -> Simulation:
    return platform().sim


def clock_text(sim: Simulation) -> str:
    return "End of evening" if sim.finished else format_hhmm(sim.now)


def identity(label: str, options: dict[str, str], key: str) -> str:
    """"Signed in as" selector that keeps its choice when the presenter visits other screens."""
    store = f"_keep_{key}"
    if st.session_state.get(store) not in options:
        st.session_state[store] = next(iter(options))
    # Streamlit keeps a selectbox's old labels when only their text changes (e.g. "Sami · 1 request"):
    # a new widget is made when the labels change, starting on the kept choice.
    labels = "|".join(options.values())
    if st.session_state.get(f"_labels_{key}") != labels:
        st.session_state.pop(key, None)
        st.session_state[f"_labels_{key}"] = labels
    if key not in st.session_state:
        st.session_state[key] = st.session_state[store]
    choice = st.selectbox(label, list(options), format_func=options.get, key=key)
    st.session_state[store] = choice
    return choice


# --- reading the simulation ------------------------------------------------------------------


def name_of(sim: Simulation, some_id: str) -> str:
    people = {p.id: p.name for p in [*sim.scenario.merchants, *sim.scenario.charities, *sim.scenario.drivers]}
    return people.get(some_id, {"env": "environment"}.get(some_id, some_id))


def posted_offers(sim: Simulation, merchant_id: str | None = None) -> list[DonationOffer]:
    """First message of each donation posted so far (answers to questions reuse the donation id)."""
    first: dict[str, DonationOffer] = {}
    for offer in sorted(sim.scenario.offers, key=lambda o: o.time):
        if offer.time < sim.now and (merchant_id is None or offer.merchant_id == merchant_id):
            first.setdefault(offer.id, offer)
    return list(first.values())


def first_offer(sim: Simulation, donation_id: str) -> DonationOffer:
    """The offer as last sent (the form, with the details and any earlier answers)."""
    return [o for o in sim.scenario.offers if o.id == donation_id][-1]


def answers(sim: Simulation, donation_id: str) -> list[DonationOffer]:
    return [o for o in sim.scenario.offers if o.id == donation_id and o.answer and o.time < sim.now]


def donation_of(sim: Simulation, donation_id: str) -> Donation | None:
    """What the system understood (after safety removed unsafe items, once it is planned)."""
    job = sim.logistics.jobs.get(donation_id)
    if job is not None:
        return job.donation
    for message in reversed(sim.bus.conversation(donation_id)):
        if "donation" in message.content:
            return Donation.model_validate(message.content["donation"])
    return None


def inbox(sim: Simulation, person_id: str) -> list[Message]:
    """Notifications, and for drivers the delivery requests and their outcome."""
    return [m for m in sim.bus.log if m.receiver == person_id and "text" in m.content
            and m.protocol in ("notification", "job-offer")]


def stops_of(sim: Simulation, driver_id: str) -> list:
    """The driver's route without deliveries they have not accepted yet."""
    jobs = sim.logistics.jobs
    return [s for s in sim.logistics.routes[driver_id].stops
            if not sim.logistics.confirmation or jobs[s.donation_id].confirmed]


def requests_for(sim: Simulation, driver_id: str) -> list:
    """Deliveries waiting for this driver's answer: (job, minute the request expires)."""
    return sim.logistics.pending_offers(driver_id) if sim.logistics.confirmation else []


def open_question(sim: Simulation, donation_id: str) -> Message | None:
    """The last question asked about this donation, if the merchant has not answered it yet."""
    questions = [m for m in sim.bus.conversation(donation_id)
                 if m.protocol == "notification" and m.content.get("question")]
    if not questions:
        return None
    last = questions[-1]
    answered = any(o.time >= last.time for o in answers(sim, donation_id))
    return None if answered else last


@dataclass(frozen=True)
class Stage:
    """Where a donation is now. `step` indexes `STEPS`; `state` says how that step is going."""

    label: str
    tone: str  # ui tone: neutral, info, good, warning, serious, critical
    step: int
    state: str = "current"  # current, done, failed, waiting

    @property
    def on_the_road(self) -> bool:
        return self.label in ("Pickup planned", "Picked up")


def stage_of(sim: Simulation, donation_id: str) -> Stage:
    """What the logistics agent recorded: "delivered" only once the delivery has really happened."""
    if donation_id in sim.logistics.lost:
        return Stage("Lost (driver cancelled)", "critical", 3, "failed")
    job = sim.logistics.jobs.get(donation_id)
    if job is not None and job.delivered_at is not None:
        return Stage("Delivered", "good", 5, "done")
    if job is not None and job.picked_up_at is not None:
        return Stage("Picked up", "info", 4)
    if job is not None and job.driver_id and sim.logistics.confirmation and not job.confirmed:
        return Stage("Waiting for a driver to accept", "warning", 3, "waiting")
    if job is not None and job.driver_id and job.delivery_eta is not None:
        return Stage("Pickup planned", "info", 3)
    if job is not None:
        return Stage("No driver", "serious", 3, "failed") if sim.finished else Stage("Finding a driver", "neutral", 3)

    stage = Stage("Received", "neutral", 1)
    for m in sim.bus.conversation(donation_id):
        if m.sender == "safety" and m.performative is Performative.REFUSE:
            stage = Stage("Rejected as unsafe", "critical", 1, "failed")
        elif (m.sender == "safety" and m.performative is Performative.REQUEST) or \
                (m.sender == "intake" and m.protocol == "follow-up"):
            stage = Stage("Needs information", "warning", 1, "waiting")
        elif m.sender == "intake" and m.performative is Performative.FAILURE:
            stage = Stage("Not understood", "serious", 1, "failed")
        elif m.sender == "intake" and m.protocol == "safety-check":
            stage = Stage("Safety check", "neutral", 1)
        elif m.sender == "safety" and m.performative is Performative.INFORM:
            stage = Stage("Finding a charity", "neutral", 2)
        elif m.sender == "matching" and m.performative is Performative.ACCEPT_PROPOSAL:
            stage = Stage("Waiting for the charity to accept", "warning", 2, "waiting")
        elif m.sender == "matching" and m.performative is Performative.FAILURE:
            stage = Stage("No charity", "serious", 2, "failed")
        elif m.sender == "matching" and m.protocol == "delivery":
            stage = Stage("Charity found", "neutral", 3)
    return stage


def declines(sim: Simulation, donation_id: str) -> list[tuple[int, str, str]]:
    """Charities that won this donation and then declined it, or did not answer: (time, charity id, reason)."""
    return [(m.time, m.content["charity_id"], m.content["reason"]) for m in sim.bus.conversation(donation_id)
            if m.sender == "matching" and m.protocol == "award"]


def driver_status(sim: Simulation, driver_id: str) -> tuple[str, str]:
    """(label, tone) for a driver right now."""
    if driver_id in sim.logistics.paused:
        until = sim.logistics.paused[driver_id][0]
        return f"Paused until {format_hhmm(until)}", "warning"
    if driver_id not in sim.logistics.active:
        return "Off duty", "critical"
    if requests_for(sim, driver_id):
        return "Request waiting", "warning"
    stops = stops_of(sim, driver_id)
    if any(s.eta is not None and s.eta >= sim.now for s in stops):
        return "On a route", "info"
    return "Available", "good"


def dispatcher_decisions(sim: Simulation, driver_id: str | None = None) -> list[dict]:
    return [d for d in sim.dispatcher.decisions if driver_id is None or d["driver_id"] == driver_id]


def decision_text(decision: dict) -> str:
    if decision.get("outcome"):
        return decision["outcome"]
    args = decision.get("args") or {}
    return {"pause_driver": f"pause for {args.get('minutes')} min", "cancel_driver": "take the driver off tonight",
            "ask_driver": "ask the driver a question"}.get(decision["tool"], decision["tool"])
