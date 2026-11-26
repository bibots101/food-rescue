"""Runs one simulated day.

At every tick:
  1. the bus delivers the messages sent during the previous tick,
  2. the environment injects the events due now (merchant offers, driver cancellations),
  3. every agent runs its perceive -> decide -> act cycle.

Merchants and drivers are people outside the system: they only get an inbox, where the
communication agent leaves their notifications (shown on their dashboard screen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from foodrescue.agents.charity import CharityAgent
from foodrescue.agents.communication import CommunicationAgent
from foodrescue.agents.dispatcher import DispatcherAgent
from foodrescue.agents.driver import DriverAgent
from foodrescue.agents.intake import IntakeAgent
from foodrescue.agents.logistics import LogisticsAgent
from foodrescue.agents.matching import MatchingAgent
from foodrescue.agents.safety import SafetyAgent
from foodrescue.config import Config, load_config
from foodrescue.core.agent import Agent
from foodrescue.core.bus import MessageBus
from foodrescue.core.clock import SimClock
from foodrescue.learning.bandit import AcceptanceLearner
from foodrescue.llm.client import LLMClient
from foodrescue.models import (
    Charity,
    Directory,
    Disruption,
    DisruptionKind,
    Donation,
    DonationOffer,
    Driver,
    Merchant,
    Message,
    Performative,
    Place,
    Route,
    Scenario,
)
from foodrescue.toolbox import build_toolbox

ENVIRONMENT = "env"


@dataclass
class SimulationResult:
    scenario: Scenario
    messages: list[Message]
    routes: dict[str, Route]
    awards: dict[str, str]  # donation id -> charity id
    donations: dict[str, Donation]  # what was actually carried (unsafe items already removed)
    lost_to_cancellation: set[str] = field(default_factory=set)  # only in the "no replanning" ablation


class Simulation:
    def __init__(self, scenario: Scenario, config: Config | None = None,
                 learner: AcceptanceLearner | None = None) -> None:
        """`learner`: pass the same learner to consecutive days so the system keeps what it learned.
        With `features.learning` on and no learner given, a fresh one is created."""
        self.config = config or load_config()
        if learner is None and self.config.features.learning:
            learner = AcceptanceLearner(seed=scenario.seed)
        self.learner = learner
        self.scenario = scenario.model_copy(deep=True)  # offers added during a run never leak into the original
        self.bus = MessageBus()
        self.tools = build_toolbox(self.config)
        self.clock = SimClock(scenario.start, scenario.end, self.config.simulation.step_minutes)
        directory = Directory.from_scenario(scenario)
        self.directory = directory  # shared by the agents: account edits reach all of them

        confirmation = self.config.features.driver_confirmation
        # With driver confirmation each driver has an agent (their phone app) instead of a plain inbox.
        for person in [*scenario.merchants, *([] if confirmation else scenario.drivers)]:
            self.bus.register(person.id)
        self.drivers = [DriverAgent(d.id, self.bus, policy=self.config.simulation.driver_policy, seed=scenario.seed)
                        for d in scenario.drivers] if confirmation else []

        self.matching = MatchingAgent("matching", self.bus, self.tools, directory, weights=self.config.matching,
                                      cfp_timeout=self.config.simulation.cfp_timeout_minutes, learner=self.learner,
                                      confirm_timeout=self.config.simulation.confirm_timeout_minutes)
        self.matching.contract_net = self.config.features.contract_net
        self.logistics = LogisticsAgent("logistics", self.bus, self.tools, directory, scenario.drivers,
                                        geo=self.config.geo, routing=self.config.routing,
                                        replanning=self.config.features.replanning, confirmation=confirmation,
                                        offer_timeout=self.config.simulation.offer_timeout_minutes)
        policy = "llm" if self.config.features.llm_dispatcher else self.config.simulation.dispatcher_policy
        self.dispatcher = DispatcherAgent("dispatcher", self.bus, policy=policy,
                                          llm=LLMClient.from_config(self.config.llm) if policy == "llm" else None)
        self.agents: list[Agent] = [
            self.dispatcher,
            IntakeAgent("intake", self.bus, self.tools,
                        llm=LLMClient.from_config(self.config.llm) if self.config.features.llm_intake else None),
            SafetyAgent("safety", self.bus, self.tools, explainer=self._safety_explainer()),
            self.matching,
            *(CharityAgent(charity, self.bus, self.tools, directory, seed=scenario.seed,
                           policy=self.config.simulation.charity_policy) for charity in scenario.charities),
            *self.drivers,
            self.logistics,
            CommunicationAgent("communication", self.bus, self.tools, directory),
        ]

        self._offers: list[DonationOffer] = sorted(self.scenario.offers, key=lambda o: o.time)
        self._disruptions: list[Disruption] = sorted(self.scenario.disruptions, key=lambda d: d.time)

    def _safety_explainer(self):
        """RAG explanations for safety verdicts, only when `features.rag_safety` is on."""
        if not self.config.features.rag_safety:
            return None
        from foodrescue.rag.corpus import load_corpus
        from foodrescue.rag.explain import explain
        from foodrescue.rag.index import Retriever

        llm_config = self.config.llm
        offline = llm_config.mode == "replay"  # use only cached vectors (BM25 for anything new)
        retriever = Retriever(load_corpus(), model=llm_config.embedding_model, host=llm_config.host, offline=offline)
        llm = LLMClient.from_config(llm_config)
        return lambda verdict, donation: explain(verdict, donation, retriever, llm)

    def add_disruption(self, disruption: Disruption) -> None:
        """Inject a disruption during the run (the dashboard's "I can't make it" button)."""
        self._disruptions.append(disruption)
        self._disruptions.sort(key=lambda d: d.time)

    # --- accounts: people create and edit their profile during the evening --------------------------

    def next_charity_id(self) -> str:
        taken = {c.id for c in self.scenario.charities}
        return next(f"C{i:02d}" for i in range(1, 100) if f"C{i:02d}" not in taken)

    def add_charity(self, charity: Charity) -> None:
        """A new charity signs up: its agent joins the charities' group and answers the next calls."""
        agent = CharityAgent(charity, self.bus, self.tools, self.directory, seed=self.scenario.seed,
                             policy=self.config.simulation.charity_policy)
        last_charity = max(i for i, a in enumerate(self.agents) if isinstance(a, CharityAgent))
        self.agents.insert(last_charity + 1, agent)
        self.scenario.charities.append(charity)
        self.directory.places[charity.id] = Place(id=charity.id, name=charity.name, location=charity.location,
                                                  opening=charity.opening)

    def update_charity(self, charity: Charity) -> None:
        """A charity edits its account: the agent answers the next calls with the new details."""
        agent = next(a for a in self.agents if isinstance(a, CharityAgent) and a.name == charity.id)
        agent.update_profile(charity)
        self.scenario.charities = [charity if c.id == charity.id else c for c in self.scenario.charities]
        self.directory.places[charity.id] = Place(id=charity.id, name=charity.name, location=charity.location,
                                                  opening=charity.opening)

    @property
    def charities(self) -> list[CharityAgent]:
        return [a for a in self.agents if isinstance(a, CharityAgent)]

    def awards_waiting(self, charity_id: str) -> list[Message]:
        """Donations this charity won and has not accepted or declined yet (the award messages)."""
        waiting = []
        for donation_id, award in self.matching.pending.items():
            if award.ranked[0].charity_id == charity_id:
                waiting.append(next(m for m in reversed(self.bus.conversation(donation_id))
                                    if m.receiver == charity_id and m.performative is Performative.ACCEPT_PROPOSAL))
        return waiting

    def answer_award(self, charity_id: str, donation_id: str, accept: bool, reason: str = "") -> bool:
        """A person accepts or declines a donation their charity won (the Charity screen). False when it is
        too late (the award already went to the next charity)."""
        award = next((m for m in self.awards_waiting(charity_id) if m.conversation_id == donation_id), None)
        if award is None:
            return False
        agent = next(a for a in self.charities if a.name == charity_id)
        self.bus.send(agent.answer_award(self.now, award, accept, reason))
        return True

    def set_charity_policy(self, policy: str) -> None:
        """Who confirms won donations: "manual" (a person) or "auto" (the charity's agent). Switching to "auto"
        also answers the awards already waiting."""
        self.config.simulation.charity_policy = policy
        for agent in self.charities:
            agent.policy = policy
            if policy == "auto":
                for award in self.awards_waiting(agent.name):
                    self.bus.send(agent._confirm(self.now, award))

    def set_driver_policy(self, policy: str) -> None:
        """Who answers delivery requests: "manual" (the presenter) or "auto" (simulated drivers). Switching to
        "auto" also accepts the requests already waiting."""
        for agent in self.drivers:
            agent.policy = policy
        if policy == "auto":
            for agent in self.drivers:
                for job, _ in self.logistics.pending_offers(agent.name):
                    self.answer_offer(agent.name, job.donation.id, accept=True)

    def answer_offer(self, driver_id: str, donation_id: str, accept: bool) -> None:
        """The driver's answer to a delivery request, from the Driver screen."""
        self.bus.send(Message(
            time=self.now, sender=driver_id, receiver="logistics",
            performative=Performative.ACCEPT_PROPOSAL if accept else Performative.REJECT_PROPOSAL,
            conversation_id=donation_id, protocol="job-offer",
            summary="I'll take it" if accept else "I can't take it"))

    def update_driver(self, driver: Driver) -> None:
        """A driver edits their profile (vehicle, what it carries, shift, home): used from the next planning."""
        self.scenario.drivers = [driver if d.id == driver.id else d for d in self.scenario.drivers]
        self.logistics.drivers[driver.id] = driver

    def set_driver_available(self, driver_id: str, available: bool) -> None:
        """The driver's on/off duty switch: the logistics agent re-plans at the next tick."""
        self.bus.send(Message(
            time=self.now, sender=driver_id, receiver="logistics", performative=Performative.REQUEST,
            conversation_id=f"duty-{driver_id}-{self.now}", protocol="availability",
            summary=f"{driver_id} is {'on' if available else 'off'} duty",
            content={"driver_id": driver_id, "available": available}))

    def next_id(self, prefix: str, people: list) -> str:
        taken = {p.id for p in people}
        width = 1 if prefix == "D" else 2
        return next(f"{prefix}{i:0{width}d}" for i in range(1, 1000) if f"{prefix}{i:0{width}d}" not in taken)

    def add_merchant(self, merchant: Merchant) -> None:
        """A merchant signs up: they can post food at once."""
        self.bus.register(merchant.id)
        self.scenario.merchants.append(merchant)
        self.directory.places[merchant.id] = Place(id=merchant.id, name=merchant.name, location=merchant.location)

    def add_driver(self, driver: Driver) -> None:
        """A driver signs up and goes on duty: logistics can give them deliveries nobody has yet."""
        if self.config.features.driver_confirmation:
            agent = DriverAgent(driver.id, self.bus, policy=self.drivers[0].policy if self.drivers
                                else self.config.simulation.driver_policy, seed=self.scenario.seed)
            self.drivers.append(agent)
            self.agents.insert(self.agents.index(self.logistics), agent)
        else:
            self.bus.register(driver.id)
        self.scenario.drivers.append(driver)
        self.logistics.add_driver(driver)
        self.set_driver_available(driver.id, True)

    def update_merchant(self, merchant: Merchant) -> None:
        """A merchant edits their profile: new donations use it; food already planned keeps its pickup place."""
        self.scenario.merchants = [merchant if m.id == merchant.id else m for m in self.scenario.merchants]
        self.directory.places[merchant.id] = Place(id=merchant.id, name=merchant.name, location=merchant.location)

    def add_offer(self, offer: DonationOffer) -> None:
        """Inject a merchant message during the run (the dashboard's merchant screen)."""
        self.scenario.offers.append(offer)
        self._offers.append(offer)
        self._offers.sort(key=lambda o: o.time)

    @property
    def now(self) -> int:
        return self.clock.now

    @property
    def finished(self) -> bool:
        return self.clock.finished

    def run_until(self, time: int) -> None:
        """Advance the simulation up to `time` (included), for step-by-step demos."""
        while not self.clock.finished and self.clock.now <= time:
            self.step()

    def result(self) -> SimulationResult:
        """The state so far (the whole day once `run` has finished)."""
        return SimulationResult(self.scenario, list(self.bus.log), self.logistics.routes,
                                dict(self.matching.awards),
                                {job_id: job.donation for job_id, job in self.logistics.jobs.items()},
                                set(self.logistics.lost))

    def step(self) -> None:
        now = self.clock.now
        self.bus.deliver()
        self._inject_events(now)
        for agent in self.agents:
            agent.step(now)
        self.clock.advance()

    def run(self) -> SimulationResult:
        while not self.clock.finished:
            self.step()
        return self.result()

    def _inject_events(self, now: int) -> None:
        while self._offers and self._offers[0].time <= now:
            offer = self._offers.pop(0)
            self.bus.send(Message(
                time=now, sender=offer.merchant_id, receiver="intake",
                performative=Performative.REQUEST, conversation_id=offer.id, protocol="intake",
                summary=f'"{_shorten(offer.text)}"', content={"offer": offer.model_dump(mode="json")},
            ))
        while self._disruptions and self._disruptions[0].time <= now:
            disruption = self._disruptions.pop(0)
            if disruption.kind is DisruptionKind.DRIVER_INCIDENT:
                self.bus.send(Message(
                    time=now, sender=disruption.target_id, receiver="dispatcher", performative=Performative.INFORM,
                    conversation_id=f"incident-{disruption.target_id}-{now}", protocol="incident",
                    summary=f'"{_shorten(disruption.reason)}"',
                    content={"driver_id": disruption.target_id, "text": disruption.reason,
                             "truth_minutes": disruption.minutes,  # truth: used only by the "oracle" policy
                             "form": ({"action": disruption.action, "minutes": disruption.minutes}
                                      if disruption.action else None)},
                ))
                continue
            if disruption.kind is DisruptionKind.MERCHANT_DELAY:
                self.bus.send(Message(
                    time=now, sender=ENVIRONMENT, receiver="logistics", performative=Performative.INFORM,
                    conversation_id=f"delay-{disruption.target_id}-{now}", protocol="disruption",
                    summary=f"{disruption.target_id}: food ready {disruption.minutes} min later ({disruption.reason})",
                    content={"kind": "merchant_delay", "donation_id": disruption.target_id,
                             "minutes": disruption.minutes, "reason": disruption.reason},
                ))
                continue
            self.bus.send(Message(
                time=now, sender=ENVIRONMENT, receiver="logistics", performative=Performative.CANCEL,
                conversation_id=f"disruption-{disruption.target_id}-{now}", protocol="disruption",
                summary=f"{disruption.target_id} cancels: {disruption.reason}",
                content={"driver_id": disruption.target_id, "reason": disruption.reason},
            ))


def _shorten(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
