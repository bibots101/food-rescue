"""Demo check: the merge gate that keeps `main` ready to demo.

    uv run python -m foodrescue.demo_check

Replays the demo evening without a screen and checks every moment the presentation relies on.
It also replays the LLM reading from the recorded answers, without calling Ollama, to make sure
the demo works offline. Exit code 0 = ready to demo, 1 = something the demo needs is broken.

Run it before a presentation: it must print READY TO DEMO.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from foodrescue.config import load_config
from foodrescue.llm.client import LLMMode
from foodrescue.models import Performative, StopKind
from foodrescue.sim.compare import compare_day
from foodrescue.sim.runner import Simulation, SimulationResult
from foodrescue.sim.scenario import load_scenario


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _messages(result: SimulationResult, performative: Performative, protocol: str) -> list:
    return [m for m in result.messages if m.performative is performative and m.protocol == protocol]


def check_demo_evening() -> list[Check]:
    scenario = load_scenario("demo")
    result = Simulation(scenario, load_config()).run()
    checks = []

    refusals = _messages(result, Performative.REFUSE, "safety-check")
    checks.append(Check("Unsafe rice (O6) is rejected and cites rule R03",
                        [m.conversation_id for m in refusals] == ["O6"] and "R03" in refusals[0].summary,
                        ", ".join(m.summary for m in refusals)))

    safe = {o.id for o in scenario.offers} - {"O6"}
    bidding = {m.conversation_id for m in _messages(result, Performative.CFP, "contract-net")}
    refused_bids = _messages(result, Performative.REFUSE, "contract-net")
    checks.append(Check("Every safe donation goes through the charities' bidding", bidding == safe,
                        f"{len(bidding)} calls for proposals, {len(refused_bids)} refusals with a reason"))

    opening = {c.id: c.opening for c in scenario.charities}
    windows = {o.id: o.donation.pickup_window for o in scenario.offers}
    on_time = all(
        (windows[s.donation_id] if s.kind is StopKind.PICKUP else opening[s.place_id]).contains(s.eta)
        for route in result.routes.values() for s in route.stops)
    delivered = {s.donation_id for route in result.routes.values() for s in route.stops if s.kind is StopKind.DELIVERY}
    checks.append(Check("Every safe donation is delivered within its time windows", delivered == safe and on_time,
                        f"{len(delivered)}/{len(safe)} delivered"))

    cancel = scenario.disruptions[0]
    replans = [m for m in _messages(result, Performative.INFORM, "replan")
               if m.content["cancelled_driver"] == cancel.target_id]
    moved = replans[0].content["moves"] if replans else []
    checks.append(Check(f"The {cancel.target_id} cancellation at its scheduled time moves at least one donation",
                        bool(moved) and all(m["driver_id"] not in (None, cancel.target_id) for m in moved),
                        ", ".join(f"{m['donation_id']} -> {m['driver_name']}" for m in moved)))

    notified = {m.receiver for m in _messages(result, Performative.INFORM, "notification")}
    checks.append(Check("Every merchant gets a notification", {o.merchant_id for o in scenario.offers} <= notified))

    ours, baseline = compare_day(scenario, load_config())
    checks.append(Check("Our system rescues more than the baseline on the demo evening",
                        ours.rescued_pct > baseline.rescued_pct,
                        f"ours {ours.rescued_pct}% vs baseline {baseline.rescued_pct}%"))
    return checks


def check_llm_replay() -> list[Check]:
    """The LLM reading of the demo must work from the recorded answers alone (no Ollama)."""
    config = load_config()
    config.features.llm_intake = True
    config.llm.mode = LLMMode.REPLAY
    result = Simulation(load_scenario("demo"), config).run()
    read = [m for m in result.messages if m.sender == "intake" and m.protocol == "safety-check"]
    by_llm = [m for m in read if m.content.get("read_by") == "llm"]
    missing = sorted(m.conversation_id for m in read if m.content.get("read_by") != "llm")
    return [
        Check("Every demo message has a recorded LLM answer (offline demo)", len(by_llm) == len(read) == 8,
              "missing: " + ", ".join(missing) + ". Re-record: delete data/llm_cache/*.json, then run "
              "`python -m foodrescue run --llm` with Ollama running" if missing else "8/8 replayed"),
        Check("With the LLM reading, at least 6 donations reach a charity", len(result.awards) >= 6,
              f"{len(result.awards)} awarded"),
        Check("With the LLM reading, the unsafe rice (O6) is still rejected",
              any(m.sender == "safety" and m.performative is Performative.REFUSE and m.conversation_id == "O6"
                  for m in result.messages)),
    ]


def check_rag_replay() -> list[Check]:
    """Safety explanations with sources must work offline (cached vectors and recorded answers)."""
    config = load_config()
    config.features.rag_safety = True
    config.llm.mode = LLMMode.REPLAY
    result = Simulation(load_scenario("demo"), config).run()
    refusal = next((m for m in result.messages if m.sender == "safety" and m.performative is Performative.REFUSE),
                   None)
    cited = refusal.content.get("explanation") if refusal else None
    ok = bool(cited and cited["grounded"] and cited["retrieval"] == "embeddings" and cited["source"].startswith("WHO"))
    detail = (f"{cited['source']}: \"{cited['quote']}\"" if ok else
              "re-record: run `python -m foodrescue run --rag` once with Ollama running")
    return [Check("The rejected rice is explained with a grounded WHO quote, offline", ok, detail)]


def check_driver_confirmation() -> list[Check]:
    """The dashboard asks drivers to accept each delivery; with simulated drivers answering, the demo
    evening must still deliver every safe donation, each one accepted before it is picked up."""
    config = load_config()
    config.features.driver_confirmation = True
    config.simulation.driver_policy = "auto"
    simulation = Simulation(load_scenario("demo"), config)
    result = simulation.run()
    safe = {o.id for o in result.scenario.offers} - {"O6"}
    delivered = {job_id for job_id, job in simulation.logistics.jobs.items() if job.delivered_at is not None}
    accepted = {m.conversation_id for m in result.messages
                if m.protocol == "job-offer" and m.performative is Performative.ACCEPT_PROPOSAL}
    return [Check("With driver confirmation (drivers answer automatically), every safe donation is still delivered",
                  delivered == safe and delivered <= accepted, f"{len(delivered)}/{len(safe)} delivered, "
                                                               f"{len(accepted)} accepted")]


SECTIONS: list[tuple[str, Callable[[], list[Check]]]] = [
    ("Demo evening", check_demo_evening),
    ("LLM reading, replayed offline", check_llm_replay),
    ("Safety explanations with sources (RAG), offline", check_rag_replay),
    ("Drivers accept each delivery", check_driver_confirmation),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    failures = 0
    for title, run_section in SECTIONS:
        print(title)
        try:
            checks = run_section()
        except Exception as exc:  # a crash is a failed check, reported, not a traceback during a demo
            checks = [Check("runs without crashing", False, f"{type(exc).__name__}: {exc}")]
        for check in checks:
            failures += not check.ok
            print(f"  {'✓' if check.ok else '✗'} {check.name}" + (f"  ({check.detail})" if check.detail else ""))
    print()
    print("READY TO DEMO" if failures == 0 else f"NOT READY: {failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
