"""Command line.

    python -m foodrescue run --scenario demo                # every message of the demo evening
    python -m foodrescue run --protocol contract-net        # only the charities' bidding
    python -m foodrescue run --scenario random:7 --quiet    # summary of a random day
    python -m foodrescue run --llm --protocol safety-check  # merchant messages read by the LLM
    python -m foodrescue compare                            # our system vs the baseline, demo evening
    python -m foodrescue compare --days 20                  # ... on 20 random days
"""

from __future__ import annotations

import argparse
import sys

from foodrescue.config import load_config
from foodrescue.sim.compare import compare_day, compare_many, format_comparison
from foodrescue.sim.runner import Simulation
from foodrescue.sim.scenario import load_scenario
from foodrescue.sim.trace import format_summary, format_trace


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic text on the Windows console

    parser = argparse.ArgumentParser(prog="foodrescue", description="Food Rescue multi-agent system")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="simulate a day and print the agents' messages")
    run.add_argument("--scenario", default="demo", help="'demo', 'random:<seed>' or a YAML file")
    run.add_argument("--protocol", help="only show one protocol, e.g. contract-net, delivery, replan")
    run.add_argument("--quiet", action="store_true", help="only print the summary")
    run.add_argument("--llm", action="store_true",
                     help="read merchant messages with the LLM (same as features.llm_intake: true)")
    run.add_argument("--rag", action="store_true",
                     help="explain safety verdicts with quotes from the WHO/EU documents (features.rag_safety)")
    compare = commands.add_parser("compare", help="our system vs the phone-call baseline, same day(s)")
    compare.add_argument("--scenario", default="demo", help="'demo', 'random:<seed>' or a YAML file")
    compare.add_argument("--days", type=int, help="instead: compare on N random days (seeds 1..N)")
    args = parser.parse_args(argv)

    config = load_config()
    if args.command == "compare":
        if args.days:
            print(compare_many([load_scenario(f"random:{seed}") for seed in range(1, args.days + 1)], config))
        else:
            print(format_comparison(*compare_day(load_scenario(args.scenario), config)))
        return 0

    if args.llm:
        config.features.llm_intake = True
    if args.rag:
        config.features.rag_safety = True
    result = Simulation(load_scenario(args.scenario), config).run()
    if not args.quiet:
        print(format_trace(result.messages, args.protocol))
        print()
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
