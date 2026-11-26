"""Assessment: what really happened on the platform tonight, and the study on generated days."""

from __future__ import annotations

from collections import Counter, defaultdict

import altair as alt
import pandas as pd
import streamlit as st

from foodrescue import impact
from foodrescue.app import state, ui
from foodrescue.config import PROJECT_ROOT, load_config
from foodrescue.core.clock import format_hhmm
from foodrescue.models import Performative
from foodrescue.sim.compare import compare_day
from foodrescue.sim.scenario import load_scenario

OURS, BASELINE = "Our system", "Baseline (phone calls)"
COLORS = {OURS: "#2a78d6", BASELINE: "#eb6834"}  # same colors as the slide charts
FIGURES = PROJECT_ROOT / "data" / "figures"  # charts of the 100-day study (see the README)
DAYS = 20

sim = state.current()


def tonight() -> pd.DataFrame:
    """One row per donation posted on the platform, with what the agents recorded about it."""
    rows = []
    for offer in state.posted_offers(sim):
        donation = state.donation_of(sim, offer.id)
        stage = state.stage_of(sim, offer.id)
        job = sim.logistics.jobs.get(offer.id)
        charity = sim.matching.awards.get(offer.id)
        driver = job.confirmed_by or job.driver_id if job else None
        rows.append({
            "Donation": offer.id,
            "Merchant": state.name_of(sim, offer.merchant_id),
            "Posted": format_hhmm(offer.time),
            "Food": donation.describe() if donation else "",
            "kg": donation.total_kg if donation else 0.0,
            "Outcome": stage.label,
            "Charity": state.name_of(sim, charity) if charity else "",
            "Driver": state.name_of(sim, driver) if driver else "",
            "Picked up": format_hhmm(job.picked_up_at) if job and job.picked_up_at is not None else "",
            "Delivered": format_hhmm(job.delivered_at) if job and job.delivered_at is not None else "",
            "Wait (min)": job.picked_up_at - offer.time if job and job.picked_up_at is not None else None,
            "AI read details": bool(offer.details),
        })
    return pd.DataFrame(rows, columns=["Donation", "Merchant", "Posted", "Food", "kg", "Outcome", "Charity",
                                       "Driver", "Picked up", "Delivered", "Wait (min)", "AI read details"])


def driver_requests() -> Counter:
    """Delivery requests sent to drivers and how they ended (job-offer protocol)."""
    counts: Counter = Counter()
    for m in sim.bus.log:
        if m.protocol != "job-offer":
            continue
        if m.performative is Performative.PROPOSE:
            counts["sent"] += 1
        elif m.performative is Performative.ACCEPT_PROPOSAL:
            counts["accepted"] += 1
        elif m.performative is Performative.REJECT_PROPOSAL:
            counts["declined"] += 1
        elif m.performative is Performative.CANCEL and "expired" in m.content.get("text", ""):
            counts["expired"] += 1
    return counts


def real_use() -> None:
    data = tonight()
    if data.empty:
        ui.html('<div class="fr-empty">Nothing posted yet this evening. Post food from the Merchant screen: '
                "everything that happens to it is measured here.</div>")
        return
    unsafe = data["Outcome"] == "Rejected as unsafe"
    delivered = data["Outcome"] == "Delivered"
    safe_kg, delivered_kg = data.loc[~unsafe, "kg"].sum(), data.loc[delivered, "kg"].sum()
    waits = data.loc[delivered, "Wait (min)"].dropna()
    ui.tiles([
        ("Food posted", f"{data['kg'].sum():.1f} kg", f"{ui.count(len(data), 'donation')} from "
                                                       f"{ui.count(data['Merchant'].nunique(), 'merchant')}"),
        ("Safe food delivered", f"{100 * delivered_kg / safe_kg:.0f} %" if safe_kg else "-",
         f"{delivered_kg:.1f} of {safe_kg:.1f} kg safe to give"),
        ("Meals rescued", f"{impact.meals(delivered_kg)} meals", "≈ 420 g of food per meal (WRAP)",
         impact.source("meals")),
        ("Mean wait for pickup", f"{waits.mean():.0f} min" if len(waits) else "-", "from posting to pickup"),
    ])
    st.write("")

    left, right = st.columns([1, 1], gap="medium")
    with left, ui.card("outcomes"):
        ui.section("Where each donation stands", f"Platform time {state.clock_text(sim)}")
        by_outcome = data.groupby("Outcome")["kg"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        total = data["kg"].sum()
        ui.html("".join(ui.meter(outcome, row["sum"], total, f"{row['sum']:.1f} kg · {int(row['count'])}")
                        for outcome, row in by_outcome.iterrows()))
    with right, ui.card("agents"):
        ui.section("What the agents decided", "Counted from the messages they exchanged")
        safety = Counter(m.performative for m in sim.bus.log if m.sender == "safety" and m.protocol == "safety-check")
        requests = driver_requests()
        ai_reads = sum(1 for m in sim.bus.log if m.sender == "intake" and m.protocol == "safety-check"
                       and "AI" in str(m.content.get("read_by", "")))
        reports = [d for d in state.dispatcher_decisions(sim) if d.get("how") != "driver form"]
        rows = [
            ["Safety check", f"{safety[Performative.INFORM]} accepted, {safety[Performative.REFUSE]} refused, "
                             f"{ui.count(safety[Performative.REQUEST], 'question')} to the merchant"],
            ["AI reading merchants' details", ui.count(ai_reads, "message")],
            ["Charity bidding", f"{len(sim.matching.awards)} won, "
                                f"{sum(1 for o in data['Outcome'] if o == 'No charity')} with no charity"],
            ["Delivery requests to drivers", f"{requests['sent']} sent, {requests['accepted']} accepted, "
                                             f"{requests['declined']} declined, {requests['expired']} expired"],
            ["Drivers' free-text reports read by the AI", ui.count(len(reports), "report")],
        ]
        ui.table(["Agent", "Tonight"], [[ui.e(a), ui.e(b)] for a, b in rows])

    st.write("")
    left, right = st.columns([1, 1], gap="medium")
    with left, ui.card("charities"):
        ui.section("Received by charity", "Delivered food only")
        received = data[delivered].groupby("Charity")["kg"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        ui.table(["Charity", "Donations", "kg", "Meals"],
                 [[ui.e(name), str(int(row["count"])), f"{row['sum']:.1f}", str(impact.meals(row["sum"]))]
                  for name, row in received.iterrows()], numeric={1, 2, 3}, empty="Nothing delivered yet.")
    with right, ui.card("merchants"):
        ui.section("Given by merchant", "Posted and delivered")
        given: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for _, row in data.iterrows():
            given[row["Merchant"]][0] += row["kg"]
            given[row["Merchant"]][1] += row["kg"] if row["Outcome"] == "Delivered" else 0.0
        ui.table(["Merchant", "Posted (kg)", "Delivered (kg)"],
                 [[ui.e(name), f"{p:.1f}", f"{d:.1f}"] for name, (p, d) in sorted(given.items())], numeric={1, 2})

    st.write("")
    with ui.card("donations"):
        ui.section("Every donation", "Download the table to analyse the evening yourself")
        st.dataframe(data, hide_index=True, use_container_width=True)
        st.download_button("Download as CSV", data.to_csv(index=False).encode("utf-8"),
                           file_name=f"food-rescue-evening-{format_hhmm(sim.now).replace(':', 'h')}.csv",
                           mime="text/csv", icon=":material/download:")


@st.cache_data(show_spinner="Running both systems on the same generated days...")
def comparison(days: int) -> pd.DataFrame:
    config = load_config()
    rows = []
    for seed in range(1, days + 1):
        ours_day, base_day = compare_day(load_scenario(f"random:{seed}"), config)
        rows += [{"day": seed, "system": OURS, "rescued": ours_day.rescued_pct},
                 {"day": seed, "system": BASELINE, "rescued": base_day.rescued_pct}]
    return pd.DataFrame(rows)


def study() -> None:
    ui.html('<div class="fr-text">Generated days, run with the same safety rules and the same data for our agents '
            "and for a phone-call baseline: the difference comes from matching, routing and replanning. These "
            "days do not use the AI (it reads nothing), so they measure the planning only.</div>")
    st.write("")
    days = comparison(DAYS)
    means = days.groupby("system")["rescued"].mean()
    ui.tiles([
        ("Safe food rescued, our system", f"{means[OURS]:.1f} %", f"mean of {DAYS} generated days"),
        ("Safe food rescued, baseline", f"{means[BASELINE]:.1f} %", f"mean of {DAYS} generated days"),
        ("Difference", f"{means[OURS] - means[BASELINE]:+.1f} points", "same days for both"),
    ])
    st.write("")
    with ui.card("days"):
        ui.section(f"{DAYS} generated days: % of safe food rescued", "Each day is a seed of the generator")
        chart = alt.Chart(days).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                         width=alt.RelativeBandSize(0.9)).encode(
            x=alt.X("day:O", title="Day (seed)", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("system:N", sort=[OURS, BASELINE]),
            y=alt.Y("rescued:Q", title="% rescued", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(gridColor="#f4f4f5", domain=False, tickCount=5)),
            color=alt.Color("system:N", sort=[OURS, BASELINE],
                            scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                            legend=alt.Legend(title=None, orient="top", direction="horizontal")),
            tooltip=[alt.Tooltip("day:O", title="Day"), alt.Tooltip("system:N", title="System"),
                     alt.Tooltip("rescued:Q", title="% rescued", format=".1f")],
        ).properties(height=300).configure_view(stroke=None).configure_axis(
            labelColor="#71717a", titleColor="#71717a", tickColor="#e4e4e7", domainColor="#e4e4e7",
        ).configure_legend(labelColor="#52525b")
        st.altair_chart(chart, use_container_width=True)
        with st.expander("Table view"):
            st.dataframe(days.pivot(index="day", columns="system", values="rescued")[[OURS, BASELINE]],
                         use_container_width=True)

    st.write("")
    with ui.card("study"):
        ui.section("The full study (100 days)", "100 simulated days per disruption level, with 95% confidence "
                                                 "intervals")
        figures = {
            "Rescued by disruption level": "1_rescued_by_disruption.png",
            "Gain over the baseline": "2_gain_vs_baseline.png",
            "Where food is lost": "3_where_food_is_lost.png",
            "Intake accuracy by language": "4_intake_by_language.png",
            "Learning charities' preferences": "5_learning_curve.png",
            "Sensitivity of the results": "6_sensitivity.png",
            "Routing benchmark (Li & Lim)": "7_routing_benchmark.png",
        }
        available = {title: FIGURES / name for title, name in figures.items() if (FIGURES / name).exists()}
        if not available:
            ui.html('<div class="fr-empty">No figure found in data/figures.</div>')
        else:
            for tab, path in zip(st.tabs(list(available)), available.values(), strict=True):
                with tab:
                    st.image(str(path), use_container_width=True)


ui.page_header("Assessment", "What the platform achieved",
               "Tonight's numbers come from real use: every donation posted in the Merchant screen, every answer "
               "from drivers. The study runs our agents and a baseline on generated days.")
st.write("")
real_tab, study_tab = st.tabs(["This evening (real use)", "Study (generated days)"])
with real_tab:
    real_use()
with study_tab:
    study()
