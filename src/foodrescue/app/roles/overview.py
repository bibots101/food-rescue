"""Overview: the operator's view of the whole evening (map, drivers, donations, agent messages)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from foodrescue import impact
from foodrescue.app import state, ui
from foodrescue.app.maps import driver_color, route_map
from foodrescue.core.clock import format_hhmm
from foodrescue.models import Message, Performative

NEGATIVE = {Performative.REFUSE, Performative.REJECT_PROPOSAL, Performative.FAILURE, Performative.CANCEL}

sim = state.current()
offers = state.posted_offers(sim)
stages = {o.id: state.stage_of(sim, o.id) for o in offers}
kg = {o.id: (d.total_kg if (d := state.donation_of(sim, o.id)) else 0.0) for o in offers}

ui.page_header("Overview", "Tonight's operations",
               f"{sim.scenario.city or 'City'} · {len(sim.scenario.merchants)} merchants · "
               f"{len(sim.scenario.charities)} charities · {len(sim.scenario.drivers)} drivers · "
               f"simulated time {state.clock_text(sim)}")
st.write("")

delivered = [i for i, s in stages.items() if s.label == "Delivered"]
on_road = [i for i, s in stages.items() if s.on_the_road]
rejected = [i for i, s in stages.items() if s.label == "Rejected as unsafe"]
km = sum(route.distance_km for route in sim.logistics.routes.values())
delivered_kg = sum(kg[i] for i in delivered)
ui.tiles([
    ("Offers received", str(len(offers)), "messages from merchants"),
    ("Delivered", f"{sum(kg[i] for i in delivered):.1f} kg",
     f"{ui.count(len(delivered), 'donation')} delivered"),
    ("On the road", f"{sum(kg[i] for i in on_road):.1f} kg",
     f"{ui.count(len(on_road), 'donation')} planned or picked up"),
    ("Meals rescued", f"{impact.meals(delivered_kg)} meals", "≈ 420 g of food per meal (WRAP)",
     impact.source("meals")),
    ("Rejected as unsafe", str(len(rejected)), "by the safety rules"),
    ("Distance planned", f"{km:.1f} km", "all drivers"),
])
st.write("")

left, right = st.columns([2.1, 1], gap="medium")
with left, ui.card("map"):
    ui.section("Live map", "Planned routes and each driver's last known position")
    choice = st.segmented_control(
        "Show routes", ["All drivers", *(d.id for d in sim.scenario.drivers)], default="All drivers",
        format_func=lambda i: i if i == "All drivers" else state.name_of(sim, i), key="map-driver",
        label_visibility="collapsed")
    route_map(sim, show_driver=None if choice in (None, "All drivers") else choice, height=430)

with right:
    with ui.card("drivers"):
        ui.section("Drivers")
        rows = []
        for driver in sim.scenario.drivers:
            label, tone = state.driver_status(sim, driver.id)
            stops = state.stops_of(sim, driver.id)
            done = sum(1 for s in stops if s.eta is not None and s.eta <= sim.now)
            swatch = f'<i class="fr-swatch" style="background:{driver_color(sim, driver.id)};margin-right:8px"></i>'
            rows.append([f'{swatch}<b>{ui.e(driver.name)}</b> <span class="fr-muted">{driver.id}</span>',
                         ui.pill(label, tone), f"{done}/{len(stops)}"])
        ui.table(["Driver", "Status", "Stops"], rows, numeric={2})
    with ui.card("dispatcher"):
        ui.section("Dispatcher decisions", "Drivers' problem reports and what the dispatcher agent did")
        ui.list_items([(format_hhmm(d["time"]),
                        f'<span class="fr-who">{ui.e(state.name_of(sim, d["driver_id"]))}</span> '
                        f'<span class="fr-muted">wrote</span><div class="fr-text">"{ui.e(d["text"])}"</div>'
                        f'<div class="fr-text">{ui.tag(d["tool"])} {ui.e(state.decision_text(d))} '
                        f'<span class="fr-muted">({ui.e(d["how"])})</span></div>')
                       for d in reversed(state.dispatcher_decisions(sim))], empty="No incident reported yet.")

st.write("")
board, activity, follow = st.tabs(["Donations", "Agent messages", "Follow a donation"])


def message_rows(messages: list[Message]) -> list[tuple[str, str]]:
    rows = []
    for m in messages:
        tone = "serious" if m.performative in NEGATIVE else ""
        rows.append((format_hhmm(m.time),
                     f'<span class="fr-who">{ui.e(state.name_of(sim, m.sender))}</span>'
                     f'<span class="fr-arrow">to</span><span class="fr-who">{ui.e(state.name_of(sim, m.receiver))}'
                     f'</span> &nbsp;{ui.tag(m.performative.value.upper(), tone)} '
                     f'<span class="fr-muted">{ui.e(m.conversation_id)} · {ui.e(m.protocol)}</span>'
                     f'<div class="fr-text">{ui.e(m.summary)}</div>'))
    return rows


with board, ui.card("board"):
    ui.section("All donations", "Planned, then picked up, then delivered: each step is recorded by the logistics "
                                "agent when it happens")
    rows = []
    for offer in reversed(offers):
        stage, donation, job = stages[offer.id], state.donation_of(sim, offer.id), sim.logistics.jobs.get(offer.id)
        charity = sim.matching.awards.get(offer.id)
        asked = sim.matching.pending.get(offer.id)  # won, waiting for that charity to accept
        charity_text = (ui.e(state.name_of(sim, charity)) if charity else
                        f'<span class="fr-muted">asking</span> {ui.e(state.name_of(sim, asked.ranked[0].charity_id))}'
                        if asked else "-")
        declined ="".join(f'<div class="fr-muted">declined by {ui.e(state.name_of(sim, c))}: {ui.e(r)}</div>'
                           for _, c, r in state.declines(sim, offer.id))
        rows.append([
            f"<b>{ui.e(offer.id)}</b>", ui.e(state.name_of(sim, offer.merchant_id)),
            ui.e(donation.describe() if donation else "..."), f"{kg[offer.id]:g}",
            ui.pill(stage.label, stage.tone), charity_text + declined,
            ui.e(state.name_of(sim, job.driver_id) if job and job.driver_id else "-"),
            format_hhmm(job.pickup_eta) if job and job.pickup_eta is not None else "-",
            format_hhmm(job.delivery_eta) if job and job.delivery_eta is not None else "-",
        ])
    with st.container(height=440, border=False):
        ui.table(["Id", "Merchant", "Food", "Kg", "Status", "Charity", "Driver", "Pickup", "Delivery"], rows,
                 numeric={3}, empty="No donation yet: move the clock forward.")

with activity, ui.card("activity"):
    protocols = sorted({m.protocol for m in sim.bus.log if m.protocol})
    chosen = st.pills("Protocols", protocols, selection_mode="multi",
                      default=[p for p in protocols if p != "notification"], label_visibility="collapsed")
    shown = [m for m in sim.bus.log if m.protocol in (chosen or [])][-80:]
    with st.container(height=460, border=False):
        ui.list_items(message_rows(list(reversed(shown))), empty="No message yet: move the clock forward.")

with follow, ui.card("follow"):
    if not offers:
        ui.html('<div class="fr-empty">No donation yet: move the clock forward.</div>')
    else:
        merchant_of = {o.id: state.name_of(sim, o.merchant_id) for o in offers}
        donation_id = st.selectbox("Donation", list(merchant_of), index=len(offers) - 1,
                                   format_func=lambda i: f"{i} · {merchant_of[i]}")
        stage = stages[donation_id]
        ui.html(f'<div class="fr-row"><div class="fr-muted">Status</div>{ui.pill(stage.label, stage.tone)}</div>'
                + ui.stepper(state.STEPS, stage.step, stage.state))
        conversation = [m for m in sim.bus.conversation(donation_id) if m.protocol != "notification"]
        talk, scores = st.columns([1.6, 1], gap="large")
        with talk:
            ui.section("Conversation between agents")
            ui.list_items(message_rows(conversation))
        with scores:
            # the matching agent's award (a driver accepting the delivery is an ACCEPT_PROPOSAL too)
            accept = next((m for m in reversed(conversation) if m.performative is Performative.ACCEPT_PROPOSAL
                           and m.sender == "matching" and "scores" in m.content), None)
            ui.section("Contract Net award", "Award score = need 50% + closeness 30% + fairness 20%")
            refusals = state.declines(sim, donation_id)
            if refusals:
                ui.list_items([(format_hhmm(t), f'<span class="fr-who">{ui.e(state.name_of(sim, c))}</span> '
                                f'<span class="fr-muted">declined after winning</span>'
                                f'<div class="fr-text">{ui.e(r)}</div>') for t, c, r in refusals])
            if accept is None:
                ui.html('<div class="fr-empty">No award yet.</div>')
            else:
                winner = accept.receiver
                declined = {c for _, c, _ in refusals}
                awarded = sim.matching.awards.get(donation_id)

                def role(c: str) -> str:
                    if c == awarded:
                        return "Winner"
                    if c in declined:
                        return "Declined"
                    return "Asked" if c == winner else "Other bidder"

                data = pd.DataFrame([{"charity": state.name_of(sim, c), "score": s, "role": role(c)}
                                     for c, s in accept.content["scores"].items()])
                data["label"] = data.apply(lambda r: f"{r.score:.2f}" + (f"  {r.role.lower()}" if r.role !=
                                                                          "Other bidder" else ""), axis=1)
                base = alt.Chart(data).encode(
                    y=alt.Y("charity:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
                    x=alt.X("score:Q", title="Award score", scale=alt.Scale(domain=[0, 1.15]),
                            axis=alt.Axis(grid=True, gridColor="#f4f4f5", domain=False, tickCount=5)),
                    tooltip=[alt.Tooltip("charity:N", title="Charity"), alt.Tooltip("score:Q", format=".2f"),
                             alt.Tooltip("role:N", title="")],
                )
                bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
                    color=alt.Color("role:N", scale=alt.Scale(domain=["Winner", "Asked", "Declined", "Other bidder"],
                                                              range=["#18181b", "#a16207", "#fca5a5", "#d4d4d8"]),
                                    legend=None))
                labels = base.mark_text(align="left", dx=6, color="#52525b", fontSize=11).encode(text="label:N")
                st.altair_chart((bars + labels).properties(height=40 * len(data) + 30).configure_view(stroke=None)
                                .configure_axis(labelColor="#71717a", titleColor="#71717a"),
                                use_container_width=True)
