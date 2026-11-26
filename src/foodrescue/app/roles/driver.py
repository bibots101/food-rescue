"""Driver app: the next stop, tonight's route, a way to report a problem, and the driver's profile."""

from __future__ import annotations

import streamlit as st

from foodrescue.app import forms, state, ui
from foodrescue.app.maps import route_map
from foodrescue.core.clock import format_hhmm
from foodrescue.models import (
    VEHICLE_CAPACITY,
    VEHICLES,
    Disruption,
    DisruptionKind,
    Driver,
    Location,
    StopKind,
    TimeWindow,
)

sim = state.current()
drivers = {d.id: d for d in sim.scenario.drivers}

NEW = "__new__"


def driver_form(existing: Driver | None) -> Driver | None:
    """Sign-up (no `existing`) or profile: who the driver is, where they start from, what they drive and how much
    it carries, when they work. Returns the driver to save, if valid."""
    key = existing.id if existing else "new"
    places = [*sim.scenario.merchants, *sim.scenario.charities]
    centre = Location(lat=sum(p.location.lat for p in places) / len(places),
                      lon=sum(p.location.lon for p in places) / len(places))
    lat, lon = forms.location_picker(f"driver-{key}", existing.home if existing else centre,
                                     "Where you start from (home)")
    # Outside the form, so the suggested load follows the vehicle at once.
    vehicle = st.segmented_control("You deliver by", list(VEHICLES), format_func=VEHICLES.get,
                                   default=existing.vehicle if existing else "car", key=f"vehicle-{key}")
    vehicle = vehicle or (existing.vehicle if existing else "car")
    same = existing is not None and vehicle == existing.vehicle
    with st.form(f"driver-form-{key}"):
        name = st.text_input("Your name", value=existing.name if existing else "")
        phone = st.text_input("Phone", value=existing.phone if existing else "")
        cols = st.columns(3)
        capacity = cols[0].number_input(f"Your {VEHICLES[vehicle].lower()} carries (kg)", min_value=1.0, step=5.0,
                                        value=existing.capacity_kg if same else VEHICLE_CAPACITY[vehicle],
                                        key=f"capacity-{key}-{vehicle}")
        start = cols[1].time_input("Available from", forms.to_time(existing.shift.start if existing
                                                                   else max(sim.now, sim.scenario.start)), step=900)
        end = cols[2].time_input("Until", forms.to_time(existing.shift.end if existing else 23 * 60), step=900)
        saved = st.form_submit_button("Save changes" if existing else "Create my account", type="primary")
    if not saved:
        return None
    problems = [] if name.strip() else ["Give your name."]
    if forms.to_minutes(end) <= forms.to_minutes(start):
        problems.append("Your shift must end after it starts.")
    for problem in problems:
        st.error(problem)
    if problems:
        return None
    return Driver(id=existing.id if existing else sim.next_id("D", sim.scenario.drivers), name=name.strip(),
                  phone=phone.strip(), home=Location(lat=lat, lon=lon), capacity_kg=capacity, vehicle=vehicle,
                  shift=TimeWindow(start=forms.to_minutes(start), end=forms.to_minutes(end)))


head, who = st.columns([3, 1], vertical_alignment="bottom")
with who:
    waiting = {i: len(state.requests_for(sim, i)) for i in drivers}
    options = {i: f"{d.name} ({i})" + (f" · {ui.count(waiting[i], 'request')}" if waiting[i] else "")
               for i, d in drivers.items()}
    driver_id = state.identity("Signed in as", options | {NEW: "Register as a driver"}, "driver_id")
if driver_id == NEW:
    with head:
        ui.page_header("Driver app", "Drive for Food Rescue",
                       "You get delivery requests near you and accept the ones you can do.")
    st.write("")
    with ui.card("register"):
        created = driver_form(None)
    if created is not None:
        sim.add_driver(created)
        forms.forget_location("driver-new")
        st.session_state["_keep_driver_id"] = created.id
        st.session_state.pop("driver_id", None)
        state.advance(1)  # logistics puts the new driver on duty
        st.toast(f"Welcome, {created.name}. You are on duty.")
        st.rerun()
    st.stop()
driver = drivers[driver_id]
state.announce(sim, driver_id)
with head:
    ui.page_header("Driver app", driver.name,
                   f"{driver_id} · {VEHICLES[driver.vehicle].lower()}, carries {driver.capacity_kg:g} kg · shift "
                   f"{format_hhmm(driver.shift.start)} to {format_hhmm(driver.shift.end)}")
    on_duty = driver_id in sim.logistics.active
    on_board = sorted(job_id for job_id, job in sim.logistics.jobs.items() if job.driver_id == driver_id
                      and job.picked_up_at is not None and job.delivered_at is None)
    wanted = st.toggle("On duty" if on_duty else "Off duty", value=on_duty, key=f"duty-{driver_id}-{on_duty}",
                       disabled=sim.finished or (on_duty and bool(on_board)),
                       help=("Deliver the food on board first" if on_duty and on_board else
                             "Off duty: your planned pickups go to other drivers. On duty: you can get new "
                             "deliveries."))
    if wanted != on_duty:
        sim.set_driver_available(driver_id, wanted)
        state.advance(2)  # logistics re-plans, communication tells everyone concerned
        st.rerun()
st.write("")

stops = state.stops_of(sim, driver_id)
upcoming = [s for s in stops if s.eta is not None and s.eta > sim.now]
next_stop = upcoming[0] if upcoming else None


MISHAPS = ["Flat tire", "Car breakdown", "Accident", "Traffic jam", "Something else"]
EFFECTS = {"delay": "I'll be late", "cancel": "I can't continue: give my deliveries to another driver"}


def food_of(donation_id: str) -> str:
    donation = state.donation_of(sim, donation_id)
    return f"{donation.describe()} · {donation.total_kg:g} kg" if donation else donation_id


def pickup_note(stop) -> str:
    """The merchant's pickup instructions (their profile), shown for pickup stops."""
    if stop.kind is not StopKind.PICKUP:
        return ""
    merchant = next((m for m in sim.scenario.merchants if m.id == stop.place_id), None)
    return merchant.pickup_notes if merchant else ""


# One container whatever the number of requests: the maps below keep their place and are not remounted.
with st.container():
    for job, deadline in state.requests_for(sim, driver_id):
        plan = sim.logistics._plan(job)
        left = max(0, deadline - sim.now)
        merchant = next((m for m in sim.scenario.merchants if m.id == job.donation.merchant_id), None)
        with ui.card(f"request-{job.donation.id}"):
            note = merchant.pickup_notes if merchant else ""
            ui.html(f'<div class="fr-row"><div class="fr-section">New delivery request · {ui.e(job.donation.id)}</div>'
                    f'{ui.pill(f"Answer within {left} min", "warning")}</div>'
                    f'<div class="fr-kv"><div class="k">Pick up</div><div class="v">'
                    f'<b>{ui.e(plan["pickup_name"])}</b> at '
                    f'{format_hhmm(job.pickup_eta) if job.pickup_eta is not None else "-"}'
                    + (f'<div class="fr-muted">{ui.e(note)}</div>' if note else "")
                    + '</div><div class="k">Deliver to</div><div class="v">'
                    f'<b>{ui.e(state.name_of(sim, job.charity_id))}</b>'
                    f' at {format_hhmm(job.delivery_eta) if job.delivery_eta is not None else "-"}</div>'
                    f'<div class="k">Food</div><div class="v">{ui.e(food_of(job.donation.id))}</div></div>'
                    '<div style="height:10px"></div>')
            yes, no = st.columns(2)
            if yes.button("Accept", key=f"accept-{job.donation.id}", type="primary", icon=":material/check:",
                          use_container_width=True, disabled=sim.finished):
                sim.answer_offer(driver_id, job.donation.id, accept=True)
                state.advance(2)  # logistics plans the trip from here, merchant and charity are told
                st.rerun()
            if no.button("Decline", key=f"decline-{job.donation.id}", icon=":material/close:", use_container_width=True,
                         disabled=sim.finished):
                sim.answer_offer(driver_id, job.donation.id, accept=False)
                state.advance(2)  # it goes to another driver
                st.rerun()
        st.write("")

tonight_tab, profile_tab = st.tabs(["Tonight", "Profile"])

with tonight_tab:
    main, side = st.columns([1.5, 1], gap="medium")
    with main:
        with st.container(key="hero-next"):
            if next_stop is None:
                ui.html('<div class="fr-hero-eyebrow">Next stop</div><div class="fr-hero-title">'
                        + ("Nothing planned right now" if driver_id in sim.logistics.active
                           else "You are off duty tonight")
                        + '</div><div class="fr-hero-sub">New stops appear here as soon as the logistics agent plans '
                          'them.</div>')
            else:
                verb = "Pick up at" if next_stop.kind is StopKind.PICKUP else "Deliver to"
                ui.html(f'<div class="fr-row"><div><div class="fr-hero-eyebrow">Next stop</div>'
                        f'<div class="fr-hero-title">{verb} {ui.e(state.name_of(sim, next_stop.place_id))}</div>'
                        f'<div class="fr-hero-sub">{ui.e(next_stop.donation_id)} · '
                        f'{ui.e(food_of(next_stop.donation_id))}'
                        + (f'<br>Note: {ui.e(pickup_note(next_stop))}' if pickup_note(next_stop) else "")
                        + f'</div></div><div style="text-align:right"><div class="fr-hero-eyebrow">Arrive</div>'
                        f'<div class="fr-hero-time">{format_hhmm(next_stop.eta)}</div><div class="fr-hero-sub">in '
                        f'{next_stop.eta - sim.now} min</div></div></div>')
        st.write("")
        done_stops = [s for s in stops if s.eta is not None and s.eta <= sim.now]
        with ui.card("route"):
            if not upcoming:
                ui.section("Current trip", "No trip right now")
                ui.html('<div class="fr-empty">When you accept a delivery, your trip appears here.</div>')
            else:
                ui.section("Current trip", f"{ui.count(len(upcoming), 'stop')} to go, from where you are")
                items = []
                for stop in upcoming:
                    cls = "next" if stop is next_stop else ""
                    verb = "Pick up at" if stop.kind is StopKind.PICKUP else "Deliver to"
                    note = pickup_note(stop)
                    items.append(f'<div class="fr-tl-item {cls}"><div class="fr-time">{format_hhmm(stop.eta)}</div>'
                                 f'<div class="dot"></div><div><div class="what">{verb} '
                                 f'{ui.e(state.name_of(sim, stop.place_id))}</div><div class="detail">'
                                 f'{ui.e(stop.donation_id)} · {ui.e(food_of(stop.donation_id))}'
                                 + (f" · {ui.e(note)}" if note else "") + '</div></div></div>')
                ui.html(f'<div class="fr-tl">{"".join(items)}</div>')
                route_map(sim, show_driver=driver_id, only_places_on_route=True, height=320, part="current")
        st.write("")
        with ui.card("history"):
            # A switch, not an expander: a map drawn while hidden gets a wrong size and zoom.
            show = st.toggle(f"Show trip history ({ui.count(len(done_stops), 'stop')} done)",
                             key=f"history-{driver_id}")
            if show and not done_stops:
                ui.html('<div class="fr-empty">Nothing done yet tonight.</div>')
            elif show:
                rows = []
                for stop in reversed(done_stops):
                    donation = state.donation_of(sim, stop.donation_id)
                    rows.append([format_hhmm(stop.eta),
                                 "Picked up at" if stop.kind is StopKind.PICKUP else "Delivered to",
                                 ui.e(state.name_of(sim, stop.place_id)),
                                 f"<b>{ui.e(stop.donation_id)}</b> {ui.e(donation.describe() if donation else '')}",
                                 f"{donation.total_kg:g}" if donation else "-"])
                ui.table(["Time", "Did", "Where", "Food", "Kg"], rows, numeric={4})
                km = sim.logistics.routes[driver_id].distance_km
                ui.html(f'<div class="fr-muted" style="margin:6px 0 10px">{km:.1f} km planned or driven tonight</div>')
                route_map(sim, show_driver=driver_id, only_places_on_route=True, height=300, part="history")

    with side:
        if driver_id in sim.logistics.active and not sim.finished:
            with ui.card("report"):
                ui.section("Report a problem",
                           "Tell the dispatcher what happened and what it means for your deliveries")
                with st.form(f"report-form-{driver_id}", clear_on_submit=True):
                    what = st.selectbox("What happened", MISHAPS)
                    effect = st.radio("What now", ["delay", "cancel"], format_func=EFFECTS.get)
                    minutes = st.slider("How late (minutes)", 5, 120, 30, step=5)
                    details = st.text_area("Details (needed for \"Something else\")", height=68,
                                           placeholder="Le pont est fermé, je repars dans 40 minutes")
                    sent = st.form_submit_button("Send to the dispatcher", type="primary", icon=":material/send:",
                                                 use_container_width=True)
                st.caption("\"Something else\" is read by the AI dispatcher, which decides what to do.")
                if sent:
                    if what == "Something else" and not details.strip():
                        st.error("Describe what happened.")
                    else:
                        if what == "Something else":  # free text: the LLM dispatcher reads it
                            incident = Disruption(time=sim.now, kind=DisruptionKind.DRIVER_INCIDENT,
                                                  target_id=driver_id, reason=details.strip())
                        else:
                            said = f"I'll be late by {minutes} minutes" if effect == "delay" else "I can't continue"
                            incident = Disruption(time=sim.now, kind=DisruptionKind.DRIVER_INCIDENT,
                                                  target_id=driver_id,
                                                  reason=f"{what}: {said}" + (f". {details.strip()}"
                                                                              if details.strip() else ""),
                                                  minutes=minutes if effect == "delay" else 0, action=effect)
                        sim.add_disruption(incident)
                        state.advance(3)  # dispatcher decides, logistics replans, communication warns everyone
                        st.rerun()
        decisions = state.dispatcher_decisions(sim, driver_id)
        if decisions:
            with ui.card("decisions"):
                ui.section("Dispatcher decisions")
                ui.list_items([(format_hhmm(d["time"]), f'<div class="fr-text" style="margin:0">You wrote '
                                f'"{ui.e(d["text"])}"</div><div class="fr-text">{ui.tag(d["tool"])} '
                                f'{ui.e(state.decision_text(d))}</div>')
                               for d in reversed(decisions)])
        with ui.card("inbox"):
            ui.section("Notifications", "Messages from the platform")
            ui.list_items([(format_hhmm(m.time),
                            f'<div class="fr-text" style="margin:0">{ui.e(m.content["text"])}</div>'
                            + (f'<div style="margin-top:4px">{ui.pill("Question", "warning")}</div>'
                               if m.content.get("question") else ""))
                           for m in reversed(state.inbox(sim, driver_id))], empty="Nothing yet.")

with profile_tab, ui.card("profile"):
    ui.section("Your profile", "Changes apply to the next deliveries the logistics agent plans")
    edited = driver_form(driver)
    if edited is not None:
        sim.update_driver(edited)
        forms.forget_location(f"driver-{driver_id}")
        st.toast("Profile saved.")
        st.rerun()
