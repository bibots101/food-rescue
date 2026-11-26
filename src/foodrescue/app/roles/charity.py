"""Charity app: sign up, keep the account up to date, follow tonight's deliveries and offers."""

from __future__ import annotations

import streamlit as st

from foodrescue import impact
from foodrescue.agents.charity import CharityAgent
from foodrescue.app import forms, state, ui
from foodrescue.core.clock import format_hhmm
from foodrescue.models import CATEGORY_LABELS, Charity, FoodCategory, Location, Performative, TimeWindow

NEW = "__new__"
CATEGORIES = [c for c in CATEGORY_LABELS if c is not FoodCategory.OTHER]

# Kinds of food a charity can only take with cold storage (WHO Five Keys: keep perishable food cold).
COLD_KINDS = {FoodCategory.DAIRY: "a fridge", FoodCategory.MEAT_FISH: "a fridge or a freezer"}

sim = state.current()
agents = {a.name: a for a in sim.agents if isinstance(a, CharityAgent)}


def can_store(category: FoodCategory, fridge: bool, freezer: bool) -> bool:
    if category is FoodCategory.DAIRY:
        return fridge
    if category is FoodCategory.MEAT_FISH:
        return fridge or freezer
    return True


def account_form(existing: Charity | None) -> Charity | None:
    """The sign-up form (no `existing`) or the account form. Returns the charity to save, if valid."""
    key = existing.id if existing else "new"
    places = [*sim.scenario.merchants, *sim.scenario.charities]
    centre = Location(lat=sum(p.location.lat for p in places) / len(places),
                      lon=sum(p.location.lon for p in places) / len(places))
    lat, lon = forms.location_picker(f"charity-{key}", existing.location if existing else centre,
                                     "Where food is delivered")
    # Outside the form, so the food kinds that need cold storage are unlocked or locked at once.
    st.markdown("**Cold storage**")
    storage = st.columns(2)
    fridge = storage[0].checkbox("We have a fridge (chilled food)", key=f"fridge-{key}",
                                 value=existing.has_fridge if existing else False)
    freezer = storage[1].checkbox("We have a freezer (frozen food)", key=f"freezer-{key}",
                                  value=existing.has_freezer if existing else False)
    with st.form(f"charity-form-{key}"):
        name = st.text_input("Charity name", value=existing.name if existing else "")
        address = st.text_input("Address", value=existing.location.address if existing else "",
                                placeholder="Street, neighbourhood")
        phone = st.text_input("Phone", value=existing.phone if existing else "")
        hours = st.columns(3)
        opens = hours[0].time_input("Receives food from", forms.to_time(existing.opening.start if existing
                                                                          else 17 * 60), step=900)
        closes = hours[1].time_input("Until", forms.to_time(existing.opening.end if existing else 22 * 60),
                                     step=900)
        capacity = hours[2].number_input("Can receive today (kg)", min_value=0.0, step=5.0,
                                         value=float(existing.capacity_kg) if existing else 50.0)
        st.markdown("**Food we accept, and how much we still need today**")
        accepted, needs = set(), {}
        for category in CATEGORIES:
            storable = can_store(category, fridge, freezer)
            row = st.columns([2, 1], vertical_alignment="center")
            # A locked kind is a new widget (key with the lock), so it shows unticked rather than a stale tick.
            takes = row[0].checkbox(CATEGORY_LABELS[category], key=f"accept-{key}-{category.value}-{storable}",
                                    value=storable and ((category in existing.accepted) if existing else True),
                                    disabled=not storable,
                                    help=None if storable else f"Needs {COLD_KINDS[category]}: tick it above first")
            need = row[1].number_input("Needed today (kg)", min_value=0.0, step=5.0,
                                       key=f"need-{key}-{category.value}-{storable}",
                                       value=float(existing.needs.get(category, 0.0)) if existing and storable
                                       else 0.0, disabled=not storable, label_visibility="collapsed")
            if takes and storable:
                accepted.add(category)
                if need > 0:
                    needs[category] = need
        notes = st.text_area("Anything else drivers or donors should know", value=existing.notes if existing else "",
                             height=70)
        saved = st.form_submit_button("Save changes" if existing else "Create my account", type="primary")
    if not saved:
        return None
    problems = []
    if not name.strip():
        problems.append("Give the charity's name.")
    if forms.to_minutes(opens) >= forms.to_minutes(closes):
        problems.append("The opening time must come before the closing time.")
    if capacity <= 0:
        problems.append("Say how many kg you can receive today.")
    if not accepted:
        problems.append("Tick at least one kind of food.")
    for problem in problems:
        st.error(problem)
    if problems:
        return None
    accepted.add(FoodCategory.OTHER)
    return Charity(
        id=existing.id if existing else sim.next_charity_id(), name=name.strip(),
        location=Location(lat=lat, lon=lon, address=address.strip()),
        opening=TimeWindow(start=forms.to_minutes(opens), end=forms.to_minutes(closes)),
        capacity_kg=capacity, has_fridge=fridge, has_freezer=freezer, accepted=accepted, needs=needs,
        phone=phone.strip(), notes=notes.strip(),
        acceptance=existing.acceptance if existing else {},
    )


head, who = st.columns([3, 1], vertical_alignment="bottom")
with who:
    options = {i: a.charity.name for i, a in agents.items()} | {NEW: "Register a new charity"}
    charity_id = state.identity("Signed in as", options, "charity_id")

if charity_id == NEW:
    with head:
        ui.page_header("Charity app", "Register your charity",
                       "Tell us where you are, when you can receive food and what you can store. Your details "
                       "stay private: only your charity's agent uses them to answer offers.")
    st.write("")
    with ui.card("register"):
        created = account_form(None)
    if created is not None:
        sim.add_charity(created)
        forms.forget_location("charity-new")
        st.session_state["_keep_charity_id"] = created.id
        st.session_state.pop("charity_id", None)
        st.toast(f"Welcome, {created.name}. You will receive the next offers.")
        st.rerun()
    st.stop()

agent = agents[charity_id]
charity = agent.charity
state.announce(sim, charity_id)
with head:
    ui.page_header("Charity app", charity.name,
                   f"Receives food {format_hhmm(charity.opening.start)} to {format_hhmm(charity.opening.end)}"
                   f"{' · ' + charity.location.address if charity.location.address else ''} · "
                   f"simulated time {state.clock_text(sim)}")
st.write("")

won = [d for d, c in sim.matching.awards.items() if c == charity_id]
stages = {d: state.stage_of(sim, d) for d in won}
arrived = [d for d in won if stages[d].label == "Delivered"]
coming = [d for d in won if stages[d].on_the_road]
kg = {d: (x.total_kg if (x := state.donation_of(sim, d)) else 0.0) for d in won}
storage = ("Fridge and freezer" if charity.has_fridge and charity.has_freezer else "Fridge" if charity.has_fridge
           else "Freezer only" if charity.has_freezer else "None")
ui.tiles([
    ("Room left today", f"{agent.remaining_kg:g} kg", f"of {charity.capacity_kg:g} kg"),
    ("Cold storage", storage, "decides which offers you can take"),
    ("On the way", f"{sum(kg[d] for d in coming):.1f} kg",
     f"{ui.count(len(coming), 'delivery', 'deliveries')} planned"),
    ("Received", f"{sum(kg[d] for d in arrived):.1f} kg",
     f"{ui.count(len(arrived), 'delivery', 'deliveries')} · ≈ {impact.meals(sum(kg[d] for d in arrived))} meals",
     impact.source("meals")),
])
st.write("")

DECLINE_REASONS = ["No room left after all", "No volunteer to receive it", "We cannot use this food",
                   "We close before it can arrive", "Something else"]


def reasons_html(message) -> str:
    """The charity agent's arguments for a bid, as a short list."""
    reasons = message.content.get("reasons") or [message.summary]
    return '<ul class="fr-reasons" style="margin:0;padding-left:18px">' + "".join(
        f"<li>{ui.e(r[0].upper() + r[1:])}</li>" for r in reasons) + "</ul>"


# One container whatever the number of awards: the elements below keep their place (the map is not remounted).
with st.container():
    for award in sim.awards_waiting(charity_id):
        donation_id = award.conversation_id
        donation = state.donation_of(sim, donation_id)
        merchant_id = next((o.merchant_id for o in sim.scenario.offers if o.id == donation_id), "")
        bid = next((m for m in sim.bus.conversation(donation_id) if m.sender == charity_id
                    and m.performative is Performative.PROPOSE), None)
        deadline = award.time + sim.matching.confirm_timeout
        with ui.card(f"award-{donation_id}"):
            window = donation.pickup_window if donation else None
            ui.html(f'<div class="fr-row"><div class="fr-section">You won a donation · {ui.e(donation_id)}</div>'
                    f'{ui.pill(f"Answer within {max(0, deadline - sim.now)} min", "warning")}</div>'
                    f'<div class="fr-kv"><div class="k">Food</div><div class="v"><b>'
                    f'{ui.e(donation.describe() if donation else donation_id)}</b> · '
                    f'{donation.total_kg if donation else 0:g} kg</div>'
                    f'<div class="k">From</div><div class="v">{ui.e(state.name_of(sim, merchant_id))}'
                    + (f', pickup {format_hhmm(window.start)} to {format_hhmm(window.end)}' if window else "")
                    + '</div><div class="k">Why you</div><div class="v">'
                    + (reasons_html(bid) if bid else "-")
                    + '</div></div><div class="fr-muted" style="margin:6px 0 10px">Nothing is delivered until you '
                      'accept. If you decline, the next charity in the bidding gets it.</div>')
            yes, no = st.columns(2)
            if yes.button("Accept the donation", key=f"award-yes-{donation_id}", type="primary",
                          icon=":material/check:", use_container_width=True, disabled=sim.finished):
                if sim.answer_award(charity_id, donation_id, accept=True):
                    state.advance(2)  # matching confirms, logistics plans the delivery
                else:
                    st.toast("Too late: it went to the next charity.")
                st.rerun()
            with no.popover("Decline", icon=":material/close:", use_container_width=True, disabled=sim.finished):
                why = st.selectbox("Why?", DECLINE_REASONS, key=f"award-why-{donation_id}")
                other = st.text_input("Details", key=f"award-other-{donation_id}",
                                      placeholder="Our fridge broke down tonight") if why == "Something else" else ""
                if st.button("Decline and pass it on", key=f"award-no-{donation_id}", type="primary",
                             use_container_width=True):
                    reason = (other.strip() or "no reason given") if why == "Something else" else why.lower()
                    if sim.answer_award(charity_id, donation_id, accept=False, reason=reason):
                        state.advance(2)  # matching offers it to the next bidder
                    st.rerun()
        st.write("")

tonight, offers, account = st.tabs(["Tonight", "Offers", "Account"])

with tonight:
    main, side = st.columns([1.7, 1], gap="medium")
    with main, ui.card("deliveries"):
        ui.section("Your deliveries tonight", "Donations your agent won and confirmed")
        rows = []
        for donation_id in reversed(won):
            job, donation, stage = sim.logistics.jobs.get(donation_id), state.donation_of(sim, donation_id), \
                stages[donation_id]
            merchant = next((o.merchant_id for o in sim.scenario.offers if o.id == donation_id), "")
            rows.append([f"<b>{ui.e(donation_id)}</b>", ui.e(donation.describe() if donation else "..."),
                         f"{kg[donation_id]:g}", ui.e(state.name_of(sim, merchant)),
                         ui.e(state.name_of(sim, job.driver_id) if job and job.driver_id else "-"),
                         format_hhmm(job.delivery_eta) if job and job.delivery_eta is not None else "-",
                         ui.pill(stage.label, stage.tone)])
        ui.table(["Id", "Food", "Kg", "From", "Driver", "Arrives", "Status"], rows, numeric={2},
                 empty="Nothing won yet tonight.")
    with side:
        with ui.card("needs"):
            ui.section("Still needed today", "Private: only your agent knows this")
            if charity.needs:
                ui.html("".join(ui.meter(CATEGORY_LABELS[category], need - agent.remaining_needs.get(category, 0.0),
                                         need, f"{agent.remaining_needs.get(category, 0.0):g} of {need:g} kg")
                                for category, need in charity.needs.items()))
                ui.html('<div class="fr-muted">The bar fills as donations cover the need.</div>')
            else:
                ui.html('<div class="fr-empty">No specific need declared.</div>')
            refused = [CATEGORY_LABELS[c] for c in CATEGORIES if c not in charity.accepted]
            if refused:
                ui.html(f'<div class="fr-muted" style="margin-top:8px">Not accepted: {ui.e(", ".join(refused))}</div>')
        with ui.card("inbox"):
            ui.section("Notifications", "Messages from the platform")
            ui.list_items([(format_hhmm(m.time), f'<div class="fr-text" style="margin:0">{ui.e(m.content["text"])}'
                            '</div>') for m in reversed(state.inbox(sim, charity_id))], empty="Nothing yet.")

with offers, ui.card("bids"):
    ui.section("Offers announced tonight", "Every call for proposals, and how your agent answered with your "
                                           "private details")
    rows = []
    for cfp in [m for m in sim.bus.log if m.performative is Performative.CFP]:
        conversation = sim.bus.conversation(cfp.conversation_id)
        answer = next((m for m in conversation if m.sender == charity_id and m.protocol == "contract-net"
                       and m.performative in (Performative.PROPOSE, Performative.REFUSE)), None)
        if answer is None and cfp.time < sim.now - 2:
            continue  # announced before this charity joined
        accepted = any(m.receiver == charity_id and m.performative is Performative.ACCEPT_PROPOSAL
                       for m in conversation)
        declined = any(m.sender == charity_id and m.performative is Performative.FAILURE for m in conversation)
        timed_out = any(m.receiver == charity_id and m.performative is Performative.CANCEL for m in conversation)
        not_chosen = any(m.receiver == charity_id and m.performative is Performative.REJECT_PROPOSAL
                         for m in conversation)
        if answer is None:
            result = ui.pill("Answering", "neutral")
        elif answer.performative is Performative.REFUSE:
            result = ui.pill("Did not bid", "neutral")
        elif declined:
            result = ui.pill("Declined after winning", "serious")
        elif timed_out:
            result = ui.pill("Not answered in time", "serious")
        elif accepted and cfp.conversation_id in won:
            result = ui.pill("Won", "good")
        elif accepted:
            result = ui.pill("Won: accept or decline above", "warning")
        elif not_chosen:
            result = ui.pill("Not chosen", "neutral")
        else:
            result = ui.pill("Waiting", "neutral")
        offered = cfp.content["donation"]
        food = ", ".join(i["name"] for i in offered["items"])
        total = sum(i["kg"] for i in offered["items"])
        if answer is None:
            said = "..."
        elif answer.performative is Performative.PROPOSE:
            said = f'<div class="fr-muted">Bid, because:</div>{reasons_html(answer)}'
        else:
            said = f'<div class="fr-muted">No bid:</div>{ui.e(answer.summary)}'
        rows.append([format_hhmm(cfp.time), f"<b>{ui.e(cfp.conversation_id)}</b> {ui.e(food)}", f"{total:g}",
                     said, result])
    ui.table(["Time", "Offer", "Kg", "Your agent answered", "Result"], rows[::-1], numeric={2},
             empty="No offer announced yet.")

with account, ui.card("account"):
    ui.section("Your account", "Changes apply to the next offers your agent answers")
    edited = account_form(charity)
    if edited is not None:
        sim.update_charity(edited)
        st.toast("Account saved.")
        st.rerun()
