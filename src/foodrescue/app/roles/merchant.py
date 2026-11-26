"""Merchant app: post leftover food in plain words, follow each donation, keep the profile up to date."""

from __future__ import annotations

import streamlit as st

from foodrescue import impact
from foodrescue.app import food_form, forms, state, ui
from foodrescue.core.clock import format_hhmm
from foodrescue.models import Disruption, DisruptionKind, Location, Merchant, Performative

sim = state.current()
merchants = {m.id: m for m in sim.scenario.merchants}

KINDS = ["bakery", "restaurant", "supermarket", "greengrocer", "hotel", "caterer", "other"]
NEW = "__new__"


def merchant_form(existing: Merchant | None) -> Merchant | None:
    """The sign-up form (no `existing`) or the profile form. Returns the merchant to save, if valid."""
    key = existing.id if existing else "new"
    places = [*sim.scenario.merchants, *sim.scenario.charities]
    centre = Location(lat=sum(p.location.lat for p in places) / len(places),
                      lon=sum(p.location.lon for p in places) / len(places))
    lat, lon = forms.location_picker(f"merchant-{key}", existing.location if existing else centre,
                                     "Where drivers pick up")
    with st.form(f"merchant-form-{key}"):
        name = st.text_input("Business name", value=existing.name if existing else "")
        kind = existing.kind if existing and existing.kind in KINDS else "other" if existing else "restaurant"
        kind = st.selectbox("Type", KINDS, index=KINDS.index(kind), format_func=str.capitalize)
        address = st.text_input("Address", value=existing.location.address if existing else "",
                                placeholder="Street, neighbourhood")
        phone = st.text_input("Phone", value=existing.phone if existing else "")
        notes = st.text_area("Pickup instructions for drivers", value=existing.pickup_notes if existing else "",
                             height=70, placeholder="Back door, ask for the manager")
        language = st.radio("Messages in", ["en", "fr"], horizontal=True,
                            index=0 if existing is None or existing.language != "fr" else 1,
                            format_func={"en": "English", "fr": "French"}.get)
        saved = st.form_submit_button("Save changes" if existing else "Create my account", type="primary")
    if not saved:
        return None
    if not name.strip():
        st.error("Give the business name.")
        return None
    return Merchant(id=existing.id if existing else sim.next_id("M", sim.scenario.merchants), name=name.strip(),
                    kind=kind, phone=phone.strip(), pickup_notes=notes.strip(), language=language,
                    location=Location(lat=lat, lon=lon, address=address.strip()))


head, who = st.columns([3, 1], vertical_alignment="bottom")
with who:
    merchant_id = state.identity("Signed in as", {i: m.name for i, m in merchants.items()}
                                 | {NEW: "Register a new merchant"}, "merchant_id")
if merchant_id == NEW:
    with head:
        ui.page_header("Merchant app", "Register your shop or restaurant",
                       "Then post leftover food whenever you have some; drivers pick it up for nearby charities.")
    st.write("")
    with ui.card("register"):
        created = merchant_form(None)
    if created is not None:
        sim.add_merchant(created)
        forms.forget_location("merchant-new")
        st.session_state["_keep_merchant_id"] = created.id
        st.session_state.pop("merchant_id", None)
        st.toast(f"Welcome, {created.name}. You can post food now.")
        st.rerun()
    st.stop()
merchant = merchants[merchant_id]
state.announce(sim, merchant_id)
with head:
    ui.page_header("Merchant app", merchant.name,
                   f"{merchant.kind.title()} · {merchant.location.address or sim.scenario.city} · "
                   f"simulated time {state.clock_text(sim)}")
st.write("")

offers = state.posted_offers(sim, merchant_id)
stages = {o.id: state.stage_of(sim, o.id) for o in offers}
picked = [i for i, s in stages.items() if s.label in ("Picked up", "Delivered")]
delivered_kg = sum(d.total_kg for i, s in stages.items() if s.label == "Delivered" and (d := state.donation_of(sim, i)))
ui.tiles([
    ("Donations tonight", str(len(offers)), "posted from this shop"),
    ("Collected", f"{sum(d.total_kg for i in picked if (d := state.donation_of(sim, i))):.1f} kg",
     f"{ui.count(len(picked), 'donation')} picked up"),
    ("Your impact tonight", f"{impact.meals(delivered_kg)} meals",
     f"from {delivered_kg:.1f} kg of your food delivered to charities (≈ 420 g per meal)",
     impact.source("meals")),
    ("Waiting for you", str(sum(1 for o in offers if state.open_question(sim, o.id))), "questions to answer"),
])
st.write("")

donations_tab, profile_tab = st.tabs(["Donations", "Profile"])

with donations_tab:
    main, side = st.columns([1.7, 1], gap="medium")
    with main:
        with ui.card("compose"):
            ui.section("Post leftover food", "The form says what the safety check and the drivers need; add "
                                             "anything else in the details, the AI reads them")
            offer = food_form.post_food_form(merchant_id, sim.now, state.next_donation_id(sim),
                                             disabled=sim.finished)
            if offer is not None:
                sim.add_offer(offer)
                food_form.clear(merchant_id)
                state.advance(4)  # intake (the AI reads the details), safety check, then any question
                st.rerun()

        st.write("")
        ui.section("Your donations", "Newest first")
        if not offers:
            ui.html('<div class="fr-empty">No donation posted yet tonight.</div>')
        for offer in reversed(offers):
            stage = stages[offer.id]
            job = sim.logistics.jobs.get(offer.id)
            with ui.card(f"donation-{offer.id}"):
                ui.html(f'<div class="fr-row"><div><b>{ui.e(offer.id)}</b> <span class="fr-muted">posted at '
                        f'{format_hhmm(offer.time)}</span></div>{ui.pill(stage.label, stage.tone)}</div>'
                        + ui.stepper(state.STEPS, stage.step, stage.state)
                        + f'<div class="fr-quote">"{ui.e(offer.text)}"</div>'
                        + "".join(f'<div class="fr-quote">Your answer at {format_hhmm(a.time)}: '
                                  f'"{ui.e(a.answer)}"</div>'
                                  for a in state.answers(sim, offer.id)))

                details = []
                for m in sim.bus.conversation(offer.id):
                    if m.sender == "intake" and m.protocol == "safety-check":
                        how = {"llm": "read by the LLM", "form": "from the form", "form + AI read the details":
                               "from the form; the AI read your details"}.get(m.content.get("read_by"), "")
                        details.append(("Understood", f'{ui.e(m.summary.split("] ", 1)[-1])} '
                                                      f'<span class="fr-muted">({how})</span>'))
                    elif m.sender == "safety":
                        word = {Performative.INFORM: "Accepted", Performative.REFUSE: "Refused",
                                Performative.REQUEST: "More information needed"}.get(m.performative, "")
                        details.append(("Safety", f"<b>{word}</b>: {ui.e(m.summary)}"))
                donation = state.donation_of(sim, offer.id)
                if donation is not None and donation.details_changes:
                    details.append(("AI read", ui.e("; ".join(donation.details_changes))))
                for time, declined, reason in state.declines(sim, offer.id):
                    when = f'<span class="fr-muted">at {format_hhmm(time)}: {ui.e(reason)}</span>'
                    details.append(("Declined", f"{ui.e(state.name_of(sim, declined))} {when}"))
                charity = sim.matching.awards.get(offer.id)
                if charity:
                    details.append(("Goes to", ui.e(state.name_of(sim, charity))))
                if job and job.driver_id and job.pickup_eta is not None:
                    driver = state.name_of(sim, job.driver_id)
                    details.append(("Pickup", f"{ui.e(driver)} at {format_hhmm(job.pickup_eta)}"))
                if details:
                    ui.html('<div class="fr-kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>'
                                                            for k, v in details) + "</div>")
                for m in sim.bus.conversation(offer.id):
                    cited = m.content.get("explanation") if m.sender == "safety" else None
                    if cited and cited.get("grounded"):
                        ui.html(f'<div class="fr-quote cite">{ui.e(cited["text"])}<div class="fr-muted" '
                                f'style="margin-top:6px">{ui.e(cited["source"])}: "{ui.e(cited["quote"])}"</div></div>')

                question = state.open_question(sim, offer.id)
                if question is not None:
                    ui.html(f'<div class="fr-quote ask"><b>Question from the '
                            f'platform</b><br>{ui.e(question.content["text"])}</div>')
                    reply = st.text_input("Your answer", key=f"answer-{offer.id}",
                                          placeholder="Yes, it is still sealed and kept in the fridge")
                    if st.button("Reply", key=f"reply-{offer.id}", disabled=sim.finished, icon=":material/reply:"):
                        if not reply.strip():
                            st.error("Write an answer first.")
                        else:  # the AI reads the details again, with the answer
                            first = state.first_offer(sim, offer.id)
                            answer = f"Answer to your question: {reply.strip()}"
                            details = "\n".join(part for part in (first.details, answer) if part)
                            sim.add_offer(first.model_copy(update={"time": sim.now, "answer": reply.strip(),
                                                                   "details": details}))
                            state.advance(4)
                            st.rerun()
                if job and job.driver_id and job.pickup_eta and job.pickup_eta > sim.now and not sim.finished:
                    if st.button("The food will be 30 min late", key=f"late-{offer.id}", icon=":material/schedule:"):
                        sim.add_disruption(Disruption(time=sim.now, kind=DisruptionKind.MERCHANT_DELAY,
                                                      target_id=offer.id, minutes=30,
                                                      reason="reported on the dashboard"))
                        state.advance(2)  # logistics replans, communication warns the driver and the charity
                        st.rerun()

    with side, ui.card("inbox"):
        ui.section("Notifications", "Messages from the platform")
        ui.list_items([(format_hhmm(m.time), f'<div class="fr-text" style="margin:0">{ui.e(m.content["text"])}</div>'
                        + (f'<div style="margin-top:4px">{ui.pill("Question", "warning")}</div>'
                           if m.content.get("question") else ""))
                       for m in reversed(state.inbox(sim, merchant_id))], empty="Nothing yet.")

with profile_tab, ui.card("profile"):
    ui.section("Your shop", "Drivers and charities see these details. New donations use them at once; food "
                            "already planned keeps its pickup place.")
    edited = merchant_form(merchant)
    if edited is not None:
        sim.update_merchant(edited)
        st.toast("Profile saved.")
        st.rerun()
