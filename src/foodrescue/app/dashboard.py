"""Food Rescue platform: one interface per role, plus the presenter's controls.

    uv run streamlit run src/foodrescue/app/dashboard.py

The top bar switches between the five interfaces (overview, merchant, charity, driver, assessment),
each in its own file under `roles/`. The sidebar is the presenter's panel: the simulated clock, live mode and
a new evening. Every tab and device shows the same evening (`state.platform`).
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from foodrescue.app import forms, state, ui
from foodrescue.core.clock import format_hhmm

ROLES = Path(__file__).parent / "roles"
SPEEDS = [0.2, 0.5, 1, 2, 5]  # live mode: simulated minutes per real second

st.set_page_config(page_title="Food Rescue", page_icon=":material/eco:", layout="wide")
ui.apply_theme()


def ticker() -> None:
    """Runs every second in every tab (always the same fragment: a timed fragment that disappears between page runs
    makes Streamlit rerun the page with that fragment's widget values only, which resets the other switches).

    Live mode: move the shared clock on (at most once per second whichever tabs are live, so two tabs never double
    the speed) and redraw this tab. Otherwise: redraw only when something happened elsewhere (a new message, a new
    evening), not on every clock tick, so a person clicking here is not interrupted. The fragment also runs during
    every full page run; it redraws only when the evening changed since this tab drew it, so a redraw never
    triggers another one at once."""
    shared = state.platform()
    if st.session_state.get("live") and not shared.sim.finished:
        with shared.lock:
            sim = shared.sim
            speed = st.session_state.get("live_speed", 1)
            # Fast: several minutes each second. Slow: one minute every few seconds (the fragment runs every second).
            every = 0.9 if speed >= 1 else 1 / speed - 0.1
            if time.monotonic() - shared.last_tick >= every:
                shared.last_tick = time.monotonic()
                state.advance(max(1, round(speed)), spinner=False)
                if sim.finished:
                    st.session_state["live"] = False
        if state.stamp() != st.session_state.get("_drawn"):
            st.rerun(scope="app")
    elif state.news() != st.session_state.get("_drawn_news"):
        st.rerun(scope="app")


def set_driver_policy(key: str) -> None:
    with state.platform().lock:
        state.current().set_driver_policy("auto" if st.session_state[key] else "manual")


def set_charity_policy(key: str) -> None:
    with state.platform().lock:
        state.current().set_charity_policy("auto" if st.session_state[key] else "manual")


def presenter_panel(ai: tuple[bool, str]) -> None:
    with st.sidebar:
        ui.html('<div class="fr-section" style="margin-bottom:10px">Presenter</div>')
        sim = state.current()

        span = max(1, sim.scenario.end - sim.scenario.start)
        done = 1.0 if sim.finished else (sim.now - sim.scenario.start) / span
        ui.html(f'<div class="fr-clock"><div class="fr-eyebrow">Platform time</div>'
                f'<div class="value">{ui.e(state.clock_text(sim))}</div>'
                f'<div class="fr-progress"><div style="width:{done * 100:.0f}%"></div></div>'
                f'<div class="fr-muted">Evening {format_hhmm(sim.scenario.start)} to '
                f'{format_hhmm(sim.scenario.end)}</div></div>')
        cols = [*st.columns(2, gap="small"), *st.columns(2, gap="small")]
        steps = [("+5 min", 5), ("+15 min", 15), ("+1 hour", 60), ("End", 24 * 60)]
        for col, (label, minutes) in zip(cols, steps, strict=True):
            if col.button(label, key=f"clock-{minutes}", use_container_width=True, disabled=sim.finished,
                          type="primary" if minutes == 5 else "secondary"):
                state.advance(minutes)
                st.rerun()

        # Live mode: the clock runs by itself; people's screens update as things happen.
        live, speed = st.columns([1, 1.4], vertical_alignment="center")
        live.toggle("Live", key="live", disabled=sim.finished,
                    help="The clock runs by itself, like a real evening sped up. Pause it to look at something.")
        # Slow speeds leave people time to act: at 1 min / 5 s a delivery request (5 minutes) lasts 25 seconds.
        speed.select_slider("Speed", options=SPEEDS, value=1, key="live_speed", label_visibility="collapsed",
                            format_func=lambda s: f"1 min / {1 / s:g} s" if s < 1 else f"{s:g} min/s")
        # Same fragment in both modes (its id does not depend on the interval): every second in live mode,
        # every 2 seconds otherwise.
        st.fragment(ticker, run_every=1 if st.session_state.get("live") and not sim.finished else 2)()
        key = f"auto-drivers-{id(sim)}"
        st.toggle("Drivers answer by themselves", key=key, value=bool(sim.drivers) and sim.drivers[0].policy == "auto",
                  on_change=set_driver_policy, args=(key,),
                  help="Off: every delivery request waits for a person in the Driver screen. On: simulated drivers "
                       "accept within a minute or two (useful to fast-forward the evening).")
        key = f"auto-charities-{id(sim)}"
        st.toggle("Charities accept by themselves", key=key, value=sim.config.simulation.charity_policy == "auto",
                  on_change=set_charity_policy, args=(key,),
                  help="Off: a charity that wins a donation accepts or declines it in the Charity screen. On: each "
                       "charity's agent decides at once.")

        ok, detail = ai
        light = ui.pill("AI online" if ok else "AI offline", "good" if ok else "critical")
        ui.html(f'<div style="margin-top:12px">{light}'
                f'<div class="fr-muted" style="margin-top:4px">{ui.e(detail)}</div></div>')
        st.divider()
        st.time_input("A new evening starts at", key="evening_start", value=forms.to_time(18 * 60), step=900)
        if st.button("Start a new evening", icon=":material/restart_alt:", use_container_width=True,
                     help="For everyone using the platform: same accounts, no food, the clock at the chosen time"):
            state.start(forms.to_minutes(st.session_state["evening_start"]))
            st.rerun()


ai = state.ai_status()  # before the lock: waiting for Ollama must never hold the evening for the other screens
with state.platform().lock:  # nobody changes the evening while this tab draws it
    if state.current() is None:
        state.start(forms.to_minutes(st.session_state.get("evening_start", forms.to_time(18 * 60))))
    st.session_state["_drawn"], st.session_state["_drawn_news"] = state.stamp(), state.news()
    presenter_panel(ai)
    state.announce_requests(state.current())
    pages = [
        st.Page(ROLES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(ROLES / "merchant.py", title="Merchant", icon=":material/flatware:"),
        st.Page(ROLES / "charity.py", title="Charity", icon=":material/volunteer_activism:"),
        st.Page(ROLES / "driver.py", title="Driver", icon=":material/local_shipping:"),
        st.Page(ROLES / "assessment.py", title="Assessment", icon=":material/insights:"),
    ]
    st.navigation(pages, position="top").run()
