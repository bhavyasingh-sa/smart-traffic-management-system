"""app.py - Streamlit dashboard entry point; orchestrates layout and session state, all rendering logic lives in ui/."""

import time

import streamlit as st
import streamlit.components.v1 as components

from simulation.adaptive_simulator import AdaptiveTrafficSimulation

from ui.styles import get_global_css, render_html
from ui.intersection_view import build_intersection_html
from ui.toolbar import render_simulation_toolbar
from ui.panels import (
    render_current_control_panel,
    render_ml_panel,
    render_model_performance_panel,
    render_rag_explanation_panel,
)
from ui.movement_table import render_movement_table


st.set_page_config(
    page_title=(
        "Smart Traffic Management and Decision Support System"
    ),
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_html(get_global_css())


if "simulator" not in st.session_state:
    st.session_state.simulator = AdaptiveTrafficSimulation()

if "running" not in st.session_state:
    st.session_state.running = False

simulator = st.session_state.simulator
state = simulator.get_state()

# The intersection view animates congestion color, queue numbers, wait
# numbers and signal glows from their previous values to their current
# ones. The previous get_state() snapshot is held only here in session
# state, for display purposes - it never feeds back into the simulator.
previous_state = st.session_state.get(
    "previous_render_state"
)

if (
    previous_state is not None
    and state["tick"] < previous_state["tick"]
):
    # Reset / time-of-day jump: don't animate from a stale run.
    previous_state = None

if "simulation_time_of_day" not in st.session_state:
    st.session_state.simulation_time_of_day = state["simulated_hour"]


render_html(f"""
<div class="stc-header">
    <div class="stc-title">
        SMART TRAFFIC MANAGEMENT AND DECISION SUPPORT SYSTEM
    </div>
    <div class="stc-subtitle">
        {state['city'].upper()} / INTERSECTION
        {state['intersection_id']:03d} /
        ML + IR ADAPTIVE SIGNAL CONTROLLER
    </div>
    <div class="stc-badge-row">
        <div class="stc-badge">
            {"SIMULATION COMPLETE" if state['finished'] else "OPERATIONAL"}
        </div>
        <div class="stc-badge">
            MOVEMENT-AWARE CONTROLLER (12 MOVEMENTS)
        </div>
        <div class="stc-badge">
            TICK {state['tick']:04d} /
            {state['simulation_duration']}
        </div>
    </div>
</div>
""")


toolbar = render_simulation_toolbar(state)

# The simulator exposes its initial hour through construction. Replacing
# the session-scoped instance on an explicit UI time change reloads the
# hour-specific arrival rates and ML/IR evidence without altering
# backend logic. 24:00 is the end-of-day slider value and maps to 00:00.
selected_simulator_hour = toolbar["selected_hour"] % 24

if selected_simulator_hour != simulator.hour:

    st.session_state.running = False
    st.session_state.simulator = AdaptiveTrafficSimulation(
        hour=selected_simulator_hour,
    )
    st.rerun()

if toolbar["play"]:
    st.session_state.running = True

if toolbar["pause"]:
    st.session_state.running = False

if toolbar["reset"]:

    st.session_state.running = False
    simulator.reset()
    st.rerun()

state = simulator.get_state()


# Primary view: intersection + current status, matching what an
# actual operator would look at. Model internals (ML scores, priority
# breakdown) sit in one collapsed section below, not mixed in here.
intersection_column, control_column = st.columns(
    [2.3, 1.0],
    gap="medium",
)

with intersection_column:

    components.html(
        build_intersection_html(state, previous_state),
        height=700,
        scrolling=False,
    )

    # This render's state becomes the next render's "previous" - saved
    # after building the view so the animation always spans exactly
    # one visible update.
    st.session_state.previous_render_state = state

with control_column:

    render_current_control_panel(state)


# AI explanation gets its own full-width, centered section rather than
# being squeezed into the sidebar - it's the human-facing payoff of
# the ML/IR evidence, so it gets the visual weight.
render_rag_explanation_panel(state)


with st.expander("Model & Analysis Detail (ML / Movement Table)"):

    render_ml_panel(state)
    render_movement_table(state)

with st.expander("Model Performance (Accuracy / F1 Scores)"):

    render_model_performance_panel(simulator.model_bundle)


if st.session_state.running and not simulator.finished:

    simulator.step()

    time.sleep(toolbar["tick_interval_seconds"])

    st.rerun()

if simulator.finished:
    st.session_state.running = False
