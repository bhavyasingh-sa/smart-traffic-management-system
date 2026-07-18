"""ui/toolbar.py - dashboard controls (Play/Pause/Reset, time, speed); returns choices, app.py applies them."""

import streamlit as st


# Base tick interval at 1x speed. ~1.4s per simulated tick: slow
# enough that a signal change and the queue response to it are
# individually observable. The speed slider scales this - moving
# it right divides the interval down (faster), left multiplies it
# up (slower), so the slider always reads as "speed", not "delay".
BASE_TICK_INTERVAL_SECONDS = 1.4

MIN_SPEED_MULTIPLIER = 0.25
MAX_SPEED_MULTIPLIER = 4.0
DEFAULT_SPEED_MULTIPLIER = 1.0


def format_simulation_time(hour):
    """Format a slider hour without exposing the 24:00 display alias."""

    return f"{hour:02d}:00"


def render_simulation_toolbar(state):
    """
    Render the compact simulation toolbar and return user-selected controls.

    The time-of-day slider intentionally includes ``24`` as a familiar end
    label. The caller normalizes it to simulator hour ``0`` before creating
    a simulator, because the traffic data is indexed from 00:00 through
    23:00.
    """

    (
        play_column,
        pause_column,
        reset_column,
        time_column,
        display_column,
    ) = st.columns(
        [0.9, 0.9, 0.9, 3.8, 1.3], gap="small"
    )

    with play_column:
        play_clicked = st.button("Play", use_container_width=True)

    with pause_column:
        pause_clicked = st.button("Pause", use_container_width=True)

    with reset_column:
        reset_clicked = st.button("Reset", use_container_width=True)

    with time_column:
        selected_hour = st.slider(
            "Time of day",
            min_value=0,
            max_value=24,
            step=1,
            key="simulation_time_of_day",
            format="%02d:00",
        )

    with display_column:
        st.markdown(
            "<div class=\"stc-time-readout\">"
            "<span>SIMULATION TIME</span>"
            f"<strong>{format_simulation_time(state['simulated_hour'])}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

    (
        _speed_spacer_column,
        speed_column,
        speed_display_column,
    ) = st.columns(
        [2.7, 3.8, 1.3], gap="small"
    )

    with speed_column:
        speed_multiplier = st.slider(
            "Simulation speed",
            min_value=MIN_SPEED_MULTIPLIER,
            max_value=MAX_SPEED_MULTIPLIER,
            value=DEFAULT_SPEED_MULTIPLIER,
            step=0.25,
            key="simulation_speed_multiplier",
            format="%.2gx",
        )

    tick_interval_seconds = (
        BASE_TICK_INTERVAL_SECONDS / speed_multiplier
    )

    with speed_display_column:
        st.markdown(
            "<div class=\"stc-time-readout\">"
            "<span>TICK INTERVAL</span>"
            f"<strong>{tick_interval_seconds:.2f}s</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

    return {
        "play": play_clicked,
        "pause": pause_clicked,
        "reset": reset_clicked,
        "tick_interval_seconds": tick_interval_seconds,
        "selected_hour": selected_hour,
    }
