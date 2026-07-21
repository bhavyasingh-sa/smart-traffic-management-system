"""ui/intersection_view.py - renders the live intersection as an SVG (Traffic Operations Dashboard view, no vehicles drawn)."""

from simulation.movement_definitions import (
    PHYSICAL_APPROACH_SIDE,
)

from ui.styles import (
    ASPHALT,
    LANE_MARKING,
    SIGNAL_RED,
    SIGNAL_YELLOW,
    SIGNAL_GREEN,
    SIGNAL_OFF,
    CONGESTION_LOW,
    CONGESTION_MODERATE,
    CONGESTION_HIGH,
    CONGESTION_SEVERE,
)


# CRITICAL: movement/approach IDs are named by DIRECTION OF TRAVEL
# (NB/SB/EB/WB), not physical side - a vehicle queuing on the NORTH
# leg is travelling south, so it's the SB approach, not NB. Getting
# this backwards here would reintroduce a heading-convention bug this
# project has already caught and fixed once. Derived (not hardcoded)
# from simulation.movement_definitions.PHYSICAL_APPROACH_SIDE so it
# can never drift out of sync with the backend's own mapping.
PHYSICAL_SIDE_TO_DIRECTION = {
    side: direction
    for direction, side in PHYSICAL_APPROACH_SIDE.items()
}

SIDE_ROTATION_DEGREES = {
    "south": 0,
    "west": 90,
    "north": 180,
    "east": 270,
}

CONGESTION_COLOURS = {
    "LOW": CONGESTION_LOW,
    "MODERATE": CONGESTION_MODERATE,
    "HIGH": CONGESTION_HIGH,
    "SEVERE": CONGESTION_SEVERE,
}

VIEWBOX_SIZE = 1000
CENTER = VIEWBOX_SIZE / 2  # 500

# Roads stop short of the canvas edge on all 4 arms instead of
# running full-bleed, so there's a visible margin of background
# around the whole intersection rather than markings running flush
# to the frame.
ROAD_MARGIN = 50
ROAD_FAR_EDGE = VIEWBOX_SIZE - ROAD_MARGIN

INTERSECTION_HALF_WIDTH = 130
STOP_LINE = CENTER + INTERSECTION_HALF_WIDTH  # intersection edge
LANE_WIDTH = 22
LANES_PER_CARRIAGEWAY = 3
ROAD_WIDTH = LANE_WIDTH * LANES_PER_CARRIAGEWAY * 2
ROAD_EDGE = CENTER - ROAD_WIDTH / 2

# Vertical layout bands (base template - traffic flows upward toward
# the intersection edge at y = STOP_LINE). Ordered from the
# intersection outward, exactly as a driver encounters them in
# reverse: intersection -> zebra crossing -> stop line -> traffic
# light. Each band's start is derived from the previous band's end,
# so nothing can overlap.
ZEBRA_STRIPE_HEIGHT = 5
ZEBRA_STRIPE_GAP = 6
ZEBRA_STRIPE_COUNT = 5
ZEBRA_START = STOP_LINE + 6
ZEBRA_END = ZEBRA_START + (
    ZEBRA_STRIPE_COUNT * (ZEBRA_STRIPE_HEIGHT + ZEBRA_STRIPE_GAP)
)

STOP_BAR_Y = ZEBRA_END + 8

SIGNAL_HEAD_Y = STOP_BAR_Y + 14
SIGNAL_HEAD_WIDTH = 12
SIGNAL_HEAD_HEIGHT = 40

LANE_ARROW_Y = SIGNAL_HEAD_Y + SIGNAL_HEAD_HEIGHT + 42

INFO_BLOCK_Y = LANE_ARROW_Y + 60

# How long (real seconds) the color fade / number count / signal glow
# transitions take. Independent of the simulation tick interval.
COLOR_TRANSITION_SECONDS = 1.0
NUMBER_COUNT_SECONDS = 0.6
SIGNAL_FADE_SECONDS = 0.4


def _render_signal_heads(
    lane_signals_for_side, lane_specs, previous_signals_for_side
):
    """
    One compact three-aspect head per inbound lane, slightly smaller
    than before. The active lamp carries a soft glow; inactive lamps
    are flat dark gray with no glow. A lamp changing state fades in/
    out via the previous/current + JS technique described in the
    module docstring.
    """

    lamp_colour = {
        "RED": SIGNAL_RED,
        "YELLOW": SIGNAL_YELLOW,
        "GREEN": SIGNAL_GREEN,
    }

    heads = []
    head_y = SIGNAL_HEAD_Y
    head_width = SIGNAL_HEAD_WIDTH
    head_height = SIGNAL_HEAD_HEIGHT

    for lane_suffix, _, lane_x in lane_specs:

        active_state = lane_signals_for_side[lane_suffix]
        previous_state = previous_signals_for_side.get(
            lane_suffix, active_state
        )

        head_x = lane_x - head_width / 2
        lamps = []

        for index, lamp_name in enumerate(["RED", "YELLOW", "GREEN"]):

            is_lit = active_state == lamp_name
            was_lit = previous_state == lamp_name
            colour = lamp_colour[lamp_name]

            from_opacity = "1" if was_lit else "0"
            to_opacity = "1" if is_lit else "0"

            cy = head_y + 9 + index * 11.5

            # Base (unlit) lamp - always drawn, flat, no glow.
            lamps.append(
                f'<circle cx="{lane_x}" cy="{cy}" r="2.9" '
                f'fill="{SIGNAL_OFF}"></circle>'
            )

            # Lit overlay - fades between previous/current states.
            lamps.append(
                f'<circle cx="{lane_x}" cy="{cy}" r="2.9" '
                f'fill="{colour}" '
                f'data-anim-opacity-from="{from_opacity}" '
                f'data-anim-opacity-to="{to_opacity}" '
                f'style="opacity:{from_opacity}; '
                f'filter: drop-shadow(0 0 3px {colour});">'
                f"</circle>"
            )

        heads.append(
            f'<g filter="url(#signal-shadow)">'
            f'<rect x="{head_x}" y="{head_y}" '
            f'width="{head_width}" height="{head_height}" rx="6" '
            f'fill="#080a0c" stroke="#4c5660" stroke-width="0.8"></rect>'
            f"{''.join(lamps)}"
            f"</g>"
        )

    return "".join(heads)


def _lane_arrow(movement_type):

    arrows = {
        "LEFT": "\u2190",
        "STRAIGHT": "\u2191",
        "RIGHT": "\u2192",
    }

    return arrows[movement_type]


def _render_road_surface():
    """The two continuous asphalt corridors behind all approaches, inset by ROAD_MARGIN on every arm."""

    return f"""
    <rect x="{ROAD_EDGE}" y="{ROAD_MARGIN}" width="{ROAD_WIDTH}"
        height="{VIEWBOX_SIZE - 2 * ROAD_MARGIN}" fill="{ASPHALT}"></rect>
    <rect x="{ROAD_MARGIN}" y="{ROAD_EDGE}" width="{VIEWBOX_SIZE - 2 * ROAD_MARGIN}"
        height="{ROAD_WIDTH}" fill="{ASPHALT}"></rect>
    """


def _render_junction_markings():
    """South-facing zebra crossing and lane leads (rotated per side).

    The zebra band sits between the intersection edge (STOP_LINE)
    and the stop bar (STOP_BAR_Y), so approaching traffic meets:
    traffic light -> stop line -> zebra -> intersection. Lane leads
    connect the intersection interior to its edge.
    """

    crossing = "".join(
        f'<rect x="{ROAD_EDGE}" '
        f'y="{ZEBRA_START + index * (ZEBRA_STRIPE_HEIGHT + ZEBRA_STRIPE_GAP)}" '
        f'width="{ROAD_WIDTH}" height="{ZEBRA_STRIPE_HEIGHT}" '
        f'fill="{LANE_MARKING}" opacity="0.72"></rect>'
        for index in range(ZEBRA_STRIPE_COUNT)
    )

    lane_leads = "".join(
        f'<line x1="{CENTER + LANE_WIDTH * boundary}" y1="{CENTER + 24}" '
        f'x2="{CENTER + LANE_WIDTH * boundary}" y2="{STOP_LINE - 4}" '
        f'stroke="{LANE_MARKING}" stroke-width="2" '
        f'stroke-dasharray="10,8" opacity="0.42"></line>'
        for boundary in range(1, LANES_PER_CARRIAGEWAY)
    )

    return lane_leads + crossing


def _render_approach_group(
    direction,
    physical_side,
    signals,
    previous_signals,
    movement_congestion,
    previous_movement_congestion,
    movement_queues,
    previous_movement_queues,
    approach_average_wait,
    previous_approach_average_wait,
):
    """
    Builds the <g> for one approach in the BASE (south, 0-degree)
    orientation; the caller wraps it in a rotate() transform for the
    other three sides.
    """

    lane_specs = [
        ("LEFT", "LEFT", CENTER + LANE_WIDTH * 0.5),
        ("STRAIGHT", "STRAIGHT", CENTER + LANE_WIDTH * 1.5),
        ("RIGHT", "RIGHT", CENTER + LANE_WIDTH * 2.5),
    ]

    outgoing_lane_xs = [
        CENTER - LANE_WIDTH * (index + 0.5)
        for index in range(LANES_PER_CARRIAGEWAY)
    ]

    parts = []

    parts.append(_render_junction_markings())

    # One lane, one movement, one colour: three physical lanes per
    # incoming carriageway (Left / Straight / Right), coloured from
    # that movement's own live congestion. Painted before markings so
    # dashes/arrows stay crisp on top of it. The outgoing carriageway
    # stays neutral asphalt (no fill at all).
    lane_fill_width = ROAD_WIDTH / 2 / LANES_PER_CARRIAGEWAY

    for lane_index, (_, movement_type, _) in enumerate(lane_specs):

        movement_id = f"{direction}_{movement_type}"

        current_colour = CONGESTION_COLOURS[
            movement_congestion[movement_id]["level"]
        ]

        previous_colour = CONGESTION_COLOURS[
            previous_movement_congestion[movement_id]["level"]
        ]

        lane_x = CENTER + lane_fill_width * lane_index

        parts.append(
            f'<rect x="{lane_x}" y="{STOP_LINE}" '
            f'width="{lane_fill_width}" '
            f'height="{ROAD_FAR_EDGE - STOP_LINE}" '
            f'data-anim-color-from="{previous_colour}" '
            f'data-anim-color-to="{current_colour}" '
            f'fill="{previous_colour}" opacity="0.40"></rect>'
        )

    # Dashed dividers for both carriageways - one line per internal
    # lane boundary (LANES_PER_CARRIAGEWAY - 1 of them per side).
    for boundary in range(1, LANES_PER_CARRIAGEWAY):
        for divider_x in (
            CENTER + LANE_WIDTH * boundary,
            CENTER - LANE_WIDTH * boundary,
        ):

            parts.append(
                f'<line x1="{divider_x}" y1="{STOP_LINE}" '
                f'x2="{divider_x}" y2="{ROAD_FAR_EDGE}" '
                f'stroke="{LANE_MARKING}" stroke-width="2" '
                f'stroke-dasharray="14,10" opacity="0.55"></line>'
            )

    # Double-yellow centreline (median) between the carriageways.
    for centreline_x in (CENTER - 2, CENTER + 2):
        parts.append(
            f'<line x1="{centreline_x}" y1="{CENTER + 20}" '
            f'x2="{centreline_x}" y2="{ROAD_FAR_EDGE}" '
            f'stroke="#b89a3c" stroke-width="1.4" opacity="0.75"></line>'
        )

    # Stop bar - incoming carriageway only, beyond the zebra band.
    parts.append(
        f'<rect x="{CENTER}" '
        f'y="{STOP_BAR_Y}" width="{ROAD_WIDTH / 2}" '
        f'height="4" fill="{LANE_MARKING}" opacity="0.85">'
        f"</rect>"
    )

    # Outgoing-direction hint arrows (neutral carriageway).
    for lane_x in outgoing_lane_xs:
        parts.append(
            f'<text x="{lane_x}" y="{CENTER + 180}" '
            f'font-size="16" fill="#75818b" text-anchor="middle" '
            f'opacity="0.58" font-family="monospace">\u2193</text>'
        )

    # Per-lane turn arrows, placed past the signal heads so nothing
    # overlaps.
    for lane_suffix, movement_type, lane_x in lane_specs:

        parts.append(
            f'<text x="{lane_x}" y="{LANE_ARROW_Y}" '
            f'font-size="26" fill="{LANE_MARKING}" '
            f'text-anchor="middle" opacity="0.75" '
            f'font-family="monospace">'
            f"{_lane_arrow(movement_type)}</text>"
        )

    lane_signals_for_side = {
        lane_suffix: signals[f"{direction}_{movement_type}"]
        for lane_suffix, movement_type, _ in lane_specs
    }

    previous_lane_signals_for_side = {
        lane_suffix: previous_signals[
            f"{direction}_{movement_type}"
        ]
        for lane_suffix, movement_type, _ in lane_specs
    }

    parts.append(
        _render_signal_heads(
            lane_signals_for_side,
            lane_specs,
            previous_lane_signals_for_side,
        )
    )

    # One queue number per real movement (L / S / R), directly under
    # that movement's own lane - a true 1:1 mapping now that the road
    # only has three lanes.
    info_x = CENTER + ROAD_WIDTH / 4
    info_y = INFO_BLOCK_Y

    parts.append(
        f'<text x="{info_x}" y="{info_y}" font-size="12" '
        f'fill="#9aa7b1" text-anchor="middle" '
        f'letter-spacing="1.4" font-family="monospace">'
        f"QUEUE BY MOVEMENT</text>"
    )

    lane_letter = {"LEFT": "L", "STRAIGHT": "S", "RIGHT": "R"}

    movement_columns = [
        (lane_letter[movement_type], movement_type, lane_x)
        for _, movement_type, lane_x in lane_specs
    ]

    for label, movement_type, column_x in movement_columns:

        movement_id = f"{direction}_{movement_type}"

        current_queue = movement_queues[movement_id]
        previous_queue = previous_movement_queues[movement_id]

        parts.append(
            f'<text x="{column_x}" y="{info_y + 22}" '
            f'font-size="11" fill="#7c8894" '
            f'text-anchor="middle" font-family="monospace">'
            f"{label}</text>"
        )

        parts.append(
            f'<text x="{column_x}" y="{info_y + 48}" '
            f'font-size="24" font-weight="700" fill="#f2f5f7" '
            f'text-anchor="middle" font-family="monospace" '
            f'data-anim-number-from="{previous_queue}" '
            f'data-anim-number-to="{current_queue}">'
            f"{previous_queue}</text>"
        )

    parts.append(
        f'<text x="{info_x}" y="{info_y + 76}" font-size="11" '
        f'fill="#9aa7b1" text-anchor="middle" '
        f'letter-spacing="1.2" font-family="monospace">'
        f"AVG WAIT</text>"
    )

    parts.append(
        f'<text x="{info_x}" y="{info_y + 100}" font-size="19" '
        f'font-weight="600" fill="#cfd8de" text-anchor="middle" '
        f'font-family="monospace" '
        f'data-anim-number-from="{previous_approach_average_wait:.0f}" '
        f'data-anim-number-to="{approach_average_wait:.0f}" '
        f'data-suffix="s">'
        f"{previous_approach_average_wait:.0f}s</text>"
    )

    parts.append(
        f'<text x="{CENTER}" y="{ROAD_FAR_EDGE - 14}" '
        f'font-size="11" fill="#8a97a3" text-anchor="middle" '
        f'letter-spacing="1.5" font-family="monospace">'
        f"{direction}</text>"
    )

    rotation = SIDE_ROTATION_DEGREES[physical_side]

    return (
        f'<g transform="rotate({rotation} {CENTER} {CENTER})">'
        + "".join(parts)
        + "</g>"
    )


def _render_transition_script():
    """
    Animates every element carrying data-anim-* attributes from its
    "from" value to its "to" value. Needed because Streamlit
    re-embeds this whole document on every rerun, so a plain CSS
    transition has nothing to animate from - the old DOM is gone.
    This paints the previous value first, then flips to the current
    value one frame later so the browser's native tween runs within
    this document's own lifetime. Purely presentational - it never
    touches simulator state.
    """

    return f"""
    <script>
    (function() {{
        function raf2(fn) {{
            requestAnimationFrame(function() {{
                requestAnimationFrame(fn);
            }});
        }}

        document.querySelectorAll('[data-anim-color-to]').forEach(
            function(el) {{
                el.style.transition =
                    'fill {COLOR_TRANSITION_SECONDS}s ease';
                raf2(function() {{
                    el.setAttribute(
                        'fill',
                        el.getAttribute('data-anim-color-to')
                    );
                }});
            }}
        );

        document.querySelectorAll('[data-anim-opacity-to]').forEach(
            function(el) {{
                el.style.transition =
                    'opacity {SIGNAL_FADE_SECONDS}s ease';
                raf2(function() {{
                    el.style.opacity =
                        el.getAttribute('data-anim-opacity-to');
                }});
            }}
        );

        document.querySelectorAll('[data-anim-number-to]').forEach(
            function(el) {{
                var from = parseFloat(
                    el.getAttribute('data-anim-number-from')
                );
                var to = parseFloat(
                    el.getAttribute('data-anim-number-to')
                );
                var suffix = el.getAttribute('data-suffix') || '';
                var duration = {NUMBER_COUNT_SECONDS} * 1000;

                if (from === to) {{
                    el.textContent = to + suffix;
                    return;
                }}

                var start = null;

                function step(timestamp) {{
                    if (!start) {{ start = timestamp; }}
                    var progress = Math.min(
                        (timestamp - start) / duration, 1
                    );
                    var value = Math.round(
                        from + (to - from) * progress
                    );
                    el.textContent = value + suffix;
                    if (progress < 1) {{
                        requestAnimationFrame(step);
                    }}
                }}

                requestAnimationFrame(step);
            }}
        );
    }})();
    </script>
    """


def build_intersection_svg(state, previous_state=None):

    if previous_state is None:
        previous_state = state

    approach_groups = []

    for physical_side, direction in (
        PHYSICAL_SIDE_TO_DIRECTION.items()
    ):

        approach_groups.append(
            _render_approach_group(
                direction=direction,
                physical_side=physical_side,
                signals=state["movement_signals"],
                previous_signals=previous_state[
                    "movement_signals"
                ],
                movement_congestion=state[
                    "movement_live_congestion"
                ],
                previous_movement_congestion=previous_state[
                    "movement_live_congestion"
                ],
                movement_queues=state["movement_queues"],
                previous_movement_queues=previous_state[
                    "movement_queues"
                ],
                approach_average_wait=state[
                    "approach_average_waits"
                ][direction],
                previous_approach_average_wait=previous_state[
                    "approach_average_waits"
                ][direction],
            )
        )

    svg = f"""
    <svg viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}"
        xmlns="http://www.w3.org/2000/svg"
        style="width:100%; height:auto; background:#0a0e12;">

        <defs>
            <filter id="signal-shadow" x="-40%" y="-30%" width="180%" height="180%">
                <feDropShadow dx="0" dy="1.5" stdDeviation="1.3" flood-color="#000000" flood-opacity="0.72"></feDropShadow>
            </filter>
        </defs>
        {_render_road_surface()}
        {"".join(approach_groups)}

    </svg>
    """

    return svg


def build_intersection_html(state, previous_state=None):

    svg = build_intersection_svg(state, previous_state)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    html, body {{
        margin: 0;
        padding: 0;
        background: #0a0e12;
    }}
</style>
</head>
<body>
{svg}
{_render_transition_script()}
</body>
</html>
"""
