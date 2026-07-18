"""movement_definitions.py - the 12 real movements, physical-side mapping, and 4-phase signal plan."""

# Verified against data/raw/train.csv for Atlanta Intersection 84:
# exactly these 12 movements exist, nothing more, nothing fabricated.
# (Run analysis/movement_audit.py to reproduce this list from scratch.)

MOVEMENT_IDS = [
    "NB_STRAIGHT",
    "NB_LEFT",
    "NB_RIGHT",

    "SB_STRAIGHT",
    "SB_LEFT",
    "SB_RIGHT",

    "EB_STRAIGHT",
    "EB_LEFT",
    "EB_RIGHT",

    "WB_STRAIGHT",
    "WB_LEFT",
    "WB_RIGHT",
]

TRAVEL_DIRECTIONS = [
    "NB",
    "SB",
    "EB",
    "WB",
]

MOVEMENT_TYPES = [
    "STRAIGHT",
    "LEFT",
    "RIGHT",
]

DIRECTION_MOVEMENTS = {
    direction: [
        f"{direction}_{movement_type}"
        for movement_type in MOVEMENT_TYPES
    ]
    for direction in TRAVEL_DIRECTIONS
}

# EntryHeading/ExitHeading describe direction of TRAVEL, not the
# side a vehicle arrives from - verified against the real dataset
# (see analysis/movement_audit.py).
# A "northbound" (NB) vehicle is travelling north, so it physically
# queues on the SOUTH leg of the intersection, waiting to go north.

PHYSICAL_APPROACH_SIDE = {
    "NB": "south",   # northbound traffic queues on the south leg
    "SB": "north",   # southbound traffic queues on the north leg
    "EB": "west",    # eastbound traffic queues on the west leg
    "WB": "east",    # westbound traffic queues on the east leg
}

# Standard 4-phase protected-left signal plan. Two opposing travel
# directions share a phase for their through+right movements, and
# share a separate phase for their protected left turns. This is a
# documented modelling assumption, not something proven safe by code
# here - _run_sanity_checks() below only proves the 12 movements are
# covered exactly once, it does not prove physical conflict safety.
# Right turns are assumed PERMISSIVE (run concurrently with their own
# direction's through movement); this doesn't account for
# pedestrians, cyclists, or jurisdiction-specific turn restrictions.
#
# Reasoning per phase is documented below rather than assumed.

PHASES = {

    "NS_THROUGH": [
        "NB_STRAIGHT",
        "NB_RIGHT",
        "SB_STRAIGHT",
        "SB_RIGHT",
    ],

    "NS_LEFT": [
        "NB_LEFT",
        "SB_LEFT",
    ],

    "EW_THROUGH": [
        "EB_STRAIGHT",
        "EB_RIGHT",
        "WB_STRAIGHT",
        "WB_RIGHT",
    ],

    "EW_LEFT": [
        "EB_LEFT",
        "WB_LEFT",
    ],
}

PHASE_NAMES = list(PHASES.keys())

PHASE_COMPATIBILITY_NOTES = {

    "NS_THROUGH": (
        "NB_STRAIGHT and SB_STRAIGHT travel straight through in "
        "opposite directions on the same street (Cheshire Bridge Rd) "
        "and do not cross paths. NB_RIGHT and SB_RIGHT turn away from "
        "opposing traffic without crossing any opposing lane, so they "
        "run permissively alongside their own direction's through "
        "movement - standard practice for right turns."
    ),

    "NS_LEFT": (
        "NB_LEFT and SB_LEFT are opposing left turns. Opposing left "
        "turns do not cross each other's paths (each curves toward "
        "its own side of the box), so they can run together safely. "
        "They cannot run with NS_THROUGH because a left turn crosses "
        "the path of the opposing through movement (e.g. SB_LEFT "
        "crosses NB_STRAIGHT's path) - this is exactly why left turns "
        "get their own protected phase."
    ),

    "EW_THROUGH": (
        "Same reasoning as NS_THROUGH, rotated 90 degrees: EB_STRAIGHT/"
        "WB_STRAIGHT pass straight through without crossing, and "
        "EB_RIGHT/WB_RIGHT turn away from opposing traffic."
    ),

    "EW_LEFT": (
        "Same reasoning as NS_LEFT: EB_LEFT and WB_LEFT are opposing "
        "left turns that don't cross each other, but do conflict with "
        "EW_THROUGH."
    ),
}

# Every NS_* phase conflicts with every EW_* phase (perpendicular
# traffic crossing the intersection box - the same fundamental
# conflict a simple NS/EW signal split models).
#
# Within the same axis, the THROUGH phase and the LEFT phase conflict
# with each other (a left turn crosses the opposing through path),
# so they must run sequentially, never simultaneously.

CONFLICTING_PHASE_PAIRS = [
    ("NS_THROUGH", "EW_THROUGH"),
    ("NS_THROUGH", "EW_LEFT"),
    ("NS_LEFT", "EW_THROUGH"),
    ("NS_LEFT", "EW_LEFT"),
    ("NS_THROUGH", "NS_LEFT"),
    ("EW_THROUGH", "EW_LEFT"),
]


def _run_sanity_checks():

    all_phase_movements = []

    for phase_name in PHASE_NAMES:

        all_phase_movements.extend(
            PHASES[phase_name]
        )

    # Every one of the 12 real movements must appear in exactly
    # one phase - no movement forgotten, none duplicated.

    if sorted(all_phase_movements) != sorted(MOVEMENT_IDS):

        raise ValueError(
            "Phase plan does not exactly cover the 12 real "
            "movements once each. Check PHASES."
        )

    # No phase pair should be listed as both compatible (sharing
    # a phase) and conflicting.

    for phase_name, movements in PHASES.items():

        if len(movements) != len(set(movements)):

            raise ValueError(
                f"Phase {phase_name} lists a movement more "
                "than once."
            )


_run_sanity_checks()
