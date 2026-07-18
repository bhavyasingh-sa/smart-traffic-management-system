"""test_controller_core.py - invariant tests for the movement-aware controller_core.py."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from simulation.movement_definitions import (
    MOVEMENT_IDS,
    TRAVEL_DIRECTIONS,
    PHASES,
    PHASE_NAMES,
)

from simulation.controller_core import (
    ENTRY_HEADING_LETTER,
    MOVEMENT_TO_PHASE,
    PHASE_TIE_BREAK_ORDER,
    MIN_GREEN,
    MAX_GREEN,
    derive_exit_heading,
    parse_movement_id,
    load_turning_proportions,
    build_ml_input,
    calculate_movement_priority,
    calculate_phase_priority,
    should_switch_phase,
    _pick_best_phase,
)


PASSED = []
FAILED = []


def check(name, condition):

    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")

    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


# Scope: pure-function tests only (movement definitions, ML/IR
# construction, priority formulas, phase-decision logic). Invariants that
# require a running simulation - total arrivals equalling movement
# arrivals, departures never exceeding arrivals, queue lengths never
# going negative, every generated vehicle having exactly one movement -
# depend on the tick-by-tick simulator and are tested in
# test_adaptive_simulator.py instead.
def test_twelve_movements():

    check(
        "exactly 12 movement IDs exist",
        len(MOVEMENT_IDS) == 12,
    )

    check(
        "movement IDs are unique",
        len(set(MOVEMENT_IDS)) == 12,
    )


def test_every_movement_in_exactly_one_phase():

    all_phase_movements = [
        movement_id
        for phase_name in PHASE_NAMES
        for movement_id in PHASES[phase_name]
    ]

    check(
        "every movement appears in the phase plan exactly once",
        sorted(all_phase_movements) == sorted(MOVEMENT_IDS),
    )

    check(
        "MOVEMENT_TO_PHASE covers all 12 movements",
        sorted(MOVEMENT_TO_PHASE.keys()) == sorted(MOVEMENT_IDS),
    )

    for movement_id in MOVEMENT_IDS:

        phase_name = MOVEMENT_TO_PHASE[movement_id]

        check(
            f"MOVEMENT_TO_PHASE[{movement_id}] "
            f"({phase_name}) actually lists it",
            movement_id in PHASES[phase_name],
        )


def test_heading_mapping_matches_real_audit():
    """
    Cross-checked against the real movement matrix produced by
    analysis/movement_audit.py - these are the actual observed
    EntryHeading -> ExitHeading pairs.
    """

    expected = {
        "NB_STRAIGHT": ("N", "N"),
        "NB_LEFT": ("N", "W"),
        "NB_RIGHT": ("N", "E"),
        "SB_STRAIGHT": ("S", "S"),
        "SB_LEFT": ("S", "E"),
        "SB_RIGHT": ("S", "W"),
        "EB_STRAIGHT": ("E", "E"),
        "EB_LEFT": ("E", "N"),
        "EB_RIGHT": ("E", "S"),
        "WB_STRAIGHT": ("W", "W"),
        "WB_LEFT": ("W", "S"),
        "WB_RIGHT": ("W", "N"),
    }

    for movement_id, (
        expected_entry,
        expected_exit,
    ) in expected.items():

        direction, movement_type = parse_movement_id(
            movement_id
        )

        entry_letter = ENTRY_HEADING_LETTER[direction]

        exit_letter = derive_exit_heading(
            entry_letter,
            movement_type,
        )

        check(
            f"{movement_id}: entry={entry_letter} "
            f"exit={exit_letter} matches real data "
            f"({expected_entry}->{expected_exit})",
            (entry_letter, exit_letter)
            == (expected_entry, expected_exit),
        )


def test_ml_input_uses_same_heading_convention():

    for direction in TRAVEL_DIRECTIONS:

        input_row = build_ml_input(
            direction=direction,
            hour=8,
            weekend=0,
            month=6,
        )

        entry = input_row.iloc[0]["EntryHeading"]
        exit_ = input_row.iloc[0]["ExitHeading"]

        check(
            f"{direction}: ML input uses same heading "
            f"({entry} -> {exit_}), not the opposite-heading bug",
            entry == exit_,
        )


def test_turning_proportions_sum_to_one():

    proportions = load_turning_proportions()

    check(
        "turning proportions cover all 12 movements",
        sorted(proportions.keys()) == sorted(MOVEMENT_IDS),
    )

    for direction in TRAVEL_DIRECTIONS:

        direction_movements = [
            movement_id
            for movement_id in MOVEMENT_IDS
            if movement_id.startswith(direction + "_")
        ]

        direction_total = sum(
            proportions[movement_id]
            for movement_id in direction_movements
        )

        check(
            f"{direction}: turning proportions sum to "
            f"{direction_total:.4f} (~1.0)",
            abs(direction_total - 1.0) < 0.001,
        )


def test_movement_uses_parent_ml_severity():

    ml_predictions = {
        "NB": {"class": "Severe", "severity": 0.90},
        "SB": {"class": "Low", "severity": 0.10},
        "EB": {"class": "Moderate", "severity": 0.50},
        "WB": {"class": "High", "severity": 0.70},
    }

    empty_queues = {
        movement_id: [] for movement_id in MOVEMENT_IDS
    }

    zero_starvation = {
        movement_id: 0 for movement_id in MOVEMENT_IDS
    }

    for movement_id in MOVEMENT_IDS:

        direction, _ = parse_movement_id(movement_id)

        # With empty queue/wait/starvation and ML-only mode, the
        # priority is entirely the (renormalized) ML score, so it
        # should scale monotonically with the parent's severity.
        priority = calculate_movement_priority(
            movement_id=movement_id,
            movement_queues=empty_queues,
            current_tick=0,
            ml_predictions=ml_predictions,
            starvation=zero_starvation,
            ir_predictions=None,
        )

        expected_severity = ml_predictions[direction]["severity"]

        # ML-only renormalized weight for ML component:
        total_base_weight = 0.45 + 0.20 + 0.15 + 0.10
        expected_priority = (
            0.15 / total_base_weight
        ) * expected_severity

        check(
            f"{movement_id} priority ({priority:.4f}) reflects "
            f"parent {direction}'s ML severity "
            f"({expected_priority:.4f})",
            abs(priority - expected_priority) < 1e-9,
        )


def test_min_green_respected():

    ml_predictions = {
        direction: {"class": "Severe", "severity": 0.95}
        for direction in TRAVEL_DIRECTIONS
    }

    # Deliberately overload every competing movement's queue to try
    # to force a switch, and confirm MIN_GREEN still holds anyway.
    heavy_queues = {
        movement_id: list(range(50))
        for movement_id in MOVEMENT_IDS
    }

    heavy_starvation = {
        movement_id: 999 for movement_id in MOVEMENT_IDS
    }

    for green_elapsed in [0, 1, 5, MIN_GREEN - 1]:

        should_switch, reason, target, _details = (
            should_switch_phase(
                current_phase="NS_THROUGH",
                green_elapsed=green_elapsed,
                movement_queues=heavy_queues,
                current_tick=100,
                ml_predictions=ml_predictions,
                starvation=heavy_starvation,
                ir_predictions=None,
            )
        )

        check(
            f"green_elapsed={green_elapsed} "
            f"(< MIN_GREEN={MIN_GREEN}): held, no switch",
            should_switch is False
            and reason is None
            and target is None,
        )


def test_max_green_forces_switch():

    ml_predictions = {
        direction: {"class": "Low", "severity": 0.10}
        for direction in TRAVEL_DIRECTIONS
    }

    empty_queues = {
        movement_id: [] for movement_id in MOVEMENT_IDS
    }

    zero_starvation = {
        movement_id: 0 for movement_id in MOVEMENT_IDS
    }

    should_switch, reason, target, _details = should_switch_phase(
        current_phase="NS_THROUGH",
        green_elapsed=MAX_GREEN,
        movement_queues=empty_queues,
        current_tick=100,
        ml_predictions=ml_predictions,
        starvation=zero_starvation,
        ir_predictions=None,
    )

    check(
        "MAX_GREEN reached: forces a switch even with empty queues",
        should_switch is True
        and reason == "maximum_green_reached"
        and target in PHASE_NAMES
        and target != "NS_THROUGH",
    )


def test_current_phase_empty_triggers_switch():

    ml_predictions = {
        direction: {"class": "Low", "severity": 0.10}
        for direction in TRAVEL_DIRECTIONS
    }

    queues = {
        movement_id: [] for movement_id in MOVEMENT_IDS
    }

    # Give EB_STRAIGHT some vehicles so a competing phase is
    # non-empty, while NS_THROUGH (current) is completely empty.
    queues["EB_STRAIGHT"] = [1, 2, 3]

    zero_starvation = {
        movement_id: 0 for movement_id in MOVEMENT_IDS
    }

    should_switch, reason, target, _details = should_switch_phase(
        current_phase="NS_THROUGH",
        green_elapsed=MIN_GREEN,
        movement_queues=queues,
        current_tick=100,
        ml_predictions=ml_predictions,
        starvation=zero_starvation,
        ir_predictions=None,
    )

    check(
        "current phase empty, competitor has traffic: switches",
        should_switch is True
        and reason == "current_phase_empty"
        and target == "EW_THROUGH",
    )


def test_deterministic_same_inputs():

    ml_predictions = {
        "NB": {"class": "Severe", "severity": 0.80},
        "SB": {"class": "Moderate", "severity": 0.40},
        "EB": {"class": "High", "severity": 0.60},
        "WB": {"class": "Low", "severity": 0.15},
    }

    queues = {
        movement_id: [10, 20, 30]
        for movement_id in MOVEMENT_IDS
    }

    starvation = {
        movement_id: 15 for movement_id in MOVEMENT_IDS
    }

    result_1 = should_switch_phase(
        current_phase="NS_THROUGH",
        green_elapsed=MIN_GREEN,
        movement_queues=queues,
        current_tick=50,
        ml_predictions=ml_predictions,
        starvation=starvation,
        ir_predictions=None,
    )

    result_2 = should_switch_phase(
        current_phase="NS_THROUGH",
        green_elapsed=MIN_GREEN,
        movement_queues=queues,
        current_tick=50,
        ml_predictions=ml_predictions,
        starvation=starvation,
        ir_predictions=None,
    )

    check(
        "should_switch_phase is deterministic for identical inputs",
        result_1 == result_2,
    )


def test_deterministic_tie_break():

    # Construct a contrived exact tie between EW_THROUGH and
    # EW_LEFT to prove the tie-break follows PHASE_TIE_BREAK_ORDER,
    # not incidental dict ordering.
    tied_priorities = {
        "NS_THROUGH": 0.10,
        "NS_LEFT": 0.10,
        "EW_THROUGH": 0.50,
        "EW_LEFT": 0.50,
    }

    winner = _pick_best_phase(
        ["EW_THROUGH", "EW_LEFT"],
        tied_priorities,
    )

    expected_winner = next(
        phase_name
        for phase_name in PHASE_TIE_BREAK_ORDER
        if phase_name in ["EW_THROUGH", "EW_LEFT"]
    )

    check(
        f"tied phases resolved via PHASE_TIE_BREAK_ORDER "
        f"(picked {winner}, expected {expected_winner})",
        winner == expected_winner,
    )

    # Run it 20 times to rule out any hidden nondeterminism
    # (e.g. accidental set iteration order).
    repeats = [
        _pick_best_phase(
            ["EW_THROUGH", "EW_LEFT"],
            tied_priorities,
        )
        for _ in range(20)
    ]

    check(
        "tie-break gives the same answer across repeated calls",
        len(set(repeats)) == 1,
    )


def test_target_phase_always_valid():

    ml_predictions = {
        direction: {"class": "Moderate", "severity": 0.5}
        for direction in TRAVEL_DIRECTIONS
    }

    queues = {
        movement_id: [1, 2]
        for movement_id in MOVEMENT_IDS
    }

    starvation = {
        movement_id: 0 for movement_id in MOVEMENT_IDS
    }

    for current_phase in PHASE_NAMES:

        for green_elapsed in [
            0,
            MIN_GREEN,
            MAX_GREEN,
        ]:

            should_switch, reason, target, _details = (
                should_switch_phase(
                    current_phase=current_phase,
                    green_elapsed=green_elapsed,
                    movement_queues=queues,
                    current_tick=100,
                    ml_predictions=ml_predictions,
                    starvation=starvation,
                    ir_predictions=None,
                )
            )

            if should_switch:

                check(
                    f"{current_phase} @ green={green_elapsed}: "
                    f"switching to valid, different phase "
                    f"({target})",
                    target in PHASE_NAMES
                    and target != current_phase,
                )

            else:

                check(
                    f"{current_phase} @ green={green_elapsed}: "
                    f"holding, target is None",
                    target is None and reason is None,
                )


def main():

    print("\nSTAGE 2 CONTROLLER_CORE INVARIANT TESTS")

    print("\n1. Exactly 12 movement IDs exist")
    test_twelve_movements()

    print("\n2. Every movement belongs to exactly one approved phase")
    test_every_movement_in_exactly_one_phase()

    print("\n3. EntryHeading/ExitHeading mapping matches real audit")
    test_heading_mapping_matches_real_audit()

    print("\n4. ML uses corrected same-heading convention")
    test_ml_input_uses_same_heading_convention()

    print("\n5. Turning proportions sum to 1.0 per direction")
    test_turning_proportions_sum_to_one()

    print("\n6. Each movement uses its correct parent ML severity")
    test_movement_uses_parent_ml_severity()

    print("\n7. MIN_GREEN is respected (no exception)")
    test_min_green_respected()

    print("\n8. MAX_GREEN forces a switch")
    test_max_green_forces_switch()

    print("\n9. current_phase_empty triggers correctly")
    test_current_phase_empty_triggers_switch()

    print("\n10. Deterministic results for identical inputs")
    test_deterministic_same_inputs()

    print("\n11. Deterministic tie-break rule")
    test_deterministic_tie_break()

    print("\n12. target_phase is always valid")
    test_target_phase_always_valid()

    print()
    print(
        f"RESULTS: {len(PASSED)} passed, "
        f"{len(FAILED)} failed"
    )

    if FAILED:

        print("\nFAILED CHECKS:")

        for name in FAILED:
            print(f"  - {name}")

        sys.exit(1)

    print("\nAll invariants passed.")


if __name__ == "__main__":
    main()
