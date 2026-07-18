"""test_adaptive_simulator.py - Stage 3 invariant tests (A-T) for simulation/adaptive_simulator.py."""

import sys
from pathlib import Path
from collections import deque

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
    MIN_GREEN,
    MAX_GREEN,
    derive_exit_heading,
    parse_movement_id,
)

from simulation.adaptive_simulator import (
    AdaptiveTrafficSimulation,
    Vehicle,
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


REQUIRED_STATE_KEYS = [
    "tick", "simulation_duration", "finished", "city",
    "intersection_id", "hour", "weekend", "month",
    "current_phase", "target_phase", "signal_state",
    "phase_elapsed", "green_elapsed", "yellow_elapsed",
    "movement_queues", "approach_queue_totals",
    "movement_average_waits", "approach_average_waits",
    "movement_signals", "approach_signals",
    "movement_live_congestion", "approach_live_congestion",
    "ml_predictions", "ir_predictions", "turning_proportions",
    "movement_starvation", "movement_priorities", "phase_priorities",
    "last_decision", "adaptive_switches", "switch_reasons",
    "total_arrivals", "total_departures",
    "movement_total_arrivals", "movement_total_departures",
    "movement_max_queues", "total_waiting",
]


def _contains_deque_or_vehicle(value, path=""):
    """Recursively check for deque/Vehicle objects anywhere in a state dict."""

    if isinstance(value, deque) or isinstance(value, Vehicle):
        return path

    if isinstance(value, dict):

        for k, v in value.items():

            found = _contains_deque_or_vehicle(
                v, f"{path}.{k}"
            )

            if found:
                return found

    if isinstance(value, (list, tuple, set)):

        for i, v in enumerate(value):

            found = _contains_deque_or_vehicle(
                v, f"{path}[{i}]"
            )

            if found:
                return found

    return None


def test_a_twelve_queues():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    check(
        "A: exactly 12 movement queues exist",
        sorted(sim.movement_queues.keys())
        == sorted(MOVEMENT_IDS),
    )


def test_b_c_vehicle_consistency():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    cross_queue_duplicate_found = False
    inconsistent_found = False
    id_counter_decreased = False

    previous_next_id = sim._next_vehicle_id

    for _ in range(100):

        sim.step()

        if sim._next_vehicle_id < previous_next_id:
            id_counter_decreased = True

        previous_next_id = sim._next_vehicle_id

        # Per-tick snapshot: no vehicle_id may appear in more than
        # one queue AT THE SAME TIME.
        ids_this_tick = []

        for movement_id in MOVEMENT_IDS:

            for vehicle in sim.movement_queues[movement_id]:

                ids_this_tick.append(vehicle.vehicle_id)

                expected_entry = ENTRY_HEADING_LETTER[
                    vehicle.direction
                ]

                expected_exit = derive_exit_heading(
                    expected_entry,
                    vehicle.movement_type,
                )

                expected_movement_id = (
                    f"{vehicle.direction}_"
                    f"{vehicle.movement_type}"
                )

                if (
                    vehicle.entry_heading != expected_entry
                    or vehicle.exit_heading != expected_exit
                    or vehicle.movement_id
                    != expected_movement_id
                    or vehicle.movement_id != movement_id
                ):
                    inconsistent_found = True

        if len(ids_this_tick) != len(set(ids_this_tick)):
            cross_queue_duplicate_found = True

    check(
        "B: no vehicle_id appears in more than one queue "
        "at the same time",
        not cross_queue_duplicate_found,
    )

    check(
        "B: vehicle_id counter never decreases "
        "(IDs are never reused)",
        not id_counter_decreased,
    )

    check(
        "C: every vehicle's direction/entry/exit/movement_id "
        "are mutually consistent",
        not inconsistent_found,
    )


def test_d_e_f_accounting():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    while not sim.finished:
        sim.step()

    state = sim.get_state()

    check(
        "D: movement arrivals sum to total_arrivals",
        sum(state["movement_total_arrivals"].values())
        == state["total_arrivals"],
    )

    check(
        "E: movement departures sum to total_departures",
        sum(state["movement_total_departures"].values())
        == state["total_departures"],
    )

    all_ok = all(
        state["movement_total_departures"][m]
        <= state["movement_total_arrivals"][m]
        for m in MOVEMENT_IDS
    )

    check(
        "F: every movement has departures <= arrivals",
        all_ok,
    )


def test_g_h_every_tick_invariants():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    negative_found = False
    mismatch_found = False

    for _ in range(300):

        state = sim.step()

        if any(
            v < 0
            for v in state["movement_queues"].values()
        ):
            negative_found = True

        if (
            state["total_arrivals"]
            - state["total_departures"]
            != state["total_waiting"]
        ):
            mismatch_found = True

    check(
        "G: queue lengths are never negative (every tick)",
        not negative_found,
    )

    check(
        "H: total_arrivals - total_departures == "
        "total_waiting (every tick)",
        not mismatch_found,
    )


def test_i_j_only_active_green_departs():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    violation_found = False

    for _ in range(300):

        phase_before = sim.current_phase
        signal_before = sim.signal_state

        departures_before = dict(
            sim.movement_total_departures
        )

        sim.step()

        departures_after = sim.movement_total_departures

        for movement_id in MOVEMENT_IDS:

            departed = (
                departures_after[movement_id]
                - departures_before[movement_id]
            )

            if departed > 0:

                # A departure this tick must mean the OLD
                # (pre-step) state was GREEN and this movement
                # was in the phase active at that time.
                if (
                    signal_before != "GREEN"
                    or movement_id
                    not in PHASES[phase_before]
                ):
                    violation_found = True

    check(
        "I/J: every departure occurred only while GREEN "
        "and only for the active phase's movements "
        "(zero departures during YELLOW)",
        not violation_found,
    )


def test_k_switch_through_yellow():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    direct_change_found = False
    previous_phase = sim.current_phase

    for _ in range(300):

        state = sim.step()

        if state["current_phase"] != previous_phase:

            # The only way current_phase changes is via the
            # "ACTIVATE" action inside _run_signal_control, which
            # only fires after yellow_elapsed >= YELLOW_DURATION.
            if state["last_decision"]["action"] != "ACTIVATE":
                direct_change_found = True

        previous_phase = state["current_phase"]

    check(
        "K: every current_phase change is via an ACTIVATE "
        "action (i.e. passed through YELLOW)",
        not direct_change_found,
    )


def test_l_m_green_duration_bounds():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    green_elapsed_at_switch = []
    max_green_seen = 0

    for _ in range(300):

        state = sim.step()

        max_green_seen = max(
            max_green_seen,
            state["green_elapsed"],
        )

        if state["last_decision"]["action"] == "SWITCH":

            # green_elapsed was already incremented for HOLD
            # ticks but not for the SWITCH tick itself - the
            # decision used sim.green_elapsed as passed into
            # should_switch_phase BEFORE this step's HOLD
            # increment, so state['green_elapsed'] at a SWITCH
            # tick reflects the elapsed green time that triggered
            # the decision.
            green_elapsed_at_switch.append(
                state["green_elapsed"]
            )

    check(
        f"L: every SWITCH occurred with green_elapsed >= "
        f"MIN_GREEN={MIN_GREEN} "
        f"(observed minimum: "
        f"{min(green_elapsed_at_switch) if green_elapsed_at_switch else 'n/a'})",
        all(
            g >= MIN_GREEN
            for g in green_elapsed_at_switch
        ),
    )

    check(
        f"M: green_elapsed never exceeded MAX_GREEN={MAX_GREEN} "
        f"(observed max: {max_green_seen})",
        max_green_seen <= MAX_GREEN,
    )


def test_n_o_determinism():

    sim_a = AdaptiveTrafficSimulation(random_seed=42, generate_rag_explanations=False)
    sim_b = AdaptiveTrafficSimulation(random_seed=42, generate_rag_explanations=False)

    while not sim_a.finished:
        sim_a.step()

    while not sim_b.finished:
        sim_b.step()

    state_a = sim_a.get_state()
    state_b = sim_b.get_state()

    check(
        "N: same seed (42) produces identical total_arrivals/"
        "departures/switches",
        (
            state_a["total_arrivals"],
            state_a["total_departures"],
            state_a["adaptive_switches"],
        )
        == (
            state_b["total_arrivals"],
            state_b["total_departures"],
            state_b["adaptive_switches"],
        ),
    )

    sim_c = AdaptiveTrafficSimulation(random_seed=7, generate_rag_explanations=False)

    while not sim_c.finished:
        sim_c.step()

    state_c = sim_c.get_state()

    check(
        "O: different seed (7 vs 42) produces a "
        "different total_arrivals count",
        state_a["total_arrivals"] != state_c["total_arrivals"],
    )


def test_p_q_parent_ml_and_heading():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    check(
        "P: ml_predictions has exactly the 4 travel directions, "
        "not 12 movements",
        sorted(sim.ml_predictions.keys())
        == sorted(TRAVEL_DIRECTIONS),
    )

    all_consistent = True

    for _ in range(50):

        sim.step()

        for movement_id in MOVEMENT_IDS:

            for vehicle in sim.movement_queues[movement_id]:

                direction, _ = parse_movement_id(movement_id)

                if vehicle.direction != direction:
                    all_consistent = False

    check(
        "Q: every queued vehicle's direction matches its "
        "movement queue (correct EntryHeading/movement mapping)",
        all_consistent,
    )


def test_r_s_state_shape():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    state = sim.step()

    missing_keys = [
        key
        for key in REQUIRED_STATE_KEYS
        if key not in state
    ]

    check(
        f"R: get_state() contains all required keys "
        f"(missing: {missing_keys if missing_keys else 'none'})",
        not missing_keys,
    )

    deque_or_vehicle_path = _contains_deque_or_vehicle(state)

    check(
        f"S: get_state() contains no deque/Vehicle objects "
        f"(found at: {deque_or_vehicle_path or 'none'})",
        deque_or_vehicle_path is None,
    )


def test_t_full_run():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    ok = True

    while not sim.finished:

        state = sim.step()

        if (
            state["total_arrivals"]
            - state["total_departures"]
            != state["total_waiting"]
        ):
            ok = False

        if any(
            v < 0 for v in state["movement_queues"].values()
        ):
            ok = False

    check(
        "T: full tick-1-to-completion run holds all "
        "accounting invariants throughout",
        ok and sim.finished,
    )


def main():

    print("\nSTAGE 3 ADAPTIVE SIMULATOR INVARIANT TESTS")

    print("\nA. Exactly 12 movement queues exist")
    test_a_twelve_queues()

    print("\nB/C. Vehicle membership and consistency")
    test_b_c_vehicle_consistency()

    print("\nD/E/F. Arrival/departure accounting")
    test_d_e_f_accounting()

    print("\nG/H. Non-negative queues, arrivals-departures=waiting")
    test_g_h_every_tick_invariants()

    print("\nI/J. Only active GREEN movements depart")
    test_i_j_only_active_green_departs()

    print("\nK. Every phase switch passes through YELLOW")
    test_k_switch_through_yellow()

    print("\nL/M. MIN_GREEN / MAX_GREEN bounds")
    test_l_m_green_duration_bounds()

    print("\nN/O. Determinism (same seed / different seed)")
    test_n_o_determinism()

    print("\nP/Q. Correct parent ML and heading mapping")
    test_p_q_parent_ml_and_heading()

    print("\nR/S. get_state() shape and JSON-safety")
    test_r_s_state_shape()

    print("\nT. Full run without accounting failure")
    test_t_full_run()

    print()
    print(
        f"RESULTS: {len(PASSED)} passed, {len(FAILED)} failed"
    )

    if FAILED:

        print("\nFAILED CHECKS:")

        for name in FAILED:
            print(f"  - {name}")

        sys.exit(1)

    print("\nAll Stage 3 invariants passed.")


if __name__ == "__main__":
    main()
