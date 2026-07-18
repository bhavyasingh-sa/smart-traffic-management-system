"""test_starvation_safeguard.py - tests for the starvation-override safeguard (controller_core.py) and ML/IR hourly refresh (adaptive_simulator.py)."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from simulation.movement_definitions import (
    MOVEMENT_IDS,
    PHASE_NAMES,
    PHASES,
)

from simulation.controller_core import (
    MIN_GREEN,
    check_starvation_override,
    should_switch_phase,
)

from simulation.adaptive_simulator import AdaptiveTrafficSimulation


PASSED = []
FAILED = []


def check(name, condition):

    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")

    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


def _empty_queues():
    return {m: [] for m in MOVEMENT_IDS}


def _flat_priorities(value=0.5):
    return {p: value for p in PHASE_NAMES}


def test_no_override_below_threshold():

    queues = _empty_queues()
    queues["EB_STRAIGHT"] = [1, 2, 3]

    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 119

    should_override, target = check_starvation_override(
        current_phase="NS_THROUGH",
        movement_queues=queues,
        starvation=starvation,
        phase_priorities=_flat_priorities(),
        max_starvation_time=120,
    )

    check(
        "no override fires when starvation (119) is "
        "just below max_starvation_time (120)",
        should_override is False and target is None,
    )


def test_override_at_threshold():

    queues = _empty_queues()
    queues["EB_STRAIGHT"] = [1, 2, 3]

    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 120

    should_override, target = check_starvation_override(
        current_phase="NS_THROUGH",
        movement_queues=queues,
        starvation=starvation,
        phase_priorities=_flat_priorities(),
        max_starvation_time=120,
    )

    check(
        "override fires when starvation == max_starvation_time "
        "exactly (>=, not >)",
        should_override is True
        and target == "EW_THROUGH",
    )

    starvation["EB_STRAIGHT"] = 500

    should_override_2, target_2 = check_starvation_override(
        current_phase="NS_THROUGH",
        movement_queues=queues,
        starvation=starvation,
        phase_priorities=_flat_priorities(),
        max_starvation_time=120,
    )

    check(
        "override still fires well above threshold (500)",
        should_override_2 is True
        and target_2 == "EW_THROUGH",
    )


def test_override_requires_nonempty_queue():

    queues = _empty_queues()
    # EB_STRAIGHT has HIGH starvation but an EMPTY queue - should
    # not trigger an override (nothing to serve).
    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 999

    should_override, target = check_starvation_override(
        current_phase="NS_THROUGH",
        movement_queues=queues,
        starvation=starvation,
        phase_priorities=_flat_priorities(),
        max_starvation_time=120,
    )

    check(
        "no override for a starved movement with an EMPTY queue",
        should_override is False and target is None,
    )


def test_min_green_still_respected_with_safeguard():

    queues = _empty_queues()
    queues["EB_STRAIGHT"] = [1, 2, 3]

    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 999

    ml_predictions = {
        d: {"class": "Low", "severity": 0.1}
        for d in ["NB", "SB", "EB", "WB"]
    }

    for green_elapsed in [0, 5, MIN_GREEN - 1]:

        should_switch, reason, target, _details = (
            should_switch_phase(
                current_phase="NS_THROUGH",
                green_elapsed=green_elapsed,
                movement_queues=queues,
                current_tick=1000,
                ml_predictions=ml_predictions,
                starvation=starvation,
                ir_predictions=None,
                max_starvation_time=120,
            )
        )

        check(
            f"MIN_GREEN still holds at green_elapsed="
            f"{green_elapsed} even with extreme starvation (999) "
            "and the safeguard enabled",
            should_switch is False
            and reason is None
            and target is None,
        )


def test_starvation_override_reason_recorded():

    queues = _empty_queues()
    queues["EB_STRAIGHT"] = [1, 2, 3]

    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 150

    ml_predictions = {
        d: {"class": "Low", "severity": 0.1}
        for d in ["NB", "SB", "EB", "WB"]
    }

    should_switch, reason, target, _details = should_switch_phase(
        current_phase="NS_THROUGH",
        green_elapsed=MIN_GREEN,
        movement_queues=queues,
        current_tick=1000,
        ml_predictions=ml_predictions,
        starvation=starvation,
        ir_predictions=None,
        max_starvation_time=120,
    )

    check(
        "reason is recorded exactly as 'starvation_override'",
        should_switch is True
        and reason == "starvation_override"
        and target == "EW_THROUGH",
    )


def test_deterministic_tie_breaking():

    # Two competing phases both qualify with the SAME max
    # starvation - tie-break must go to phase priority, then
    # PHASE_TIE_BREAK_ORDER, deterministically.
    queues = _empty_queues()
    queues["EB_STRAIGHT"] = [1]
    queues["WB_LEFT"] = [1]

    starvation = {m: 0 for m in MOVEMENT_IDS}
    starvation["EB_STRAIGHT"] = 200
    starvation["WB_LEFT"] = 200

    # EW_THROUGH given a higher phase priority than EW_LEFT -
    # tie-break rule 2 (higher phase priority) should pick it.
    priorities = _flat_priorities(0.3)
    priorities["EW_THROUGH"] = 0.9
    priorities["EW_LEFT"] = 0.5

    should_override, target = check_starvation_override(
        current_phase="NS_THROUGH",
        movement_queues=queues,
        starvation=starvation,
        phase_priorities=priorities,
        max_starvation_time=120,
    )

    check(
        "tied max-starvation phases broken by higher phase "
        "priority (EW_THROUGH over EW_LEFT)",
        should_override is True and target == "EW_THROUGH",
    )

    # Now make priorities ALSO tie - must fall through to
    # PHASE_TIE_BREAK_ORDER deterministically, and repeat several
    # times to rule out hidden nondeterminism.
    tied_priorities = _flat_priorities(0.5)

    repeats = [
        check_starvation_override(
            current_phase="NS_THROUGH",
            movement_queues=queues,
            starvation=starvation,
            phase_priorities=tied_priorities,
            max_starvation_time=120,
        )
        for _ in range(20)
    ]

    check(
        "fully-tied phases resolve identically across 20 "
        "repeated calls (deterministic, not dict-order dependent)",
        len(set(r[1] for r in repeats)) == 1,
    )


def test_no_direct_green_to_green_with_safeguard():

    sim = AdaptiveTrafficSimulation(
        arrival_rate_overrides={"EB": 2.5, "WB": 3.5},
        max_starvation_time=120,
        generate_rag_explanations=False,
    )

    violation_found = False
    previous_phase = sim.current_phase

    for _ in range(1500):

        sim.finished = False
        sim.step()

        if sim.current_phase != previous_phase:

            if sim.last_decision["action"] != "ACTIVATE":
                violation_found = True

        previous_phase = sim.current_phase

    check(
        "every phase change (including starvation_override "
        "switches) still passes through an ACTIVATE action "
        "(i.e. through YELLOW) over 3000 ticks",
        not violation_found,
    )


def test_starvation_switch_count_correct():

    sim = AdaptiveTrafficSimulation(
        arrival_rate_overrides={"EB": 2.5, "WB": 3.5},
        max_starvation_time=120,
        generate_rag_explanations=False,
    )

    manual_count = 0

    for _ in range(1500):

        sim.finished = False
        state = sim.step()

        if (
            state["last_decision"]["action"] == "SWITCH"
            and state["last_decision"]["reason"]
            == "starvation_override"
        ):
            manual_count += 1

    check(
        f"starvation_override_switches ({sim.starvation_override_switches}) "
        f"matches manually counted SWITCH+starvation_override "
        f"decisions ({manual_count})",
        sim.starvation_override_switches == manual_count,
    )


def test_normal_simulation_completes_with_safeguard():

    sim = AdaptiveTrafficSimulation(max_starvation_time=120, generate_rag_explanations=False)

    accounting_ok = True

    while not sim.finished:

        state = sim.step()

        if (
            state["total_arrivals"]
            - state["total_departures"]
            != state["total_waiting"]
        ):
            accounting_ok = False

        if any(
            v < 0 for v in state["movement_queues"].values()
        ):
            accounting_ok = False

    check(
        "normal 300-tick simulation completes with the "
        "safeguard enabled (max_starvation_time=120), accounting "
        "invariants held throughout",
        sim.finished and accounting_ok,
    )


def test_ml_ir_hourly_refresh():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    initial_ml = dict(sim.ml_predictions)
    initial_ir = dict(sim.ir_predictions)

    # Run exactly 3600 ticks - hour should NOT have changed yet
    # (refresh happens strictly AFTER 3600 ticks, at tick 3601).
    for _ in range(3600):

        sim.finished = False
        sim.step()

    check(
        "ML/IR do NOT refresh before 3600 ticks have elapsed "
        f"(simulated_hour stayed at {sim.simulated_hour}, "
        f"started at {sim.hour})",
        sim.simulated_hour == sim.hour
        and len(sim.ml_ir_refresh_history) == 1,
    )

    # One more tick crosses into the next simulated hour.
    sim.finished = False
    sim.step()

    check(
        f"ML/IR refreshed exactly at tick 3601 "
        f"(simulated_hour is now {sim.simulated_hour}, "
        f"was {sim.hour})",
        sim.simulated_hour == (sim.hour + 1) % 24
        and len(sim.ml_ir_refresh_history) == 2
        and sim.ml_ir_refresh_history[-1]["tick"] == 3601,
    )

    check(
        "ML predictions actually changed after the hourly refresh "
        "(not just relabeled)",
        sim.ml_predictions != initial_ml,
    )

    check(
        "IR predictions actually changed after the hourly refresh",
        sim.ir_predictions != initial_ir,
    )


def test_ml_ir_do_not_refresh_unnecessarily():

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    refresh_call_count_before = len(
        sim.ml_ir_refresh_history
    )

    # A normal 300-tick run should never cross an hour boundary.
    for _ in range(300):
        sim.step()

    check(
        "ML/IR do not refresh at all during a normal 300-tick run "
        "(well under the 3600-tick hour boundary)",
        len(sim.ml_ir_refresh_history)
        == refresh_call_count_before,
    )


def test_ml_ir_refresh_uses_correct_hour():

    from simulation.controller_core import (
        load_ml_model,
        load_ir_retriever,
        build_ml_predictions,
        build_ir_predictions,
    )

    sim = AdaptiveTrafficSimulation(generate_rag_explanations=False)

    for _ in range(3601):
        sim.finished = False
        sim.step()

    # Independently recompute what hour 9's ML/IR predictions
    # should be, using the same public functions, and compare -
    # this proves the refresh used the CORRECT new hour, not just
    # that *something* changed. Severity is compared with a small
    # tolerance, not exact equality: the classifier's n_jobs=-1
    # (ml/train.py) aggregates 250 trees across threads, and
    # parallel float summation isn't guaranteed bit-identical
    # between two separate calls (~1e-15 relative drift observed) -
    # a real property of the model, not something a correct refresh
    # could avoid. The predicted class label, which is what actually
    # drives the controller, IS compared exactly.
    expected_ml = build_ml_predictions(
        model_bundle=sim.model_bundle,
        hour=(sim.hour + 1) % 24,
        weekend=sim.weekend,
        month=sim.month,
    )

    ml_predictions_match = all(
        sim.ml_predictions[direction]["class"]
        == expected_ml[direction]["class"]
        and abs(
            sim.ml_predictions[direction]["severity"]
            - expected_ml[direction]["severity"]
        )
        < 1e-6
        for direction in expected_ml
    )

    check(
        "post-refresh ML predictions match an independent "
        "recomputation for the new hour (class exact, "
        "severity within 1e-6)",
        ml_predictions_match,
    )


TEST_FUNCTIONS = [
    ("1. No override below threshold", test_no_override_below_threshold),
    ("2. Override occurs at or above threshold", test_override_at_threshold),
    ("3. Override requires a non-empty queue", test_override_requires_nonempty_queue),
    ("4. MIN_GREEN still respected with safeguard enabled", test_min_green_still_respected_with_safeguard),
    ("5. starvation_override reason recorded correctly", test_starvation_override_reason_recorded),
    ("6. Deterministic target-phase selection / tie-breaking", test_deterministic_tie_breaking),
    ("7. No direct GREEN-to-GREEN change (via simulator)", test_no_direct_green_to_green_with_safeguard),
    ("8. Starvation-forced switch count is correct", test_starvation_switch_count_correct),
    ("9. Normal simulation completes, accounting holds", test_normal_simulation_completes_with_safeguard),
    ("10. ML/IR hourly refresh fires at the right tick", test_ml_ir_hourly_refresh),
    ("11. ML/IR do not refresh unnecessarily (normal run)", test_ml_ir_do_not_refresh_unnecessarily),
    ("12. ML/IR refresh uses the correct new hour", test_ml_ir_refresh_uses_correct_hour),
]


def main(only_indices=None):

    print("\nSTAGE 3.5 STARVATION SAFEGUARD + ML/IR REFRESH TESTS")

    indices = (
        range(len(TEST_FUNCTIONS))
        if only_indices is None
        else only_indices
    )

    for i in indices:

        label, func = TEST_FUNCTIONS[i]
        print(f"\n{label}")
        func()

    print()
    print(
        f"RESULTS: {len(PASSED)} passed, {len(FAILED)} failed"
    )

    if FAILED:

        print("\nFAILED CHECKS:")

        for name in FAILED:
            print(f"  - {name}")

        sys.exit(1)

    print("\nAll requested tests passed.")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        indices = [int(x) - 1 for x in sys.argv[1:]]
        main(only_indices=indices)
    else:
        main()
