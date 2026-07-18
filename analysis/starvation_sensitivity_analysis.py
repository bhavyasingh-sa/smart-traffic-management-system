"""starvation_sensitivity_analysis.py - evaluates the hard starvation safeguard across 5 threshold configurations (None/60/90/120/180) under 3 scenarios, using the exact same deterministic traffic schedule within each scenario."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

import numpy as np

from simulation.movement_definitions import (
    MOVEMENT_IDS,
    PHASES,
    PHASE_NAMES,
)

from simulation.controller_core import MIN_GREEN, MAX_GREEN
from simulation.adaptive_simulator import AdaptiveTrafficSimulation


THRESHOLD_CONFIGS = [
    ("A_no_override", None),
    ("B_60s", 60),
    ("C_90s", 90),
    ("D_120s", 120),
    ("E_180s", 180),
]

SCENARIOS = [
    {
        "name": "Scenario 1: Normal (300 ticks)",
        "ticks": 300,
        "arrival_rate_overrides": None,
    },
    {
        "name": "Scenario 2: Long-run (3000 ticks)",
        "ticks": 3000,
        "arrival_rate_overrides": None,
    },
    {
        "name": "Scenario 3: Heavy-demand stress (6000 ticks)",
        "ticks": 6000,
        # Identical to the stress scenario in tests/test_starvation_stress.py: EB/WB boosted 6x.
        "arrival_rate_overrides_factory": lambda base: {
            "EB": base["EB"] * 6.0,
            "WB": base["WB"] * 6.0,
        },
    },
]


def run_one(ticks, arrival_rate_overrides, max_starvation_time):

    sim = AdaptiveTrafficSimulation(
        arrival_rate_overrides=arrival_rate_overrides,
        max_starvation_time=max_starvation_time,
    )

    phase_green_ticks = {p: 0 for p in PHASE_NAMES}
    previous_phase = sim.current_phase

    max_starvation_overall = 0
    max_starvation_per_movement = {m: 0 for m in MOVEMENT_IDS}

    accounting_ok = True
    min_green_ok = True
    max_green_ok = True
    yellow_ok = True

    # NOTE: deliberately reading sim's own lightweight attributes
    # directly during this bulk loop instead of calling
    # sim.get_state() every tick - get_state() rebuilds a large
    # dashboard-facing dict (including the 12-movement priority
    # breakdown) which is fine for real dashboard use (a few times
    # a second) but wastefully expensive across tens of thousands
    # of ticks in this analysis. get_state() is still called once
    # at the end for the final summary snapshot.
    for _ in range(ticks):

        sim.finished = False
        sim.step()

        total_arrivals = sum(
            sim.movement_total_arrivals.values()
        )

        total_departures = sum(
            sim.movement_total_departures.values()
        )

        total_waiting = sum(
            len(sim.movement_queues[m])
            for m in MOVEMENT_IDS
        )

        if (
            total_arrivals - total_departures
            != total_waiting
        ):
            accounting_ok = False

        if any(
            len(sim.movement_queues[m]) < 0
            for m in MOVEMENT_IDS
        ):
            accounting_ok = False

        if sim.green_elapsed > MAX_GREEN:
            max_green_ok = False

        if sim.last_decision["action"] == "SWITCH":

            if sim.green_elapsed < MIN_GREEN:
                min_green_ok = False

        if (
            sim.current_phase != previous_phase
            and sim.last_decision["action"] != "ACTIVATE"
        ):
            yellow_ok = False

        previous_phase = sim.current_phase

        if sim.signal_state == "GREEN":
            phase_green_ticks[sim.current_phase] += 1

        for movement_id in MOVEMENT_IDS:

            s = sim.movement_starvation[movement_id]

            if s > max_starvation_overall:
                max_starvation_overall = s

            if s > max_starvation_per_movement[movement_id]:
                max_starvation_per_movement[movement_id] = s

    state = sim.get_state()

    all_completed_waits = []

    for movement_id in MOVEMENT_IDS:
        all_completed_waits.extend(
            sim.completed_waits[movement_id]
        )

    average_completed_wait = (
        float(np.mean(all_completed_waits))
        if all_completed_waits
        else 0.0
    )

    max_completed_wait = (
        float(np.max(all_completed_waits))
        if all_completed_waits
        else 0.0
    )

    throughput_pct = (
        state["total_departures"]
        / state["total_arrivals"]
        * 100.0
        if state["total_arrivals"] > 0
        else 0.0
    )

    return {
        "total_arrivals": state["total_arrivals"],
        "total_departures": state["total_departures"],
        "total_waiting": state["total_waiting"],
        "throughput_pct": throughput_pct,
        "average_completed_wait": average_completed_wait,
        "max_completed_wait": max_completed_wait,
        "max_starvation_overall": max_starvation_overall,
        "max_starvation_per_movement": (
            max_starvation_per_movement
        ),
        "adaptive_switches": state["adaptive_switches"],
        "starvation_override_switches": (
            state["starvation_override_switches"]
        ),
        "switch_reasons": state["switch_reasons"],
        "phase_green_ticks": phase_green_ticks,
        "movement_total_arrivals": dict(
            state["movement_total_arrivals"]
        ),
        "movement_total_departures": dict(
            state["movement_total_departures"]
        ),
        "movement_queues": dict(state["movement_queues"]),
        "accounting_ok": accounting_ok,
        "min_green_ok": min_green_ok,
        "max_green_ok": max_green_ok,
        "yellow_ok": yellow_ok,
        "ticks_run": ticks,
    }


import json

CACHE_DIR = (
    Path(__file__).resolve().parent
    / "_sensitivity_cache"
)


def run_single_and_cache(scenario_index, config_name):

    scenario = SCENARIOS[scenario_index]
    ticks = scenario["ticks"]

    max_starvation_time = dict(THRESHOLD_CONFIGS)[
        config_name
    ]

    if "arrival_rate_overrides_factory" in scenario:

        probe_sim = AdaptiveTrafficSimulation()

        arrival_rate_overrides = scenario[
            "arrival_rate_overrides_factory"
        ](probe_sim.arrival_rates)

    else:

        arrival_rate_overrides = scenario[
            "arrival_rate_overrides"
        ]

    result = run_one(
        ticks=ticks,
        arrival_rate_overrides=arrival_rate_overrides,
        max_starvation_time=max_starvation_time,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = (
        CACHE_DIR
        / f"scenario{scenario_index}_{config_name}.json"
    )

    with open(cache_path, "w") as f:
        json.dump(result, f)

    print(
        f"Cached scenario={scenario_index} "
        f"config={config_name} -> {cache_path}"
    )

    print(
        f"  arrivals={result['total_arrivals']} "
        f"departures={result['total_departures']} "
        f"waiting={result['total_waiting']} "
        f"max_starvation={result['max_starvation_overall']} "
        f"switches={result['adaptive_switches']} "
        f"(override={result['starvation_override_switches']})"
    )


def main(scenario_indices=None):

    print("\nSTAGE 3.5 — STARVATION SAFEGUARD SENSITIVITY ANALYSIS")

    all_results = {}

    scenarios_to_run = (
        SCENARIOS
        if scenario_indices is None
        else [SCENARIOS[i] for i in scenario_indices]
    )

    for scenario in scenarios_to_run:

        scenario_name = scenario["name"]
        ticks = scenario["ticks"]

        print(f"\n\n{scenario_name}")

        scenario_results = {}

        # Resolve arrival_rate_overrides once per scenario (need a
        # base simulator instance just to read real arrival rates
        # for the boosted-scenario factory).
        if "arrival_rate_overrides_factory" in scenario:

            probe_sim = AdaptiveTrafficSimulation()

            arrival_rate_overrides = scenario[
                "arrival_rate_overrides_factory"
            ](probe_sim.arrival_rates)

        else:

            arrival_rate_overrides = scenario[
                "arrival_rate_overrides"
            ]

        for config_name, max_starvation_time in (
            THRESHOLD_CONFIGS
        ):

            print(
                f"\n--- Running {config_name} "
                f"(max_starvation_time={max_starvation_time}) ---"
            )

            result = run_one(
                ticks=ticks,
                arrival_rate_overrides=arrival_rate_overrides,
                max_starvation_time=max_starvation_time,
            )

            scenario_results[config_name] = result

            print(
                f"  arrivals={result['total_arrivals']} "
                f"departures={result['total_departures']} "
                f"waiting={result['total_waiting']} "
                f"throughput={result['throughput_pct']:.1f}% "
                f"avg_wait={result['average_completed_wait']:.2f}s "
                f"max_wait={result['max_completed_wait']:.0f}s "
                f"max_starvation={result['max_starvation_overall']} "
                f"switches={result['adaptive_switches']} "
                f"(override={result['starvation_override_switches']})"
            )

        # Confirms the RNG-ordering claim empirically: arrival/movement-
        # assignment rng consumption happens entirely inside
        # _generate_arrivals(), before any phase-decision logic runs, so
        # the starvation-safeguard threshold cannot affect the arrival
        # schedule - all 5 configs should therefore be identical here.
        arrival_schedules = [
            scenario_results[name]["movement_total_arrivals"]
            for name, _ in THRESHOLD_CONFIGS
        ]

        schedule_identical = all(
            schedule == arrival_schedules[0]
            for schedule in arrival_schedules
        )

        print(
            f"\nFAIRNESS CHECK — identical arrival schedule "
            f"across all 5 configs: {schedule_identical}"
        )

        if not schedule_identical:

            print(
                "  WARNING: arrival schedules differ between "
                "configs - results below are NOT a fair "
                "comparison for this scenario."
            )

        for config_name, _ in THRESHOLD_CONFIGS:

            r = scenario_results[config_name]

            all_ok = (
                r["accounting_ok"]
                and r["min_green_ok"]
                and r["max_green_ok"]
                and r["yellow_ok"]
            )

            print(
                f"  {config_name}: accounting_ok="
                f"{r['accounting_ok']} min_green_ok="
                f"{r['min_green_ok']} max_green_ok="
                f"{r['max_green_ok']} yellow_ok={r['yellow_ok']} "
                f"-> {'OK' if all_ok else 'FAILED'}"
            )

        print(f"\n{scenario_name} — FULL RESULTS TABLE\n")

        header = (
            f"{'Config':14s} {'Arrivals':>9} {'Depart':>8} "
            f"{'Waiting':>8} {'Thrpt%':>7} {'AvgWait':>8} "
            f"{'MaxWait':>8} {'MaxStarve':>10} {'Switches':>9} "
            f"{'Override':>9}"
        )

        print(header)
        print("-" * len(header))

        for config_name, _ in THRESHOLD_CONFIGS:

            r = scenario_results[config_name]

            print(
                f"{config_name:14s} "
                f"{r['total_arrivals']:>9} "
                f"{r['total_departures']:>8} "
                f"{r['total_waiting']:>8} "
                f"{r['throughput_pct']:>6.1f}% "
                f"{r['average_completed_wait']:>7.2f}s "
                f"{r['max_completed_wait']:>7.0f}s "
                f"{r['max_starvation_overall']:>10} "
                f"{r['adaptive_switches']:>9} "
                f"{r['starvation_override_switches']:>9}"
            )

        print(
            "\nPhase green-time share (% of run):"
        )

        for config_name, _ in THRESHOLD_CONFIGS:

            r = scenario_results[config_name]

            shares = ", ".join(
                f"{p}={r['phase_green_ticks'][p] / ticks * 100:.1f}%"
                for p in PHASE_NAMES
            )

            print(f"  {config_name:14s}: {shares}")

        print(
            "\nTargeted low-volume phase (NS_LEFT) per-movement "
            "detail:"
        )

        for config_name, _ in THRESHOLD_CONFIGS:

            r = scenario_results[config_name]

            for movement_id in PHASES["NS_LEFT"]:

                print(
                    f"  {config_name:14s} {movement_id:10s}: "
                    f"arrivals={r['movement_total_arrivals'][movement_id]:>4} "
                    f"departures={r['movement_total_departures'][movement_id]:>4} "
                    f"remaining={r['movement_queues'][movement_id]:>4} "
                    f"max_starvation="
                    f"{r['max_starvation_per_movement'][movement_id]:>5}"
                )

        all_results[scenario_name] = scenario_results

    print("\nSENSITIVITY ANALYSIS COMPLETE")

    return all_results


if __name__ == "__main__":

    if len(sys.argv) == 3 and sys.argv[1] == "--single":
        # Usage: script.py --single "0:A_no_override"
        scenario_index_str, config_name = sys.argv[2].split(
            ":"
        )
        run_single_and_cache(
            int(scenario_index_str), config_name
        )
    elif len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
        main(scenario_indices=indices)
    else:
        main()
