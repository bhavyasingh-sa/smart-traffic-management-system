"""aggregate_sensitivity_results.py - reads cached per-run results from analysis/_sensitivity_cache/ (produced by starvation_sensitivity_analysis.py --single ...) and prints the full Stage 3.5 sensitivity report."""

import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from simulation.movement_definitions import PHASES, PHASE_NAMES

CACHE_DIR = (
    Path(__file__).resolve().parent
    / "_sensitivity_cache"
)

SCENARIO_NAMES = [
    "Scenario 1: Normal (300 ticks)",
    "Scenario 2: Long-run (3000 ticks)",
    "Scenario 3: Heavy-demand stress (6000 ticks)",
]

SCENARIO_TICKS = [300, 3000, 6000]

THRESHOLD_CONFIGS = [
    "A_no_override",
    "B_60s",
    "C_90s",
    "D_120s",
    "E_180s",
]


def load(scenario_index, config_name):

    path = (
        CACHE_DIR
        / f"scenario{scenario_index}_{config_name}.json"
    )

    with open(path) as f:
        return json.load(f)


def main():

    print("\nSTAGE 3.5 — FULL SENSITIVITY ANALYSIS REPORT")

    for scenario_index, scenario_name in enumerate(
        SCENARIO_NAMES
    ):

        ticks = SCENARIO_TICKS[scenario_index]

        print(f"\n\n{scenario_name}")

        results = {
            config_name: load(scenario_index, config_name)
            for config_name in THRESHOLD_CONFIGS
        }

        arrival_schedules = [
            results[c]["movement_total_arrivals"]
            for c in THRESHOLD_CONFIGS
        ]

        schedule_identical = all(
            s == arrival_schedules[0]
            for s in arrival_schedules
        )

        print(
            f"\nFAIRNESS CHECK - identical arrival schedule "
            f"across all 5 configs: {schedule_identical}"
        )

        print("\nInvariant checks:")

        for config_name in THRESHOLD_CONFIGS:

            r = results[config_name]

            all_ok = (
                r["accounting_ok"]
                and r["min_green_ok"]
                and r["max_green_ok"]
                and r["yellow_ok"]
            )

            print(
                f"  {config_name:14s}: accounting_ok="
                f"{r['accounting_ok']} min_green_ok="
                f"{r['min_green_ok']} max_green_ok="
                f"{r['max_green_ok']} yellow_ok={r['yellow_ok']} "
                f"-> {'OK' if all_ok else 'FAILED'}"
            )

        print(f"\n{scenario_name} - FULL RESULTS TABLE\n")

        header = (
            f"{'Config':14s} {'Arrivals':>9} {'Depart':>8} "
            f"{'Waiting':>8} {'Thrpt%':>7} {'AvgWait':>8} "
            f"{'MaxWait':>8} {'MaxStarve':>10} {'Switches':>9} "
            f"{'Override':>9}"
        )

        print(header)
        print("-" * len(header))

        for config_name in THRESHOLD_CONFIGS:

            r = results[config_name]

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

        print("\nSwitch reasons breakdown:")

        for config_name in THRESHOLD_CONFIGS:

            r = results[config_name]

            print(f"  {config_name:14s}: {r['switch_reasons']}")

        print("\nPhase green-time share (% of run):")

        for config_name in THRESHOLD_CONFIGS:

            r = results[config_name]

            shares = ", ".join(
                f"{p}={r['phase_green_ticks'][p] / ticks * 100:.1f}%"
                for p in PHASE_NAMES
            )

            print(f"  {config_name:14s}: {shares}")

        print(
            "\nTargeted low-volume phase (NS_LEFT) "
            "per-movement detail:"
        )

        for config_name in THRESHOLD_CONFIGS:

            r = results[config_name]

            for movement_id in PHASES["NS_LEFT"]:

                print(
                    f"  {config_name:14s} {movement_id:10s}: "
                    f"arrivals="
                    f"{r['movement_total_arrivals'][movement_id]:>5} "
                    f"departures="
                    f"{r['movement_total_departures'][movement_id]:>5} "
                    f"remaining="
                    f"{r['movement_queues'][movement_id]:>5} "
                    f"max_starvation="
                    f"{r['max_starvation_per_movement'][movement_id]:>5}"
                )

    print("\nREPORT COMPLETE")


if __name__ == "__main__":
    main()
