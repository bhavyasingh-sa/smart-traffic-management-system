"""movement_audit.py - single source of truth: derives Atlanta Intersection 84's real movement matrix and turning proportions directly from data/raw/train.csv."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

MOVEMENT_MATRIX_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movement_matrix.csv"
)

TURNING_PROPORTIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "turning_proportions.csv"
)

CITY = "Atlanta"
INTERSECTION_ID = 84

# Compass heading -> degrees, used to derive movement type
# mathematically instead of guessing.
HEADING_DEGREES = {
    "N": 0,
    "NE": 45,
    "E": 90,
    "SE": 135,
    "S": 180,
    "SW": 225,
    "W": 270,
    "NW": 315,
}

# DATASET CONVENTION (verified against data, not assumed): EntryHeading /
# ExitHeading describe the DIRECTION OF TRAVEL, not the physical arrival
# side - e.g. EntryHeading="N" means travelling north, i.e. the vehicle
# physically approaches FROM THE SOUTH. Straight-through traffic keeps
# the SAME heading letter at entry and exit (verified: "Cheshire Bridge
# Road N -> Cheshire Bridge Road N" is the single most common movement
# at this intersection, 329 records - a real straight flow, not a U-turn).
#
# EntryHeading letter -> travel-direction code used everywhere
# else in this project (movement IDs, dashboard, controller).
TRAVEL_DIRECTION_CODES = {
    "N": "NB",
    "S": "SB",
    "E": "EB",
    "W": "WB",
}

# Travel-direction code -> the physical side of the intersection
# a vehicle in that direction actually approaches from. This is
# what the dashboard should use to draw each movement on the
# correct physical leg.
PHYSICAL_APPROACH_SIDE = {
    "NB": "south",
    "SB": "north",
    "EB": "west",
    "WB": "east",
}


def classify_movement(entry_heading, exit_heading):
    """
    Derive LEFT / STRAIGHT / RIGHT / U-TURN / UNKNOWN mathematically
    from the compass-degree difference between entry and exit
    heading. No hardcoded left/right mapping per direction - it
    falls straight out of the angle.
    """

    if (
        entry_heading not in HEADING_DEGREES
        or exit_heading not in HEADING_DEGREES
    ):

        return "UNKNOWN"

    delta = (
        HEADING_DEGREES[exit_heading]
        - HEADING_DEGREES[entry_heading]
    ) % 360

    if delta == 0:
        return "STRAIGHT"

    if delta == 270:
        return "LEFT"

    if delta == 90:
        return "RIGHT"

    if delta == 180:
        return "U-TURN"

    return "UNKNOWN"


def load_intersection_data():

    dataframe = pd.read_csv(
        RAW_DATA_PATH
    )

    intersection_data = dataframe[
        (dataframe["City"] == CITY)
        & (
            dataframe["IntersectionId"]
            == INTERSECTION_ID
        )
    ].copy()

    if intersection_data.empty:

        raise ValueError(
            f"No records found for "
            f"{CITY} intersection "
            f"{INTERSECTION_ID}."
        )

    return intersection_data


def build_movement_matrix(
    intersection_data
):
    """
    One row per real (EntryStreetName, EntryHeading, ExitStreetName,
    ExitHeading) combination actually observed at this intersection,
    with record counts, traffic share, derived movement type, and
    an ML-suitability flag.
    """

    total_records = len(
        intersection_data
    )

    grouped = (
        intersection_data
        .groupby(
            [
                "EntryStreetName",
                "EntryHeading",
                "ExitStreetName",
                "ExitHeading",
            ]
        )
        .size()
        .reset_index(name="records")
    )

    grouped["movement_type"] = grouped.apply(
        lambda row: classify_movement(
            row["EntryHeading"],
            row["ExitHeading"],
        ),
        axis=1,
    )

    grouped["travel_direction"] = grouped[
        "EntryHeading"
    ].map(TRAVEL_DIRECTION_CODES)

    grouped["movement_id"] = (
        grouped["travel_direction"]
        + "_"
        + grouped["movement_type"]
    )

    grouped["traffic_share_pct"] = (
        grouped["records"]
        / total_records
        * 100
    ).round(2)

    # A movement is flagged ML-suitable only as a loose signal for
    # discussion purposes - Version 2's actual design (Option C)
    # does NOT train independent per-movement classifiers, exactly
    # because these counts are too thin to justify it. This column
    # exists to make that reasoning visible in the CSV, not to
    # trigger per-movement model training.
    grouped["ml_suitable_standalone"] = (
        grouped["records"] >= 200
    )

    grouped = grouped.sort_values(
        "records",
        ascending=False,
    ).reset_index(drop=True)

    return grouped, total_records


def build_turning_proportions(
    movement_matrix
):
    """
    Aggregate the movement matrix up to travel-direction level
    (NB / SB / EB / WB) and compute the real historical share of
    STRAIGHT / LEFT / RIGHT traffic for each direction.

    These proportions are what Version 2 uses to split an
    approach-level ML severity prediction across that approach's
    three movement queues (NB_STRAIGHT / NB_LEFT / NB_RIGHT, etc.)
    - NOT independently trained per-movement ML predictions.
    """

    rows = []

    for direction in [
        "NB",
        "SB",
        "EB",
        "WB",
    ]:

        direction_rows = movement_matrix[
            movement_matrix[
                "travel_direction"
            ]
            == direction
        ]

        direction_total = direction_rows[
            "records"
        ].sum()

        for movement_type in [
            "STRAIGHT",
            "LEFT",
            "RIGHT",
        ]:

            matching = direction_rows[
                direction_rows[
                    "movement_type"
                ]
                == movement_type
            ]

            movement_records = int(
                matching["records"].sum()
            )

            proportion = (
                movement_records
                / direction_total
                if direction_total > 0
                else 0.0
            )

            rows.append(
                {
                    "travel_direction": direction,
                    "physical_approach_side": (
                        PHYSICAL_APPROACH_SIDE[
                            direction
                        ]
                    ),
                    "movement_type": movement_type,
                    "movement_id": (
                        f"{direction}_{movement_type}"
                    ),
                    "records": movement_records,
                    "direction_total_records": int(
                        direction_total
                    ),
                    "proportion": round(
                        proportion,
                        4,
                    ),
                    "proportion_pct": round(
                        proportion * 100,
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


def main():

    print(
        f"\nMOVEMENT AUDIT — "
        f"{CITY} INTERSECTION {INTERSECTION_ID}"
    )

    intersection_data = (
        load_intersection_data()
    )

    (
        movement_matrix,
        total_records,
    ) = build_movement_matrix(
        intersection_data
    )

    print(
        f"\nTotal records at this intersection: "
        f"{total_records:,}"
    )

    print(
        f"Distinct real movement combinations: "
        f"{len(movement_matrix)}"
    )

    print(
        f"\nMonths present: "
        f"{sorted(intersection_data['Month'].unique().tolist())}"
    )

    unknown_count = int(
        (
            movement_matrix["movement_type"]
            == "UNKNOWN"
        ).sum()
    )

    uturn_count = int(
        (
            movement_matrix["movement_type"]
            == "U-TURN"
        ).sum()
    )

    print(
        f"\nUNKNOWN movement rows: {unknown_count}"
    )

    print(
        f"U-TURN movement rows: {uturn_count}"
    )

    print(
        "\n--- FULL MOVEMENT MATRIX ---\n"
    )

    print(
        movement_matrix[
            [
                "EntryStreetName",
                "EntryHeading",
                "ExitStreetName",
                "ExitHeading",
                "movement_id",
                "movement_type",
                "records",
                "traffic_share_pct",
                "ml_suitable_standalone",
            ]
        ].to_string(index=False)
    )

    turning_proportions = (
        build_turning_proportions(
            movement_matrix
        )
    )

    print(
        "\n--- REAL TURNING PROPORTIONS "
        "(by travel direction) ---\n"
    )

    print(
        turning_proportions.to_string(
            index=False
        )
    )

    MOVEMENT_MATRIX_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    movement_matrix.to_csv(
        MOVEMENT_MATRIX_OUTPUT_PATH,
        index=False,
    )

    turning_proportions.to_csv(
        TURNING_PROPORTIONS_OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved movement matrix to:\n"
        f"{MOVEMENT_MATRIX_OUTPUT_PATH}"
    )

    print(
        f"\nSaved turning proportions to:\n"
        f"{TURNING_PROPORTIONS_OUTPUT_PATH}"
    )

    print("\nMOVEMENT AUDIT COMPLETE")


if __name__ == "__main__":
    main()
