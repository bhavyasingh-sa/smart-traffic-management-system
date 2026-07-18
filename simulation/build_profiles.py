from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import (
    calculate_scale_limits,
    create_congestion_score,
    congestion_level,
)

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

PROFILE_PATH = (
    OUTPUT_DIRECTORY
    / "traffic_profiles.csv"
)

SELECTED_CITY = "Atlanta"

SELECTED_INTERSECTION_ID = 84

APPROACHES = [
    "N",
    "S",
    "E",
    "W",
]


def print_section(title):

    print(f"\n{title}")


def heading_to_approach(heading):
    """
    Convert the eight compass headings in the Geotab dataset
    into four main traffic approaches.

    Mapping:

        N, NE -> N
        S, SW -> S
        E, SE -> E
        W, NW -> W

    This gives the simulator four simplified approaches:
        North, South, East, West
    """

    heading = str(heading).strip().upper()

    mapping = {
        "N": "N",
        "NE": "N",

        "S": "S",
        "SW": "S",

        "E": "E",
        "SE": "E",

        "W": "W",
        "NW": "W",
    }

    return mapping.get(
        heading,
        "Unknown"
    )


def severity_to_arrival_rate(severity):
    """
    Convert historical congestion severity into an expected
    vehicle-arrival rate for the simulation.

    IMPORTANT:
    This is a simulation parameter. It is not a measured
    vehicle count from the Geotab dataset.

    Output:
        Expected vehicles arriving per simulation tick.

    Range:
        Approximately 0.8 to 6.0 vehicles per tick.
    """

    severity = float(
        np.clip(
            severity,
            0.0,
            1.0
        )
    )

    minimum_arrivals = 0.8
    maximum_arrivals = 6.0

    arrival_rate = (

        minimum_arrivals

        +

        severity
        * (
            maximum_arrivals
            - minimum_arrivals
        )
    )

    return float(arrival_rate)

print_section(
    "SMART TRAFFIC AI — HISTORICAL PROFILE BUILDER"
)

if not TRAIN_PATH.exists():

    raise FileNotFoundError(
        f"\ntrain.csv not found at:\n"
        f"{TRAIN_PATH}"
    )

print("\nLoading historical Geotab data...")

data = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Loaded {len(data):,} records."
)

print_section(
    "SELECTED INTERSECTION"
)

intersection_data = data[

    (data["City"] == SELECTED_CITY)

    &

    (
        data["IntersectionId"]
        == SELECTED_INTERSECTION_ID
    )

].copy()


print(
    f"\nCity: "
    f"{SELECTED_CITY}"
)

print(
    f"Intersection ID: "
    f"{SELECTED_INTERSECTION_ID}"
)

print(
    f"Historical records: "
    f"{len(intersection_data):,}"
)


if intersection_data.empty:

    raise ValueError(
        "\nNo historical records found "
        "for the selected intersection."
    )

# We calculate limits from the selected intersection itself
# because these profiles are specifically for the local
# traffic simulation.

scale_limits = calculate_scale_limits(
    intersection_data
)


print("\nLocal scale limits:")

for column, value in scale_limits.items():

    print(
        f"  {column}: "
        f"{value:.4f}"
    )

intersection_data[
    "HistoricalSeverity"
] = create_congestion_score(

    intersection_data,

    scale_limits,
)


intersection_data[
    "CongestionLevel"
] = intersection_data[

    "HistoricalSeverity"

].apply(
    congestion_level
)

intersection_data[
    "Approach"
] = intersection_data[

    "EntryHeading"

].apply(
    heading_to_approach
)


intersection_data = intersection_data[

    intersection_data["Approach"].isin(
        APPROACHES
    )

].copy()


print_section(
    "APPROACH DISTRIBUTION"
)


approach_counts = (

    intersection_data["Approach"]

    .value_counts()

    .reindex(
        APPROACHES,
        fill_value=0
    )
)


for approach in APPROACHES:

    print(
        f"{approach}: "
        f"{approach_counts[approach]:,} records"
    )

print_section(
    "BUILDING HOURLY TRAFFIC PROFILES"
)


profiles = (

    intersection_data

    .groupby(
        [
            "Hour",
            "Weekend",
            "Approach",
        ],
        as_index=False,
    )

    .agg(

        Records=(
            "RowId",
            "count"
        ),

        AverageSeverity=(
            "HistoricalSeverity",
            "mean"
        ),

        MedianSeverity=(
            "HistoricalSeverity",
            "median"
        ),

        SeverityStd=(
            "HistoricalSeverity",
            "std"
        ),

        AverageStoppedTime=(
            "TotalTimeStopped_p50",
            "mean"
        ),

        AverageDelay=(
            "TimeFromFirstStop_p50",
            "mean"
        ),

        AverageDistanceToStop=(
            "DistanceToFirstStop_p50",
            "mean"
        ),
    )
)

profiles[
    "SeverityStd"
] = profiles[
    "SeverityStd"
].fillna(0.0)

profiles[
    "ExpectedArrivalRate"
] = profiles[

    "AverageSeverity"

].apply(
    severity_to_arrival_rate
)

profiles[
    "CongestionLevel"
] = profiles[

    "AverageSeverity"

].apply(
    congestion_level
)

profiles = profiles.sort_values(

    by=[
        "Weekend",
        "Hour",
        "Approach",
    ]

).reset_index(
    drop=True
)

columns_to_round = [

    "AverageSeverity",
    "MedianSeverity",
    "SeverityStd",
    "AverageStoppedTime",
    "AverageDelay",
    "AverageDistanceToStop",
    "ExpectedArrivalRate",
]


for column in columns_to_round:

    profiles[column] = (

        profiles[column]

        .round(4)
    )

profiles.to_csv(

    PROFILE_PATH,

    index=False,
)

print(
    f"\nCreated "
    f"{len(profiles):,} historical profiles."
)


print(
    "\nProfile dimensions:"
)


print(
    f"  Hours: "
    f"{profiles['Hour'].nunique()}"
)


print(
    f"  Weekend states: "
    f"{profiles['Weekend'].nunique()}"
)


print(
    f"  Approaches: "
    f"{profiles['Approach'].nunique()}"
)

print_section(
    "EXAMPLE — WEEKDAY AT 8 AM"
)


example = profiles[

    (profiles["Hour"] == 8)

    &

    (profiles["Weekend"] == 0)

].copy()


if example.empty:

    print(
        "\nNo weekday 8 AM profiles found."
    )

else:

    display_columns = [

        "Approach",
        "Records",
        "AverageSeverity",
        "CongestionLevel",
        "ExpectedArrivalRate",
        "AverageStoppedTime",
    ]


    print(
        "\n"
        + example[
            display_columns
        ].to_string(
            index=False
        )
    )

print_section(
    "TOP 10 HIGHEST-CONGESTION PROFILES"
)


top_profiles = (

    profiles

    .sort_values(

        by="AverageSeverity",

        ascending=False,

    )

    .head(10)
)


display_columns = [

    "Hour",
    "Weekend",
    "Approach",
    "Records",
    "AverageSeverity",
    "CongestionLevel",
    "ExpectedArrivalRate",
]


print(
    "\n"
    + top_profiles[
        display_columns
    ].to_string(
        index=False
    )
)

print_section(
    "HISTORICAL PROFILE BUILD COMPLETE"
)


print(
    "\nTraffic profiles saved successfully to:"
)


print(
    PROFILE_PATH
)


print(
    "\nNext step:"
)


print(
    "Build the traffic simulation engine that uses "
    "these historically informed arrival rates to "
    "generate changing N/S/E/W traffic queues."
)