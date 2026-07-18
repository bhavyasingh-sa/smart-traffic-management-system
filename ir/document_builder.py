import pandas as pd


MONTH_NAMES = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}


HEADING_NAMES = {
    "N": "north",
    "NE": "northeast",
    "E": "east",
    "SE": "southeast",
    "S": "south",
    "SW": "southwest",
    "W": "west",
    "NW": "northwest",
}


def get_time_period(hour):
    """
    Convert hour into a human-readable time period.
    """

    hour = int(hour)

    if 5 <= hour < 12:
        return "morning"

    if 12 <= hour < 17:
        return "afternoon"

    if 17 <= hour < 21:
        return "evening"

    return "night"


def get_congestion_level(
    stopped_time,
    delay,
    distance,
):
    """
    Create a descriptive historical congestion level.

    This function is used only as a fallback when the input
    row does not already contain a CongestionLevel value.

    It is NOT an ML prediction.
    """

    stopped_time = float(stopped_time)
    delay = float(delay)
    distance = float(distance)

    score = (
        min(stopped_time / 60.0, 1.0) * 0.40
        + min(delay / 120.0, 1.0) * 0.35
        + min(distance / 200.0, 1.0) * 0.25
    )

    if score < 0.20:
        return "low"

    if score < 0.40:
        return "moderate"

    if score < 0.65:
        return "high"

    return "severe"


def safe_street_name(value):
    """
    Handle missing street names.
    """

    if pd.isna(value):
        return "unknown road"

    value = str(value).strip()

    if not value:
        return "unknown road"

    return value


def build_document(row):
    """
    Convert one structured Geotab traffic row into a
    searchable text document.

    If CongestionLevel has already been calculated by the
    IR index builder, that exact label is used so document
    text and metadata remain consistent.
    """

    city = str(
        row["City"]
    ).lower()

    intersection_id = int(
        row["IntersectionId"]
    )

    hour = int(
        row["Hour"]
    )

    month_number = int(
        row["Month"]
    )

    month = MONTH_NAMES.get(
        month_number,
        f"month {month_number}",
    )

    day_type = (
        "weekend"
        if int(row["Weekend"]) == 1
        else "weekday"
    )

    time_period = get_time_period(
        hour
    )

    entry_heading_code = str(
        row["EntryHeading"]
    ).upper()

    exit_heading_code = str(
        row["ExitHeading"]
    ).upper()

    entry_heading = HEADING_NAMES.get(
        entry_heading_code,
        entry_heading_code.lower(),
    )

    exit_heading = HEADING_NAMES.get(
        exit_heading_code,
        exit_heading_code.lower(),
    )

    entry_street = safe_street_name(
        row["EntryStreetName"]
    )

    exit_street = safe_street_name(
        row["ExitStreetName"]
    )

    stopped_time = float(
        row["TotalTimeStopped_p50"]
    )

    delay = float(
        row["TimeFromFirstStop_p50"]
    )

    distance = float(
        row["DistanceToFirstStop_p50"]
    )

    if (
        "CongestionLevel" in row.index
        and pd.notna(
            row["CongestionLevel"]
        )
    ):
        congestion = str(
            row["CongestionLevel"]
        ).lower()

    else:
        congestion = get_congestion_level(
            stopped_time=stopped_time,
            delay=delay,
            distance=distance,
        )

    historical_severity = None

    if (
        "HistoricalSeverity" in row.index
        and pd.notna(
            row["HistoricalSeverity"]
        )
    ):
        historical_severity = float(
            row["HistoricalSeverity"]
        )

    document_parts = [
        city,
        f"intersection {intersection_id}",
        day_type,
        month,
        f"hour {hour}",
        time_period,
        f"{entry_heading} approach",
        f"entry {entry_heading}",
        f"exit {exit_heading}",
        f"from {entry_street}",
        f"to {exit_street}",
        f"{congestion} congestion",
    ]

    if historical_severity is not None:
        document_parts.append(
            f"historical severity "
            f"{historical_severity:.4f}"
        )

    document_parts.extend(
        [
            (
                f"stopped time "
                f"{stopped_time:.1f} seconds"
            ),
            (
                f"delay "
                f"{delay:.1f} seconds"
            ),
            (
                f"traffic distance "
                f"{distance:.1f} meters"
            ),
        ]
    )

    document = " ".join(
        document_parts
    )

    return document


def build_documents(dataframe):
    """
    Build searchable text documents for an entire DataFrame.
    """

    documents = []

    for _, row in dataframe.iterrows():

        document = build_document(
            row
        )

        documents.append(
            document
        )

    return documents