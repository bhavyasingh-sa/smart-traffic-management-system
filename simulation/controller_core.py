"""controller_core.py - stateless decision logic (priority formulas, phase rules, ML/IR loaders) for the movement-aware controller; owns no per-tick state, that lives in adaptive_simulator.py."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ml.features import prepare_features, probabilities_to_severity

from simulation.movement_definitions import (
    MOVEMENT_IDS,
    TRAVEL_DIRECTIONS,
    MOVEMENT_TYPES,
    DIRECTION_MOVEMENTS,
    PHYSICAL_APPROACH_SIDE,
    PHASES,
    PHASE_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ml"
    / "congestion_classifier.pkl"
)

IR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ir"
    / "traffic_ir.pkl"
)

TURNING_PROPORTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "turning_proportions.csv"
)

TRAFFIC_PROFILES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "traffic_profiles.csv"
)

# V1's ARRIVAL_RATES (0.4092/0.2654/0.3878/0.5452 vehicles/sec for
# N/S/E/W) were traced (not guessed) to exactly:
#     traffic_profiles.csv's ExpectedArrivalRate(Hour=8, Weekend=0)
#     / ARRIVAL_RATE_SCALING_DIVISOR
# confirmed to 4 decimal places against V1's live source. This
# divisor reproduces that exact, already-existing methodology.
ARRIVAL_RATE_SCALING_DIVISOR = 10.0

CITY = "Atlanta"
INTERSECTION_ID = 84

SIMULATION_HOUR = 8
WEEKEND = 0
MONTH = 6

SIMULATION_DURATION = 300

# Unchanged from V1 — not modified without approval.
MIN_GREEN = 20
MAX_GREEN = 60
YELLOW_DURATION = 3

SERVICE_RATE = 1.0

# Unchanged from V1 — preserved exactly, per requirement.
QUEUE_WEIGHT = 0.45
WAIT_WEIGHT = 0.20
ML_WEIGHT = 0.15
IR_WEIGHT = 0.10
STARVATION_WEIGHT = 0.10

#
# The 45/20/15/10/10 weighted formula above is UNCHANGED - this is
# an ADDITIONAL fairness constraint layered on top of it, not a
# replacement. The Stage 3 stress test proved that under sustained
# heavy competing demand, the 10%-weighted starvation component
# alone does not guarantee bounded waiting (a phase was starved for
# 5,859 of 6,000 ticks). This safeguard forces a switch to a
# starving movement's phase once its wait exceeds a hard ceiling,
# but ONLY at the earliest SAFE opportunity (after MIN_GREEN, still
# through YELLOW, never a direct GREEN->GREEN change).
#
# SELECTED VALUE: 120 seconds, chosen from a 5-way sensitivity
# analysis (None/60/90/120/180) across 3 scenarios (normal
# 300-tick, long-run 3000-tick, heavy-demand 6000-tick stress),
# each run under an IDENTICAL deterministic traffic schedule per
# scenario (same seed, same arrival overrides across all 5
# configs - verified by comparing movement_total_arrivals across
# runs). Full results: analysis/starvation_sensitivity_analysis.py
# and analysis/aggregate_sensitivity_results.py.
#
# Why 120s and not another tested value:
#   - 60s is too aggressive: under the heavy-demand stress
#     scenario it produced the WORST throughput (31.7%, worse
#     than no override at all) and WORST max completed wait
#     (4886s, worse than doing nothing) because 247/250 switches
#     became forced overrides - the controller degenerates into a
#     rigid round-robin, discarding almost all of its weighted
#     intelligence. It also intervened 8 times even under the
#     NORMAL 300-tick scenario, where no intervention should be
#     needed.
#   - 180s is too lax: under the stress scenario it left one
#     movement (NB_LEFT) with 210 vehicles never served by the end
#     of the run - it does not reliably bound starvation.
#   - 90s and 120s both fully resolved starvation under stress
#     (0 vehicles left unserved in the target phase) while staying
#     completely silent under normal/long-run conditions (0-1
#     overrides). 120s edged out 90s on throughput (37.9% vs
#     36.0%) and max completed wait (4532s vs 4646s) under stress,
#     at the cost of a higher (but still bounded and far smaller
#     than uncontrolled) max starvation (147 vs 123 ticks).
#
# None still means "disabled" if explicitly passed - the
# AdaptiveTrafficSimulation constructor accepts max_starvation_time
# as an override for testing (see tests/test_starvation_safeguard.py).
MAX_STARVATION_TIME = 120

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

CONGESTION_SEVERITY = {
    "low": 0.10,
    "moderate": 0.30,
    "high": 0.55,
    "severe": 0.85,
}

HEADING_NAMES = {
    "N": "north", "S": "south", "E": "east", "W": "west",
}

# Explicit, documented deterministic tie-break order. This exists
# specifically so that phase selection never depends on incidental
# Python dict/set ordering — ties are broken by this list's order,
# always, and this is the only place that order is defined.
PHASE_TIE_BREAK_ORDER = [
    "NS_THROUGH",
    "NS_LEFT",
    "EW_THROUGH",
    "EW_LEFT",
]

# Priority values this close are treated as an exact tie for the
# purpose of PHASE_TIE_BREAK_ORDER. Real priority scores are
# continuous weighted sums, so exact float ties are effectively
# impossible outside of deliberately constructed test scenarios —
# this epsilon exists for those tests, not because it's expected
# to matter in a real run.
TIE_BREAK_EPSILON = 1e-9

# Verified mapping (see analysis/movement_audit.py and the Phase 1
# audit): a travel direction's EntryHeading is simply its own
# compass letter. This is the ONLY place this mapping is defined —
# every other function in this file goes through it.
ENTRY_HEADING_LETTER = {
    "NB": "N",
    "SB": "S",
    "EB": "E",
    "WB": "W",
}

# Real street names for Atlanta Intersection 84's straight-through
# movements (EntryHeading == ExitHeading), read directly from
# data/raw/train.csv - not invented. "Unknown" never appears in the
# training data for EntryStreetName/ExitStreetName (verified: 0
# occurrences), so passing "Unknown" at inference feeds the model's
# OneHotEncoder(handle_unknown="ignore") an all-zero vector for 2 of
# its 4 categorical features on every single prediction - real
# signal the model was trained on but never actually receives.
STRAIGHT_THROUGH_STREET_NAMES = {
    "NB": (
        "Cheshire Bridge Road",
        "Cheshire Bridge Road",
    ),
    "SB": (
        "Cheshire Bridge Road",
        "Cheshire Bridge Road",
    ),
    "EB": (
        "Lindbergh Drive Northeast",
        "Lavista Road Northeast",
    ),
    "WB": (
        "Lavista Road Northeast",
        "Lindbergh Drive Northeast",
    ),
}

# Reverse lookup: which phase serves a given movement.
MOVEMENT_TO_PHASE = {
    movement_id: phase_name
    for phase_name, movement_list in PHASES.items()
    for movement_id in movement_list
}

HEADING_DEGREES = {
    "N": 0, "E": 90, "S": 180, "W": 270,
}

DEGREES_TO_HEADING = {
    degrees: letter
    for letter, degrees in HEADING_DEGREES.items()
}


def parse_movement_id(movement_id):
    """
    Split "NB_LEFT" into ("NB", "LEFT"). This is the single place
    that parsing happens, so every caller agrees on the format.
    """

    direction, movement_type = movement_id.split("_", 1)

    return direction, movement_type


def derive_exit_heading(entry_heading_letter, movement_type):
    """
    Derive the real exit heading letter for a movement, using the
    same compass-degree math as analysis/movement_audit.py's
    classify_movement() (inverted: given entry heading + movement
    type, what exit heading does that imply).

    STRAIGHT: delta 0     LEFT: delta 270     RIGHT: delta 90

    This was verified against all 12 real rows in the Phase 1 audit
    (e.g. NB_LEFT: N=0 -> (0+270)%360=270=W, matching the real
    Cheshire Bridge Rd N -> Lindbergh Dr NE W record).
    """

    entry_degrees = HEADING_DEGREES[
        entry_heading_letter
    ]

    if movement_type == "STRAIGHT":
        delta = 0

    elif movement_type == "LEFT":
        delta = 270

    elif movement_type == "RIGHT":
        delta = 90

    else:
        raise ValueError(
            f"Unknown movement_type: {movement_type}"
        )

    exit_degrees = (
        entry_degrees + delta
    ) % 360

    return DEGREES_TO_HEADING[exit_degrees]

# Loaded from disk, not duplicated as literals anywhere in this
# project — data/processed/turning_proportions.csv is generated by
# analysis/movement_audit.py directly from the raw dataset.

_turning_proportions_cache = None


def load_turning_proportions():
    """
    Load the real historical turning proportions, generated
    reproducibly by analysis/movement_audit.py.

    Returns: {movement_id: proportion} for all 12 movements,
    where the 3 movements of each travel direction sum to 1.0.
    """

    global _turning_proportions_cache

    if _turning_proportions_cache is not None:
        return _turning_proportions_cache

    if not TURNING_PROPORTIONS_PATH.exists():

        raise FileNotFoundError(
            "\nTurning proportions file not found.\n\n"
            f"Expected path:\n{TURNING_PROPORTIONS_PATH}\n\n"
            "Run analysis/movement_audit.py first to generate it "
            "reproducibly from data/raw/train.csv."
        )

    proportions_df = pd.read_csv(
        TURNING_PROPORTIONS_PATH
    )

    proportions = {
        row["movement_id"]: float(row["proportion"])
        for _, row in proportions_df.iterrows()
    }

    missing = set(MOVEMENT_IDS) - set(proportions.keys())

    if missing:

        raise ValueError(
            "turning_proportions.csv is missing movements: "
            f"{sorted(missing)}. Re-run analysis/movement_audit.py."
        )

    # Sanity check: each direction's 3 movements should sum to
    # ~1.0 (they're proportions of that direction's real traffic).
    for direction in TRAVEL_DIRECTIONS:

        direction_total = sum(
            proportions[movement_id]
            for movement_id in DIRECTION_MOVEMENTS[direction]
        )

        if abs(direction_total - 1.0) > 0.01:

            raise ValueError(
                f"Turning proportions for {direction} sum to "
                f"{direction_total:.4f}, expected ~1.0. Check "
                "turning_proportions.csv."
            )

    _turning_proportions_cache = proportions

    return proportions


def load_arrival_rates(hour=SIMULATION_HOUR, weekend=WEEKEND):
    """
    Load real, historically-derived arrival rates for the 4 travel
    directions (NB/SB/EB/WB), by reading traffic_profiles.csv
    (built by simulation/build_profiles.py directly from the raw
    dataset) and applying V1's exact, traced scaling.

    Provenance (traced, not assumed):
      traffic_profiles.csv's "Approach" column is built by
      build_profiles.py's heading_to_approach(EntryHeading) - i.e.
      it is ALREADY keyed by travel direction (EntryHeading), the
      same convention this project verified and uses throughout.
      So Approach "N" maps directly to NB, "S" to SB, "E" to EB,
      "W" to WB - no swap needed, unlike the turning-proportions
      mislabeling found during Stage 1/2.

      ExpectedArrivalRate itself comes from real historical
      congestion severity (via severity_to_arrival_rate()), NOT a
      fabricated number. Dividing by ARRIVAL_RATE_SCALING_DIVISOR
      reproduces V1's exact hardcoded constants (verified to 4
      decimal places): N/NB=0.4092, S/SB=0.2654, E/EB=0.3878,
      W/WB=0.5452 vehicles/second at Hour=8, Weekend=0.
    """

    if not TRAFFIC_PROFILES_PATH.exists():

        raise FileNotFoundError(
            "\ntraffic_profiles.csv not found.\n\n"
            f"Expected path:\n{TRAFFIC_PROFILES_PATH}\n\n"
            "Run simulation/build_profiles.py first to generate it "
            "reproducibly from data/raw/train.csv."
        )

    profiles = pd.read_csv(TRAFFIC_PROFILES_PATH)

    matching_rows = profiles[
        (profiles["Hour"] == hour)
        & (profiles["Weekend"] == weekend)
    ]

    approach_to_direction = {
        "N": "NB", "S": "SB", "E": "EB", "W": "WB",
    }

    rates = {}

    for _, row in matching_rows.iterrows():

        direction = approach_to_direction.get(
            row["Approach"]
        )

        if direction is None:
            continue

        rates[direction] = float(
            row["ExpectedArrivalRate"]
            / ARRIVAL_RATE_SCALING_DIVISOR
        )

    missing = set(TRAVEL_DIRECTIONS) - set(rates.keys())

    if missing:

        raise ValueError(
            f"No arrival-rate profile found for directions "
            f"{sorted(missing)} at hour={hour}, weekend={weekend}. "
            "Check traffic_profiles.csv."
        )

    return rates


def load_ml_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nML model not found.\n\n"
            f"Expected path:\n{MODEL_PATH}\n\n"
            "Train the ML model first."
        )

    print("\nLoading ML model...")

    with open(MODEL_PATH, "rb") as file:
        model_bundle = pickle.load(file)

    print("ML model loaded successfully.")
    print(f"  Name: {model_bundle['model_name']}")
    print(f"  Scope: {model_bundle['model_scope']}")

    return model_bundle


def build_ml_input(direction, hour, weekend, month):
    """
    Build the synthetic inference row for one travel direction.

    Corrected convention (verified in the Phase 1/2 audit): a
    straight-through movement keeps the SAME heading at entry and
    exit, so this uses ENTRY_HEADING_LETTER[direction] for BOTH
    EntryHeading and ExitHeading:

        NB -> EntryHeading N, ExitHeading N
        SB -> EntryHeading S, ExitHeading S
        EB -> EntryHeading E, ExitHeading E
        WB -> EntryHeading W, ExitHeading W

    V1 used the opposite letter for ExitHeading (e.g. N -> S), which
    the audit showed represents a U-turn under the dataset's real
    convention and was never seen in training. V2 does not repeat
    that mismatch.

    Street names use STRAIGHT_THROUGH_STREET_NAMES (the real names
    for this intersection's straight movements, read from
    data/raw/train.csv) rather than "Unknown" - "Unknown" never
    appears in training data, so it fed the model's
    OneHotEncoder(handle_unknown="ignore") an all-zero vector for 2
    of its 4 categorical features on every prediction, discarding
    real signal the model was trained on.
    """

    heading_letter = ENTRY_HEADING_LETTER[direction]

    entry_street, exit_street = (
        STRAIGHT_THROUGH_STREET_NAMES[direction]
    )

    return pd.DataFrame([{
        "City": CITY,
        "IntersectionId": INTERSECTION_ID,
        "EntryStreetName": entry_street,
        "ExitStreetName": exit_street,
        "EntryHeading": heading_letter,
        "ExitHeading": heading_letter,
        "Hour": hour,
        "Weekend": weekend,
        "Month": month,
    }])


def build_ml_predictions(model_bundle, hour, weekend, month):
    """
    One ML severity prediction PER TRAVEL DIRECTION (4 total:
    NB, SB, EB, WB). This is deliberately NOT 12 predictions —
    see the module docstring (Option C).
    """

    pipeline = model_bundle["pipeline"]
    classes = model_bundle["classes"]
    model_scope = model_bundle["model_scope"]

    predictions = {}

    for direction in TRAVEL_DIRECTIONS:

        input_data = build_ml_input(
            direction=direction,
            hour=hour,
            weekend=weekend,
            month=month,
        )

        features = prepare_features(
            input_data,
            model_scope=model_scope,
        )

        predicted_class = pipeline.predict(features)[0]
        probabilities = pipeline.predict_proba(features)[0]

        severity = probabilities_to_severity(
            probabilities=probabilities,
            classes=classes,
        )

        predictions[direction] = {
            "class": predicted_class,
            "severity": float(severity),
        }

    return predictions


def load_ir_retriever():

    if not IR_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nIR model not found.\n\n"
            f"Expected path:\n{IR_MODEL_PATH}\n\n"
            "Build the IR index first."
        )

    print("\nLoading IR retriever...")

    with open(IR_MODEL_PATH, "rb") as file:
        retriever = pickle.load(file)

    print("IR retriever loaded successfully.")

    return retriever


def build_ir_query(movement_id, hour, weekend, month):
    """
    Build a movement-aware free-text IR query.

    IMPORTANT (traced against the actual retriever, not assumed):
    retriever.search() has no structured "exit heading" or
    "movement type" filter parameter — it only structurally
    filters/boosts on a single `approach` value (matched against
    each document's EntryHeading). Movement-level differentiation
    instead comes from the FREE-TEXT query matching the real
    "entry {heading} exit {heading}" words that every indexed
    document already contains (verified directly from the index).
    This query embeds the real entry AND exit heading so TF-IDF
    similarity favors documents describing the same real movement.
    """

    direction, movement_type = parse_movement_id(movement_id)

    entry_letter = ENTRY_HEADING_LETTER[direction]
    exit_letter = derive_exit_heading(
        entry_letter,
        movement_type,
    )

    entry_heading_word = HEADING_NAMES[entry_letter]
    exit_heading_word = HEADING_NAMES[exit_letter]

    day_type = "weekend" if weekend == 1 else "weekday"
    month_text = MONTH_NAMES.get(int(month), f"month {month}")

    movement_word = {
        "STRAIGHT": "straight",
        "LEFT": "left turn",
        "RIGHT": "right turn",
    }[movement_type]

    return (
        f"{CITY.lower()} intersection {INTERSECTION_ID} "
        f"{day_type} {month_text} hour {hour} "
        f"entry {entry_heading_word} exit {exit_heading_word} "
        f"{movement_word}"
    )


def extract_historical_severity(result):
    """
    Identical logic to V1's extract_historical_severity — unchanged,
    since the audit found no issue with this function (it never
    touches EntryHeading/ExitHeading at all).
    """

    metadata = result.get("metadata", {})

    numeric_keys = [
        "HistoricalSeverity", "historical_severity",
        "AverageSeverity", "average_severity",
        "CongestionScore", "congestion_score",
        "Severity", "severity",
    ]

    for key in numeric_keys:

        value = metadata.get(key)

        if value is None:
            continue

        try:
            return float(np.clip(float(value), 0.0, 1.0))

        except (TypeError, ValueError):
            continue

    level_keys = [
        "CongestionLevel", "congestion_level",
        "Congestion", "congestion",
        "TrafficLevel", "traffic_level",
        "Level", "level",
    ]

    for key in level_keys:

        value = metadata.get(key)

        if value is None:
            continue

        normalized = str(value).strip().lower()

        if normalized in CONGESTION_SEVERITY:
            return CONGESTION_SEVERITY[normalized]

    document = str(result.get("document", "")).lower()

    for level in ["severe", "high", "moderate", "low"]:

        if f"{level} congestion" in document:
            return CONGESTION_SEVERITY[level]

    return None


def build_ir_predictions(
    retriever,
    hour,
    weekend,
    month,
    ml_predictions=None,
    top_k=5,
):
    """
    One IR severity prediction PER MOVEMENT (12 total), each
    grounded in real retrieved historical documents for Atlanta
    Intersection 84 (city/intersection_id are hard filters —
    verified against the actual retriever code).
    """

    predictions = {}

    for movement_id in MOVEMENT_IDS:

        direction, _movement_type = parse_movement_id(
            movement_id
        )

        query = build_ir_query(
            movement_id=movement_id,
            hour=hour,
            weekend=weekend,
            month=month,
        )

        entry_letter = ENTRY_HEADING_LETTER[direction]

        target_severity = None

        if (
            ml_predictions is not None
            and direction in ml_predictions
        ):
            target_severity = float(
                ml_predictions[direction]["severity"]
            )

        try:
            results = retriever.search(
                query=query,
                top_k=top_k,
                city=CITY,
                intersection_id=INTERSECTION_ID,
                approach=entry_letter,
                hour=hour,
                weekend=weekend,
                month=month,
                target_severity=target_severity,
                strict_context=False,
                candidate_limit=500,
            )

        except TypeError:

            results = retriever.search(
                query=query,
                top_k=top_k,
            )

        severity_values = []
        similarity_weights = []
        usable_cases = 0

        for result in results:

            severity = extract_historical_severity(result)

            if severity is None:
                continue

            similarity = float(result.get("score", 0.0))

            if similarity <= 0:
                continue

            severity_values.append(severity)
            similarity_weights.append(max(similarity, 0.000001))
            usable_cases += 1

        if severity_values:

            ir_severity = float(
                np.average(
                    severity_values,
                    weights=similarity_weights,
                )
            )

        else:

            ir_severity = 0.50

        predictions[movement_id] = {
            "severity": float(ir_severity),
            "query": query,
            "retrieved_cases": len(results),
            "usable_cases": usable_cases,
            "top_results": results,
        }

    return predictions


def get_average_wait(queue, current_tick):

    if not queue:
        return 0.0

    waits = [current_tick - arrival_tick for arrival_tick in queue]

    return float(np.mean(waits))


def calculate_movement_priority(
    movement_id,
    movement_queues,
    current_tick,
    ml_predictions,
    starvation,
    ir_predictions=None,
):
    """
    Movement Priority =
        45% movement-specific live queue
      + 20% movement-specific waiting time
      + 15% PARENT direction's ML severity  <- approach-level, not per-movement
      + 10% movement-specific IR severity
      + 10% movement-specific starvation

    ir_predictions=None runs an ML-only mode (IR's 10% weight
    redistributed proportionally across the other four), mirroring
    V1's design, kept here for a future ablation comparison (Stage 4)
    without duplicating this formula a second time.
    """

    direction, _movement_type = parse_movement_id(movement_id)

    queue_length = len(movement_queues[movement_id])

    average_wait = get_average_wait(
        movement_queues[movement_id],
        current_tick,
    )

    queue_score = min(queue_length / 10.0, 1.0)
    wait_score = min(average_wait / 60.0, 1.0)

    # ML is approach-level: every movement of a direction shares
    # the SAME ml_score, taken from its parent direction.
    ml_score = float(ml_predictions[direction]["severity"])

    starvation_score = min(
        starvation[movement_id] / 60.0,
        1.0,
    )

    if ir_predictions is None:

        total_base_weight = (
            QUEUE_WEIGHT + WAIT_WEIGHT + ML_WEIGHT + STARVATION_WEIGHT
        )

        priority = (
            (QUEUE_WEIGHT / total_base_weight) * queue_score
            + (WAIT_WEIGHT / total_base_weight) * wait_score
            + (ML_WEIGHT / total_base_weight) * ml_score
            + (STARVATION_WEIGHT / total_base_weight) * starvation_score
        )

    else:

        ir_score = float(
            ir_predictions[movement_id]["severity"]
        )

        priority = (
            QUEUE_WEIGHT * queue_score
            + WAIT_WEIGHT * wait_score
            + ML_WEIGHT * ml_score
            + IR_WEIGHT * ir_score
            + STARVATION_WEIGHT * starvation_score
        )

    return float(priority)


def calculate_movement_priority_breakdown(
    movement_id,
    movement_queues,
    current_tick,
    ml_predictions,
    starvation,
    ir_predictions=None,
):
    """
    Same formula as calculate_movement_priority(), but returns the
    5 weighted contributions separately instead of only the total.
    New in Stage 3.5, purely additive - does not change
    calculate_movement_priority()'s behavior or signature.

    Exists so the future dashboard/RAG layer can show exactly how
    much each of queue/wait/ML/IR/starvation contributed to a
    movement's priority, per the Stage 3.5 dashboard-facing
    requirements.
    """

    direction, _movement_type = parse_movement_id(movement_id)

    queue_length = len(movement_queues[movement_id])

    average_wait = get_average_wait(
        movement_queues[movement_id],
        current_tick,
    )

    queue_score = min(queue_length / 10.0, 1.0)
    wait_score = min(average_wait / 60.0, 1.0)
    ml_score = float(ml_predictions[direction]["severity"])
    starvation_score = min(
        starvation[movement_id] / 60.0,
        1.0,
    )

    if ir_predictions is None:

        total_base_weight = (
            QUEUE_WEIGHT + WAIT_WEIGHT + ML_WEIGHT + STARVATION_WEIGHT
        )

        queue_contribution = (
            QUEUE_WEIGHT / total_base_weight
        ) * queue_score

        wait_contribution = (
            WAIT_WEIGHT / total_base_weight
        ) * wait_score

        ml_contribution = (
            ML_WEIGHT / total_base_weight
        ) * ml_score

        ir_contribution = 0.0

        starvation_contribution = (
            STARVATION_WEIGHT / total_base_weight
        ) * starvation_score

    else:

        ir_score = float(
            ir_predictions[movement_id]["severity"]
        )

        queue_contribution = QUEUE_WEIGHT * queue_score
        wait_contribution = WAIT_WEIGHT * wait_score
        ml_contribution = ML_WEIGHT * ml_score
        ir_contribution = IR_WEIGHT * ir_score
        starvation_contribution = (
            STARVATION_WEIGHT * starvation_score
        )

    total = (
        queue_contribution
        + wait_contribution
        + ml_contribution
        + ir_contribution
        + starvation_contribution
    )

    return {
        "queue_contribution": float(queue_contribution),
        "wait_contribution": float(wait_contribution),
        "ml_contribution": float(ml_contribution),
        "ir_contribution": float(ir_contribution),
        "starvation_contribution": float(
            starvation_contribution
        ),
        "total": float(total),
    }


def calculate_phase_priority(
    phase_name,
    movement_queues,
    current_tick,
    ml_predictions,
    starvation,
    ir_predictions=None,
):
    """
    Phase priority = ARITHMETIC MEAN of the priorities of the
    movements that phase serves.

    Why mean, not sum or max (explicit, as required):
      - MEAN keeps every phase's priority on the same 0-1 scale
        regardless of how many movements it serves. NS_THROUGH
        serves 4 movements, NS_LEFT serves 2 - summing would
        structurally bias every decision toward the 4-movement
        phases regardless of actual congestion.
      - MAX would let a single congested movement force a switch
        even when the rest of its phase is empty, and would ignore
        the phase's overall demand - more volatile, harder to
        reason about in a viva.
      - MEAN is also what V1 used (np.mean across a phase's
        approaches), so this preserves V1's conceptual behavior
        exactly, just applied to however many movements a phase
        now serves.
    """

    priorities = [
        calculate_movement_priority(
            movement_id=movement_id,
            movement_queues=movement_queues,
            current_tick=current_tick,
            ml_predictions=ml_predictions,
            starvation=starvation,
            ir_predictions=ir_predictions,
        )
        for movement_id in PHASES[phase_name]
    ]

    return float(np.mean(priorities))


def _phase_total_queue(phase_name, movement_queues):

    return sum(
        len(movement_queues[movement_id])
        for movement_id in PHASES[phase_name]
    )


def _pick_best_phase(candidate_phases, phase_priorities):
    """
    Deterministic tie-break: among candidate_phases, return the one
    with the highest priority. If two or more are tied within
    TIE_BREAK_EPSILON, return whichever of them appears first in
    PHASE_TIE_BREAK_ORDER — an explicit, documented constant, never
    incidental dict/set ordering.
    """

    best_priority = max(
        phase_priorities[phase_name]
        for phase_name in candidate_phases
    )

    tied_phases = [
        phase_name
        for phase_name in candidate_phases
        if abs(phase_priorities[phase_name] - best_priority)
        <= TIE_BREAK_EPSILON
    ]

    for phase_name in PHASE_TIE_BREAK_ORDER:

        if phase_name in tied_phases:
            return phase_name

    # Should be unreachable if PHASE_TIE_BREAK_ORDER covers all
    # phases (it does — see the sanity check at the bottom of this
    # file), but fail loudly rather than silently if it ever isn't.
    raise RuntimeError(
        "No tie-break candidate found — "
        "PHASE_TIE_BREAK_ORDER may be incomplete."
    )


def check_starvation_override(
    current_phase,
    movement_queues,
    starvation,
    phase_priorities,
    max_starvation_time,
):
    """
    Stage 3.5 fairness safeguard. Independent of, and layered on
    top of, the 45/20/15/10/10 weighted formula (unchanged).

    If any movement NOT in current_phase has both:
      (a) a non-empty queue, and
      (b) starvation >= max_starvation_time,
    then its phase becomes MANDATORY at the earliest safe
    opportunity (caller still enforces MIN_GREEN before calling
    this - see should_switch_phase).

    Deterministic tie-break when multiple phases qualify:
      1. Highest max movement starvation among that phase's
         qualifying movements.
      2. Then highest phase priority.
      3. Then PHASE_TIE_BREAK_ORDER (explicit, never dict order).

    Returns (should_override, target_phase) - (False, None) if
    max_starvation_time is None (safeguard disabled) or nothing
    qualifies.
    """

    if max_starvation_time is None:
        return False, None

    phase_max_starvation = {}

    for movement_id in MOVEMENT_IDS:

        phase_name = MOVEMENT_TO_PHASE[movement_id]

        if phase_name == current_phase:
            continue

        has_queue = len(movement_queues[movement_id]) > 0

        is_over_threshold = (
            starvation[movement_id] >= max_starvation_time
        )

        if has_queue and is_over_threshold:

            phase_max_starvation[phase_name] = max(
                phase_max_starvation.get(phase_name, 0),
                starvation[movement_id],
            )

    if not phase_max_starvation:
        return False, None

    qualifying_phases = list(phase_max_starvation.keys())

    def sort_key(phase_name):

        return (
            -phase_max_starvation[phase_name],
            -phase_priorities[phase_name],
            PHASE_TIE_BREAK_ORDER.index(phase_name),
        )

    target_phase = sorted(
        qualifying_phases,
        key=sort_key,
    )[0]

    return True, target_phase


def should_switch_phase(
    current_phase,
    green_elapsed,
    movement_queues,
    current_tick,
    ml_predictions,
    starvation,
    ir_predictions=None,
    max_starvation_time=MAX_STARVATION_TIME,
):
    """
    Decide whether to switch away from current_phase right now.

    Returns (should_switch, reason, target_phase, details):
      - should_switch: bool
      - reason: "starvation_override", "current_phase_empty",
        "maximum_green_reached", "competing_phase_higher_priority",
        or None if holding.
      - target_phase: the phase name to switch TO, or None if holding.
      - details: {phase_name: priority, ...} for ALL 4 phases, for
        display/logging.

    Starvation handling (two layers, both real, neither replacing
    the other):
      1. The weighted formula's STARVATION_WEIGHT=10% component
         (unchanged since Stage 2) - a soft, continuous influence.
      2. This function's max_starvation_time hard safeguard (new in
         Stage 3.5, default None=disabled) - checked immediately
         after the MIN_GREEN gate, so it takes precedence over the
         normal current_phase_empty/max_green/competing-priority
         checks below whenever it fires. It can ONLY fire when
         signal_state == "GREEN" (the only state this function is
         ever called in), so it can never interrupt YELLOW.
    """

    # Compute every phase's priority regardless of the path taken
    # below, so `details` is always complete for display/logging.
    details = {
        phase_name: calculate_phase_priority(
            phase_name=phase_name,
            movement_queues=movement_queues,
            current_tick=current_tick,
            ml_predictions=ml_predictions,
            starvation=starvation,
            ir_predictions=ir_predictions,
        )
        for phase_name in PHASE_NAMES
    }

    if green_elapsed < MIN_GREEN:

        # MIN_GREEN safety: no exception, not even for the
        # starvation safeguard. Hold unconditionally.
        return False, None, None, details

    should_override, override_target = (
        check_starvation_override(
            current_phase=current_phase,
            movement_queues=movement_queues,
            starvation=starvation,
            phase_priorities=details,
            max_starvation_time=max_starvation_time,
        )
    )

    if should_override:

        return (
            True,
            "starvation_override",
            override_target,
            details,
        )

    competing_phases = [
        phase_name
        for phase_name in PHASE_NAMES
        if phase_name != current_phase
    ]

    current_queue = _phase_total_queue(
        current_phase,
        movement_queues,
    )

    non_empty_competitors = [
        phase_name
        for phase_name in competing_phases
        if _phase_total_queue(phase_name, movement_queues) > 0
    ]

    if current_queue == 0 and non_empty_competitors:

        target_phase = _pick_best_phase(
            non_empty_competitors,
            details,
        )

        return True, "current_phase_empty", target_phase, details

    if green_elapsed >= MAX_GREEN:

        target_phase = _pick_best_phase(
            competing_phases,
            details,
        )

        return True, "maximum_green_reached", target_phase, details

    best_competing_phase = _pick_best_phase(
        competing_phases,
        details,
    )

    if (
        details[best_competing_phase]
        > details[current_phase]
    ):

        return (
            True,
            "competing_phase_higher_priority",
            best_competing_phase,
            details,
        )

    return False, None, None, details


def _run_sanity_checks():

    if sorted(PHASE_TIE_BREAK_ORDER) != sorted(PHASE_NAMES):

        raise ValueError(
            "PHASE_TIE_BREAK_ORDER does not exactly match "
            "PHASE_NAMES. Check both lists."
        )

    if sorted(MOVEMENT_TO_PHASE.keys()) != sorted(MOVEMENT_IDS):

        raise ValueError(
            "MOVEMENT_TO_PHASE does not cover exactly the 12 "
            "movement IDs."
        )

    for direction in TRAVEL_DIRECTIONS:

        if direction not in ENTRY_HEADING_LETTER:

            raise ValueError(
                f"ENTRY_HEADING_LETTER is missing direction "
                f"{direction}."
            )


_run_sanity_checks()
