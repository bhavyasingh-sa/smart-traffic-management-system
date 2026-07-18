import numpy as np
import pandas as pd


# These columns are used ONLY to create the historical
# congestion target.
#
# They must NOT be used as input features because that
# would cause target leakage.

TARGET_COLUMNS = [
    "TotalTimeStopped_p50",
    "TimeFromFirstStop_p50",
    "DistanceToFirstStop_p50",
]


GLOBAL_CATEGORICAL_FEATURES = [
    "City",
    "EntryHeading",
    "ExitHeading",
    "EntryStreetName",
    "ExitStreetName",
]


GLOBAL_NUMERICAL_FEATURES = [
    "IntersectionId",
    "Hour",
    "Weekend",
    "Month",
]


GLOBAL_MODEL_FEATURES = (
    GLOBAL_CATEGORICAL_FEATURES
    + GLOBAL_NUMERICAL_FEATURES
)


# For Atlanta Intersection 84, City and IntersectionId
# never change, so they provide no predictive information.

LOCAL_CATEGORICAL_FEATURES = [
    "EntryHeading",
    "ExitHeading",
    "EntryStreetName",
    "ExitStreetName",
]


LOCAL_NUMERICAL_FEATURES = [
    "Hour",
    "Weekend",
    "Month",
]


LOCAL_MODEL_FEATURES = (
    LOCAL_CATEGORICAL_FEATURES
    + LOCAL_NUMERICAL_FEATURES
)


CONGESTION_LABELS = [
    "Low",
    "Moderate",
    "High",
    "Severe",
]


SEVERITY_WEIGHTS = {
    "Low": 0.00,
    "Moderate": 0.33,
    "High": 0.66,
    "Severe": 1.00,
}


def prepare_features(
    dataframe,
    model_scope="global"
):
    """
    Select and clean model input features.

    model_scope:
        "global" -> all cities/intersections
        "local"  -> selected Atlanta Intersection 84
    """

    if model_scope == "global":

        categorical_features = (
            GLOBAL_CATEGORICAL_FEATURES
        )

        numerical_features = (
            GLOBAL_NUMERICAL_FEATURES
        )

        model_features = (
            GLOBAL_MODEL_FEATURES
        )

    elif model_scope == "local":

        categorical_features = (
            LOCAL_CATEGORICAL_FEATURES
        )

        numerical_features = (
            LOCAL_NUMERICAL_FEATURES
        )

        model_features = (
            LOCAL_MODEL_FEATURES
        )

    else:

        raise ValueError(
            "model_scope must be "
            "'global' or 'local'."
        )


    features = dataframe[
        model_features
    ].copy()


    for column in categorical_features:

        features[column] = (
            features[column]
            .fillna("Unknown")
            .astype(str)
        )


    for column in numerical_features:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        ).fillna(0)


    return features


def calculate_scale_limits(dataframe):
    """
    Calculate robust upper limits using the 95th percentile.

    These limits should be calculated from training data only
    to prevent test-data leakage.
    """

    limits = {}


    for column in TARGET_COLUMNS:

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).fillna(0)


        limit = float(
            values.quantile(0.95)
        )


        if limit <= 0:

            limit = 1.0


        limits[column] = limit


    return limits


def create_congestion_score(
    dataframe,
    scale_limits
):
    """
    Create a normalized historical congestion severity score.

    Formula:

        40% stopped-time severity
        35% delay severity
        25% traffic-distance severity

    Output range:
        approximately 0.0 to 1.0

    These weights are project design choices and are not
    claimed as a universal traffic-engineering standard.
    """

    stopped = pd.to_numeric(
        dataframe["TotalTimeStopped_p50"],
        errors="coerce"
    ).fillna(0)


    delay = pd.to_numeric(
        dataframe["TimeFromFirstStop_p50"],
        errors="coerce"
    ).fillna(0)


    distance = pd.to_numeric(
        dataframe["DistanceToFirstStop_p50"],
        errors="coerce"
    ).fillna(0)


    stopped_scaled = np.clip(

        stopped
        / scale_limits[
            "TotalTimeStopped_p50"
        ],

        0,
        1,
    )


    delay_scaled = np.clip(

        delay
        / scale_limits[
            "TimeFromFirstStop_p50"
        ],

        0,
        1,
    )


    distance_scaled = np.clip(

        distance
        / scale_limits[
            "DistanceToFirstStop_p50"
        ],

        0,
        1,
    )


    congestion_score = (

        0.40 * stopped_scaled

        +

        0.35 * delay_scaled

        +

        0.25 * distance_scaled
    )


    return congestion_score


def congestion_level(score):
    """
    Convert a continuous congestion score to a category.
    """

    score = float(score)


    if score < 0.20:

        return "Low"


    if score < 0.40:

        return "Moderate"


    if score < 0.65:

        return "High"


    return "Severe"


def create_congestion_labels(
    dataframe,
    scale_limits
):
    """
    Create categorical ML targets:

        Low
        Moderate
        High
        Severe
    """

    scores = create_congestion_score(
        dataframe,
        scale_limits
    )


    labels = pd.Series(

        [
            congestion_level(score)
            for score in scores
        ],

        index=dataframe.index,

        name="CongestionLevel",
    )


    return labels


def probabilities_to_severity(
    probabilities,
    classes
):
    """
    Convert classifier probabilities into a continuous
    severity score between approximately 0 and 1.

    Example:

        Low      = 0.05
        Moderate = 0.15
        High     = 0.55
        Severe   = 0.25

    Severity:

        0.00 * 0.05
        + 0.33 * 0.15
        + 0.66 * 0.55
        + 1.00 * 0.25
    """

    severity = 0.0


    for probability, class_name in zip(
        probabilities,
        classes
    ):

        weight = SEVERITY_WEIGHTS.get(
            class_name,
            0.0
        )


        severity += (
            float(probability)
            * weight
        )


    return float(
        np.clip(
            severity,
            0.0,
            1.0
        )
    )