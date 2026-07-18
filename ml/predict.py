from pathlib import Path
import pickle

import pandas as pd

from ml.features import (
    CONGESTION_LABELS,
    prepare_features,
    probabilities_to_severity,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ml"
    / "congestion_classifier.pkl"
)


def print_section(title):
    print(f"\n{title}")


print_section(
    "SMART TRAFFIC AI — ML CONGESTION PREDICTION"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(

        "\nClassifier model not found.\n\n"

        "Run this first:\n"

        "python3 -m ml.train"
    )


print("\nLoading classifier...")


with open(

    MODEL_PATH,

    "rb",

) as file:

    model_bundle = pickle.load(
        file
    )


pipeline = model_bundle[
    "pipeline"
]


model_scope = model_bundle[
    "model_scope"
]


classes = model_bundle[
    "classes"
]


print(
    "Classifier loaded successfully."
)


print(
    f"\nSelected model: "
    f"{model_bundle['model_name'].upper()}"
)


print(
    f"Model scope: "
    f"{model_scope}"
)


DEFAULT_VALUES = {

    "City": "Atlanta",

    "IntersectionId": 84,

    "EntryStreetName": (
        "Unknown"
    ),

    "ExitStreetName": (
        "Unknown"
    ),

    "EntryHeading": "W",

    "ExitHeading": "E",

    "Hour": 8,

    "Weekend": 0,

    "Month": 6,
}


print(
    "\nPress Enter to use each default value."
)


values = {}


for field, default in DEFAULT_VALUES.items():

    user_input = input(

        f"{field} [{default}]: "

    ).strip()


    if not user_input:

        values[field] = default


    elif field in {

        "IntersectionId",

        "Hour",

        "Weekend",

        "Month",

    }:

        values[field] = int(
            user_input
        )


    else:

        values[field] = user_input


if model_scope == "local":

    selected_city = model_bundle[
        "selected_city"
    ]


    selected_intersection = model_bundle[
        "selected_intersection_id"
    ]


    if (

        values["City"]
        != selected_city

        or

        values["IntersectionId"]
        != selected_intersection

    ):

        raise ValueError(

            "\nThe selected final model is local "
            "to:\n"

            f"City: {selected_city}\n"

            f"Intersection: "
            f"{selected_intersection}\n\n"

            "Use those values for prediction."
        )


input_data = pd.DataFrame(
    [values]
)


features = prepare_features(

    input_data,

    model_scope=model_scope,
)


predicted_class = pipeline.predict(
    features
)[0]


probabilities = pipeline.predict_proba(
    features
)[0]


probability_map = {

    class_name: float(
        probability
    )

    for class_name, probability in zip(

        classes,

        probabilities,
    )
}


severity_score = probabilities_to_severity(

    probabilities=probabilities,

    classes=classes,
)


print_section(
    "ML PREDICTION RESULT"
)


print(
    f"\nCity:          "
    f"{values['City']}"
)


print(
    f"Intersection:  "
    f"{values['IntersectionId']}"
)


print(
    f"Hour:          "
    f"{values['Hour']}"
)


print(
    f"Weekend:       "
    f"{values['Weekend']}"
)


print(
    f"Month:         "
    f"{values['Month']}"
)


print(
    f"Entry heading: "
    f"{values['EntryHeading']}"
)


print(
    f"Exit heading:  "
    f"{values['ExitHeading']}"
)


print(
    f"\nPredicted congestion category: "
    f"{predicted_class}"
)


print("\nClass probabilities:\n")


for label in CONGESTION_LABELS:

    probability = probability_map.get(
        label,
        0.0
    )


    print(
        f"{label:<10}: "
        f"{probability * 100:>6.2f}%"
    )


print(
    f"\nContinuous ML severity score: "
    f"{severity_score:.4f}"
)


print(
    "\nInterpretation:"
)


if severity_score < 0.20:

    interpretation = (
        "Very low traffic priority."
    )


elif severity_score < 0.40:

    interpretation = (
        "Moderate traffic priority."
    )


elif severity_score < 0.65:

    interpretation = (
        "High traffic priority."
    )


else:

    interpretation = (
        "Severe traffic priority. "
        "The adaptive controller should "
        "strongly consider additional green time."
    )


print(
    interpretation
)


print(
    "\nThis ML severity score will later be "
    "combined with the current simulated traffic "
    "state, IR-retrieved historical cases, and "
    "the adaptive signal decision engine."
)