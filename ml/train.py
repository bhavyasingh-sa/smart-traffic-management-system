from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
)

from ml.features import (
    CONGESTION_LABELS,
    GLOBAL_CATEGORICAL_FEATURES,
    GLOBAL_NUMERICAL_FEATURES,
    LOCAL_CATEGORICAL_FEATURES,
    LOCAL_NUMERICAL_FEATURES,
    calculate_scale_limits,
    create_congestion_labels,
    prepare_features,
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


MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "ml"
)


MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIRECTORY
    / "congestion_classifier.pkl"
)


RANDOM_STATE = 42

MAX_GLOBAL_TRAINING_ROWS = 250_000


SELECTED_CITY = "Atlanta"

SELECTED_INTERSECTION_ID = 84


def print_section(title):
    print(f"\n{title}")


def create_classifier_pipeline(
    model_scope
):

    if model_scope == "global":

        categorical_features = (
            GLOBAL_CATEGORICAL_FEATURES
        )

        numerical_features = (
            GLOBAL_NUMERICAL_FEATURES
        )


    elif model_scope == "local":

        categorical_features = (
            LOCAL_CATEGORICAL_FEATURES
        )

        numerical_features = (
            LOCAL_NUMERICAL_FEATURES
        )


    else:

        raise ValueError(
            "model_scope must be "
            "'global' or 'local'."
        )


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),

                categorical_features,
            ),

            (
                "numerical",

                "passthrough",

                numerical_features,
            ),
        ]
    )


    classifier = RandomForestClassifier(

        n_estimators=250,

        max_depth=20,

        min_samples_leaf=2,

        max_features="sqrt",

        class_weight="balanced",

        n_jobs=-1,

        random_state=RANDOM_STATE,
    )


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "classifier",
                classifier,
            ),
        ]
    )


    return pipeline


def print_class_distribution(
    labels,
    title
):

    print(f"\n{title}:\n")


    counts = (

        pd.Series(labels)

        .value_counts()

        .reindex(
            CONGESTION_LABELS,
            fill_value=0,
        )
    )


    total = len(labels)


    for label in CONGESTION_LABELS:

        count = int(
            counts[label]
        )


        percentage = (

            count
            / total
            * 100

            if total > 0

            else 0.0
        )


        print(
            f"{label:<10} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )


def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test
):

    print_section(
        f"{model_name} — EVALUATION"
    )


    predictions = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )


    macro_precision = precision_score(

        y_test,

        predictions,

        labels=CONGESTION_LABELS,

        average="macro",

        zero_division=0,
    )


    macro_recall = recall_score(

        y_test,

        predictions,

        labels=CONGESTION_LABELS,

        average="macro",

        zero_division=0,
    )


    macro_f1 = f1_score(

        y_test,

        predictions,

        labels=CONGESTION_LABELS,

        average="macro",

        zero_division=0,
    )


    weighted_f1 = f1_score(

        y_test,

        predictions,

        labels=CONGESTION_LABELS,

        average="weighted",

        zero_division=0,
    )


    print(
        f"\nAccuracy:          "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )


    print(
        f"Macro precision:   "
        f"{macro_precision:.4f}"
    )


    print(
        f"Macro recall:      "
        f"{macro_recall:.4f}"
    )


    print(
        f"Macro F1:          "
        f"{macro_f1:.4f}"
    )


    print(
        f"Weighted F1:       "
        f"{weighted_f1:.4f}"
    )


    print("\nClassification report:\n")


    print(

        classification_report(

            y_test,

            predictions,

            labels=CONGESTION_LABELS,

            zero_division=0,
        )
    )


    matrix = confusion_matrix(

        y_test,

        predictions,

        labels=CONGESTION_LABELS,
    )


    matrix_df = pd.DataFrame(

        matrix,

        index=[
            f"Actual_{label}"
            for label in CONGESTION_LABELS
        ],

        columns=[
            f"Predicted_{label}"
            for label in CONGESTION_LABELS
        ],
    )


    print("\nConfusion matrix:\n")


    print(
        matrix_df.to_string()
    )


    return {

        "accuracy": float(
            accuracy
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),
    }


print_section(
    "SMART TRAFFIC AI — CLASSIFIER TRAINING"
)


if not TRAIN_PATH.exists():

    raise FileNotFoundError(
        f"train.csv not found at:\n"
        f"{TRAIN_PATH}"
    )


print("\nLoading train.csv...")


data = pd.read_csv(
    TRAIN_PATH
)


print(
    f"Loaded {len(data):,} "
    f"historical records."
)


available_months = sorted(

    data["Month"]

    .dropna()

    .unique()
)


print(
    "\nAvailable months: "
    f"{[
        int(month)
        for month in available_months
    ]}"
)


if len(available_months) < 2:

    raise ValueError(
        "At least two months are required."
    )


test_month = available_months[-1]


print(
    f"Test month: "
    f"{int(test_month)}"
)


print_section(
    "GLOBAL DATASET PREPARATION"
)


global_train = data[

    data["Month"] < test_month

].copy()


global_test = data[

    data["Month"] == test_month

].copy()


print(
    f"\nGlobal training records before sampling: "
    f"{len(global_train):,}"
)


print(
    f"Global test records: "
    f"{len(global_test):,}"
)


if (
    len(global_train)
    > MAX_GLOBAL_TRAINING_ROWS
):

    global_train = global_train.sample(

        n=MAX_GLOBAL_TRAINING_ROWS,

        random_state=RANDOM_STATE,
    )


    print(
        f"\nGlobal training data sampled to "
        f"{len(global_train):,} records."
    )


# Use the global training set only.
# This avoids leakage from the unseen test month.

scale_limits = calculate_scale_limits(
    global_train
)


print("\nCongestion scale limits:")


for column, value in scale_limits.items():

    print(
        f"  {column}: "
        f"{value:.4f}"
    )


X_global_train = prepare_features(

    global_train,

    model_scope="global",
)


X_global_test = prepare_features(

    global_test,

    model_scope="global",
)


y_global_train = create_congestion_labels(

    global_train,

    scale_limits,
)


y_global_test = create_congestion_labels(

    global_test,

    scale_limits,
)


print_class_distribution(

    y_global_train,

    "Global training distribution",
)


print_class_distribution(

    y_global_test,

    "Global test distribution",
)


print_section(
    "TRAINING GLOBAL BALANCED CLASSIFIER"
)


global_pipeline = create_classifier_pipeline(
    model_scope="global"
)


start_time = time.time()


global_pipeline.fit(

    X_global_train,

    y_global_train,
)


global_training_time = (

    time.time()

    - start_time
)


print(
    f"\nGlobal training completed in "
    f"{global_training_time:.2f} seconds."
)


global_metrics = evaluate_model(

    model_name="GLOBAL CLASSIFIER",

    pipeline=global_pipeline,

    X_test=X_global_test,

    y_test=y_global_test,
)


print_section(
    "LOCAL INTERSECTION 84 DATASET"
)


local_data = data[

    (data["City"] == SELECTED_CITY)

    &

    (
        data["IntersectionId"]
        == SELECTED_INTERSECTION_ID
    )

].copy()


print(
    f"\nSelected city: "
    f"{SELECTED_CITY}"
)


print(
    f"Selected intersection: "
    f"{SELECTED_INTERSECTION_ID}"
)


print(
    f"Total local records: "
    f"{len(local_data):,}"
)


if local_data.empty:

    raise ValueError(
        "Selected local intersection "
        "contains no records."
    )


local_train = local_data[

    local_data["Month"] < test_month

].copy()


local_test = local_data[

    local_data["Month"] == test_month

].copy()


print(
    f"\nLocal training records: "
    f"{len(local_train):,}"
)


print(
    f"Local test records: "
    f"{len(local_test):,}"
)


if local_train.empty:

    raise ValueError(
        "Local training set is empty."
    )


if local_test.empty:

    raise ValueError(
        "Local test set is empty."
    )


X_local_train = prepare_features(

    local_train,

    model_scope="local",
)


X_local_test = prepare_features(

    local_test,

    model_scope="local",
)


y_local_train = create_congestion_labels(

    local_train,

    scale_limits,
)


y_local_test = create_congestion_labels(

    local_test,

    scale_limits,
)


print_class_distribution(

    y_local_train,

    "Local training distribution",
)


print_class_distribution(

    y_local_test,

    "Local test distribution",
)


print_section(
    "TRAINING LOCAL BALANCED CLASSIFIER"
)


local_pipeline = create_classifier_pipeline(
    model_scope="local"
)


start_time = time.time()


local_pipeline.fit(

    X_local_train,

    y_local_train,
)


local_training_time = (

    time.time()

    - start_time
)


print(
    f"\nLocal training completed in "
    f"{local_training_time:.2f} seconds."
)


local_metrics = evaluate_model(

    model_name=(
        "LOCAL ATLANTA INTERSECTION 84 CLASSIFIER"
    ),

    pipeline=local_pipeline,

    X_test=X_local_test,

    y_test=y_local_test,
)


print_section(
    "GLOBAL MODEL ON ATLANTA INTERSECTION 84"
)


X_local_test_global_format = prepare_features(

    local_test,

    model_scope="global",
)


global_on_local_metrics = evaluate_model(

    model_name=(
        "GLOBAL CLASSIFIER ON INTERSECTION 84"
    ),

    pipeline=global_pipeline,

    X_test=X_local_test_global_format,

    y_test=y_local_test,
)


print_section(
    "GLOBAL VS LOCAL MODEL COMPARISON"
)


comparison = pd.DataFrame(

    [

        {
            "Model": (
                "Global model on Intersection 84"
            ),

            "Accuracy": (
                global_on_local_metrics[
                    "accuracy"
                ]
            ),

            "Balanced Accuracy": (
                global_on_local_metrics[
                    "balanced_accuracy"
                ]
            ),

            "Macro F1": (
                global_on_local_metrics[
                    "macro_f1"
                ]
            ),
        },

        {
            "Model": (
                "Local Intersection 84 model"
            ),

            "Accuracy": (
                local_metrics[
                    "accuracy"
                ]
            ),

            "Balanced Accuracy": (
                local_metrics[
                    "balanced_accuracy"
                ]
            ),

            "Macro F1": (
                local_metrics[
                    "macro_f1"
                ]
            ),
        },
    ]
)


print(
    "\n"
    + comparison.to_string(
        index=False
    )
)


# Selection score:
#
# 60% balanced accuracy
# 40% macro F1
#
# Both metrics are appropriate for imbalanced classes.

global_selection_score = (

    0.60
    * global_on_local_metrics[
        "balanced_accuracy"
    ]

    +

    0.40
    * global_on_local_metrics[
        "macro_f1"
    ]
)


local_selection_score = (

    0.60
    * local_metrics[
        "balanced_accuracy"
    ]

    +

    0.40
    * local_metrics[
        "macro_f1"
    ]
)


if (
    local_selection_score
    > global_selection_score
):

    selected_model_name = (
        "local"
    )

    selected_pipeline = (
        local_pipeline
    )

    selected_metrics = (
        local_metrics
    )

    selected_scope = (
        "local"
    )


else:

    selected_model_name = (
        "global"
    )

    selected_pipeline = (
        global_pipeline
    )

    selected_metrics = (
        global_on_local_metrics
    )

    selected_scope = (
        "global"
    )


print_section(
    "FINAL MODEL SELECTION"
)


print(
    f"\nGlobal selection score: "
    f"{global_selection_score:.4f}"
)


print(
    f"Local selection score:  "
    f"{local_selection_score:.4f}"
)


print(
    f"\nSelected final model: "
    f"{selected_model_name.upper()}"
)


print(
    f"Model scope: "
    f"{selected_scope}"
)


print(
    f"Balanced accuracy: "
    f"{selected_metrics['balanced_accuracy'] * 100:.2f}%"
)


print(
    f"Macro F1: "
    f"{selected_metrics['macro_f1']:.4f}"
)


model_bundle = {

    "pipeline": selected_pipeline,

    "model_scope": selected_scope,

    "model_name": (
        selected_model_name
    ),

    "model_type": (
        "RandomForestClassifier"
    ),

    "selected_city": (
        SELECTED_CITY
    ),

    "selected_intersection_id": (
        SELECTED_INTERSECTION_ID
    ),

    "scale_limits": (
        scale_limits
    ),

    "classes": list(
        selected_pipeline.named_steps[
            "classifier"
        ].classes_
    ),

    "test_month": int(
        test_month
    ),

    "selected_metrics": (
        selected_metrics
    ),

    "global_metrics": (
        global_metrics
    ),

    "global_on_local_metrics": (
        global_on_local_metrics
    ),

    "local_metrics": (
        local_metrics
    ),
}


print_section(
    "SAVING FINAL CLASSIFIER"
)


with open(

    MODEL_PATH,

    "wb",

) as file:

    pickle.dump(

        model_bundle,

        file,
    )


print(
    "\nModel saved successfully to:"
)


print(
    MODEL_PATH
)


print_section(
    "ML CLASSIFIER TRAINING COMPLETE"
)


print(
    f"\nSelected model: "
    f"{selected_model_name.upper()}"
)


print(
    f"Selected scope: "
    f"{selected_scope}"
)


print(
    f"Balanced accuracy: "
    f"{selected_metrics['balanced_accuracy'] * 100:.2f}%"
)


print(
    f"Macro F1: "
    f"{selected_metrics['macro_f1']:.4f}"
)


print(
    "\nNext step:"
)


print(
    "Run python3 -m ml.predict "
    "to test class probabilities and "
    "continuous severity prediction."
)