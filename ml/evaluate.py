"""ml/evaluate.py - prints the trained classifier's saved accuracy/F1 scores. Run: python3 -m ml.evaluate"""

from pathlib import Path
import pickle

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "ml" / "congestion_classifier.pkl"


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No trained model found at {MODEL_PATH}.\n"
        "Run python3 -m ml.train first."
    )

with open(MODEL_PATH, "rb") as file:
    model_bundle = pickle.load(file)

selected_scope = model_bundle["model_scope"]

comparisons = [
    ("Global model (all cities)", model_bundle["global_metrics"], False),
    ("Global model on Intersection 84", model_bundle["global_on_local_metrics"], selected_scope == "global"),
    ("Local Intersection 84 model", model_bundle["local_metrics"], selected_scope == "local"),
]

table = pd.DataFrame(
    [
        {
            "Model": label + (" (SELECTED)" if is_selected else ""),
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Balanced Accuracy": f"{metrics['balanced_accuracy'] * 100:.2f}%",
            "Precision": f"{metrics['macro_precision']:.4f}",
            "Recall": f"{metrics['macro_recall']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Weighted F1": f"{metrics['weighted_f1']:.4f}",
        }
        for label, metrics, is_selected in comparisons
    ]
)

print("MODEL PERFORMANCE SUMMARY")
print(f"Classifier: {model_bundle['model_type']}")
print(f"Test month held out (unseen during training): {model_bundle['test_month']}")
print()
print(table.to_string(index=False))
print()
print(
    "Selection rule: 60% balanced accuracy + 40% macro F1, since the "
    "traffic congestion classes are imbalanced and plain accuracy alone "
    "would favor the majority class."
)
print(
    "Precision: of the times the model predicted a class, how often it "
    "was right. Recall: of all the actual cases of a class, how many the "
    "model caught. F1 is their harmonic mean. All four are macro-averaged "
    "across Low/Moderate/High/Severe so no class dominates the score."
)
print(
    f"\nFinal selected model: {selected_scope.upper()} "
    f"(Atlanta, Intersection 84)"
)
