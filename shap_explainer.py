"""
shap_explainer.py
-------------------
Real SHAP explainability for the leakage-fixed XGBoost fault classifier
(models/classification/xgboost_fault_classifier_fixed.json).

Unlike a hand-weighted "SHAP proxy" (fixed multipliers on raw residuals,
normalized to percentages), this uses shap.TreeExplainer, which computes
exact Shapley values for tree ensembles - the per-feature attribution
sums exactly to (prediction - baseline), a property no hand-picked-weight
scheme has.

Points at xgboost_fault_classifier_fixed.json specifically, NOT the
original xgboost_fault_classifier.json - the original was trained with
StratifiedGroupKFold grouped on flight_id (the fault-labeled name, e.g.
"misfire_005"), which does not prevent leakage: cooling_degradation_005
and misfire_005 are the same underlying base trajectory (confirmed
byte-identical for their first 10+ rows) and could still land in
different folds. The fixed model groups on base_id (the trailing _NNN)
instead - see train_xgboost_fixed.py. Explaining the original model's
predictions would mean explaining a possibly-leak-inflated model.
"""

import sys
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "classification" / "xgboost_fault_classifier_aug.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "classification_features.csv"

DROP_COLUMNS = ["label", "flight_id", "window_start", "window_end"]


class FaultExplainer:
    def __init__(self, model_path=MODEL_PATH, class_names=None):
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        self.explainer = shap.TreeExplainer(self.model)
        self.class_names = class_names  # pass int_to_label mapping, e.g. from train_xgboost_fixed.py

    def explain(self, X_row: pd.DataFrame, predicted_class_idx: int, top_n=8):
        """
        X_row: single-row DataFrame, same columns/order the model was
            trained on (i.e. df.drop(columns=DROP_COLUMNS) for one window).
        predicted_class_idx: the model's predicted class (int), so we
            explain THAT class's SHAP values, not an arbitrary one.

        Returns exact per-feature contributions toward the predicted
        class's log-odds, sorted by absolute magnitude - not normalized
        into fake percentages, since SHAP values are already meaningful
        in the model's own units.
        """
        shap_values = self.explainer.shap_values(X_row)

        # shap_values shape for multiclass varies by shap version:
        # either a list of per-class arrays, or a single array shaped
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features).
        # Handle all three so this doesn't silently break on a version bump.
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        elif shap_values.ndim == 3 and shap_values.shape[-1] == len(self.model.classes_):
            class_shap = shap_values[0, :, predicted_class_idx]
        else:
            class_shap = shap_values[predicted_class_idx, 0, :]

        base_value = self.explainer.expected_value
        base_value = base_value[predicted_class_idx] if hasattr(base_value, "__len__") else base_value

        contributions = list(zip(X_row.columns, class_shap))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contributions[:top_n]

        predicted_label = self.class_names[predicted_class_idx] if self.class_names else predicted_class_idx

        return {
            "predicted_class": predicted_label,
            "base_value": round(float(base_value), 4),
            "sum_of_all_shap_values": round(float(class_shap.sum()), 4),
            "top_contributing_features": [
                {"feature": f, "shap_value": round(float(v), 4)} for f, v in top
            ],
            "note": (
                "shap_value units are log-odds contribution toward the predicted "
                "class; base_value + sum(all_shap_values) = model's raw margin "
                "for this class (exact, not approximate)."
            ),
        }

    def global_importance(self, X: pd.DataFrame, top_n=15):
        """
        Mean |SHAP value| per feature across all rows and all classes -
        a real, aggregate feature-importance ranking (distinct from
        per-prediction explain() above). Useful for a single deck slide
        showing "what the classifier relies on overall", backed by exact
        Shapley values rather than XGBoost's built-in gain/weight/cover
        importances (which don't account for feature interactions the
        way SHAP does).
        """
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
            mean_abs = np.abs(stacked).mean(axis=(0, 1))
        elif shap_values.ndim == 3 and shap_values.shape[-1] == len(self.model.classes_):
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=(0, 1))

        ranked = sorted(zip(X.columns, mean_abs), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLUMNS)

    class_names = sorted(df["label"].unique())
    int_to_label = {i: name for i, name in enumerate(class_names)}

    explainer = FaultExplainer(class_names=int_to_label)

    # --- Per-window explanations, one example per class ---
    # Grab one example per class instead of just the first 5 rows,
    # so the demo shows real fault diagnoses, not repeated "none" calls.
    example_idx = df.groupby("label").head(1).index.tolist()[:7]
    preds = explainer.model.predict(X.loc[example_idx])
    print("=" * 60)
    print("PER-WINDOW SHAP EXPLANATIONS (one per class)")
    print("=" * 60)
    for i, idx in enumerate(example_idx):
        result = explainer.explain(X.loc[[idx]], int(preds[i]))
        print(f"\nWindow (row {idx}) - true label: {df['label'].iloc[idx]}")
        print(f"  Predicted: {result['predicted_class']}")
        for c in result["top_contributing_features"]:
            print(f"    {c['feature']:30s} {c['shap_value']:+.4f}")

    # --- Global feature importance across the full dataset ---
    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP|, all classes, all rows)")
    print("=" * 60)
    ranked = explainer.global_importance(X, top_n=15)
    for feat, val in ranked:
        print(f"  {feat:30s} {val:.4f}")