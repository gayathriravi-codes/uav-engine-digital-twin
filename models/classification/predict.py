import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT FEATURE EXTRACTION FUNCTIONS
# ============================================================

from models.classification.feature_extraction import (
    calculate_health_score,
    extract_window_features,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "xgboost_fault_classifier.json"
)


CLASS_NAMES = [
    "cooling_degradation",
    "misfire",
    "none",
    "oil_issue",
    "overheat",
    "sensor_drift",
    "vibration_fault",
]


# ============================================================
# OPERATOR ACTION STRATEGIES
# ============================================================

OPERATOR_ACTIONS = {
    "none": (
        "Continue normal operation while maintaining routine monitoring."
    ),

    "misfire": (
        "Reduce engine load, monitor RPM/EGT/vibration, "
        "and inspect the combustion or ignition system."
    ),

    "overheat": (
        "Reduce engine load/RPM, closely monitor EGT and CHT, "
        "and inspect the cooling and thermal system."
    ),

    "cooling_degradation": (
        "Reduce load if possible, closely monitor CHT, "
        "and inspect cooling-system performance."
    ),

    "oil_issue": (
        "Reduce engine load, monitor oil pressure and temperature, "
        "and inspect the lubrication system."
    ),

    "sensor_drift": (
        "Treat the affected sensor as potentially unreliable, "
        "compare it with correlated sensors, and recalibrate or replace "
        "the sensor before relying on it for critical decisions."
    ),

    "vibration_fault": (
        "Reduce RPM/load if possible, monitor vibration continuously, "
        "and inspect rotating or mechanical components."
    ),
}


# ============================================================
# ENGINE CONDITION STATUS
# ============================================================

def determine_condition_status(health_score):
    """
    Determine physical engine condition from the interpretable
    health score.
    """

    if health_score >= 90:
        return "NORMAL"

    elif health_score >= 75:
        return "DEGRADED"

    else:
        return "CRITICAL"


# ============================================================
# FAULT SEVERITY
# ============================================================

def determine_fault_severity(fault):
    """
    Determine advisory severity from the diagnosed fault type.

    This is an advisory classification for the project and
    should not be interpreted as certified autonomous
    flight-control logic.
    """

    severity_map = {
        "none": "NORMAL",
        "sensor_drift": "CAUTION",
        "misfire": "WARNING",
        "cooling_degradation": "WARNING",
        "vibration_fault": "WARNING",
        "overheat": "CRITICAL",
        "oil_issue": "CRITICAL",
    }

    return severity_map.get(fault, "CAUTION")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained XGBoost fault-classification model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    return model


# ============================================================
# CONVERT EXTRACTED FEATURES TO DATAFRAME
# ============================================================

def prepare_features(window):
    """
    Extract features from a telemetry window and convert them
    into the DataFrame format expected by XGBoost.
    """

    feature_dict = extract_window_features(window)

    # extract_window_features() returns a dictionary.
    # XGBoost requires a 2D structure for prediction.
    features = pd.DataFrame([feature_dict])

    return features


# ============================================================
# FAULT PREDICTION
# ============================================================

def predict_fault(window):
    """
    Predict the engine fault for a 30-row telemetry window.

    Parameters
    ----------
    window : pandas.DataFrame
        A telemetry window containing exactly 30 rows.

    Returns
    -------
    dict
        Health score, engine condition, predicted fault,
        confidence, advisory severity, and operator action.
    """

    # --------------------------------------------------------
    # Validate input type
    # --------------------------------------------------------

    if not isinstance(window, pd.DataFrame):
        raise TypeError(
            "window must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Required sensor columns
    # --------------------------------------------------------

    required_sensors = [
        "rpm",
        "egt",
        "cht",
        "oil_pressure",
        "oil_temp",
        "vibration",
        "fuel_flow",
    ]

    missing_columns = [
        column
        for column in required_sensors
        if column not in window.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required sensor columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Validate window size
    # --------------------------------------------------------

    if len(window) != 30:
        raise ValueError(
            f"Expected exactly 30 rows, but received {len(window)} rows."
        )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Extract and prepare features
    # --------------------------------------------------------

    features = prepare_features(window)

    # --------------------------------------------------------
    # Ensure feature order matches the trained model
    # --------------------------------------------------------

    expected_features = model.get_booster().feature_names

    if expected_features is not None:

        missing_features = [
            feature
            for feature in expected_features
            if feature not in features.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        # Keep exactly the same order as training
        features = features[expected_features]

    # --------------------------------------------------------
    # Calculate health score
    # --------------------------------------------------------

    health_scores = calculate_health_score(window)

    mean_health = float(health_scores.mean())

    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(features)

    predicted_class_index = int(
        probabilities[0].argmax()
    )

    confidence = float(
        probabilities[0][predicted_class_index]
    )

    predicted_fault = CLASS_NAMES[predicted_class_index]

    # --------------------------------------------------------
    # Determine physical engine condition
    # --------------------------------------------------------

    condition_status = determine_condition_status(
        mean_health
    )

    # --------------------------------------------------------
    # Determine fault severity
    # --------------------------------------------------------

    severity = determine_fault_severity(
        predicted_fault
    )

    # --------------------------------------------------------
    # Operator action
    # --------------------------------------------------------

    operator_action = OPERATOR_ACTIONS.get(
        predicted_fault,
        "Continue monitoring the engine and investigate the detected condition."
    )

    # --------------------------------------------------------
    # Return final result
    # --------------------------------------------------------

    return {
        "health_score": round(mean_health, 2),

        "condition_status": condition_status,

        "predicted_fault": predicted_fault,

        "confidence": round(confidence, 4),

        "severity": severity,

        "operator_action": operator_action,
    }


# ============================================================
# COMMAND-LINE DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UAV ENGINE DIGITAL TWIN - FAULT PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Example telemetry file
    # --------------------------------------------------------

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "overheat_000.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found:\n{csv_path}"
        )

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Select a 30-row window
    # --------------------------------------------------------

    window = df.iloc[130:160].copy()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predict_fault(window)

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPrediction Result")
    print("-" * 60)

    print(
        f"Health Score       : {result['health_score']}"
    )

    print(
        f"Condition Status   : {result['condition_status']}"
    )

    print(
        f"Predicted Fault    : {result['predicted_fault']}"
    )

    print(
        f"Confidence         : {result['confidence']:.2%}"
    )

    print(
        f"Advisory Severity  : {result['severity']}"
    )

    print("\nOperator Action")
    print("-" * 60)

    print(result["operator_action"])

    print("\n" + "=" * 60)