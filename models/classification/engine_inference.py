import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXISTING MODELS
# ============================================================

from models.classification.predict import predict_fault

from models.classification.recalibration_speed import (
    compute_recalibration_speed,
)

from models.rul.train_rul_ensemble import (
    load_ensemble,
    build_inference_window,
    predict_rul_ensemble,
)


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 30

REQUIRED_SENSORS = [
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temp",
    "vibration",
    "fuel_flow",
]


# ============================================================
# END-TO-END ENGINE INFERENCE
# ============================================================

def run_engine_inference(df):
    """
    Run the complete UAV Engine Digital Twin inference pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Full telemetry history for one engine flight.

    Returns
    -------
    dict
        Combined fault, health, RUL, and recovery information.
    """

    # --------------------------------------------------------
    # Validate input type
    # --------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Validate minimum data length
    # --------------------------------------------------------

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} telemetry rows are required."
        )

    # --------------------------------------------------------
    # Validate required sensor columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_SENSORS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required sensor columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Sort telemetry chronologically
    # --------------------------------------------------------

    data = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Select latest 30-row window
    # --------------------------------------------------------

    current_window = (
        data
        .iloc[-WINDOW_SIZE:]
        .copy()
    )

    # ========================================================
    # 1. FAULT CLASSIFICATION + HEALTH SCORE
    # ========================================================

    fault_result = predict_fault(
        current_window
    )

    # ========================================================
    # 2. RUL PREDICTION
    # ========================================================

    raw_window = (
        current_window[
            REQUIRED_SENSORS
        ]
        .to_numpy()
    )

    # Convert seven raw sensors into the ten
    # features expected by the trained RUL model.
    inference_window = build_inference_window(
        raw_window
    )

    # Load trained RUL ensemble.
    models, scaler, dropped_idx_list = load_ensemble()

    # Use calibrated RUL prediction.
    rul_result = predict_rul_ensemble(
        inference_window,
        models,
        scaler,
        dropped_idx_list,
        calibrated=True,
    )

    # ========================================================
    # 3. RECALIBRATION / RECOVERY DETECTION
    # ========================================================

    # IMPORTANT:
    # The recovery detector needs the full flight history,
    # not just the latest 30-row window.
    recovery_result = compute_recalibration_speed(
        data
    )

    # ========================================================
    # 4. COMBINE ALL RESULTS
    # ========================================================

    result = {
        # ----------------------------------------------------
        # Engine health
        # ----------------------------------------------------

        "health_score": fault_result[
            "health_score"
        ],

        "condition_status": fault_result[
            "condition_status"
        ],

        # ----------------------------------------------------
        # Fault diagnosis
        # ----------------------------------------------------

        "predicted_fault": fault_result[
            "predicted_fault"
        ],

        "confidence": fault_result[
            "confidence"
        ],

        "severity": fault_result[
            "severity"
        ],

        "operator_action": fault_result[
            "operator_action"
        ],

        # ----------------------------------------------------
        # Remaining useful life
        #
        # The underlying model target is
        # true_rul_timesteps.
        #
        # The original RUL module uses legacy
        # *_minutes keys, so we expose the correct
        # unit at this integration layer.
        # ----------------------------------------------------

        "rul_estimate_timesteps": round(
            float(
                rul_result[
                    "point_estimate_minutes"
                ]
            ),
            2,
        ),

        "rul_lower_bound_timesteps": round(
            float(
                rul_result[
                    "rul_lower_bound_minutes"
                ]
            ),
            2,
        ),

        "rul_uncertainty_timesteps": round(
            float(
                rul_result[
                    "std_minutes"
                ]
            ),
            2,
        ),

        # ----------------------------------------------------
        # Recovery / recalibration
        # ----------------------------------------------------

        "recovery_rate": recovery_result[
            "recovery_rate"
        ],

        "recovered": recovery_result[
            "recovered"
        ],
    }

    return result


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "UAV ENGINE DIGITAL TWIN - END-TO-END INFERENCE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Test using final overheat flight
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
    # Load final telemetry
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    print(
        f"\nLoaded telemetry rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Run complete inference
    # --------------------------------------------------------

    result = run_engine_inference(df)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nENGINE STATE")
    print("-" * 70)

    for key, value in result.items():
        print(
            f"{key:35}: {value}"
        )

    print("\n" + "=" * 70)
    print("END-TO-END INFERENCE COMPLETED")
    print("=" * 70)