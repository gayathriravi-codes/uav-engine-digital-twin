import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT ACTUAL INFERENCE FUNCTION
# =========================================================

from models.classification.predict import predict_fault


# =========================================================
# SETTINGS
# =========================================================

DATA_DIR = PROJECT_ROOT / "data" / "raw"

WINDOW_SIZE = 30
STRIDE = 5

# Same special rule used for intermittent misfire windows
MISFIRE_MIN_ROWS = 5


# =========================================================
# FAULT TYPES
# =========================================================

FAULT_TYPES = {
    "cooling_degradation",
    "misfire",
    "oil_issue",
    "overheat",
    "sensor_drift",
    "vibration_fault",
}


# =========================================================
# GET FLIGHT FAULT FROM FILENAME
# =========================================================

def get_flight_fault(file_name):

    name = Path(file_name).stem

    if name.startswith("healthy_"):
        return "none"

    for fault in FAULT_TYPES:

        if name.startswith(fault + "_"):
            return fault

    raise ValueError(
        f"Could not determine fault type from filename: "
        f"{file_name}"
    )


# =========================================================
# DETERMINE EXPECTED LABEL FOR A WINDOW
# =========================================================

def get_expected_window_label(
    window,
    flight_fault,
    onset_idx
):

    # Healthy flight
    if flight_fault == "none":
        return "none"

    # Window has not reached fault onset
    window_end = int(window.index[-1])

    if window_end < onset_idx:
        return None

    # -----------------------------------------------------
    # Misfire is intermittent.
    # Use the same >=5 faulty rows rule as feature extraction.
    # -----------------------------------------------------

    if flight_fault == "misfire":

        misfire_count = (
            window["fault_type"]
            .eq("misfire")
            .sum()
        )

        if misfire_count >= MISFIRE_MIN_ROWS:
            return "misfire"

        return None

    # -----------------------------------------------------
    # Other injected faults remain active after onset.
    # -----------------------------------------------------

    return flight_fault


# =========================================================
# MAIN
# =========================================================

def main():

    csv_files = sorted(
        DATA_DIR.glob("*.csv")
    )

    print("=" * 75)
    print(
        "UAV ENGINE DIGITAL TWIN - "
        "FAULT-WINDOW VALIDATION"
    )
    print("=" * 75)

    print(
        f"\nDataset directory : {DATA_DIR}"
    )

    print(
        f"CSV files found   : {len(csv_files)}"
    )

    if not csv_files:
        print("\nNo CSV files found.")
        return


    # =====================================================
    # STORAGE
    # =====================================================

    true_labels = []
    predicted_labels = []

    results = []

    flight_results = []


    # =====================================================
    # PROCESS EVERY FLIGHT
    # =====================================================

    for file in csv_files:

        df = pd.read_csv(file)

        flight_fault = get_flight_fault(
            file.name
        )

        # -------------------------------------------------
        # Get onset index
        # -------------------------------------------------

        if "fault_onset_idx" in df.columns:

            onset_values = (
                df["fault_onset_idx"]
                .dropna()
                .unique()
            )

            if len(onset_values) > 0:
                onset_idx = int(
                    onset_values[0]
                )
            else:
                onset_idx = -1

        else:
            onset_idx = -1


        flight_predictions = []


        # =================================================
        # WINDOW THROUGH FLIGHT
        # =================================================

        for start in range(
            0,
            len(df) - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window = df.iloc[
                start:end
            ].copy()


            # -------------------------------------------------
            # Determine whether this is a valid evaluation
            # window.
            # -------------------------------------------------

            expected_label = (
                get_expected_window_label(
                    window,
                    flight_fault,
                    onset_idx
                )
            )

            # None means:
            # - pre-fault window
            # - insufficient misfire activity
            #
            # We don't count these as fault-detection tests.

            if expected_label is None:
                continue


            # =================================================
            # RUN ACTUAL MODEL
            # =================================================

            try:

                prediction = predict_fault(
                    window
                )

            except Exception as e:

                print(
                    f"\nERROR: {file.name}"
                    f" | window {start}-{end-1}"
                    f" | {e}"
                )

                continue


            predicted_fault = (
                prediction["predicted_fault"]
            )


            # =================================================
            # STORE RESULTS
            # =================================================

            true_labels.append(
                expected_label
            )

            predicted_labels.append(
                predicted_fault
            )

            flight_predictions.append(
                predicted_fault
            )

            results.append({

                "flight_id":
                    file.stem,

                "window_start":
                    start,

                "window_end":
                    end - 1,

                "actual_fault":
                    expected_label,

                "predicted_fault":
                    predicted_fault,

                "confidence":
                    prediction["confidence"],

                "health_score":
                    prediction["health_score"],

            })


        # =================================================
        # FLIGHT-LEVEL RESULT
        # =================================================

        if flight_predictions:

            prediction_counts = (
                pd.Series(
                    flight_predictions
                )
                .value_counts()
            )

            flight_prediction = (
                prediction_counts.idxmax()
            )

            correct = (
                flight_prediction ==
                flight_fault
            )

            flight_results.append({

                "flight_id":
                    file.stem,

                "actual_fault":
                    flight_fault,

                "predicted_fault":
                    flight_prediction,

                "correct":
                    correct,

                "evaluated_windows":
                    len(flight_predictions),

            })

            print(
                f"{file.stem:30s} "
                f"Actual: {flight_fault:22s} "
                f"Predicted: {flight_prediction:22s} "
                f"{'✓' if correct else '✗'}"
            )


    # =====================================================
    # WINDOW-LEVEL RESULTS
    # =====================================================

    print("\n" + "=" * 75)
    print("FAULT-WINDOW RESULTS")
    print("=" * 75)


    if not true_labels:

        print(
            "\nNo fault-containing windows "
            "were available for evaluation."
        )

        return


    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )


    print(
        f"\nFault-containing windows : "
        f"{len(true_labels)}"
    )

    print(
        f"Accuracy                 : "
        f"{accuracy:.4f}"
    )

    print(
        f"Accuracy (%)             : "
        f"{accuracy * 100:.2f}%"
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            true_labels,
            predicted_labels,
            zero_division=0
        )
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    labels = sorted(
        set(true_labels) |
        set(predicted_labels)
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels
    )

    print(
        "Confusion Matrix:\n"
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)


    # =====================================================
    # FLIGHT-LEVEL RESULTS
    # =====================================================

    flight_df = pd.DataFrame(
        flight_results
    )

    if not flight_df.empty:

        flight_accuracy = (
            flight_df["correct"]
            .mean()
        )

        print("\n" + "=" * 75)
        print("FLIGHT-LEVEL FAULT DETECTION")
        print("=" * 75)

        print(
            f"\nFlights evaluated : "
            f"{len(flight_df)}"
        )

        print(
            f"Flight accuracy   : "
            f"{flight_accuracy:.4f}"
        )

        print(
            f"Flight accuracy % : "
            f"{flight_accuracy * 100:.2f}%"
        )


    # =====================================================
    # SAVE WINDOW RESULTS
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    window_output = (
        PROJECT_ROOT
        / "models"
        / "classification"
        / "fault_window_validation.csv"
    )

    results_df.to_csv(
        window_output,
        index=False
    )


    # =====================================================
    # SAVE FLIGHT RESULTS
    # =====================================================

    flight_output = (
        PROJECT_ROOT
        / "models"
        / "classification"
        / "flight_validation.csv"
    )

    flight_df.to_csv(
        flight_output,
        index=False
    )


    print(
        f"\nWindow results saved to:"
        f"\n{window_output}"
    )

    print(
        f"\nFlight results saved to:"
        f"\n{flight_output}"
    )


    print("\n" + "=" * 75)
    print(
        "VALIDATION COMPLETE"
    )
    print("=" * 75)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()