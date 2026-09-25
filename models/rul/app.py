"""
app.py -- Minimal Flask API server exposing the RUL ensemble to the frontend.

Run:  python app.py
Then it listens on http://localhost:5000

Loads the ensemble ONCE at startup (not per-request -- loading models on
every request would be very slow and is unnecessary).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

from train_rul_ensemble import load_ensemble, predict_rul_ensemble, build_inference_window

app = Flask(__name__)
CORS(app)  # allows the frontend (different port) to call this API

print("Loading trained ensemble ...")
models, scaler, dropped_idx_list = load_ensemble()
print("Ensemble loaded. Server ready.")


@app.route("/api/health", methods=["GET"])
def health():
    """Quick check that the server + model are up. Hit this first from the browser."""
    return jsonify({"status": "ok", "n_variants": len(models)})


@app.route("/api/rul/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
      {
        "raw_window": [[s1,s2,...,s7], [s1,s2,...,s7], ...],  # window_size rows x 7 sensors
        "elapsed_fraction": 0.42   # REQUIRED, see build_inference_window's docstring
      }

    Returns JSON:
      {
        "point_estimate_timesteps": ...,
        "rul_lower_bound_timesteps": ...,
        "std_timesteps": ...,
        ... (plus _minutes aliases, same values)
      }
    """
    data = request.get_json()

    if "raw_window" not in data:
        return jsonify({"error": "missing 'raw_window' in request body"}), 400
    if "elapsed_fraction" not in data:
        return jsonify({"error": "missing 'elapsed_fraction' in request body -- required, see backend notes"}), 400

    raw_window = np.array(data["raw_window"], dtype=float)
    elapsed_fraction = float(data["elapsed_fraction"])

    try:
        featurized_window = build_inference_window(raw_window, elapsed_fraction)
        result = predict_rul_ensemble(featurized_window, models, scaler, dropped_idx_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rul/mock", methods=["GET"])
def mock():
    """
    Fallback/demo-safe endpoint: returns a plausible-looking fake prediction
    with NO model call at all. Use this if the real model pipeline has any
    issue mid-demo and you need something that will not crash on stage.
    """
    return jsonify({
        "point_estimate_timesteps": 187.4,
        "rul_lower_bound_timesteps": 162.1,
        "std_timesteps": 12.8,
        "point_estimate_minutes": 187.4,
        "rul_lower_bound_minutes": 162.1,
        "std_minutes": 12.8,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)