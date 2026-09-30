/**
 * src/data/simSource.js
 * In-browser telemetry simulator generating plausible aero piston engine dynamics
 * for healthy flight and 6 injection fault modes.
 */

import { normalizeTick } from "./normalizeTick";

// Deterministic Pseudo-Random Number Generator (Mulberry32)
export function createRNG(seed = 42) {
  let s = (Math.abs(seed) || 42) >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a single simulated tick at timestep t
 */
export function generateSimTick({
  t = 0,
  fault = "none",
  severity = 0.5,
  seed = 42,
  onsetTick = 2,
  prevTick = null,
  tickSeconds = 1.0,
}) {
  const rng = createRNG(seed + t * 997);
  const noise = (scale = 1) => (rng() - 0.5) * 2 * scale;
  const gaussianApprox = (scale = 1) => (rng() + rng() + rng() - 1.5) * (2 / 3) * scale;

  // Normalized ramp factor [0, 1] once past onsetTick
  const progress = Math.max(0, Math.min(1, (t - onsetTick) / 4));
  const effectiveSeverity = t >= onsetTick ? severity * Math.max(0.4, progress) : 0;

  // Healthy baseline parameters (Rotax 914 / Boxer 4-cylinder UAV profile)
  let rpm = 5000 + noise(12);
  let egt = 715 + noise(3.5);
  let cht = 168 + noise(1.8);
  let oil_pressure = 65 + noise(0.8);
  let oil_temp = 96 + noise(0.9);
  let fuel_flow = 21.2 + noise(0.25);
  let vibration = 1.15 + noise(0.06);

  // Trust flags map
  const trust = {
    rpm: true,
    egt: true,
    cht: true,
    oil_pressure: true,
    oil_temp: true,
    fuel_flow: true,
    vibration: true,
  };
  const untrusted_sensors = [];

  // Fault injection dynamics
  if (t >= onsetTick && severity > 0) {
    switch (fault) {
      case "misfire": {
        // Combustion instability: RPM drop with intermittent spikes, EGT fluctuations, vibration jump
        const misfireCycle = (t % 3 === 0) || rng() < (0.35 * effectiveSeverity);
        if (misfireCycle) {
          rpm -= 280 * effectiveSeverity + noise(40);
          egt += 75 * effectiveSeverity + noise(20);
          vibration += 1.8 * effectiveSeverity + noise(0.3);
          fuel_flow += 1.8 * effectiveSeverity;
        } else {
          rpm -= 90 * effectiveSeverity;
          egt -= 35 * effectiveSeverity;
          vibration += 0.6 * effectiveSeverity;
        }
        break;
      }

      case "overheat": {
        // Thermal surge: CHT & EGT climb dramatically, oil temp follows
        cht += 82 * effectiveSeverity + gaussianApprox(4);
        egt += 180 * effectiveSeverity + gaussianApprox(7);
        oil_temp += 36 * effectiveSeverity + gaussianApprox(2);
        oil_pressure -= 12 * effectiveSeverity;
        break;
      }

      case "cooling_degradation": {
        // Gradual thermal drift: CHT and oil temp creep upwards at steady rate
        cht += 54 * effectiveSeverity + noise(2);
        oil_temp += 28 * effectiveSeverity + noise(1.5);
        egt += 25 * effectiveSeverity + noise(3);
        break;
      }

      case "oil_issue": {
        // Lubrication breakdown: Oil pressure drops sharply, oil temp escalates, friction increases
        oil_pressure -= 45 * effectiveSeverity + noise(2);
        oil_pressure = Math.max(12, oil_pressure);
        oil_temp += 52 * effectiveSeverity + noise(3);
        cht += 24 * effectiveSeverity + noise(2);
        vibration += 0.8 * effectiveSeverity;
        break;
      }

      case "sensor_drift": {
        // Instrument sensor drift / freeze:
        // CHT reading drifts upward or freezes, while physical engine remains nominal!
        // Sensor trust flags mark it untrusted!
        cht += 110 * effectiveSeverity; // The sensor is drifting / lying
        trust.cht = false;
        untrusted_sensors.push("cht");
        // physical engine RPM, oil, vibration, EGT stay healthy
        break;
      }

      case "vibration_fault": {
        // Mechanical imbalance / bearing wear: vibration surges, RPM slight flutter
        vibration += 5.5 * effectiveSeverity + gaussianApprox(0.4);
        rpm += noise(25) * effectiveSeverity;
        break;
      }

      case "none":
      default:
        // Stable baseline flight
        break;
    }
  }

  // Calculate Health Score (100 -> 0)
  let healthDegradation = 0;
  if (fault === "sensor_drift") {
    // Sensor drift does not impair mechanical health, but lowers sensor network score slightly
    healthDegradation = effectiveSeverity * 12;
  } else {
    // Mechanical faults impair physical health proportionally to effective severity
    healthDegradation = effectiveSeverity * 85;
  }
  const health_score = Math.max(8, Math.round(100 - healthDegradation - Math.abs(noise(1.5))));

  // Condition Status
  let condition_status = "Nominal";
  if (health_score <= 35) {
    condition_status = "Critical Failure";
  } else if (health_score <= 60) {
    condition_status = "Degraded";
  } else if (health_score <= 82) {
    condition_status = "Watchlist Advisory";
  }

  // Fault classification & confidence
  const predicted_fault = (t >= onsetTick && effectiveSeverity > 0.15) ? fault : "none";
  let confidence = 0.94;
  if (fault === "none" || t < onsetTick) {
    confidence = 0.98 - rng() * 0.05;
  } else {
    // When severity is moderate (~0.5 - 0.7), confidence can be around 0.75 - 0.82 to test operator confirmation gate!
    if (severity < 0.75) {
      confidence = 0.78 + (rng() * 0.04); // Lower confidence (< 0.85) to trigger operator gate on RTB
    } else {
      confidence = 0.91 + (rng() * 0.06);
    }
  }

  // Class probability distribution across 7 classes
  const class_probs = {
    none: 0, misfire: 0, overheat: 0,
    cooling_degradation: 0, oil_issue: 0,
    sensor_drift: 0, vibration_fault: 0
  };

  if (predicted_fault === 'none' || t < onsetTick) {
    class_probs.none = 0.85 + rng() * 0.12;
    // distribute rest among faults
    const remaining = 1 - class_probs.none;
    const faultKeys = ['misfire','overheat','cooling_degradation','oil_issue','sensor_drift','vibration_fault'];
    const weights = faultKeys.map(() => rng());
    const wSum = weights.reduce((a,b) => a+b, 0);
    faultKeys.forEach((k,i) => { class_probs[k] = (remaining * weights[i] / wSum); });
  } else {
    // Active fault gets ~confidence probability
    class_probs[predicted_fault] = confidence;
    const remaining = 1 - confidence;
    const otherKeys = Object.keys(class_probs).filter(k => k !== predicted_fault);
    const weights = otherKeys.map(() => rng());
    const wSum = weights.reduce((a,b) => a+b, 0);
    otherKeys.forEach((k,i) => { class_probs[k] = (remaining * weights[i] / wSum); });
  }

  // Calibrated Remaining Useful Life (RUL) in TIMESTEPS
  // Base healthy UAV mission life ~320 timesteps
  let baseRul = Math.max(10, 320 - t);
  if (fault !== "none" && t >= onsetTick) {
    if (fault === "sensor_drift") {
      // Sensor drift doesn't degrade actual engine lifespan
      baseRul = Math.max(10, baseRul - 10);
    } else {
      const dropFactor = (fault === "overheat" || fault === "oil_issue") ? 2.4 : 1.6;
      baseRul = Math.max(
        12,
        Math.round(baseRul - (effectiveSeverity * 240 * dropFactor * Math.min(1, (t - onsetTick) / 25)))
      );
    }
  }
  const rul_estimate_timesteps = baseRul;
  // Calibrated uncertainty band in timesteps
  const rul_uncertainty_timesteps = Math.max(
    5,
    Math.round(rul_estimate_timesteps * (0.12 + 0.15 * effectiveSeverity))
  );
  // Calibrated lower bound in timesteps
  const rul_lower_bound_timesteps = Math.max(
    0,
    rul_estimate_timesteps - rul_uncertainty_timesteps
  );

  // Risk Score (0 - 100)
  let risk_score = Math.min(
    100,
    Math.round(
      fault === "sensor_drift"
        ? 15 + effectiveSeverity * 25
        : (100 - health_score) * 1.05 + effectiveSeverity * 15
    )
  );
  risk_score = Math.max(4, risk_score);

  // Mission Action progression: CONTINUE -> REDUCE_LOAD -> RTB -> ABORT
  let action = "CONTINUE";
  const reasons = [];

  if (fault === "none" || t < onsetTick || effectiveSeverity < 0.15) {
    action = "CONTINUE";
    reasons.push("All engine parameters within certified flight limits.");
    reasons.push("Telemetry baseline nominal, zero anomalous spectral spikes.");
  } else if (effectiveSeverity < 0.45) {
    action = "REDUCE_LOAD";
    reasons.push(`Early signature detected: ${fault.replace("_", " ")} onset.`);
    reasons.push("Throttle reduction to 70% recommended to attenuate stress.");
  } else if (effectiveSeverity < 0.8) {
    action = "RTB";
    reasons.push(`Progressive degradation: ${fault.replace("_", " ")} escalating.`);
    reasons.push(`Calibrated RUL lower bound decreased to ${rul_lower_bound_timesteps} timesteps.`);
    reasons.push("Exceeds loiter margin; abort mission objective and return to airfield.");
  } else {
    action = "ABORT";
    reasons.push(`Severe critical fault: ${fault.replace("_", " ")} breach.`);
    reasons.push("Structural integrity threshold breached; imminent component seizure.");
    reasons.push("Immediate forced glide / recovery parachute deployment suggested.");
  }

  // Operator confirmation gate:
  // When action is RTB or ABORT and confidence is below 0.85, set requires_confirmation = true
  const requires_confirmation = (action === "RTB" || action === "ABORT") && confidence < 0.85;

  // SHAP Feature Attribution Explanations
  const explanationsByFault = {
    none: [
      { feature: "Cylinder head temp (CHT)", contribution: 0.12 },
      { feature: "Vibration baseline", contribution: 0.10 },
      { feature: "Oil pressure stability", contribution: 0.08 },
      { feature: "Exhaust gas temp (EGT)", contribution: 0.05 },
    ],
    overheat: [
      { feature: "CHT gradient surge", contribution: 0.52 },
      { feature: "EGT collector spike", contribution: 0.31 },
      { feature: "Oil scavenge temp delta", contribution: 0.12 },
      { feature: "Combustion efficiency drop", contribution: 0.05 },
    ],
    cooling_degradation: [
      { feature: "CHT upward creep", contribution: 0.44 },
      { feature: "Oil sump temp retention", contribution: 0.36 },
      { feature: "Heat dissipation rate", contribution: 0.14 },
      { feature: "Exhaust manifold coupling", contribution: 0.06 },
    ],
    misfire: [
      { feature: "RPM spectral jitter", contribution: 0.48 },
      { feature: "EGT intermittent drop", contribution: 0.32 },
      { feature: "Torsional vibration harmonics", contribution: 0.15 },
      { feature: "Fuel flow fluctuation", contribution: 0.05 },
    ],
    oil_issue: [
      { feature: "Oil pressure loss", contribution: 0.58 },
      { feature: "Oil scavenge thermal rise", contribution: 0.24 },
      { feature: "Hydrodynamic friction drag", contribution: 0.12 },
      { feature: "Bearing vibration shift", contribution: 0.06 },
    ],
    sensor_drift: [
      { feature: "CHT physically implausible rate-of-change", contribution: 0.62 },
      { feature: "CHT cross-sensor divergence", contribution: 0.26 },
      { feature: "Oil temp physical coherence", contribution: 0.08 },
      { feature: "Sensor trust anomaly flag", contribution: 0.04 },
    ],
    vibration_fault: [
      { feature: "Crankcase accelerometer amplitude", contribution: 0.65 },
      { feature: "High-order rotational harmonics", contribution: 0.21 },
      { feature: "Propeller balance index", contribution: 0.10 },
      { feature: "Torsional shaft displacement", contribution: 0.04 },
    ],
  };

  const explanation = explanationsByFault[predicted_fault] || explanationsByFault.none;

  const raw = {
    t,
    tick_seconds: tickSeconds,
    sensors: {
      rpm,
      egt,
      cht,
      oil_pressure,
      oil_temp,
      fuel_flow,
      vibration,
    },
    trust,
    untrusted_sensors,
    health_score,
    condition_status,
    predicted_fault,
    class_probs,
    confidence,
    severity,
    rul_estimate_timesteps,
    rul_lower_bound_timesteps,
    rul_uncertainty_timesteps,
    recovery_rate: 0.0,
    recovered: false,
    action,
    risk_score,
    reasons,
    requires_confirmation,
    explanation,
  };

  return normalizeTick(raw, prevTick);
}

/**
 * Generate full sequence of flight ticks for replay scrubbing or what-if comparison
 */
export function generateFlight({
  fault = "none",
  severity = 0.5,
  seed = 42,
  totalTicks = 90,
  onsetTick = 2,
  tickSeconds = 1.0,
}) {
  const ticks = [];
  let prev = null;
  for (let t = 0; t < totalTicks; t++) {
    const tick = generateSimTick({
      t,
      fault,
      severity,
      seed,
      onsetTick,
      prevTick: prev,
      tickSeconds,
    });
    ticks.push(tick);
    prev = tick;
  }
  return ticks;
}
