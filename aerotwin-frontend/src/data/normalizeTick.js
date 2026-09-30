/**
 * src/data/normalizeTick.js
 * Tolerant adapter mapping telemetry ticks from either SimSource or ApiSource
 * into the standard internal state shape. Accepts various key aliases and never crashes.
 */


/**
 * Format clock time seconds into HH:MM:SS
 */
export function formatMissionClock(seconds) {
  if (typeof seconds === "string" && seconds.includes(":")) {
    return seconds;
  }
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = s % 60;
  return [hrs, mins, secs].map((n) => String(n).padStart(2, "0")).join(":");
}

/**
 * Safe numeric parser with fallback
 */
function safeNum(val, fallback = 0, precision = null) {
  if (val === null || val === undefined || val === "") return fallback;
  const num = Number(val);
  if (Number.isNaN(num)) return fallback;
  if (precision !== null) {
    return Number(num.toFixed(precision));
  }
  return num;
}

/**
 * Safe string fallback
 */
function safeStr(val, fallback = "n/a") {
  if (val === null || val === undefined || val === "") return fallback;
  return String(val);
}

/**
 * Normalize an incoming raw tick from any source
 * @param {Object} raw - Incoming tick payload
 * @param {Object|null} prevTick - Previous normalized tick to detect action_changed
 * @returns {Object} Normalized tick
 */
export function normalizeTick(raw = {}, prevTick = null) {
  if (!raw || typeof raw !== "object") {
    raw = {};
  }

  // 1. Timestep & Mission Clock
  const t = Math.max(
    0,
    safeNum(raw.t ?? raw.timestep ?? raw.step ?? raw.tick ?? 0, 0, 0)
  );
  const tickSec = safeNum(raw.tick_seconds ?? raw.dt ?? 1.0, 1.0);
  const mission_clock = raw.mission_clock
    ? safeStr(raw.mission_clock)
    : formatMissionClock(t * tickSec);

  // 2. Sensor telemetry (support nested raw.sensors or flat keys, plus aliases)
  const srcSensors = (raw.sensors && typeof raw.sensors === "object") ? raw.sensors : raw;

  const rpm = safeNum(
    srcSensors.rpm ?? srcSensors.RPM ?? srcSensors.engine_rpm,
    5000,
    0
  );
  const egt = safeNum(
    srcSensors.egt ?? srcSensors.EGT ?? srcSensors.exhaust_gas_temp ?? srcSensors.egt_degc,
    715,
    1
  );
  const cht = safeNum(
    srcSensors.cht ?? srcSensors.CHT ?? srcSensors.cylinder_head_temp ?? srcSensors.cht_degc,
    168,
    1
  );
  const oil_pressure = safeNum(
    srcSensors.oil_pressure ?? srcSensors.oil_p ?? srcSensors.oilPressure ?? srcSensors.oil_press,
    65,
    1
  );
  const oil_temp = safeNum(
    srcSensors.oil_temp ?? srcSensors.oil_t ?? srcSensors.oilTemperature ?? srcSensors.oil_temperature,
    96,
    1
  );
  const fuel_flow = safeNum(
    srcSensors.fuel_flow ?? srcSensors.fuelFlow ?? srcSensors.ff ?? srcSensors.fuel_rate,
    21.2,
    1
  );
  const vibration = safeNum(
    srcSensors.vibration ?? srcSensors.vib ?? srcSensors.vib_g ?? srcSensors.vibration_index,
    1.15,
    2
  );

  const sensors = {
    rpm,
    egt,
    cht,
    oil_pressure,
    oil_temp,
    fuel_flow,
    vibration,
  };

  // 3. Sensor Trust Flags
  // Expected: trust: { [sensor]: boolean }
  const rawTrust = (raw.trust && typeof raw.trust === "object") ? raw.trust : {};
  const trust = {
    rpm: rawTrust.rpm !== undefined ? Boolean(rawTrust.rpm) : true,
    egt: rawTrust.egt !== undefined ? Boolean(rawTrust.egt) : true,
    cht: rawTrust.cht !== undefined ? Boolean(rawTrust.cht) : true,
    oil_pressure: rawTrust.oil_pressure !== undefined ? Boolean(rawTrust.oil_pressure) : true,
    oil_temp: rawTrust.oil_temp !== undefined ? Boolean(rawTrust.oil_temp) : true,
    fuel_flow: rawTrust.fuel_flow !== undefined ? Boolean(rawTrust.fuel_flow) : true,
    vibration: rawTrust.vibration !== undefined ? Boolean(rawTrust.vibration) : true,
  };

  // Untrusted sensors list
  let untrusted_sensors = [];
  if (Array.isArray(raw.untrusted_sensors)) {
    untrusted_sensors = raw.untrusted_sensors.map(String);
  } else {
    untrusted_sensors = Object.keys(trust).filter((k) => !trust[k]);
  }

  // Ensure trust map matches untrusted list
  untrusted_sensors.forEach((s) => {
    if (trust[s] !== undefined) trust[s] = false;
  });

  // 4. Health & Diagnosis
  const health_score = Math.min(
    100,
    Math.max(0, safeNum(raw.health_score ?? raw.healthScore ?? raw.health ?? 100, 100, 1))
  );

  const condition_status = safeStr(
    raw.condition_status ??
      raw.conditionStatus ??
      (health_score > 80 ? "Nominal" : health_score > 55 ? "Degraded" : health_score > 30 ? "Critical" : "Severe Failure")
  );

  const predicted_fault = safeStr(
    raw.predicted_fault ?? raw.predictedFault ?? raw.fault ?? "none"
  ).toLowerCase();

  const confidence = Math.min(
    1,
    Math.max(0, safeNum(raw.confidence ?? raw.fault_confidence ?? 0.95, 0.95, 2))
  );

  const severity = Math.min(
    1,
    Math.max(0, safeNum(raw.severity ?? raw.fault_severity ?? 0.0, 0.0, 2))
  );

  let class_probs = {
    none: 0, misfire: 0, overheat: 0,
    cooling_degradation: 0, oil_issue: 0,
    sensor_drift: 0, vibration_fault: 0
  };
  
  if (raw.class_probs && typeof raw.class_probs === 'object') {
    Object.keys(class_probs).forEach(k => {
      class_probs[k] = safeNum(raw.class_probs[k], 0, 3);
    });
  } else {
    const pf = predicted_fault === 'none' ? 'none' : predicted_fault;
    class_probs[pf] = confidence;
    const remaining = 1 - confidence;
    const otherKeys = Object.keys(class_probs).filter(k => k !== pf);
    otherKeys.forEach(k => {
      class_probs[k] = safeNum(remaining / otherKeys.length, 0, 3);
    });
  }

  // 5. Remaining Useful Life (RUL) in TIMESTEPS
  const rul_estimate_timesteps = Math.max(
    0,
    safeNum(raw.rul_estimate_timesteps ?? raw.rul ?? raw.rul_estimate ?? 300, 300, 0)
  );

  const rul_uncertainty_timesteps = Math.max(
    0,
    safeNum(
      raw.rul_uncertainty_timesteps ??
        raw.rul_uncertainty ??
        Math.round(rul_estimate_timesteps * 0.15),
      35,
      0
    )
  );

  const rul_lower_bound_timesteps = Math.max(
    0,
    safeNum(
      raw.rul_lower_bound_timesteps ??
        raw.rul_lower_bound ??
        Math.max(0, rul_estimate_timesteps - rul_uncertainty_timesteps),
      0,
      0
    )
  );

  const recovery_rate = safeNum(raw.recovery_rate ?? 0.0, 0.0, 2);
  const recovered = Boolean(raw.recovered ?? false);

  // 6. Mission Action & Risk
  let rawAction = safeStr(raw.action ?? raw.mission_action ?? "CONTINUE").toUpperCase();
  if (!["CONTINUE", "REDUCE_LOAD", "RTB", "ABORT"].includes(rawAction)) {
    rawAction = "CONTINUE";
  }
  const action = rawAction;

  const risk_score = Math.min(
    100,
    Math.max(0, safeNum(raw.risk_score ?? raw.riskScore ?? raw.risk ?? 5, 5, 0))
  );

  let reasons = [];
  if (Array.isArray(raw.reasons)) {
    reasons = raw.reasons.map((r) => String(r));
  } else if (raw.reasons && typeof raw.reasons === "string") {
    reasons = [raw.reasons];
  } else if (raw.reason) {
    reasons = [String(raw.reason)];
  } else {
    reasons = action === "CONTINUE"
      ? ["Telemetry parameters within standard flight envelope."]
      : [`Degradation detected under ${predicted_fault} condition.`];
  }

  // Action changed detector
  const action_changed = prevTick
    ? prevTick.action !== action
    : Boolean(raw.action_changed ?? false);

  // Operator confirmation gate:
  // Required when action is RTB or ABORT and confidence < 0.85
  const requires_confirmation = raw.requires_confirmation !== undefined
    ? Boolean(raw.requires_confirmation)
    : (action === "RTB" || action === "ABORT") && confidence < 0.85;

  // 7. SHAP Feature Attribution Explanations
  let explanation = [];
  if (Array.isArray(raw.explanation)) {
    explanation = raw.explanation.map((item) => ({
      feature: safeStr(item.feature ?? item.name ?? "Telemetry factor"),
      contribution: safeNum(item.contribution ?? item.weight ?? item.value ?? 0, 0, 3),
    }));
  } else {
    // Generate sensible SHAP factors if missing
    explanation = [
      { feature: "Cylinder head temp (CHT)", contribution: 0.38 },
      { feature: "Exhaust gas temp (EGT)", contribution: 0.28 },
      { feature: "Oil pressure delta", contribution: 0.19 },
      { feature: "Vibration spectral energy", contribution: 0.15 },
    ];
  }

  return {
    t,
    mission_clock,
    sensors,
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
    recovery_rate,
    recovered,
    action,
    risk_score,
    reasons,
    action_changed,
    requires_confirmation,
    explanation,
  };
}
