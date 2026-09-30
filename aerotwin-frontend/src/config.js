/**
 * src/config.js
 * Central configuration for NIDAN — UAV Engine Digital Twin Operator Console.
 * The project name is defined ONLY here. Never hard-code "AeroTwin" anywhere.
 */

/* ── Project Identity ─────────────────────────────────────────── */
export const PROJECT_NAME = "NIDAN";
export const SUBTITLE = "AI-Enabled Digital Twin for Aero Piston Engine Health Monitoring";
export const HACKATHON_TAG = "Smart India Hackathon 2026 · PS 26054 · DRDO";

/* ── Team (edit these freely) ─────────────────────────────────── */
export const TEAM_NAME = "Team NIDAN";
export const TEAM_MEMBERS = [
  "Member 1",
  "Member 2",
  "Member 3",
  "Member 4",
  "Member 5",
  "Member 6",
];

/* ── API ──────────────────────────────────────────────────────── */
export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ── Page Routes ──────────────────────────────────────────────── */
export const PAGES = [
  { id: "overview",       path: "/",              label: "Overview",          icon: "LayoutDashboard" },
  { id: "live-monitor",   path: "/live-monitor",  label: "Live Monitor",      icon: "Activity" },
  { id: "diagnosis",      path: "/diagnosis",     label: "Diagnosis",         icon: "Search" },
  { id: "remaining-life", path: "/remaining-life", label: "Remaining Life",   icon: "Clock" },
  { id: "mission",        path: "/mission",       label: "Mission & Operator", icon: "Shield" },
  { id: "what-if",        path: "/what-if",       label: "What-If",           icon: "GitBranch" },
  { id: "replay",         path: "/replay",        label: "Replay",            icon: "Rewind" },
  { id: "audit-log",      path: "/audit-log",     label: "Audit Log",         icon: "FileText" },
  { id: "about",          path: "/about",         label: "About & Limits",    icon: "Info" },
];

/* ── Fault Types ──────────────────────────────────────────────── */
export const FAULT_TYPES = [
  { id: "none",                  label: "None (Healthy baseline)" },
  { id: "misfire",               label: "Misfire" },
  { id: "overheat",              label: "Overheat" },
  { id: "cooling_degradation",   label: "Cooling Degradation" },
  { id: "oil_issue",             label: "Oil Issue" },
  { id: "sensor_drift",          label: "Sensor Drift" },
  { id: "vibration_fault",       label: "Vibration Fault" },
];

/* ── Fault Library (detailed descriptions for Diagnosis page) ── */
export const FAULT_LIBRARY = {
  misfire: {
    id: "misfire",
    title: "Misfire",
    subtitle: "Combustion instability",
    description:
      "Intermittent loss of combustion in one or more cylinders causes RPM drops, EGT spikes, and elevated vibration. Often cyclic in nature.",
    affectedSensors: ["rpm", "egt", "vibration", "fuel_flow"],
    signature: "RPM jitter with periodic EGT spikes and vibration harmonics.",
  },
  overheat: {
    id: "overheat",
    title: "Overheat",
    subtitle: "Thermal surge",
    description:
      "Rapid rise in cylinder head and exhaust gas temperatures. Caused by cooling system failure, lean mixture, or excessive load. Oil temperature follows.",
    affectedSensors: ["cht", "egt", "oil_temp", "oil_pressure"],
    signature: "CHT and EGT climb dramatically; oil pressure drops under thermal stress.",
  },
  cooling_degradation: {
    id: "cooling_degradation",
    title: "Cooling Degradation",
    subtitle: "Gradual thermal creep",
    description:
      "Slow, progressive rise in CHT and oil temperature due to degraded cooling fins, blocked airflow, or thermostat failure. Harder to detect than overheat.",
    affectedSensors: ["cht", "oil_temp", "egt"],
    signature: "Steady upward drift in CHT and oil temp over many timesteps.",
  },
  oil_issue: {
    id: "oil_issue",
    title: "Oil Issue",
    subtitle: "Loss of lubrication pressure",
    description:
      "Oil pressure drops sharply due to pump failure, leak, or filter blockage. Oil temperature rises from increased friction. Bearing damage risk.",
    affectedSensors: ["oil_pressure", "oil_temp", "cht", "vibration"],
    signature: "Oil pressure plummets while oil temp and friction-induced vibration rise.",
  },
  sensor_drift: {
    id: "sensor_drift",
    title: "Sensor Drift",
    subtitle: "Instrument bias or freeze",
    description:
      "One sensor reports physically implausible values (drifting upward or frozen) while the actual engine remains healthy. Trust flag marks it as untrusted.",
    affectedSensors: ["cht"],
    signature: "CHT reading diverges from physical reality; trust flag goes untrusted.",
  },
  vibration_fault: {
    id: "vibration_fault",
    title: "Vibration Fault",
    subtitle: "Mechanical bearing or propeller imbalance",
    description:
      "Accelerometer detects excessive vibration from worn bearings, propeller imbalance, or shaft misalignment. RPM may flutter slightly.",
    affectedSensors: ["vibration", "rpm"],
    signature: "Vibration index surges well above normal range; RPM shows minor flutter.",
  },
};

/* ── Quick Scenario Buttons (for Overview page) ───────────────── */
export const QUICK_SCENARIOS = [
  { fault: "none",                severity: 0.0,  label: "Healthy Flight",       icon: "Heart" },
  { fault: "overheat",            severity: 0.8,  label: "Overheat",             icon: "Flame" },
  { fault: "oil_issue",           severity: 0.7,  label: "Oil Problem",          icon: "Droplets" },
  { fault: "sensor_drift",        severity: 0.6,  label: "Sensor Drift",         icon: "AlertTriangle" },
  { fault: "misfire",             severity: 0.7,  label: "Misfire",              icon: "Zap" },
  { fault: "cooling_degradation", severity: 0.65, label: "Cooling Degradation",  icon: "Thermometer" },
  { fault: "vibration_fault",     severity: 0.75, label: "Vibration Fault",      icon: "Radio" },
];

/* ── Mission Actions (dark-theme colours) ─────────────────────── */
export const MISSION_ACTIONS = {
  CONTINUE: {
    id: "CONTINUE",
    label: "Continue mission",
    color: "#14B8A6",
    bg: "rgba(20,184,166,0.15)",
    border: "rgba(20,184,166,0.4)",
    icon: "CheckCircle",
    description: "Parameters nominal. Engine operates within design envelope.",
  },
  REDUCE_LOAD: {
    id: "REDUCE_LOAD",
    label: "Reduce engine load",
    color: "#F59E0B",
    bg: "rgba(245,158,11,0.15)",
    border: "rgba(245,158,11,0.4)",
    icon: "AlertTriangle",
    description: "Thermal or mechanical stress detected. Throttle back to 70% cruise.",
  },
  RTB: {
    id: "RTB",
    label: "Return to base",
    color: "#F97316",
    bg: "rgba(249,115,22,0.15)",
    border: "rgba(249,115,22,0.4)",
    icon: "ArrowLeftCircle",
    description: "Progressive degradation threshold breached. Abort loiter and return.",
  },
  ABORT: {
    id: "ABORT",
    label: "Emergency abort",
    color: "#EF4444",
    bg: "rgba(239,68,68,0.15)",
    border: "rgba(239,68,68,0.4)",
    icon: "XOctagon",
    description: "Catastrophic failure imminent. Immediate forced glide.",
  },
};

/* ── Sensor Metadata ──────────────────────────────────────────── */
export const SENSOR_METADATA = {
  rpm: {
    id: "rpm",
    name: "Engine RPM",
    shortName: "RPM",
    unit: "rpm",
    healthyRange: [4800, 5200],
    warningRange: [4600, 5400],
    physicalLimits: [0, 8000],
    precision: 0,
    description: "Crankshaft rotational velocity at the front reduction gear hub.",
  },
  egt: {
    id: "egt",
    name: "Exhaust Gas Temp",
    shortName: "EGT",
    unit: "°C",
    healthyRange: [680, 760],
    warningRange: [650, 810],
    physicalLimits: [0, 1200],
    precision: 1,
    description: "Combustion exhaust temperature at the primary collector port.",
  },
  cht: {
    id: "cht",
    name: "Cylinder Head Temp",
    shortName: "CHT",
    unit: "°C",
    healthyRange: [150, 190],
    warningRange: [140, 210],
    physicalLimits: [0, 300],
    precision: 1,
    description: "Cylinder head temperature near the spark plug thermocouple boss.",
  },
  oil_pressure: {
    id: "oil_pressure",
    name: "Oil Pressure",
    shortName: "Oil Press",
    unit: "PSI",
    healthyRange: [55, 75],
    warningRange: [45, 85],
    physicalLimits: [0, 150],
    precision: 1,
    description: "Main lubrication line pressure before bearing distribution.",
  },
  oil_temp: {
    id: "oil_temp",
    name: "Oil Temperature",
    shortName: "Oil Temp",
    unit: "°C",
    healthyRange: [85, 110],
    warningRange: [75, 125],
    physicalLimits: [0, 200],
    precision: 1,
    description: "Lubricating fluid temperature inside the oil scavenge sump.",
  },
  fuel_flow: {
    id: "fuel_flow",
    name: "Fuel Flow Rate",
    shortName: "Fuel Flow",
    unit: "L/h",
    healthyRange: [18, 24],
    warningRange: [15, 28],
    physicalLimits: [0, 50],
    precision: 1,
    description: "Mass flow meter in the electronic fuel injection rail.",
  },
  vibration: {
    id: "vibration",
    name: "Vibration Index",
    shortName: "Vibration",
    unit: "g",
    healthyRange: [0.5, 2.0],
    warningRange: [0.3, 3.2],
    physicalLimits: [0, 10],
    precision: 2,
    description: "Tri-axial accelerometer on the upper crankcase spine.",
  },
};

export const SENSOR_KEYS = Object.keys(SENSOR_METADATA);

/* ── Six Pillar Cards (About page) ────────────────────────────── */
export const PILLAR_CARDS = [
  {
    title: "Complete",
    description:
      "Covers the full loop: 7 sensors, sensor trust, fault diagnosis, RUL estimation, mission action, and operator confirmation.",
  },
  {
    title: "Accurate",
    description:
      "XGBoost classifier achieves high accuracy on synthetic benchmarks. Calibrated RUL bounds quantify uncertainty honestly.",
  },
  {
    title: "Quiet",
    description:
      "Avoids false alarms through multi-level escalation (CONTINUE → REDUCE_LOAD → RTB → ABORT) and confidence thresholds.",
  },
  {
    title: "Robust",
    description:
      "Handles sensor drift and untrusted readings without crashing. Falls back gracefully from API to simulator.",
  },
  {
    title: "Trustworthy",
    description:
      "Human operator stays in the loop for high-stakes actions. All decisions are logged in an immutable audit trail.",
  },
  {
    title: "Deployable",
    description:
      "Lightweight frontend runs on any browser. Backend inference under 10ms on standard hardware. No GPU required.",
  },
];

/* ── Known Limits ─────────────────────────────────────────────── */
export const KNOWN_LIMITS = [
  "All data is synthetic. Not validated on real engines.",
  "Inference latency measured on a laptop only.",
  "RUL is least reliable near end of life.",
  "Sensor drift is hard to detect from the sensor stream alone.",
  "The anomaly detector is advisory, not authoritative.",
  "Remaining mission time is entered manually by the operator.",
];

/* ── System Limits (detailed cards) ───────────────────────────── */
export const SYSTEM_LIMITS = [
  {
    title: "Synthetic telemetry",
    text: "All streaming telemetry and counterfactual flights are synthetically generated. Not validated on certified flight hardware.",
  },
  {
    title: "Inference latency",
    text: "Model inference times measured on standard mobile workstations (~10ms CPU inference).",
  },
  {
    title: "RUL uncertainty at end-of-life",
    text: "Remaining useful life estimates carry increased variance when RUL < 30 timesteps due to nonlinear accelerated wear.",
  },
  {
    title: "Sensor trust limitations",
    text: "Single-sensor drift detection relies on physical rate-of-change and flatline tests. Slow multi-sensor correlated bias may require cross-validation.",
  },
  {
    title: "Advisory recommendations",
    text: "Automated recommendations (CONTINUE, REDUCE_LOAD, RTB, ABORT) are advisory decision aids. Final command resides with the UAV operator.",
  },
];
