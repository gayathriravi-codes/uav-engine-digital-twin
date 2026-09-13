"""
Shared data schema for the AeroTwin project.
Every module (simulator, models, dashboard, integration) imports this
so field names, types, and ranges never drift out of sync between teammates.
"""

SENSOR_FIELDS = [
    "rpm", "egt", "cht", "oil_pressure", "oil_temp", "vibration", "fuel_flow",
]

HEALTHY_RANGES = {
    "rpm": (4800, 5200),
    "egt": (680, 760),
    "cht": (150, 190),
    "oil_pressure": (55, 75),
    "oil_temp": (85, 110),
    "vibration": (0.5, 2.0),
    "fuel_flow": (18, 24),
}

FAULT_TYPES = [
    "none", "misfire", "overheat", "cooling_degradation",
    "oil_issue", "sensor_drift", "vibration_fault",
]

WINDOW_SIZE = 30
STRIDE = 5