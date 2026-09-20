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
# Physically impossible bounds - looser than HEALTHY_RANGES. A value
# outside HEALTHY_RANGES can still be a genuine fault (that's what the
# classifier is for); a value outside PHYSICAL_LIMITS means the sensor
# itself is lying, not that the engine is in a bad state. Bounds are a
# first pass based on generous margins around HEALTHY_RANGES - tune
# against real hardware datasheets if available.
PHYSICAL_LIMITS = {
    "rpm": (0, 8000),
    "egt": (0, 1200),
    "cht": (0, 300),
    "oil_pressure": (0, 150),
    "oil_temp": (0, 200),
    "vibration": (0, 10),
    "fuel_flow": (0, 50),
}

# Max plausible change between consecutive readings (one timestep).
# A jump larger than this indicates a glitch/dropout, not real engine
# dynamics - same "tune against real hardware" caveat as above.
MAX_RATE_OF_CHANGE = {
    "rpm": 500,
    "egt": 100,
    "cht": 50,
    "oil_pressure": 20,
    "oil_temp": 20,
    "vibration": 3,
    "fuel_flow": 10,
}

# Flatline detection: if a sensor's std within a window is below this,
# treat it as suspiciously constant (stuck sensor) rather than genuinely
# stable engine behavior.
FLATLINE_STD_THRESHOLD = {
    "rpm": 1.0,
    "egt": 0.5,
    "cht": 0.5,
    "oil_pressure": 0.1,
    "oil_temp": 0.1,
    "vibration": 0.01,
    "fuel_flow": 0.1,
}