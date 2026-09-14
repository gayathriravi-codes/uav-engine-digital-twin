import random

def get_mock_data():
    return {
        "timestamp": random.randint(1, 100),
        "rpm": random.randint(2000, 3000),
        "egt": random.randint(600, 750),
        "cht": random.randint(150, 220),
        "oil_pressure": random.randint(40, 70),
        "oil_temp": random.randint(80, 110),
        "vibration": round(random.uniform(0.1, 1.0), 2),
        "fuel_flow": random.randint(10, 20),
        "health_score": random.randint(40, 90),
        "recovery_rate": round(random.uniform(0.1, 1.0), 2),
        "fault_type": random.choice(["None", "Overheating", "Misfire"]),
        "fault_confidence": round(random.uniform(0.5, 0.95), 2),
        "rul_estimate_minutes": random.randint(10, 30),
        "rul_lower_bound_minutes": random.randint(5, 10),
        "suggested_actions": [
            {"action": "Reduce throttle", "projected_rul_minutes": random.randint(20, 40)}
        ]
    }