from models.rul.whatif_engine import run_whatif

r = run_whatif(fault_type='vibration_fault', onset_idx=141, duration=300, window_start=252)
for x in r:
    print(x['label'], x['point_estimate_minutes'], x['delta_minutes'])