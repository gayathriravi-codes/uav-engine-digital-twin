from models.rul.whatif_engine import run_whatif

r = run_whatif(fault_type='vibration', onset_idx=141, duration=300, window_start=290)
for x in r:
    print(x['label'], x['point_estimate_timesteps'], x['delta_minutes'])