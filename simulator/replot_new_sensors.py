"""
Re-plot the same three new fault types with a twin y-axis for the
small-scale sensor (injection_timing_deg, fuel_flow), since EGT's ~700
scale was squashing them flat in the original sanity check plot.

Run from simulator/: python replot_new_sensors.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate_healthy import simulate_healthy_flight
from battery_injection_sensors import (
    attach_healthy_battery_injection_columns,
    inject_battery_alternator_fault,
    inject_injection_timing_fault,
    inject_injector_abnormality,
)

healthy = simulate_healthy_flight(duration_timesteps=300, seed=42)
healthy = attach_healthy_battery_injection_columns(healthy, seed=42)

fig, axes = plt.subplots(3, 1, figsize=(9, 12))

# Panel 1: battery -- single scale, no twin axis needed (already looked correct)
faulted = inject_battery_alternator_fault(healthy, onset_frac=0.4, seed=7)
axes[0].plot(faulted["timestamp"], faulted["battery_voltage"], label="BATTERY_VOLTAGE")
axes[0].axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--", label="onset")
axes[0].set_title("battery_alternator_fault")
axes[0].legend(fontsize=7)

# Panel 2: injection timing (~20 scale) vs EGT (~700 scale) -- twin axis
faulted = inject_injection_timing_fault(healthy, onset_frac=0.4, seed=7)
ax2a = axes[1]
ax2b = ax2a.twinx()
l1, = ax2a.plot(faulted["timestamp"], faulted["injection_timing_deg"], color="tab:blue", label="INJECTION_TIMING_DEG")
l2, = ax2b.plot(faulted["timestamp"], faulted["egt"], color="tab:orange", alpha=0.5, label="EGT")
ax2a.axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--")
ax2a.set_ylabel("injection_timing_deg", color="tab:blue")
ax2b.set_ylabel("egt", color="tab:orange")
ax2a.set_title(f"injection_timing_fault (direction={faulted['timing_fault_direction'].iloc[0]})")
ax2a.legend(handles=[l1, l2], fontsize=7)

# Panel 3: fuel_flow (~20 scale) vs EGT (~700 scale) -- twin axis
faulted = inject_injector_abnormality(healthy, onset_frac=0.4, seed=7)
ax3a = axes[2]
ax3b = ax3a.twinx()
l1, = ax3a.plot(faulted["timestamp"], faulted["fuel_flow"], color="tab:blue", label="FUEL_FLOW")
l2, = ax3b.plot(faulted["timestamp"], faulted["egt"], color="tab:orange", alpha=0.5, label="EGT")
ax3a.axvline(faulted["fault_onset_idx"].iloc[0], color="red", linestyle="--")
ax3a.set_ylabel("fuel_flow", color="tab:blue")
ax3b.set_ylabel("egt", color="tab:orange")
ax3a.set_title("injector_abnormality")
ax3a.legend(handles=[l1, l2], fontsize=7)

plt.tight_layout()
plt.savefig("new_sensors_sanity_check_v2.png", dpi=100)
print("Saved new_sensors_sanity_check_v2.png -- open it and check the fuel_flow/timing panels are no longer flat")