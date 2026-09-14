import streamlit as st
import plotly.graph_objects as go
from mock_data import get_mock_data
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")

# ---------------- DATA ----------------
st_autorefresh(interval=2000, key="refresh")
data = get_mock_data()

# ---------------- TITLE ----------------
st.title("✈️ UAV Engine Digital Twin Dashboard")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a);
    color: white;
}

.card {
    background: rgba(17, 24, 39, 0.85);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.glow {
    box-shadow: 0 0 15px rgba(0,255,200,0.2);
}

div[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.9);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("✈️ AeroTwin")

st.sidebar.markdown("### Engine Status")

health = data["health_score"]

if health > 70:
    st.sidebar.success("🟢 Healthy")
elif health > 40:
    st.sidebar.warning("🟡 Moderate")
else:
    st.sidebar.error("🔴 Critical")

st.sidebar.markdown("---")

st.sidebar.markdown("### Quick Info")
st.sidebar.write(f"Recovery Rate: {data['recovery_rate']:.2f}")
st.sidebar.write(f"RUL: {data.get('rul_estimate_minutes', 0)} min")

st.sidebar.markdown("---")
st.sidebar.caption("Digital Twin Dashboard")

# ---------------- TOP STATUS BAR ----------------
st.markdown(f"""
<div style='background:rgba(0,255,200,0.08);
padding:10px;
border-radius:10px;
margin-bottom:20px;
border:1px solid rgba(0,255,200,0.2);'>
🟢 Live Monitoring Active | Timestamp: {data['timestamp']}
</div>
""", unsafe_allow_html=True)

# ---------------- GAUGE FUNCTION ----------------
def create_gauge(title, value, min_val, max_val):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 16, 'color': "white"}},
            number={'font': {'size': 28, 'color': "white"}},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "#00e5ff", 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [min_val, max_val*0.6], 'color': "#064e3b"},
                    {'range': [max_val*0.6, max_val*0.85], 'color': "#78350f"},
                    {'range': [max_val*0.85, max_val], 'color': "#7f1d1d"},
                ],
            }
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )

    return fig

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({"health": health})
df = pd.DataFrame(st.session_state.history)

# ---------------- ENGINE GAUGES ----------------
st.markdown("<div class='card glow'><h3>✈️ Engine Gauges</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(create_gauge("RPM", data["rpm"], 0, 6000))
with col2:
    st.plotly_chart(create_gauge("EGT", data["egt"], 0, 900))
with col3:
    st.plotly_chart(create_gauge("CHT", data["cht"], 0, 300))

col4, col5, col6 = st.columns(3)
with col4:
    st.plotly_chart(create_gauge("Oil Pressure", data["oil_pressure"], 0, 100))
with col5:
    st.plotly_chart(create_gauge("Oil Temp", data["oil_temp"], 0, 150))
with col6:
    st.plotly_chart(create_gauge("Vibration", data["vibration"], 0, 5))

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HEALTH + RECOVERY ----------------
st.markdown("<div class='card glow'><h3>🧠 Health & Recovery</h3>", unsafe_allow_html=True)

if health > 70:
    st.success(f"{health}% Healthy")
elif health > 40:
    st.warning(f"{health}% Moderate")
else:
    st.error(f"{health}% Critical")

st.metric("Recovery Speed", f"{data['recovery_rate']:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- COMPARISON ----------------
st.markdown("<div class='card glow'><h3>⚖️ Recovery Comparison</h3>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.markdown("**Engine A (Fast Recovery)**")
    st.plotly_chart(create_gauge("Health", health, 0, 100), key="A")
    st.metric("Recovery Rate", f"{data['recovery_rate'] + 0.3:.2f}")

with colB:
    st.markdown("**Engine B (Slow Recovery)**")
    st.plotly_chart(create_gauge("Health", health, 0, 100), key="B")
    st.metric("Recovery Rate", f"{data['recovery_rate'] - 0.3:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RUL ----------------
st.markdown("<div class='card glow'><h3>⏳ Remaining Useful Life</h3>", unsafe_allow_html=True)

rul_value = data.get("rul_estimate_timesteps", data["rul_estimate_minutes"])

if "rul" not in st.session_state:
    st.session_state.rul = rul_value

st.session_state.rul -= 0.1

st.markdown(f"""
<h1 style='text-align:center;color:#ff4d4d;font-size:70px;
text-shadow:0 0 20px rgba(255,0,0,0.6);'>
{round(st.session_state.rul,1)} min
</h1>
""", unsafe_allow_html=True)

st.caption("Estimated Life Remaining")
time.sleep(0.1)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ALERTS ----------------
FAULT_DISPLAY_NAMES = {
    "none": "No Fault",
    "misfire": "Misfire",
    "overheat": "Overheating",
    "cooling_degradation": "Cooling Degradation",
}

st.markdown("<div class='card glow'><h3>🚨 System Alerts</h3>", unsafe_allow_html=True)

if data["fault_type"] != "none":
    fault_name = FAULT_DISPLAY_NAMES.get(data["fault_type"], data["fault_type"])
    st.markdown(f"""
    <div style='background:rgba(255,0,0,0.15);
    padding:15px;border-radius:10px;border-left:5px solid red;
    box-shadow:0 0 10px rgba(255,0,0,0.4);'>
    ⚠️ {fault_name} detected!
    </div>
    """, unsafe_allow_html=True)
    st.write(f"Confidence: {data['fault_confidence']*100:.1f}%")
else:
    st.success("✅ No active faults")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ACTIONS ----------------
st.markdown("<div class='card glow'><h3>🛠️ Suggested Actions</h3>", unsafe_allow_html=True)

sorted_actions = sorted(
    data["suggested_actions"],
    key=lambda x: x["projected_rul_minutes"],
    reverse=True
)

for i, action in enumerate(sorted_actions):
    label = action["label"]
    rul = action["projected_rul_minutes"]
    delta = action.get("delta_minutes", 0)

    if i == 0:
        st.success(f"⭐ Recommended: {label} → {rul} min (+{delta:.1f})")
    else:
        st.info(f"{label} → {rul} min (+{delta:.1f})")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TREND ----------------
st.markdown("<div class='card glow'><h3>📈 Health Trend</h3>", unsafe_allow_html=True)
st.line_chart(df["health"])
st.markdown("</div>", unsafe_allow_html=True)
