import streamlit as st
import plotly.graph_objects as go
from mock_data import get_mock_data
from streamlit_autorefresh import st_autorefresh
import pandas as pd

def create_gauge(title, value, min_val, max_val):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 16}},
            number={'font': {'size': 28}},
            gauge={
                'axis': {
                    'range': [min_val, max_val],
                    'tickwidth': 1,
                    'tickcolor': "#888"
                },
                'bar': {'color': "#444", 'thickness': 0.25},
                'bgcolor': "white",
                'steps': [
                    {'range': [min_val, max_val*0.6], 'color': "#e8f5e9"},
                    {'range': [max_val*0.6, max_val*0.85], 'color': "#fff8e1"},
                    {'range': [max_val*0.85, max_val], 'color': "#fdecea"},
                ],
                'borderwidth': 1,
                'bordercolor': "#ddd"
            }
        )
    )

    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        font={'color': "#333"}
    )

    return fig
    
st.set_page_config(layout="wide")

st.title("✈️ UAV Engine Digital Twin Dashboard")
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Auto refresh
st_autorefresh(interval=2000, key="refresh")

data = get_mock_data()

if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "health": data["health_score"]
})

df = pd.DataFrame(st.session_state.history)

st.divider()

# Sensors
st.header("✈️ Engine Gauges")

# ROW 1
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.plotly_chart(create_gauge("RPM", data["rpm"], 0, 4000), use_container_width=False)

with col2:
    st.plotly_chart(create_gauge("EGT", data["egt"], 0, 1000), use_container_width=False)

with col3:
    st.plotly_chart(create_gauge("CHT", data["cht"], 0, 300), use_container_width=False)


# ROW 2 (same width → fixes alignment)
col4, col5, col6 = st.columns(3,gap="large")

with col4:
    st.plotly_chart(create_gauge("Oil Pressure", data["oil_pressure"], 0, 100), use_container_width=False)

with col5:
    st.plotly_chart(create_gauge("Oil Temp", data["oil_temp"], 0, 150), use_container_width=False)

with col6:
    st.plotly_chart(create_gauge("Vibration", data["vibration"], 0, 1), use_container_width=False)
    st.markdown("<br>", unsafe_allow_html=True)

# Health
st.header("🧠 Health Score")
health = data["health_score"]

if health > 70:
    st.success(f"{health}% Healthy")
elif health > 40:
    st.warning(f"{health}% Moderate")
else:
    st.error(f"{health}% Critical")

st.divider()

# Timer
import time

st.header("⏳ Remaining Useful Life")

if "rul" not in st.session_state:
    st.session_state.rul = data["rul_estimate_minutes"]

# smooth decrease
st.session_state.rul -= 0.1

st.markdown(
    f"""
    <h1 style='text-align: center; color: red; font-size:60px;'>
        {round(st.session_state.rul,1)} min
    </h1>
    """,
    unsafe_allow_html=True
)

time.sleep(0.1)
st.caption(f"Estimated Life Remaining")

st.divider()
st.header("🚨 System Alerts")

if data["fault_type"] != "None":
    st.error(f"⚠️ {data['fault_type']} detected!")
    st.write(f"Confidence: {data['fault_confidence']*100:.1f}%")
else:
    st.success("✅ No active faults")

st.subheader("🛠️ Suggested Action")

for action in data["suggested_actions"]:
    st.info(f"{action['action']} → RUL becomes {action['projected_rul_minutes']} min")

st.divider()
st.header("📈 Health Trend")

st.line_chart(df["health"])