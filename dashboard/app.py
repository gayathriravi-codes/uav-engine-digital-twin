import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from models.classification.engine_inference import run_engine_inference
from models.rul.whatif_engine import run_whatif


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroTwin | UAV Engine Digital Twin",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ASSETS_DIR = ROOT / "dashboard" / "assets"

WINDOW_SIZE = 30

REQUIRED_COLUMNS = [
    "timestamp",
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temp",
    "vibration",
    "fuel_flow",
]

SLOW_RECOVERY_THRESHOLD = 0.10


# ============================================================
# FUTURISTIC UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 50% 10%, #082b45 0%, #03111d 42%, #02070d 100%);
        color: #e7f8ff;
    }

    .main {
        background: transparent;
    }

    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #031421 0%, #020b13 100%);
        border-right: 1px solid #0b4c70;
    }

    section[data-testid="stSidebar"] * {
        color: #d8f5ff;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        color: #e9fbff !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    h2, h3 {
        color: #67dcff !important;
        letter-spacing: .4px;
    }

    /* ---------- CARDS ---------- */

    .neo-card {
        background:
            linear-gradient(
                145deg,
                rgba(5, 36, 57, .92),
                rgba(2, 15, 25, .95)
            );
        border: 1px solid #0a5278;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow:
            0 0 18px rgba(0, 183, 255, .07),
            inset 0 0 18px rgba(0, 160, 220, .025);
    }

    .live-card {
        border: 1px solid #00e5a8;
        box-shadow:
            0 0 18px rgba(0, 229, 168, .13),
            inset 0 0 20px rgba(0, 229, 168, .025);
    }

    .danger-card {
        border: 1px solid #ff3f70;
        box-shadow:
            0 0 20px rgba(255, 63, 112, .13);
    }

    .warning-card {
        border: 1px solid #ffad33;
        box-shadow:
            0 0 18px rgba(255, 173, 51, .10);
    }

    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(145deg, rgba(5,34,52,.95), rgba(2,15,25,.95));
        border: 1px solid #0a5278;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #80b9ca !important;
    }

    div[data-testid="stMetricValue"] {
        color: #e8fbff !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        background: linear-gradient(135deg, #063e5d, #052235);
        color: #dffaff;
        border: 1px solid #0876a5;
        border-radius: 9px;
        transition: .2s;
    }

    .stButton > button:hover {
        border-color: #00e5ff;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(0,229,255,.25);
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #0b405b;
    }

    /* ---------- STATUS ---------- */

    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00e5a8;
        box-shadow: 0 0 12px #00e5a8;
        margin-right: 7px;
    }

    .status-text {
        color: #00e5a8;
        font-weight: 600;
    }

    .small-muted {
        color: #7194a3;
        font-size: 12px;
    }

    .big-number {
        font-size: 32px;
        font-weight: 700;
        color: #69e7ff;
    }

    .fault-number {
        font-size: 24px;
        font-weight: 700;
        color: #ff557c;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "playing": False,
    "current_index": WINDOW_SIZE - 1,
    "selected_flight": None,
    "speed": 5,
    "timeline_slider": WINDOW_SIZE - 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data
def get_flights():
    return sorted(RAW_DIR.glob("*.csv"))


@st.cache_data
def load_flight(path):
    df = pd.read_csv(path)

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(missing)}"
        )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def get_whatif(
    fault_type,
    onset_idx,
    duration,
    window_start,
):
    try:
        return run_whatif(
            fault_type=fault_type,
            onset_idx=onset_idx,
            duration=duration,
            window_start=window_start,
            seed=42,
        )
    except Exception:
        return []


# ============================================================
# LIVE ENGINE DIGITAL TWIN
# ============================================================
def build_live_engine_twin(current, inference, sample_idx):
    """Build a visibly telemetry-responsive Plotly engine twin."""
    rpm = float(current["rpm"])
    egt = float(current["egt"])
    vibration = float(current["vibration"])
    health = float(inference["health_score"])
    recovery = float(inference["recovery_rate"])
    fault = str(inference["predicted_fault"])
    severity = str(inference["severity"])
    condition = str(inference["condition_status"])
    rul = float(inference["rul_estimate_timesteps"])

    # Model-driven state color.
    if fault != "none" and severity.upper() == "CRITICAL":
        state_color = "#ff3f70"
    elif fault != "none":
        state_color = "#ffad33"
    elif health >= 90:
        state_color = "#00e5a8"
    elif health >= 75:
        state_color = "#ffad33"
    else:
        state_color = "#ff3f70"

    # Sensor-driven thermal state.
    if egt >= 750:
        thermal_color = "#ff3f70"
    elif egt >= 650:
        thermal_color = "#ffad33"
    else:
        thermal_color = "#00d9ff"

    # Normalize against the dashboard's sensor ranges.
    rpm_norm = float(np.clip(rpm / 6000.0, 0, 1))
    egt_norm = float(np.clip((egt - 400.0) / 500.0, 0, 1))
    vib_norm = float(np.clip(vibration / 5.0, 0, 1))

    # Use both the sample position and RPM so the rotor geometry visibly changes.
    angle = np.deg2rad((sample_idx * (35.0 + 145.0 * rpm_norm)) % 360)
    turbine_angle = -angle * 1.15
    shake_x = np.sin(sample_idx * 2.2) * vib_norm * 0.28
    shake_y = np.cos(sample_idx * 1.8) * vib_norm * 0.20

    fig = go.Figure()

    # Outer casing visibly reacts to vibration and model state.
    casing_width = 2.0 + 2.0 * vib_norm
    fig.add_shape(
        type="rect", x0=-3.65, x1=3.65, y0=-1.45, y1=1.45,
        line=dict(color=state_color, width=casing_width),
        fillcolor="rgba(4,25,39,0.92)",
    )
    fig.add_shape(
        type="rect", x0=-3.42, x1=3.42, y0=-1.22, y1=1.22,
        line=dict(color="#0a5575", width=1),
        fillcolor="rgba(3,16,27,0.65)",
    )

    # Compressor and turbine rings move with vibration and change size with load.
    compressor_r = 0.78 + 0.24 * rpm_norm
    turbine_r = 0.78 + 0.22 * egt_norm
    for cx, cy, radius, color in [
        (-1.65 + shake_x, shake_y, compressor_r, "#1bc9ff"),
        (-1.65 + shake_x, shake_y, compressor_r * 0.67, "#0b7195"),
        (1.45 + shake_x, shake_y, turbine_r, thermal_color),
        (1.45 + shake_x, shake_y, turbine_r * 0.67, "#8c4055"),
    ]:
        fig.add_shape(
            type="circle", x0=cx-radius, x1=cx+radius,
            y0=cy-radius, y1=cy+radius,
            line=dict(color=color, width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    # Thermal core size and glow are directly driven by EGT.
    core_r = 0.48 + 0.34 * egt_norm
    fig.add_shape(
        type="circle", x0=-0.35-core_r/2, x1=0.75+core_r/2,
        y0=-core_r, y1=core_r,
        line=dict(color=thermal_color, width=3),
        fillcolor="rgba(255,100,40,0.18)",
    )
    inner_r = 0.26 + 0.28 * egt_norm
    fig.add_shape(
        type="circle", x0=0.275-inner_r, x1=0.275+inner_r,
        y0=-inner_r, y1=inner_r,
        line=dict(color=thermal_color, width=2),
        fillcolor="rgba(255,130,40,0.25)",
    )

    # Central shaft.
    fig.add_shape(
        type="line", x0=-2.55, x1=2.55, y0=0, y1=0,
        line=dict(color="#78eaff", width=4 + 3 * rpm_norm),
    )

    # Rotor blades: length, angle, and thickness visibly respond to RPM/load.
    for center_x, a0, blade_color, length_scale in [
        (-1.65 + shake_x, angle, "#5fe5ff", 0.75 + 0.30 * rpm_norm),
        (1.45 + shake_x, turbine_angle, thermal_color, 0.75 + 0.30 * egt_norm),
    ]:
        for blade in range(7):
            a = a0 + blade * (2 * np.pi / 7)
            a2 = a + 0.30 + 0.16 * rpm_norm
            x1 = center_x + 0.15 * np.cos(a)
            y1 = shake_y + 0.15 * np.sin(a)
            x2 = center_x + length_scale * np.cos(a2)
            y2 = shake_y + length_scale * np.sin(a2)
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2], mode="lines",
                line=dict(color=blade_color, width=4 + 3 * rpm_norm),
                hoverinfo="skip", showlegend=False,
            ))

    # HIGH-VISIBILITY ROTATION MARKERS.  These deliberately break rotational symmetry
    # so a one-second telemetry step is obvious to the eye.
    for center_x, center_y, rotor_angle, rotor_color, rotor_r in [
        (-1.65 + shake_x, shake_y, angle, "#dffaff", compressor_r),
        (1.45 + shake_x, shake_y, turbine_angle, thermal_color, turbine_r),
    ]:
        # Three unequal spokes + one bright index marker = unmistakable orientation change.
        for offset, length, width in [(0.0, 0.95, 6), (1.75, 0.72, 4), (3.95, 0.58, 3)]:
            a = rotor_angle + offset
            fig.add_trace(go.Scatter(
                x=[center_x, center_x + length * np.cos(a)],
                y=[center_y, center_y + length * np.sin(a)],
                mode="lines",
                line=dict(color=rotor_color, width=width),
                hoverinfo="skip", showlegend=False,
            ))
        marker_angle = rotor_angle + 0.18
        marker_r = rotor_r * 0.92
        fig.add_trace(go.Scatter(
            x=[center_x + marker_r * np.cos(marker_angle)],
            y=[center_y + marker_r * np.sin(marker_angle)],
            mode="markers",
            marker=dict(size=14, color=rotor_color, line=dict(color="#ffffff", width=2)),
            hoverinfo="skip", showlegend=False,
        ))
        # Outer rotating arc segment makes the angular state visible even at low RPM.
        arc_angles = np.linspace(rotor_angle - 0.55, rotor_angle + 0.55, 18)
        fig.add_trace(go.Scatter(
            x=center_x + rotor_r * 1.08 * np.cos(arc_angles),
            y=center_y + rotor_r * 1.08 * np.sin(arc_angles),
            mode="lines",
            line=dict(color=rotor_color, width=4),
            hoverinfo="skip", showlegend=False,
        ))

    # Hubs with telemetry-rich hover information.
    fig.add_trace(go.Scatter(
        x=[-1.65 + shake_x, 1.45 + shake_x],
        y=[shake_y, shake_y],
        mode="markers",
        marker=dict(
            size=[18 + 12 * rpm_norm, 18 + 12 * egt_norm],
            color=["#dffaff", thermal_color],
            line=dict(color=state_color, width=3),
        ),
        text=["Compressor", "Turbine"],
        customdata=[[rpm, egt, vibration], [rpm, egt, vibration]],
        hovertemplate=(
            "%{text}<br>RPM: %{customdata[0]:,.0f}"
            "<br>EGT: %{customdata[1]:.1f}"
            "<br>Vibration: %{customdata[2]:.3f}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Exhaust plume visibly expands/contracts with EGT and changes phase with playback.
    plume_strength = float(np.clip(0.15 + 0.85 * egt_norm, 0, 1))
    plume_x = np.linspace(2.25, 3.55, 16)
    plume_y = (
        0.22 + 0.32 * plume_strength
    ) * np.sin(np.linspace(0, 5*np.pi, 16) + sample_idx * 0.65)
    fig.add_trace(go.Scatter(
        x=plume_x, y=plume_y, mode="markers",
        marker=dict(
            size=5 + 13 * plume_strength,
            color=thermal_color,
            opacity=0.25 + 0.65 * plume_strength,
        ),
        hovertemplate=f"EGT: {egt:.1f}<br>Thermal load: {egt_norm*100:.0f}%<extra></extra>",
        showlegend=False,
    ))

    # Airflow arrows pulse through position changes during playback.
    airflow_shift = 0.10 * np.sin(sample_idx * 0.8)
    for y in (0.82, -0.82):
        fig.add_annotation(
            x=-3.05 + airflow_shift, y=y,
            ax=-3.45 + airflow_shift, ay=y,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.2, arrowwidth=2,
            arrowcolor="#1bc9ff", showarrow=True, text="",
        )

    fault_label = "NO ACTIVE FAULT" if fault == "none" else fault.replace("_", " ").upper()
    recovery_label = "RECOVERY" if recovery > SLOW_RECOVERY_THRESHOLD else "RECALIBRATION FLAG"

    fig.add_annotation(
        x=-3.25, y=1.85, text="<b>LIVE ENGINE TWIN</b>",
        showarrow=False, font=dict(size=18, color="#e9fbff"), xanchor="left",
    )
    fig.add_annotation(
        x=3.25, y=1.85, text=f"<b>● {condition}</b>",
        showarrow=False, font=dict(size=14, color=state_color), xanchor="right",
    )

    labels = [
        (-2.45, -1.85, f"RPM  <b>{rpm:,.0f}</b>"),
        (-0.75, -1.85, f"EGT  <b>{egt:.1f}</b>"),
        (0.75, -1.85, f"VIB  <b>{vibration:.3f}</b>"),
        (2.35, -1.85, f"HEALTH  <b>{health:.1f}%</b>"),
    ]
    for x, y, text in labels:
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False,
            font=dict(size=12, color="#bdeeff"),
        )

    fig.add_annotation(
        x=0, y=1.05,
        text=f"<b>COMBUSTION / THERMAL CORE</b><br>{fault_label}",
        showarrow=False, font=dict(size=11, color=thermal_color),
    )
    fig.add_annotation(
        x=0, y=-0.98,
        text=f"RUL <b>{rul:.2f} timesteps</b>  •  {recovery_label}",
        showarrow=False, font=dict(size=11, color=state_color),
    )

    fig.update_xaxes(range=[-4.05, 4.05], visible=False, fixedrange=False)
    fig.update_yaxes(
        range=[-2.15, 2.15], visible=False, fixedrange=False,
        scaleanchor="x", scaleratio=1,
    )
    fig.update_layout(
        height=430, margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,10,18,0.55)",
        hovermode="closest", showlegend=False,
    )
    return fig


# ============================================================
# HEADER
# ============================================================

head1, head2 = st.columns([4, 1])

with head1:
    st.markdown(
        "# ✈️ UAV Engine Digital Twin"
    )

    st.markdown(
        """
        <div class="small-muted">
        Live Telemetry &nbsp;•&nbsp;
        Fault Diagnosis &nbsp;•&nbsp;
        Health Monitoring &nbsp;•&nbsp;
        RUL Prediction
        </div>
        """,
        unsafe_allow_html=True,
    )

with head2:
    if st.session_state.playing:
        st.markdown(
            """
            <div style="
                text-align:right;
                padding-top:15px;
                color:#00e5a8;
                font-weight:600;">
                <span class="status-dot"></span>
                TELEMETRY PLAYBACK
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                text-align:right;
                padding-top:15px;
                color:#6f91a0;">
                <span class="status-dot"
                style="background:#6f91a0;
                box-shadow:none;"></span>
                PLAYBACK PAUSED
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🎛️ Flight & Telemetry Control"
)

flights = get_flights()

if not flights:
    st.error("No telemetry files found in data/raw.")
    st.stop()

flight_names = [x.name for x in flights]

selected = st.sidebar.selectbox(
    "Select Flight",
    flight_names,
)

if st.session_state.selected_flight != selected:

    st.session_state.selected_flight = selected
    st.session_state.current_index = WINDOW_SIZE - 1
    st.session_state.timeline_slider = WINDOW_SIZE - 1
    st.session_state.playing = False


flight_path = RAW_DIR / selected

try:
    df = load_flight(flight_path)
except Exception as e:
    st.error(str(e))
    st.stop()


max_index = len(df) - 1

st.sidebar.markdown("### Playback Controls")

c1, c2 = st.sidebar.columns(2)

with c1:
    if st.button(
        "▶ Play" if not st.session_state.playing else "▶ Playing",
        use_container_width=True,
    ):
        st.session_state.playing = True
        st.rerun()

with c2:
    if st.button(
        "⏸ Pause",
        use_container_width=True,
    ):
        st.session_state.playing = False
        st.rerun()


if st.sidebar.button(
    "↻ Reset",
    use_container_width=True,
):
    st.session_state.current_index = WINDOW_SIZE - 1
    st.session_state.timeline_slider = WINDOW_SIZE - 1
    st.session_state.playing = False
    st.rerun()


st.sidebar.markdown("### Playback Speed")

st.sidebar.radio(
    "Speed",
    [1, 2, 5, 10],
    horizontal=True,
    format_func=lambda x: f"{x}×",
    key="speed",
)


# ============================================================
# LIVE FRAGMENT
# ============================================================

run_every = "1s" if st.session_state.playing else None


@st.fragment(run_every=run_every)
def dashboard():

    # --------------------------------------------------------
    # ADVANCE PLAYBACK
    # --------------------------------------------------------

    if st.session_state.playing:

        next_idx = (
            st.session_state.current_index
            + st.session_state.speed
        )

        if next_idx >= max_index:

            st.session_state.current_index = max_index
            st.session_state.playing = False

        else:

            st.session_state.current_index = next_idx


    idx = st.session_state.current_index

    idx = max(
        WINDOW_SIZE - 1,
        min(idx, max_index),
    )

    st.session_state.current_index = idx


    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    st.sidebar.markdown("### Telemetry Timeline")

    st.session_state.timeline_slider = idx

    timeline = st.slider(
        "Telemetry Timeline",
        WINDOW_SIZE - 1,
        max_index,
        key="timeline_slider",
        label_visibility="collapsed",
    )

    if timeline != idx:
        st.session_state.current_index = timeline
        st.session_state.playing = False
        st.rerun()


    # --------------------------------------------------------
    # CURRENT WINDOW
    # --------------------------------------------------------

    start = idx - WINDOW_SIZE + 1

    window = df.iloc[
        start : idx + 1
    ].copy()

    current = df.iloc[idx]

    # --------------------------------------------------------
    # LIVE MODEL INFERENCE
    # --------------------------------------------------------
    # Fault/RUL use the latest 30 samples internally. Recovery analysis
    # needs the complete telemetry history up to the current sample.
    try:
        inference = run_engine_inference(df.iloc[:idx + 1])
    except Exception as e:
        st.error(f"Inference error: {e}")
        return

    health = float(inference["health_score"])


    # --------------------------------------------------------
    # HERO AREA
    # --------------------------------------------------------

    hero_col, health_col = st.columns(
        [3.2, 1],
        gap="medium",
    )

    with hero_col:
        st.markdown("### ⚙️ LIVE ENGINE DIGITAL TWIN")
        twin_fig = build_live_engine_twin(current, inference, idx)
        st.plotly_chart(
            twin_fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
            key="live_engine_twin",
        )
        st.caption(
            "Telemetry-driven twin • Rotor state follows RPM • "
            "Thermal state follows EGT • Fault state follows model inference"
        )

    with health_col:

        health = float(inference["health_score"])


        st.markdown(
            """
            <div class="neo-card live-card">
                <h3>❤️ Engine Health</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 32,
                        "color": "#69e7ff",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#648b9b",
                    },
                    "bar": {
                        "color": "#00e5c3",
                    },
                    "bgcolor": "#071b29",
                    "bordercolor": "#0c6385",
                },
            )
        )

        gauge.update_layout(
            height=220,
            margin=dict(
                l=10, r=10, t=10, b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#dffaff",
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption(
            f"Condition: {inference['condition_status']}"
        )


    # --------------------------------------------------------
    # LIVE SENSOR GAUGES
    # --------------------------------------------------------

    st.markdown(
        "## 📡 Live Engine Telemetry"
    )

    sensors = [
        ("RPM", "rpm", 6000),
        ("EGT", "egt", 900),
        ("CHT", "cht", 300),
        ("Oil Pressure", "oil_pressure", 100),
        ("Oil Temp", "oil_temp", 150),
        ("Vibration", "vibration", 5),
        ("Fuel Flow", "fuel_flow", 100),
    ]

    gauge_cols = st.columns(7)

    for col, (label, field, maximum) in zip(
        gauge_cols,
        sensors,
    ):

        value = float(current[field])

        with col:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    number={
                        "font": {
                            "size": 19,
                            "color": "#e7faff",
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [0, maximum],
                            "tickcolor": "#577b8c",
                        },
                        "bar": {
                            "color": "#18cce6",
                        },
                        "bgcolor": "#061724",
                        "bordercolor": "#0b5573",
                    },
                )
            )

            fig.update_layout(
                height=160,
                margin=dict(
                    l=2, r=2, t=8, b=2
                ),
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.markdown(
                f"<div style='text-align:center;"
                f"color:#80b9ca;font-size:12px;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # HEALTH + RECOVERY
    # --------------------------------------------------------

    recovery = float(
        inference["recovery_rate"]
    )

    h1, h2 = st.columns([2, 1])

    with h1:

        st.markdown(
            "## 📈 Engine Health Trend"
        )

        health_history = []

        trend_start = max(
            WINDOW_SIZE - 1,
            idx - 24,
        )

        for i in range(
            trend_start,
            idx + 1,
        ):

            w = df.iloc[
                i - WINDOW_SIZE + 1 : i + 1
            ]

            try:

                r = run_engine_inference(w)

                health_history.append(
                    {
                        "Sample": i + 1,
                        "Health": r["health_score"],
                        "Recovery": r["recovery_rate"],
                    }
                )

            except Exception:
                pass


        if health_history:

            hist = pd.DataFrame(
                health_history
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hist["Sample"],
                    y=hist["Health"],
                    mode="lines",
                    name="Health",
                    line=dict(
                        color="#00e5a8",
                        width=3,
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=hist["Sample"],
                    y=hist["Recovery"] * 100,
                    mode="lines",
                    name="Recovery",
                    line=dict(
                        color="#25a8ff",
                        width=2,
                    ),
                )
            )

            fig.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(2,18,29,.8)",
                margin=dict(
                    l=20, r=20, t=20, b=20
                ),
                xaxis_title="Telemetry Sample",
                yaxis_title="Value",
                legend=dict(
                    orientation="h",
                    y=1.08,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    with h2:

        st.markdown(
            "## 🔄 Recovery Monitor"
        )

        st.metric(
            "Recovery Rate",
            f"{recovery:.4f}",
        )

        if inference["recovered"]:

            st.success(
                "● Recovery detected"
            )

        elif recovery <= SLOW_RECOVERY_THRESHOLD:

            st.warning(
                "⚠ Recalibration Flag"
            )

        else:

            st.info(
                "● Monitoring recovery"
            )

        st.progress(
            min(
                1.0,
                max(
                    0.0,
                    recovery,
                ),
            )
        )


    # --------------------------------------------------------
    # RIGHT-SIDE DIAGNOSTICS
    # --------------------------------------------------------

    st.markdown("---")

    d1, d2, d3 = st.columns(3)

    with d1:

        st.markdown(
            "### 🚨 Fault Diagnosis"
        )

        fault = inference["predicted_fault"]

        if fault == "none":

            st.success(
                "NO ACTIVE FAULT"
            )

        else:

            st.error(
                f"**{fault.upper()}**"
            )

        st.metric(
            "Confidence",
            f"{float(inference['confidence']) * 100:.1f}%",
        )

        st.caption(
            f"Severity: {inference['severity']}"
        )


    with d2:

        st.markdown(
            "### ⏱️ RUL Estimate"
        )

        st.markdown(
            f"""
            <div class="big-number">
                {inference['rul_estimate_timesteps']:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "timesteps — model point estimate"
        )

        st.metric(
            "Lower Bound",
            f"{inference['rul_lower_bound_timesteps']:.2f}",
        )


    with d3:

        st.markdown(
            "### 🛩️ UAV Status"
        )

        st.markdown(
            """
            <div class="neo-card live-card">
            <b>● Connected to Digital Twin</b><br><br>
            UAV Engine<br>
            Telemetry Playback<br>
            Autonomous Monitoring
            </div>
            """,
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # FAULT ACTION
    # --------------------------------------------------------

    st.info(
        f"**Operator Action:** "
        f"{inference['operator_action']}"
    )


    # --------------------------------------------------------
    # ENGINE 3D / DIGITAL TWIN VISUAL
    # --------------------------------------------------------

    st.markdown(
        "## 🔬 Engine Digital Twin View"
    )

    engine_col, action_col = st.columns(
        [1.25, 1],
        gap="medium",
    )

    with engine_col:
        st.markdown("### 🛰️ Live Component State")
        component_rows = [
            ("Compressor", float(current["rpm"]) / 6000 * 100, "RPM-linked"),
            ("Thermal Core", float(current["egt"]) / 900 * 100, f"EGT {float(current['egt']):.1f}"),
            ("Cooling", float(current["cht"]) / 300 * 100, f"CHT {float(current['cht']):.1f}"),
            ("Lubrication", float(current["oil_pressure"]) / 100 * 100, f"Oil P {float(current['oil_pressure']):.1f}"),
            ("Mechanical", float(current["vibration"]) / 5 * 100, f"Vibration {float(current['vibration']):.3f}"),
        ]
        for name, pct, detail in component_rows:
            pct = float(np.clip(pct, 0, 100))
            bar_color = "#ff3f70" if pct >= 90 else "#ffad33" if pct >= 75 else "#00e5a8"
            st.markdown(
                f"""
                <div class="neo-card" style="padding:11px 14px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;">
                        <b>{name}</b>
                        <span class="small-muted">{detail}</span>
                    </div>
                    <div style="height:6px; background:#092130; border-radius:6px; margin-top:8px;">
                        <div style="width:{pct:.1f}%; height:100%; background:{bar_color}; border-radius:6px; box-shadow:0 0 8px {bar_color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    # --------------------------------------------------------
    # PANEL 5
    # --------------------------------------------------------

    with action_col:

        st.markdown(
            "### 🧠 What-If Action Recommendations"
        )

        predicted_fault = inference[
            "predicted_fault"
        ]

        if predicted_fault == "none":

            st.success(
                "No active fault detected."
            )

        else:

            onset_idx = None

            if "fault_type" in df.columns:

                matches = df.index[
                    df["fault_type"]
                    .astype(str)
                    .str.lower()
                    == str(predicted_fault).lower()
                ].tolist()

                if matches:
                    onset_idx = matches[0]


            if onset_idx is None:

                onset_idx = max(
                    0,
                    idx - WINDOW_SIZE + 1,
                )


            duration = max(
                1,
                idx - onset_idx + 1,
            )

            results = get_whatif(
                predicted_fault,
                onset_idx,
                duration,
                start,
            )

            ranked = sorted(
                results,
                key=lambda x: float(
                    x["point_estimate_timesteps"]
                ),
                reverse=True,
            )


            if not ranked:

                st.warning(
                    "No what-if results available."
                )

            else:

                for rank, result in enumerate(
                    ranked[:4]
                ):

                    projected = float(
                        result[
                            "point_estimate_timesteps"
                        ]
                    )

                    lower = float(
                        result[
                            "rul_lower_bound_timesteps"
                        ]
                    )

                    gain = float(
                        result[
                            "delta_minutes"
                        ]
                    )

                    if rank == 0:

                        st.markdown(
                            f"""
                            <div class="neo-card live-card">
                                <h3>
                                🥇 {result['label']}
                                </h3>

                                <div class="small-muted">
                                RECOMMENDED ACTION
                                </div>

                                <br>

                                <b>Projected RUL:</b>
                                {projected:.2f}

                                &nbsp;&nbsp;

                                <b>Lower Bound:</b>
                                {lower:.2f}

                                <br><br>

                                <b>RUL Gain:</b>
                                +{gain:.2f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="neo-card">
                                <b>
                                {rank + 1}. {result['label']}
                                </b>
                                <br>
                                <span class="small-muted">
                                Projected RUL:
                                {projected:.2f}
                                &nbsp; | &nbsp;
                                Lower:
                                {lower:.2f}
                                &nbsp; | &nbsp;
                                Gain:
                                +{gain:.2f}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


    # --------------------------------------------------------
    # SENSOR TELEMETRY GRAPH
    # --------------------------------------------------------

    st.markdown(
        "## 📊 Sensor Telemetry — Last 30 Samples"
    )

    fig = go.Figure()

    graph_sensors = [
        ("rpm", "RPM"),
        ("egt", "EGT"),
        ("cht", "CHT"),
        ("vibration", "Vibration"),
    ]

    for field, label in graph_sensors:

        fig.add_trace(
            go.Scatter(
                x=window["timestamp"],
                y=window[field],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        height=340,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,18,29,.8)",
        hovermode="x unified",
        margin=dict(
            l=20, r=20, t=20, b=20
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # CURRENT TELEMETRY WINDOW
    # --------------------------------------------------------

    st.markdown(
        "## 📋 Current Telemetry Window"
    )

    display_df = window.copy()

    display_df.insert(
        0,
        "#",
        range(
            start + 1,
            idx + 2,
        ),
    )

    st.dataframe(
        display_df.tail(30),
        use_container_width=True,
        height=280,
    )


    # --------------------------------------------------------
    # CURRENT STATUS
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="neo-card">
            <span class="status-dot"></span>
            <span class="status-text">
                Telemetry Playback Active
            </span>
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Sample {idx + 1} / {len(df)}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Speed {st.session_state.speed}×
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN
# ============================================================

dashboard()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="small-muted"
         style="text-align:center;">
        ⚠ Telemetry Playback uses recorded telemetry progressively
        to simulate a live monitoring workflow.
        It is not a direct aircraft connection.
        &nbsp; • &nbsp;
        AeroTwin UAV Engine Digital Twin
    </div>
    """,
    unsafe_allow_html=True,
)