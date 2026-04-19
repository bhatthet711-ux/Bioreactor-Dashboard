# app.py
"""
Bioreactor Time-Series Dashboard
A complete, interactive dashboard for visualization of bioprocess engineering metrics,
specifically designed for 48-hour fermentation cycles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import pearsonr

# ==========================================
# 1. IMPORTS & CONFIG
# ==========================================

# Set page configuration
st.set_page_config(
    page_title="BioReactor Analytics Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium aesthetic
st.markdown("""
<style>
    /* Import modern font and material symbols from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    /* Set base font securely */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Re-allow specific icon fonts safely so streamlit navigation arrow renders */
    span.material-symbols-rounded, span.material-icons, .material-symbols-rounded {
        font-family: 'Material Symbols Rounded', sans-serif !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }

    /* Main App Background Pattern - Subtle tech dots */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(#2A2E35 1px, transparent 1px);
        background-size: 24px 24px;
    }

    /* Sleek gradient headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #00FFCC, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #00FFCC !important;
        font-weight: 600 !important;
    }

    /* Fancy Premium Button Styling - Let Streamlit maintain box-sizes, just alter colors */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 255, 204, 0.1) 0%, rgba(0, 191, 255, 0.1) 100%) !important;
        color: #00FFCC !important;
        border: 1px solid rgba(0, 255, 204, 0.5) !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #00FFCC 0%, #00BFFF 100%) !important;
        color: #000000 !important;
        border-color: transparent !important;
        box-shadow: 0 8px 20px rgba(0, 255, 204, 0.4) !important;
    }

    /* File Uploader styling */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 2px dashed rgba(0, 255, 204, 0.3) !important;
        background-color: rgba(30, 33, 39, 0.5) !important;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00FFCC !important;
        background-color: rgba(30, 33, 39, 0.8) !important;
    }
    
    /* Sidebar styling for glassmorphism */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(0, 255, 204, 0.15) !important;
        background: rgba(18, 20, 24, 0.95);
        backdrop-filter: blur(20px);
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }

    /* Explicit styling for the collapsible sidebar arrow to make it extremely prominent */
    [data-testid="collapsedControl"] {
        border-radius: 50% !important;
        border: 2px solid #00FFCC !important;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4);
        background: #1E2127 !important;
        color: #00FFCC !important;
        transition: all 0.3s ease;
        margin-left: 10px;
        margin-top: 10px;
    }
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.15);
        box-shadow: 0 0 25px rgba(0, 255, 204, 0.8);
        background: #0E1117 !important;
    }
    
    /* Make the arrow inside the button larger */
    [data-testid="collapsedControl"] span {
        color: #00FFCC !important;
    }

    /* Dataframe corner styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 204, 0.2);
    }
    
    /* Checkbox nice color */
    .stCheckbox label span {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SYNTHETIC DATA GENERATOR
# ==========================================

def generate_synthetic_data() -> pd.DataFrame:
    """
    Generates synthetic 48-hour fermentation data with realistic noise.
    
    Bioprocess engineering simulation details:
      - Lag phase (0-6h): minimal cell growth, high DO, stable pH.
      - Log phase (6-36h): exponential biomass (OD600) growth, rapid DO depletion,
        acid production (pH drop).
      - Stationary phase (36-48h): carbon source depletion, biomass plateaus,
        DO rises slightly as oxygen demand decreases.
    """
    np.random.seed(42)
    t = np.arange(0.0, 48.5, 0.5)
    n = len(t)
    
    # Base profiles using logical functions for typical batch growth
    # OD600 uses a logistic growth curve representation
    L = 25.0  # max OD
    k = 0.25  # steepness
    t0 = 20.0 # midpoint
    od_base = L / (1 + np.exp(-k * (t - t0)))
    # Add lag phase minimum
    od_base = np.where(t < 6, 0.5 + (t/12.0), od_base)
    
    # pH drops from 7.0 to 6.8 during log phase
    ph_base = 7.0 - 0.2 * (od_base / L)
    
    # DO drops from 95% to ~20% as cells grow, then recovers slightly
    do_base = 95.0 - 75.0 * (od_base / L)
    do_base = np.where(t > 36, do_base + 1.5*(t-36), do_base)
    
    # Temperature stays around 37 (optimal for E. coli)
    temp_base = np.full(n, 37.0)
    
    # Add specified Gaussian noise
    od_noise = np.random.normal(0, 0.2, n)
    ph_noise = np.random.normal(0, 0.05, n)
    do_noise = np.random.normal(0, 1.5, n)
    temp_noise = np.random.normal(0, 0.3, n)
    
    df = pd.DataFrame({
        "time_hr": t,
        "pH": ph_base + ph_noise,
        "temp_C": temp_base + temp_noise,
        "DO_percent": np.clip(do_base + do_noise, 0, 100),
        "OD600": np.clip(od_base + od_noise, 0, None)
    })
    
    return df


# ==========================================
# 3. CSV LOADER & VALIDATOR
# ==========================================

REQUIRED_COLS = ["time_hr", "pH", "temp_C", "DO_percent", "OD600"]

@st.cache_data
def load_and_validate_csv(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        if list(df.columns) != REQUIRED_COLS:
            st.sidebar.error(f"Column mismatch! Expected: {REQUIRED_COLS}")
            return None
        st.sidebar.success("Data loaded & validated successfully.")
        return df
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {str(e)}")
        return None


# ==========================================
# 4. DERIVED METRIC CALCULATOR
# ==========================================

def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Avoid log(0) or negative OD
    od_safe = np.where(df["OD600"] <= 0, 1e-6, df["OD600"])
    
    # Calculate finite differences
    delta_ln_od = np.diff(np.log(od_safe))
    delta_t = np.diff(df["time_hr"])
    
    mu = np.zeros(len(df))
    # Handle division by zero nicely by using np.divide with out param
    safe_delta_t = np.where(delta_t == 0, np.nan, delta_t)
    mu[1:] = delta_ln_od / safe_delta_t
    
    # Smooth extreme artifacts from noise using a 3-point rolling median
    mu_series = pd.Series(mu).rolling(window=3, min_periods=1, center=True).median()
    df["growth_rate_mu"] = mu_series.fillna(0.0)
    
    return df


# ==========================================
# 8. PHASE ANNOTATION LOGIC (Placed before UI to use in main)
# ==========================================

def detect_phases(df: pd.DataFrame) -> dict:
    max_od = df["OD600"].max()
    threshold_lag = 0.10 * max_od
    threshold_stat = 0.90 * max_od
    
    phases = {}
    
    lag_mask = df["OD600"] < threshold_lag
    stat_mask = df["OD600"] > threshold_stat
    
    if lag_mask.any():
        phases["Lag"] = (df["time_hr"].min(), df.loc[lag_mask, "time_hr"].max())
    else:
        phases["Lag"] = (0, 0)
        
    if stat_mask.any():
        stat_start = df.loc[stat_mask, "time_hr"].min()
        phases["Stationary"] = (stat_start, df["time_hr"].max())
    else:
        phases["Stationary"] = (df["time_hr"].max(), df["time_hr"].max())
        
    # Log phase is between Lag end and Stat start
    phases["Log"] = (phases["Lag"][1], phases["Stationary"][0])
    
    return phases


# ==========================================
# 5. MAIN UI & ENGINEERING FLAGS
# ==========================================

st.title("BioReactor Analytics Suite")

# Initialize session state for data
if 'app_data' not in st.session_state:
    st.session_state.app_data = None

with st.sidebar:
    st.header("Control Panel")
    
    # F1: File Upload Panel
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
    
    if st.button("Use Demo Dataset", use_container_width=True):
        st.session_state.app_data = generate_synthetic_data()

    demo_csv = generate_synthetic_data().to_csv(index=False)
    st.download_button("Download Demo CSV", data=demo_csv, file_name="demo_data.csv", mime="text/csv", use_container_width=True)
        
    if uploaded_file is not None:
        df_uploaded = load_and_validate_csv(uploaded_file)
        if df_uploaded is not None:
             st.session_state.app_data = df_uploaded

    st.markdown("---")
    
    # F2: Parameter Control Panel
    st.write("")
    st.subheader("Visualization Settings")
    if st.session_state.app_data is not None:
        max_t = float(st.session_state.app_data["time_hr"].max())
        time_range = st.slider("Time Range (hr)", 0.0, max_t, (0.0, max_t), 0.5)
    else:
        time_range = (0.0, 48.0)
        
    st.write("")
    st.write("**Toggle Traces:**")
    show_ph = st.checkbox("pH", value=True)
    show_temp = st.checkbox("Temperature (°C)", value=True)
    show_do = st.checkbox("Dissolved Oxygen (%)", value=True)
    show_od = st.checkbox("OD600", value=True)
    show_mu = st.checkbox("Growth Rate (μ)", value=True)
    
    st.write("")
    st.write("**Overlays:**")
    show_phases = st.checkbox("Show Phase Annotations", value=True)
    show_rolling = st.checkbox("Show Rolling Average (n=3)")
    show_bands = st.checkbox("Show ±1σ Confidence Band")
    is_log_scale = st.checkbox("OD600 Log Scale")
    
    # Engineering Flags placeholder
    st.markdown("---")
    st.subheader("Process Warnings")
    warning_container = st.container()

# Ensure we have data
if st.session_state.app_data is None:
    st.info("Please load data or use the Demo Dataset from the sidebar to begin.")
    st.stop()

# Prepare Data
df_plot = st.session_state.app_data.copy()
df_plot = calculate_derived_metrics(df_plot)

# Filter by time range
df_plot = df_plot[(df_plot["time_hr"] >= time_range[0]) & (df_plot["time_hr"] <= time_range[1])]

# Apply Rolling Avg if toggled
if show_rolling:
    cols_to_smooth = ["pH", "temp_C", "DO_percent", "OD600", "growth_rate_mu"]
    df_plot[cols_to_smooth] = df_plot[cols_to_smooth].rolling(window=3, min_periods=1, center=True).mean()

# Populate Engineering Flags in Sidebar
with warning_container:
    # pH Check
    ph_outliers = df_plot[(df_plot["pH"] < 6.8) | (df_plot["pH"] > 7.2)]
    if not ph_outliers.empty:
        st.error(f"pH deviated outside optimal 6.8-7.2 range ({len(ph_outliers)} instances)")
    else:
        st.success("pH maintained within optimal limits")
        
    # Temp Check
    temp_deviations = df_plot[abs(df_plot["temp_C"] - 37.0) > 1.0]
    if not temp_deviations.empty:
        st.error(f"Temperature > 1°C deviation detected ({len(temp_deviations)} instances)")
        
    # DO Check
    do_stress = df_plot[df_plot["DO_percent"] < 20.0]
    if not do_stress.empty:
        st.warning(f"DO dropped below 20% - Potential Oxygen limitation!")
        
    # Mu check
    mu_noise = df_plot[df_plot["growth_rate_mu"] > 1.0]
    if not mu_noise.empty:
        st.warning(f"μ > 1.0 hr⁻¹ detected - Check OD sensor for noise artifacts")


# ==========================================
# 6. MULTI-AXIS CHART BUILDER
# ==========================================

def create_multiaxis_chart(df, phases):
    fig = go.Figure()
    
    # Define color scheme
    colors = {
        "OD600": "#00FF00",  # Neon Green
        "pH": "#00BBFF",     # Light Blue
        "Temp": "#FF3333",   # Red
        "DO": "#FFAA00",     # Orange
        "Mu": "#BB33FF"      # Purple
    }
    
    x = df["time_hr"]
    
    def add_trace(series_name, col_name, color, yaxis, dash='solid', width=2):
        y = df[col_name]
        
        if show_bands:
            std = y.std()
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([y + std, (y - std)[::-1]]),
                fill='toself',
                fillcolor=color,
                opacity=0.1,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                yaxis=yaxis
            ))
            
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=series_name,
            line=dict(color=color, dash=dash, width=width),
            yaxis=yaxis
        ))

    if show_od: add_trace("OD600", "OD600", colors["OD600"], "y1", width=3)
    if show_ph: add_trace("pH", "pH", colors["pH"], "y2", dash='dash')
    if show_temp: add_trace("Temp (°C)", "temp_C", colors["Temp"], "y3", dash='dot')
    if show_do: add_trace("DO %", "DO_percent", colors["DO"], "y4")
    if show_mu: add_trace("μ (hr⁻¹)", "growth_rate_mu", colors["Mu"], "y5")

    layout = dict(
        title="Bioreactor Performance — 48-Hour Fermentation Cycle",
        xaxis=dict(
            title="Time (hours)",
            domain=[0, 0.75],
            showgrid=True, gridcolor="#333", gridwidth=1
        ),
        yaxis=dict(
            title=dict(text="OD600", font=dict(color=colors["OD600"])), tickfont=dict(color=colors["OD600"]),
            type='log' if is_log_scale else 'linear', showgrid=False
        ),
        yaxis2=dict(
            title=dict(text="pH", font=dict(color=colors["pH"])), tickfont=dict(color=colors["pH"]),
            overlaying="y", side="right", position=0.75, showgrid=False
        ),
        yaxis3=dict(
            title=dict(text="Temp (°C)", font=dict(color=colors["Temp"])), tickfont=dict(color=colors["Temp"]),
            overlaying="y", side="right", position=0.83, showgrid=False
        ),
        yaxis4=dict(
            title=dict(text="DO %", font=dict(color=colors["DO"])), tickfont=dict(color=colors["DO"]),
            overlaying="y", side="right", position=0.91, showgrid=False
        ),
        yaxis5=dict(
            title=dict(text="Growth Rate μ", font=dict(color=colors["Mu"])), tickfont=dict(color=colors["Mu"]),
            overlaying="y", side="right", position=0.99, showgrid=False
        ),
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#121212",
        font=dict(color="#FFF", family="Consolas, monospace"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)")
    )
    fig.update_layout(layout)
    
    if show_phases:
        fig.add_vrect(x0=phases["Lag"][0], x1=phases["Lag"][1], fillcolor="yellow", opacity=0.1, layer="below", line_width=0, annotation_text="LAG", annotation_position="top left")
        fig.add_vrect(x0=phases["Log"][0], x1=phases["Log"][1], fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="LOG", annotation_position="top left")
        fig.add_vrect(x0=phases["Stationary"][0], x1=phases["Stationary"][1], fillcolor="grey", opacity=0.2, layer="below", line_width=0, annotation_text="STATIONARY", annotation_position="top left")

    return fig

with st.spinner("Rendering Multi-Axis Visualization..."):
    phases_dict = detect_phases(df_plot)
    fig = create_multiaxis_chart(df_plot, phases_dict)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 4. (Cont.) PHASE DETECTION TIMELINE
# ==========================================
if show_phases:
    st.markdown("### Detected Growth Phases")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Lag Phase:** {phases_dict['Lag'][0]:.1f} hr - {phases_dict['Lag'][1]:.1f} hr")
    c2.success(f"**Log Phase:** {phases_dict['Log'][0]:.1f} hr - {phases_dict['Log'][1]:.1f} hr")
    c3.error(f"**Stationary Phase:** {phases_dict['Stationary'][0]:.1f} hr - {phases_dict['Stationary'][1]:.1f} hr")


# ==========================================
# 7. STATISTICAL SUMMARY PANEL
# ==========================================
st.markdown("### Parameter Statistics")

def compute_stats(df):
    metrics = ["pH", "temp_C", "DO_percent", "OD600", "growth_rate_mu"]
    stats_list = []
    
    for m in metrics:
        if m not in df.columns: continue
        col_data = df[m]
        stats_list.append({
            "Parameter": m,
            "Min": round(col_data.min(), 3),
            "Max": round(col_data.max(), 3),
            "Mean": round(col_data.mean(), 3),
            "Std Dev": round(col_data.std(), 3),
            "Time of Peak (hr)": df.loc[col_data.idxmax(), "time_hr"]
        })
    return pd.DataFrame(stats_list)

stats_df = compute_stats(df_plot)

def highlight_anomalies(s):
    if s.name in ["Min", "Max"]:
        out = []
        for val, mean, std in zip(s, stats_df["Mean"], stats_df["Std Dev"]):
            if abs(val - mean) > 2 * std:
                out.append('background-color: rgba(255, 0, 0, 0.4)')
            else:
                out.append('')
        return out
    return [''] * len(s)

st.dataframe(stats_df.style.apply(highlight_anomalies), use_container_width=True)

colA, colB = st.columns(2)
with colA:
    st.download_button("Download Summary CSV", data=stats_df.to_csv(index=False), file_name="bioreactor_summary.csv", mime="text/csv")


# ==========================================
# 8. CORRELATION HEATMAP
# ==========================================
with st.expander("Show Correlation Analysis"):
    st.markdown("Pearson correlation matrix depicting linear relationships between parameters.")
    corr_cols = ["pH", "temp_C", "DO_percent", "OD600", "growth_rate_mu"]
    corr_matrix = df_plot[corr_cols].corr(method='pearson')
    
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu",
        title="Parameter Correlation Heatmap"
    )
    fig_corr.update_layout(plot_bgcolor="#1E1E1E", paper_bgcolor="#121212", font=dict(color="#FFF"))
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info("""
    **Interpretation of Strongest Correlations:**
    1. **OD600 vs DO (-r):** Strong negative correlation. As cell biomass exponentially increases, metabolic demand for oxygen surges, causing rapid depletion of Dissolved Oxygen in the vessel.
    2. **OD600 vs pH (-r):** Negative correlation. As cultures grow (increasing OD), they often excrete acidic metabolic byproducts (e.g., acetate in E. coli), leading to a drop in vessel pH.
    """)
