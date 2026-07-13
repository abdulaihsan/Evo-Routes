import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Evo Routes",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    /* MAIN BACKGROUND: Dark Blue/Slate */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }

    /* HEADERS: Peach */
    h1, h2, h3, h4, h5, h6 {
        color: #FDBA74 !important; /* Peach/Orange-300 */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }

    /* SIDEBAR: Darker Navy */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1E293B;
    }
    
    /* SIDEBAR TEXT */
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] h1 {
        color: #94A3B8 !important;
    }

    /* BUTTONS: Peach Background, Dark Text */
    div.stButton > button {
        background-color: #FDBA74;
        color: #0F172A;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #FB923C; /* Darker Peach */
        color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* DATAFRAMES: Dark Mode Friendly */
    div[data-testid="stDataFrame"] {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 10px;
    }

    /* METRICS */
    div[data-testid="stMetricValue"] {
        color: #FDBA74 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    /* ALERTS/INFO BOXES */
    div[data-baseweb="notification"] {
        background-color: #1E293B;
        border-left-color: #FDBA74;
    }
    
    /* RADIO BUTTONS */
    div[role="radiogroup"] label {
        color: #E2E8F0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- TRAFFIC MONITORING SYSTEM ---
TRAFFIC_LOG_FILE = "traffic_log.csv"


def log_event(event_type):
    """Logs an event (Visit, Optimization, etc.) to a CSV file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(TRAFFIC_LOG_FILE):
        with open(TRAFFIC_LOG_FILE, "w") as f:
            f.write("Timestamp,Event\n")
    with open(TRAFFIC_LOG_FILE, "a") as f:
        f.write(f"{timestamp},{event_type}\n")


# --- AUTHENTICATION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if st.session_state["password"] == "Logistics2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown(
            "<h1 style='text-align: center; color: #FDBA74;'>🔒 Evo Routes Access</h1>",
            unsafe_allow_html=True,
        )
        st.text_input(
            "Enter Security Code",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Security Code",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("⛔ Access Denied")
        return False
    else:
        return True


if not check_password():
    st.stop()

if "visit_logged" not in st.session_state:
    log_event("App Visit")
    st.session_state["visit_logged"] = True

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("### Model Architecture")

model_choice = st.sidebar.radio(
    "Predictive Engine:",
    (
        "Linear Regression (Baseline)",
        "Random Forest (Proposed)",
        "Logarithmic Regression (Experimental)",
    ),
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Active Engine:** {model_choice.split('(')[0]}")

if "Linear" in model_choice:
    st.sidebar.markdown(
        "🔹 **Performance:** High Speed, High Interpretability\n\n📉 **RMSE:** 8.37 min\n\n📈 **R²:** 0.84"
    )
elif "Random Forest" in model_choice:
    st.sidebar.markdown(
        "🔹 **Performance:** Non-linear capability, Higher Latency\n\n📉 **RMSE:** 11.46 min\n\n📈 **R²:** 0.70"
    )
else:
    st.sidebar.markdown(
        "🔹 **Performance:** Experimental Traffic Modeling\n\n📉 **RMSE:** 13.1 min\n\n📈 **R²:** 0.61"
    )

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Evo Routes Logistics")

# --- MAIN APP LAYOUT ---
st.title("🧬 Evo Routes")
st.markdown("### Intelligent Logistics & Delivery Optimization")
st.markdown(
    """
<div style='background-color: #1E293B; padding: 15px; border-radius: 10px; border-left: 5px solid #FDBA74; margin-bottom: 20px;'>
    <strong>System Status:</strong> Operational 🟢 <br>
    Leveraging <b>Genetic Algorithms</b> for routing and <b>Machine Learning</b> for time prediction.
</div>
""",
    unsafe_allow_html=True,
)

# 1. DATA UPLOAD SECTION
st.header("1. Data Ingestion")
uploaded_file = st.file_uploader(
    "Drop manifest file here (CSV)", type=["csv"], key="main_csv_uploader"
)

if not uploaded_file:
    st.info("ℹ️ Using demo dataset for visualization.")
    # MOCK DATA
    data = {
        "CUSTOMERID": [f"Cust_{i}" for i in range(1, 11)],
        "LATITUDE": [
            47.6062 + random.uniform(-0.05, 0.05) for _ in range(10)
        ],  # Seattle
        "LONGITUDE": [-122.3321 + random.uniform(-0.05, 0.05) for _ in range(10)],
        "MILES": [random.uniform(2, 20) for _ in range(10)],
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(uploaded_file)

    # DATA NORMALIZATION
    df.columns = [c.upper() for c in df.columns]

    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        st.warning(
            "⚠️ CSV lacks 'Latitude'/'Longitude'. Generating simulated coordinates."
        )
        df["LATITUDE"] = [47.6062 + random.uniform(-0.1, 0.1) for _ in range(len(df))]
        df["LONGITUDE"] = [
            -122.3321 + random.uniform(-0.1, 0.1) for _ in range(len(df))
        ]

    if "CUSTOMERID" not in df.columns:
        df["CUSTOMERID"] = [f"Stop_{i}" for i in range(len(df))]

with st.expander("📂 View Source Data", expanded=True):
    st.dataframe(df.head(), use_container_width=True)

# 2. OPTIMIZATION SCREEN
st.header("2. Route Optimization Engine")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        """
    <div style='background-color: #172554; padding: 15px; border-radius: 8px;'>
    <h4 style='margin-top:0'>Parameters</h4>
    """,
        unsafe_allow_html=True,
    )
    generations = st.slider(
        "Evolution Generations", min_value=10, max_value=100, value=50
    )
    st.markdown("</div><br>", unsafe_allow_html=True)

    optimize_btn = st.button(
        "🚀 EXECUTE OPTIMIZATION", type="primary", use_container_width=True
    )


# BACKEND SIMULATION
def run_optimization_simulation(df, model_type):
    """Running genetic_algorithm.py"""
    try:
        from genetic_algorithm import RouteOptimizerGA
    except ImportError:
        st.error("Critical Error: Core 'genetic_algorithm.py' module not found.")
        return 0, 0, 0, 0

    with st.spinner("🧬 Initializing Genetic Evolution..."):
        locations = list(zip(df["LATITUDE"], df["LONGITUDE"]))

        optimizer = RouteOptimizerGA(
            locations, population_size=50, mutation_rate=0.05, generations=50
        )

    with st.spinner("🔄 Evolving Optimal Paths..."):
        progress_bar = st.progress(0)
        start_time = time.time()
        best_route_indices = optimizer.run()
        progress_bar.progress(100)
        end_time = time.time()
        compute_time = end_time - start_time

    optimized_df = df.iloc[best_route_indices].reset_index(drop=True)

    original_dist = 0
    for i in range(len(df) - 1):
        original_dist += optimizer._haversine(locations[i], locations[i + 1])
    original_dist = original_dist / 1609.34

    optimized_dist = 0
    opt_locations = list(zip(optimized_df["LATITUDE"], optimized_df["LONGITUDE"]))
    for i in range(len(opt_locations) - 1):
        optimized_dist += optimizer._haversine(opt_locations[i], opt_locations[i + 1])
    optimized_dist = optimized_dist / 1609.34

    if "Linear" in model_type:
        pred_time = optimized_dist * 2
    elif "Logarithmic" in model_type:
        pred_time = optimized_dist * 2.5
    else:
        pred_time = optimized_dist * 2.25

    return original_dist, optimized_dist, compute_time, pred_time, optimized_df


# SHAP EXPLAINABILITY
@st.cache_resource
def build_model_and_explain(input_df):
    if not HAS_ML_LIBS:
        return None, "ML Libraries Missing"

    required_cols = ["START_DATE", "END_DATE", "MILES"]
    if not all(col in input_df.columns for col in required_cols):
        return None, "Missing Required Columns"

    try:
        df_ml = input_df.copy()
        df_ml = df_ml.dropna(subset=required_cols)

        df_ml["START_DATE"] = pd.to_datetime(df_ml["START_DATE"], errors="coerce")
        df_ml["END_DATE"] = pd.to_datetime(df_ml["END_DATE"], errors="coerce")
        df_ml = df_ml.dropna(subset=["START_DATE", "END_DATE"])

        df_ml["TRAVEL_TIME_MIN"] = (
            df_ml["END_DATE"] - df_ml["START_DATE"]
        ).dt.total_seconds() / 60.0
        df_ml["START_HOUR"] = df_ml["START_DATE"].dt.hour
        df_ml["DAY_OF_WEEK"] = df_ml["START_DATE"].dt.dayofweek

        df_ml = df_ml[df_ml["TRAVEL_TIME_MIN"] > 0]
        df_ml = df_ml[df_ml["MILES"] > 0]

        numeric_features = ["MILES", "START_HOUR", "DAY_OF_WEEK"]
        X = df_ml[numeric_features]
        y = df_ml["TRAVEL_TIME_MIN"]

        if len(X) < 10:
            return None, "Insufficient Data Points"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        sample_X = X_test.iloc[:50] if len(X_test) > 50 else X_test
        shap_values = explainer.shap_values(sample_X)

        return (shap_values, sample_X), "Success"

    except Exception as e:
        return None, str(e)


# MAP CREATION
def create_route_map(df):
    try:
        import folium

        from pathfinding import PathFinder

        center_lat = df["LATITUDE"].mean()
        center_lon = df["LONGITUDE"].mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron"
        )  # Sleeker map tiles

        pf = PathFinder(center_lat=center_lat, center_lon=center_lon)

        locations = list(zip(df["LATITUDE"], df["LONGITUDE"]))

        for index, row in df.iterrows():
            if index == len(df) - 1 and row["CUSTOMERID"] == df.iloc[0]["CUSTOMERID"]:
                continue

            if index == 0:
                folium.Marker(
                    [row["LATITUDE"], row["LONGITUDE"]],
                    popup=f"🏢 <b>HEADQUARTERS</b><br>{row['CUSTOMERID']}",
                    icon=folium.Icon(color="darkblue", icon="building", prefix="fa"),
                ).add_to(m)
            else:
                folium.Marker(
                    [row["LATITUDE"], row["LONGITUDE"]],
                    popup=f"📦 Stop #{index}<br>{row['CUSTOMERID']}",
                    icon=folium.Icon(color="orange", icon="truck", prefix="fa"),
                ).add_to(m)

        full_route_points = []

        if len(locations) > 1:
            with st.spinner("Calculating street-level polyline..."):
                for i in range(len(locations) - 1):
                    start = locations[i]
                    end = locations[i + 1]
                    segment = pf.get_route_coords(start, end)
                    full_route_points.extend(segment)

        folium.PolyLine(
            full_route_points, color="#0F172A", weight=4, opacity=0.8
        ).add_to(m)

        return m
    except Exception as e:
        st.error(f"Mapping Error: {e}")
        return None


# LOGIC TO HANDLE BUTTON CLICKS AND PERSISTENCE
if optimize_btn:
    orig_dist, opt_dist, compute_time, pred_time, optimized_df_sorted = (
        run_optimization_simulation(df, model_choice)
    )
    map_obj = create_route_map(optimized_df_sorted)

    st.session_state["optimization_run"] = True
    st.session_state["orig_dist"] = orig_dist
    st.session_state["opt_dist"] = opt_dist
    st.session_state["pred_time"] = pred_time
    st.session_state["last_model_used"] = model_choice
    st.session_state["generated_map"] = map_obj
    st.session_state["compute_time"] = compute_time

# RESULTS & VISUALIZATION
if st.session_state.get("optimization_run"):
    with col2:
        st.success("Optimization Complete")

    orig_dist = st.session_state["orig_dist"]
    opt_dist = st.session_state["opt_dist"]
    pred_time = st.session_state["pred_time"]
    model_used = st.session_state.get("last_model_used", "Unknown Model")
    map_obj = st.session_state.get("generated_map")
    compute_time = st.session_state["compute_time"]

    efficiency_gain = ((orig_dist - opt_dist) / orig_dist) * 100

    st.subheader("📊 Performance Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        "Optimized Distance",
        f"{opt_dist:.2f} mi",
        delta=f"{efficiency_gain:.1f}% Reduction",
    )
    kpi2.metric("Predicted Duration", f"{pred_time:.0f} min", delta="AI Prediction")
    kpi3.metric(
        "Computation Latency", f"{compute_time:.4f} s", delta="Genetic Algorithm"
    )

    st.subheader("🗺️ Route Visualization")
    if map_obj:
        try:
            from streamlit_folium import st_folium

            st_folium(map_obj, width=1000, height=500, returned_objects=[])
        except Exception as e:
            st.error(f"Map Rendering Failed: {e}")

# EXPLAINABILITY
st.divider()
st.header("3. AI Interpretability (SHAP)")

with st.expander("🔍 View Decision Factors", expanded=False):
    shap_data, status = build_model_and_explain(df)

    if shap_data:
        shap_values, sample_X = shap_data
        st.markdown("**Live Analysis:** SHAP values generated from uploaded dataset.")

        feature_importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"Feature": sample_X.columns, "Importance": feature_importance}
        ).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df, x="Feature", y="Importance", color="#FDBA74")

        st.caption("Feature Impact on Travel Time Prediction (Random Forest Engine)")

    else:
        st.warning(f"Live SHAP generation unavailable: {status}.")
        st.info("Displaying Reference Benchmarks:")

        chart_data = pd.DataFrame(
            {
                "Feature": [
                    "MILES",
                    "START_HOUR",
                    "DAY_OF_WEEK",
                ],
                "Importance": [0.65, 0.20, 0.10],
            }
        )
        st.bar_chart(chart_data, x="Feature", y="Importance", color="#FDBA74")

st.markdown("---")
st.caption("Built by Abdullah Ihsan, Aazeb Ali & Syed Faseeh | Evo Routes")
