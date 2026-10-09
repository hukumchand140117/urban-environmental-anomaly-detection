from pathlib import Path
from io import BytesIO
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# URBAN ENVIRONMENTAL ANOMALY DETECTION
# Historical analysis + station analysis + manual prediction
# ============================================================

st.set_page_config(
    page_title="Urban Environmental Anomaly Detection",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "PM2_5_mean", "PM2_5_max", "PM2_5_std",
    "PM10_mean", "PM10_max", "PM10_std",
    "SO2_mean", "SO2_max", "SO2_std",
    "NO2_mean", "NO2_max", "NO2_std",
    "CO_mean", "CO_max", "CO_std",
    "O3_mean", "O3_max", "O3_std",
    "TEMP_mean", "PRES_mean", "DEWP_mean",
    "RAIN_sum", "WSPM_mean"
]

POLLUTANT_GROUPS = [
    ("PM2.5", "PM2_5_mean", "PM2_5_max", "PM2_5_std", "µg/m³", 50, 100, 20),
    ("PM10", "PM10_mean", "PM10_max", "PM10_std", "µg/m³", 80, 150, 30),
    ("SO2", "SO2_mean", "SO2_max", "SO2_std", "µg/m³", 20, 50, 10),
    ("NO2", "NO2_mean", "NO2_max", "NO2_std", "µg/m³", 40, 80, 15),
    ("CO", "CO_mean", "CO_max", "CO_std", "µg/m³", 1000, 2000, 500),
    ("O3", "O3_mean", "O3_max", "O3_std", "µg/m³", 60, 120, 20),
]

MANUAL_FIELDS = [
    ("PM2_5_mean", "PM2.5 Mean", "µg/m³", "WHO 24-hour guideline: 15", 50, 0, 100000, 1),
    ("PM2_5_max", "PM2.5 Maximum", "µg/m³", "No universal maximum", 100, 0, 100000, 1),
    ("PM2_5_std", "PM2.5 Standard Deviation", "µg/m³", "Hourly variation", 20, 0, 100000, 1),
    ("PM10_mean", "PM10 Mean", "µg/m³", "WHO 24-hour guideline: 45", 80, 0, 100000, 1),
    ("PM10_max", "PM10 Maximum", "µg/m³", "No universal maximum", 150, 0, 100000, 1),
    ("PM10_std", "PM10 Standard Deviation", "µg/m³", "Hourly variation", 30, 0, 100000, 1),
    ("SO2_mean", "SO2 Mean", "µg/m³", "WHO 24-hour guideline: 40", 20, 0, 100000, 1),
    ("SO2_max", "SO2 Maximum", "µg/m³", "No universal maximum", 50, 0, 100000, 1),
    ("SO2_std", "SO2 Standard Deviation", "µg/m³", "Hourly variation", 10, 0, 100000, 1),
    ("NO2_mean", "NO2 Mean", "µg/m³", "WHO 24-hour guideline: 25", 40, 0, 100000, 1),
    ("NO2_max", "NO2 Maximum", "µg/m³", "No universal maximum", 80, 0, 100000, 1),
    ("NO2_std", "NO2 Standard Deviation", "µg/m³", "Hourly variation", 15, 0, 100000, 1),
    ("CO_mean", "CO Mean", "µg/m³", "WHO 24-hour guideline: 4000", 1000, 0, 1000000, 100),
    ("CO_max", "CO Maximum", "µg/m³", "No universal maximum", 2000, 0, 1000000, 100),
    ("CO_std", "CO Standard Deviation", "µg/m³", "Hourly variation", 500, 0, 1000000, 50),
    ("O3_mean", "O3 Mean", "µg/m³", "WHO 8-hour guideline: 100*", 60, 0, 100000, 1),
    ("O3_max", "O3 Maximum", "µg/m³", "No universal maximum", 120, 0, 100000, 1),
    ("O3_std", "O3 Standard Deviation", "µg/m³", "Hourly variation", 20, 0, 100000, 1),
    ("TEMP_mean", "Temperature Mean", "°C", "Indicative guide: −10 to 50 °C", 15, -100, 100, 1),
    ("PRES_mean", "Pressure Mean", "hPa", "Varies by altitude and weather", 1010, 0, 2000, 1),
    ("DEWP_mean", "Dew Point Mean", "°C", "Depends on humidity and temperature", 8, -100, 100, 1),
    ("RAIN_sum", "Daily Rainfall Total", "mm", "Zero means no recorded rain", 0, 0, 100000, 1),
    ("WSPM_mean", "Wind Speed Mean", "m/s", "No universal maximum", 2, 0, 200, 0.5),
]

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; }
    .main-title { font-size: 30px; font-weight: 700; color: #F1F5F9; }
    .subtitle { color: #A0A8B5; font-size: 14px; margin-bottom: 20px; }
    .section-title { color: #DCE3EC; font-size: 20px; font-weight: 600; margin: 12px 0; }
    .field-label { display:flex; justify-content:space-between; align-items:baseline;
        flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
    .field-name { color:#E2E8F0; font-size:13px; font-weight:500; }
    .field-unit { color:#A0A8B5; font-size:12px; }
    .field-reference { color:#8993A2; font-size:10px; text-align:right; }
    .muted { color:#A0A8B5; font-size:12px; }
    .html-table { width:100%; border-collapse:collapse; font-size:13px; }
    .html-table th { text-align:left; background:#202631; color:#E2E8F0; padding:9px; }
    .html-table td { border-bottom:1px solid #303744; padding:8px; color:#DCE3EC; }
    .html-table tr:nth-child(even) { background:#141922; }
    .result-normal { background:#142B25; border:1px solid #286B52; border-radius:10px;
        padding:18px; color:#A7F3D0; }
    .result-anomaly { background:#351B20; border:1px solid #8D3945; border-radius:10px;
        padding:18px; color:#FCA5A5; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🌍 Urban Environmental Anomaly Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Unsupervised machine learning · Historical anomaly analysis · '
    '12 Beijing monitoring stations · March 2013 – February 2017</div>',
    unsafe_allow_html=True
)

# -------------------- Load trained models --------------------

@st.cache_resource(show_spinner=False)
def load_models():
    required = {
        "scaler": MODEL_DIR / "scaler.pkl",
        "isolation_forest": MODEL_DIR / "isolation_forest.pkl",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
        )
    models = {
        "scaler": joblib.load(required["scaler"]),
        "isolation_forest": joblib.load(required["isolation_forest"]),
    }
    pca_path = MODEL_DIR / "pca.pkl"
    kmeans_path = MODEL_DIR / "kmeans_model.pkl"
    models["pca"] = joblib.load(pca_path) if pca_path.exists() else None
    models["kmeans"] = joblib.load(kmeans_path) if kmeans_path.exists() else None
    return models

try:
    models = load_models()
    scaler = models["scaler"]
    isolation_forest = models["isolation_forest"]
    pca = models["pca"]
    kmeans = models["kmeans"]
except Exception as exc:
    st.error("The trained models could not be loaded. Check the models folder.")
    st.code(str(exc))
    st.stop()

# -------------------- Load and process raw station data --------------------

NUMERIC_HOURLY = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

@st.cache_data(show_spinner="Loading and preparing the 12 monitoring stations...")
def build_historical_data(dataset_folder):
    pattern = str(Path(dataset_folder) / "PRSA_Data_*_20130301-20170228.csv")
    csv_files = sorted(glob.glob(pattern))
    if len(csv_files) != 12:
        raise FileNotFoundError(
            f"Expected 12 station CSV files in '{dataset_folder}', found {len(csv_files)}. "
            "Place all original PRSA_Data_*_20130301-20170228.csv files in the dataset folder."
        )

    frames = []
    for file in csv_files:
        frame = pd.read_csv(file)
        if "station" not in frame.columns:
            # The station name is encoded in the file name if the CSV lacks a station column.
            station_name = Path(file).stem.replace("PRSA_Data_", "").replace(
                "_20130301-20170228", ""
            )
            frame["station"] = station_name
        frames.append(frame)

    master = pd.concat(frames, ignore_index=True)

    required_columns = ["year", "month", "day", "hour", "station"] + NUMERIC_HOURLY
    missing_columns = [c for c in required_columns if c not in master.columns]
    if missing_columns:
        raise ValueError(
            "The station CSV files do not contain expected columns: "
            + ", ".join(missing_columns)
        )

    master["datetime"] = pd.to_datetime(
        master[["year", "month", "day", "hour"]].rename(
            columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
        ),
        errors="coerce"
    )
    master = master.dropna(subset=["datetime"]).copy()
    master = master.sort_values(["station", "datetime"]).reset_index(drop=True)

    for feature in NUMERIC_HOURLY:
        master[feature] = pd.to_numeric(master[feature], errors="coerce")
        master[feature] = master.groupby("station")[feature].transform(
            lambda series: series.interpolate(
                method="linear", limit_direction="both"
            )
        )

    master["date"] = master["datetime"].dt.normalize()

    # Reproduce the notebook's station-day aggregation.
    station_daily = (
        master.groupby(["station", "date"])
        .agg(
            **{
                "PM2.5_mean": ("PM2.5", "mean"),
                "PM2.5_max": ("PM2.5", "max"),
                "PM10_mean": ("PM10", "mean"),
                "PM10_max": ("PM10", "max"),
                "SO2_mean": ("SO2", "mean"),
                "SO2_max": ("SO2", "max"),
                "NO2_mean": ("NO2", "mean"),
                "NO2_max": ("NO2", "max"),
                "CO_mean": ("CO", "mean"),
                "CO_max": ("CO", "max"),
                "O3_mean": ("O3", "mean"),
                "O3_max": ("O3", "max"),
                "TEMP_mean": ("TEMP", "mean"),
                "PRES_mean": ("PRES", "mean"),
                "DEWP_mean": ("DEWP", "mean"),
                "RAIN_sum": ("RAIN", "sum"),
                "WSPM_mean": ("WSPM", "mean"),
            }
        )
        .reset_index()
    )

    # Reproduce the notebook's city-level daily feature engineering.
    city_daily = (
        station_daily.groupby("date")
        .agg(
            PM2_5_mean=("PM2.5_mean", "mean"),
            PM2_5_max=("PM2.5_max", "max"),
            PM2_5_std=("PM2.5_mean", "std"),
            PM10_mean=("PM10_mean", "mean"),
            PM10_max=("PM10_max", "max"),
            PM10_std=("PM10_mean", "std"),
            SO2_mean=("SO2_mean", "mean"),
            SO2_max=("SO2_max", "max"),
            SO2_std=("SO2_mean", "std"),
            NO2_mean=("NO2_mean", "mean"),
            NO2_max=("NO2_max", "max"),
            NO2_std=("NO2_mean", "std"),
            CO_mean=("CO_mean", "mean"),
            CO_max=("CO_max", "max"),
            CO_std=("CO_mean", "std"),
            O3_mean=("O3_mean", "mean"),
            O3_max=("O3_max", "max"),
            O3_std=("O3_mean", "std"),
            TEMP_mean=("TEMP_mean", "mean"),
            PRES_mean=("PRES_mean", "mean"),
            DEWP_mean=("DEWP_mean", "mean"),
            RAIN_sum=("RAIN_sum", "sum"),
            WSPM_mean=("WSPM_mean", "mean"),
        )
        .reset_index()
    )

    city_daily[FEATURES] = city_daily[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = city_daily[FEATURES].notna().all(axis=1)
    city_daily = city_daily.loc[valid_mask].copy().reset_index(drop=True)

    scaled = scaler.transform(city_daily[FEATURES])
    city_daily["anomaly_label"] = isolation_forest.predict(scaled)
    city_daily["anomaly_score"] = isolation_forest.decision_function(scaled)

    if pca is not None and kmeans is not None:
        try:
            city_daily["cluster"] = kmeans.predict(pca.transform(scaled))
        except Exception:
            city_daily["cluster"] = np.nan
    else:
        city_daily["cluster"] = np.nan

    return master, station_daily, city_daily

try:
    master_df, station_daily_df, anomaly_df = build_historical_data(str(DATASET_DIR))
except Exception as exc:
    st.error("Historical data could not be loaded or processed.")
    st.markdown(
        "Check that your project contains the **dataset** folder with all 12 original "
        "Beijing station CSV files."
    )
    st.code(str(exc))
    st.stop()

# -------------------- Utility functions --------------------

def html_table(frame, max_rows=30):
    if frame is None or frame.empty:
        st.info("No records match the current selection.")
        return
    shown = frame.head(max_rows).copy()
    html = shown.to_html(index=False, escape=True, classes="html-table", border=0)
    st.markdown(html, unsafe_allow_html=True)
    if len(frame) > max_rows:
        st.caption(f"Showing {max_rows} of {len(frame)} rows. Download the CSV for all records.")

def fig_to_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    return buffer

def season_for_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"

anomaly_df["year"] = anomaly_df["date"].dt.year
anomaly_df["month"] = anomaly_df["date"].dt.month
anomaly_df["month_name"] = anomaly_df["date"].dt.strftime("%b")
anomaly_df["season"] = anomaly_df["month"].map(season_for_month)
anomaly_df["status"] = np.where(
    anomaly_df["anomaly_label"] == -1, "Anomalous", "Normal"
)

total_days = len(anomaly_df)
anomalous_days = int((anomaly_df["anomaly_label"] == -1).sum())
normal_days = int((anomaly_df["anomaly_label"] == 1).sum())
anomaly_rate = (anomalous_days / total_days * 100) if total_days else 0
start_date = anomaly_df["date"].min().date()
end_date = anomaly_df["date"].max().date()

# -------------------- Sidebar --------------------

st.sidebar.title("Project Navigation")
st.sidebar.caption("Choose a section from the tabs on the main page.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.write("Beijing Multi-Site Air Quality")
st.sidebar.write("12 monitoring stations")
st.sidebar.write(f"{start_date} to {end_date}")
st.sidebar.markdown("---")
st.sidebar.caption("Isolation Forest is used for anomaly detection. K-Means clusters are supplementary.")

overview_tab, history_tab, temporal_tab, station_tab, feature_tab, manual_tab, download_tab = st.tabs([
    "Overview",
    "Historical Anomalies",
    "Temporal Trends",
    "Station Analysis",
    "Feature Insights",
    "Manual Prediction",
    "Download Reports"
])

# -------------------- TAB 1: Overview --------------------

with overview_tab:
    st.markdown('<div class="section-title">Historical Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City-days analyzed", f"{total_days:,}")
    c2.metric("Anomalous days", f"{anomalous_days:,}")
    c3.metric("Normal days", f"{normal_days:,}")
    c4.metric("Anomaly rate", f"{anomaly_rate:.2f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Monitoring stations", f"{station_daily_df['station'].nunique()}")
    c6.metric("Model features", f"{len(FEATURES)}")
    c7.metric("Hourly records", f"{len(master_df):,}")

    st.markdown(
        "This dashboard re-creates the notebook's station-wise cleaning, daily aggregation, "
        "and 23-feature city-level representation before applying the saved scaler and "
        "Isolation Forest model."
    )

    monthly = anomaly_df.groupby("month").agg(
        Observed_Days=("date", "size"),
        Anomalous_Days=("anomaly_label", lambda x: int((x == -1).sum()))
    ).reindex(range(1, 13), fill_value=0)
    monthly["Anomaly_Rate"] = np.where(
        monthly["Observed_Days"] > 0,
        monthly["Anomalous_Days"] / monthly["Observed_Days"] * 100,
        0
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.bar(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           monthly["Anomalous_Days"])
    ax.set_title("Anomalous Days by Calendar Month", color="#E2E8F0")
    ax.set_ylabel("Anomalous days", color="#DCE3EC")
    ax.tick_params(colors="#DCE3EC")
    for spine in ax.spines.values():
        spine.set_color("#48515E")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="section-title">Most Anomalous Historical Days</div>', unsafe_allow_html=True)
    top_days = anomaly_df[anomaly_df["anomaly_label"] == -1].sort_values("anomaly_score").head(10)
    display_cols = ["date", "anomaly_score", "PM2_5_mean", "PM10_mean", "NO2_mean", "TEMP_mean", "WSPM_mean"]
    top_view = top_days[display_cols].copy()
    top_view["date"] = top_view["date"].dt.strftime("%Y-%m-%d")
    html_table(top_view.round(3), max_rows=10)

# -------------------- TAB 2: Historical anomalies --------------------

with history_tab:
    st.markdown('<div class="section-title">Explore Historical Anomalies</div>', unsafe_allow_html=True)
    st.caption("Filter the daily city-level model results by date and classification.")

    min_d = anomaly_df["date"].min().date()
    max_d = anomaly_df["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="history_date_range"
    )
    status_choice = st.selectbox(
        "Classification",
        ["All days", "Anomalous only", "Normal only"],
        key="history_status"
    )

    filtered = anomaly_df.copy()
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        left = pd.Timestamp(date_range[0])
        right = pd.Timestamp(date_range[1])
        filtered = filtered[
            filtered["date"].between(left, right)
        ]
    if status_choice == "Anomalous only":
        filtered = filtered[filtered["anomaly_label"] == -1]
    elif status_choice == "Normal only":
        filtered = filtered[filtered["anomaly_label"] == 1]

    st.write(f"Matching city-days: **{len(filtered):,}**")
    anomaly_columns = [
        "date", "status", "anomaly_score", "PM2_5_mean", "PM2_5_max",
        "PM10_mean", "SO2_mean", "NO2_mean", "CO_mean", "O3_mean",
        "TEMP_mean", "PRES_mean", "DEWP_mean", "RAIN_sum", "WSPM_mean"
    ]
    result_table = filtered[anomaly_columns].sort_values("date", ascending=False).copy()
    result_table["date"] = result_table["date"].dt.strftime("%Y-%m-%d")
    html_table(result_table.round(3), max_rows=40)

    csv_bytes = filtered.drop(columns=["month_name", "season", "status"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered historical results (CSV)",
        data=csv_bytes,
        file_name="filtered_historical_anomaly_results.csv",
        mime="text/csv",
        key="download_filtered_history"
    )

# -------------------- TAB 3: Temporal trends --------------------

with temporal_tab:
    st.markdown('<div class="section-title">Temporal and Seasonal Patterns</div>', unsafe_allow_html=True)

    yearly = anomaly_df.groupby("year").agg(
        Total_Days=("date", "size"),
        Anomalous_Days=("anomaly_label", lambda x: int((x == -1).sum()))
    )
    yearly["Anomaly_Rate_Percent"] = yearly["Anomalous_Days"] / yearly["Total_Days"] * 100

    monthly = anomaly_df.groupby("month").agg(
        Total_Days=("date", "size"),
        Anomalous_Days=("anomaly_label", lambda x: int((x == -1).sum()))
    ).reindex(range(1, 13), fill_value=0)
    monthly["Anomaly_Rate_Percent"] = np.where(
        monthly["Total_Days"] > 0,
        monthly["Anomalous_Days"] / monthly["Total_Days"] * 100,
        0
    )
    monthly.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    seasonal_order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = anomaly_df.groupby("season").agg(
        Total_Days=("date", "size"),
        Anomalous_Days=("anomaly_label", lambda x: int((x == -1).sum()))
    ).reindex(seasonal_order)
    seasonal["Anomaly_Rate_Percent"] = seasonal["Anomalous_Days"] / seasonal["Total_Days"] * 100

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(yearly.index.astype(str), yearly["Anomalous_Days"])
        ax.set_title("Anomalous Days by Year", color="#E2E8F0")
        ax.set_ylabel("Anomalous days", color="#DCE3EC")
        ax.tick_params(colors="#DCE3EC")
        for spine in ax.spines.values():
            spine.set_color("#48515E")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(monthly.index, monthly["Anomaly_Rate_Percent"])
        ax.set_title("Anomaly Rate by Month", color="#E2E8F0")
        ax.set_ylabel("Anomaly rate (%)", color="#DCE3EC")
        ax.tick_params(colors="#DCE3EC", rotation=35)
        for spine in ax.spines.values():
            spine.set_color("#48515E")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.bar(seasonal.index, seasonal["Anomaly_Rate_Percent"])
    ax.set_title("Anomaly Rate by Season", color="#E2E8F0")
    ax.set_ylabel("Anomaly rate (%)", color="#DCE3EC")
    ax.tick_params(colors="#DCE3EC")
    for spine in ax.spines.values():
        spine.set_color("#48515E")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="section-title">Yearly Summary</div>', unsafe_allow_html=True)
    year_view = yearly.reset_index().rename(columns={"index": "Year"})
    html_table(year_view.round(2), max_rows=20)
    st.markdown('<div class="section-title">Monthly Summary</div>', unsafe_allow_html=True)
    month_view = monthly.reset_index().rename(columns={"index": "Month"})
    html_table(month_view.round(2), max_rows=12)
    st.markdown('<div class="section-title">Seasonal Summary</div>', unsafe_allow_html=True)
    season_view = seasonal.reset_index().rename(columns={"index": "Season"})
    html_table(season_view.round(2), max_rows=4)

# -------------------- TAB 4: Station analysis --------------------

with station_tab:
    st.markdown('<div class="section-title">Monitoring Station Comparison</div>', unsafe_allow_html=True)
    anomaly_dates = set(anomaly_df.loc[anomaly_df["anomaly_label"] == -1, "date"])
    station_daily = station_daily_df.copy()
    station_daily["city_status"] = np.where(
        station_daily["date"].isin(anomaly_dates), "Anomalous city-day", "Normal city-day"
    )

    normal_pm = (
        station_daily[station_daily["city_status"] == "Normal city-day"]
        .groupby("station")["PM2.5_mean"].mean()
    )
    anomaly_pm = (
        station_daily[station_daily["city_status"] == "Anomalous city-day"]
        .groupby("station")["PM2.5_mean"].mean()
    )
    station_compare = pd.DataFrame({
        "Station": normal_pm.index,
        "Normal-day PM2.5 mean": normal_pm.values,
        "Anomalous-day PM2.5 mean": anomaly_pm.reindex(normal_pm.index).values
    })
    station_compare["Difference"] = (
        station_compare["Anomalous-day PM2.5 mean"]
        - station_compare["Normal-day PM2.5 mean"]
    )
    station_compare["Increase (%)"] = np.where(
        station_compare["Normal-day PM2.5 mean"] != 0,
        station_compare["Difference"] / station_compare["Normal-day PM2.5 mean"] * 100,
        np.nan
    )
    station_compare = station_compare.sort_values("Increase (%)", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Stations analyzed", f"{station_daily['station'].nunique()}")
    c2.metric("Stations with higher PM2.5 on anomaly days",
              f"{int((station_compare['Difference'] > 0).sum())} / {len(station_compare)}")
    c3.metric("Highest relative increase",
              f"{station_compare['Increase (%)'].max():.1f}%" if station_compare["Increase (%)"].notna().any() else "N/A")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    plot_station = station_compare.sort_values("Anomalous-day PM2.5 mean")
    y = np.arange(len(plot_station))
    height = 0.38
    ax.barh(y - height/2, plot_station["Normal-day PM2.5 mean"], height, label="Normal city-days")
    ax.barh(y + height/2, plot_station["Anomalous-day PM2.5 mean"], height, label="Anomalous city-days")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_station["Station"])
    ax.set_xlabel("Average PM2.5 (µg/m³)", color="#DCE3EC")
    ax.set_title("Station-Level PM2.5: Normal vs Anomalous City-Days", color="#E2E8F0")
    ax.tick_params(colors="#DCE3EC")
    ax.legend()
    for spine in ax.spines.values():
        spine.set_color("#48515E")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    station_view = station_compare.round(2)
    html_table(station_view, max_rows=20)
    st.download_button(
        "Download station comparison (CSV)",
        data=station_compare.to_csv(index=False).encode("utf-8"),
        file_name="station_pm25_comparison.csv",
        mime="text/csv",
        key="download_station_comparison"
    )

# -------------------- TAB 5: Feature insights --------------------

with feature_tab:
    st.markdown('<div class="section-title">Which measurements differ on anomalous days?</div>', unsafe_allow_html=True)
    st.caption(
        "This is a descriptive comparison of feature averages, not a causal explanation "
        "or a model-specific feature attribution."
    )

    normal_data = anomaly_df[anomaly_df["anomaly_label"] == 1][FEATURES]
    anomaly_data = anomaly_df[anomaly_df["anomaly_label"] == -1][FEATURES]

    normal_means = normal_data.mean()
    anomaly_means = anomaly_data.mean()
    normal_std = normal_data.std().replace(0, np.nan)

    feature_comparison = pd.DataFrame({
        "Feature": FEATURES,
        "Normal-day mean": [normal_means[f] for f in FEATURES],
        "Anomalous-day mean": [anomaly_means[f] for f in FEATURES],
        "Difference": [anomaly_means[f] - normal_means[f] for f in FEATURES],
        "Standardized difference": [
            (anomaly_means[f] - normal_means[f]) / normal_std[f]
            if pd.notna(normal_std[f]) else np.nan
            for f in FEATURES
        ]
    })
    feature_comparison["Absolute standardized difference"] = feature_comparison[
        "Standardized difference"
    ].abs()
    feature_comparison = feature_comparison.sort_values(
        "Absolute standardized difference", ascending=False
    )

    top_features = feature_comparison.head(10).sort_values("Standardized difference")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.barh(top_features["Feature"], top_features["Standardized difference"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Difference in normal-day standard deviations", color="#DCE3EC")
    ax.set_title("Largest Feature-Level Differences", color="#E2E8F0")
    ax.tick_params(colors="#DCE3EC")
    for spine in ax.spines.values():
        spine.set_color("#48515E")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    html_table(
        feature_comparison[
            ["Feature", "Normal-day mean", "Anomalous-day mean", "Difference", "Standardized difference"]
        ].round(3),
        max_rows=23
    )

    st.download_button(
        "Download feature comparison (CSV)",
        data=feature_comparison.to_csv(index=False).encode("utf-8"),
        file_name="feature_level_anomaly_comparison.csv",
        mime="text/csv",
        key="download_feature_comparison"
    )

# -------------------- TAB 6: Manual prediction --------------------

with manual_tab:
    st.markdown('<div class="section-title">Predict a Manually Entered Daily Profile</div>', unsafe_allow_html=True)
    st.caption(
        "This predicts whether one manually entered 23-feature profile is unusual relative "
        "to the model's training data. It is separate from the historical dataset analysis."
    )

    with st.form("manual_prediction_form"):
        input_values = {}
        for section_title, start, end in [
            ("Particulate Matter", 0, 6),
            ("Gaseous Pollutants", 6, 18),
            ("Weather Parameters", 18, 23),
        ]:
            st.markdown(f'<div class="section-title">{section_title}</div>', unsafe_allow_html=True)
            fields = MANUAL_FIELDS[start:end]
            for row_start in range(0, len(fields), 3):
                row_fields = fields[row_start:row_start + 3]
                columns = st.columns(len(row_fields))
                for column, item in zip(columns, row_fields):
                    key, label, unit, reference, default, minimum, maximum, step = item
                    with column:
                        st.markdown(
                            f'<div class="field-label"><span class="field-name">{label} '
                            f'<span class="field-unit">({unit})</span></span>'
                            f'<span class="field-reference">{reference}</span></div>',
                            unsafe_allow_html=True
                        )
                        input_values[key] = st.number_input(
                            label,
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(default),
                            step=float(step),
                            key=f"manual_{key}",
                            label_visibility="collapsed"
                        )
        submitted = st.form_submit_button("Predict Environment", use_container_width=True)

    if submitted:
        errors = []
        for label, mean_key, max_key, std_key, unit, *_ in POLLUTANT_GROUPS:
            if input_values[max_key] < input_values[mean_key]:
                errors.append(f"{label} Maximum cannot be lower than its Mean.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            manual_df = pd.DataFrame(
                [[input_values[name] for name in FEATURES]],
                columns=FEATURES
            )
            try:
                scaled_manual = scaler.transform(manual_df)
                pred = int(isolation_forest.predict(scaled_manual)[0])
                score = float(isolation_forest.decision_function(scaled_manual)[0])
                if pred == -1:
                    st.markdown(
                        '<div class="result-anomaly"><h3>⚠️ ANOMALOUS ENVIRONMENT</h3>'
                        '<p>The model classified this input profile as unusual.</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-normal"><h3>✅ NORMAL ENVIRONMENT</h3>'
                        '<p>The model did not classify this input profile as anomalous.</p></div>',
                        unsafe_allow_html=True
                    )
                c1, c2 = st.columns(2)
                c1.metric("Model decision score", f"{score:.4f}")
                c2.metric("Features evaluated", "23")
                st.caption(
                    "Lower decision scores indicate more unusual observations relative to "
                    "the learned model boundary. The score is not a probability or a pollution "
                    "severity measure."
                )
            except Exception as exc:
                st.error("Prediction failed. Verify the saved scaler and model files.")
                st.code(str(exc))

    st.markdown(
        '<p class="muted">WHO guideline values are reference benchmarks with specific averaging periods, '
        'not universal min/max limits. The ozone guideline is an 8-hour average and is not directly '
        'equivalent to this daily mean feature.</p>',
        unsafe_allow_html=True
    )

# -------------------- TAB 7: Download reports --------------------

with download_tab:
    st.markdown('<div class="section-title">Download Project Results</div>', unsafe_allow_html=True)
    st.write("Download the processed historical data and key analytical outputs for reports or further analysis.")

    all_results = anomaly_df.drop(
        columns=["month_name", "season", "status"], errors="ignore"
    )
    st.download_button(
        "Download all historical anomaly results",
        data=all_results.to_csv(index=False).encode("utf-8"),
        file_name="anomaly_results_full.csv",
        mime="text/csv",
        key="download_all_anomalies"
    )

    anomalous_only = anomaly_df[anomaly_df["anomaly_label"] == -1].drop(
        columns=["month_name", "season", "status"], errors="ignore"
    )
    st.download_button(
        "Download anomalous days only",
        data=anomalous_only.to_csv(index=False).encode("utf-8"),
        file_name="anomalous_days_only.csv",
        mime="text/csv",
        key="download_anomalous_only"
    )

    st.download_button(
        "Download city-level daily dataset",
        data=anomaly_df[["date"] + FEATURES].to_csv(index=False).encode("utf-8"),
        file_name="city_daily_dataset.csv",
        mime="text/csv",
        key="download_city_daily"
    )

    st.download_button(
        "Download station-level daily dataset",
        data=station_daily_df.to_csv(index=False).encode("utf-8"),
        file_name="station_daily_dataset.csv",
        mime="text/csv",
        key="download_station_daily"
    )

    st.markdown("---")
    st.markdown("**Model interpretation notes**")
    st.markdown(
        "- Isolation Forest label `-1` means anomalous; `1` means normal.\n"
        "- The anomaly score is a decision-function score, not a probability.\n"
        "- K-Means clusters are supplementary and are only shown when the saved PCA and K-Means models are compatible.\n"
        "- Station-level pollution differences are descriptive associations, not proof of causation.\n"
        "- WHO reference guidelines are not used by the Isolation Forest to assign its label."
    )

st.markdown("---")
st.markdown(
    '<p class="muted">Urban Environmental Anomaly Detection · Beijing Multi-Site Air Quality Dataset · '
    'Isolation Forest · K-Means · PCA · Streamlit</p>',
    unsafe_allow_html=True
)
