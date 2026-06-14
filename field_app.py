"""
app.py
Veil Finder - Streamlit observation logging and location scoring app.

Run:
    streamlit run app.py
    (or: python -m streamlit run app.py)
"""

import os
import sys
import json
import uuid
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, time, datetime

# ── Backend path ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))
import importlib
import scorer
importlib.reload(scorer)
from scorer  import (
    score_all_observations, get_point_diagnostics, scores_to_geojson,
    compute_plant_scores, detect_high_contrast_days, score_contrast_observations,
)
import inaturalist_client
importlib.reload(inaturalist_client)
import fetcher
importlib.reload(fetcher)
import base64
from fetcher import smart_fetch, get_missing_data_points, load_master_csv
from grid_utils import generate_grid
import geometry_utils
importlib.reload(geometry_utils)
from geometry_utils import get_rotated_corners
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# ── Constants ─────────────────────────────────────────────────────────────────
HUNTS_DIR = os.path.join(_HERE, "hunts")

# Helpers to get current hunt paths
def get_hunt_dir() -> str:
    hunt = st.session_state.get("current_hunt", "veil_eight")
    return os.path.join(HUNTS_DIR, hunt)

def get_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "observations.json")

def get_cloud_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "cloud_observations.json")

def get_contrast_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "contrast_observations.json")

def get_plants_file() -> str:
    return os.path.join(get_hunt_dir(), "plants.json")

def get_export_dir() -> str:
    return os.path.join(get_hunt_dir(), "scans_export")

def get_master_parquet() -> str:
    return os.path.join(get_export_dir(), "fog_master.parquet")

def get_settings_file() -> str:
    return os.path.join(get_hunt_dir(), "settings.json")

CENTER     = {"lat": 40.31, "lon": -75.13}

# Preferred master Parquet — created/updated by the in-app fetcher
_MASTER_PARQUET = get_master_parquet()

SCORE_COLORSCALE = [
    [0.00, "rgba(60,60,70,0.35)"],
    [0.40, "rgba(60,60,70,0.35)"],
    [0.55, "#e8c830"],
    [0.72, "#ff7b00"],
    [0.88, "#ff2800"],
    [1.00, "#ff0055"],
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veil Finder",
    page_icon="fog",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.4rem;
        margin-top: 1rem;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    .stAlert { border-radius: 8px; }
    .obs-badge-fog   { background:#0d2b3e; color:#00e5ff; border-radius:4px; padding:2px 8px; font-size:0.82rem; }
    .obs-badge-clear { background:#1a2a1a; color:#4ade80; border-radius:4px; padding:2px 8px; font-size:0.82rem; }
    hr { border-color: #30363d; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Observation I/O ───────────────────────────────────────────────────────────
def load_observations() -> list[dict]:
    fpath = get_obs_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_observations(obs_list: list[dict]) -> None:
    fpath = get_obs_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obs_list, f, indent=2)


def load_cloud_observations() -> list[dict]:
    fpath = get_cloud_obs_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_cloud_observations(obs_list: list[dict]) -> None:
    fpath = get_cloud_obs_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obs_list, f, indent=2)


def load_contrast_observations() -> list[dict]:
    fpath = get_contrast_obs_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_contrast_observations(obs_list: list[dict]) -> None:
    fpath = get_contrast_obs_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obs_list, f, indent=2)


def load_plants() -> dict:
    fpath = get_plants_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_plants(selected: list, obs_dict: dict) -> None:
    fpath = get_plants_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"selected_plants": selected, "plant_obs_dict": obs_dict}, f, indent=2)


def load_settings() -> dict:
    fpath = get_settings_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(settings: dict) -> None:
    fpath = get_settings_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def invalidate_scores() -> None:
    """Clear cached scores so they are recomputed on next access."""
    st.session_state.scores = None


# ── Session state init ────────────────────────────────────────────────────────
if "current_hunt" not in st.session_state:
    st.session_state.current_hunt = "veil_eight"
if "observations" not in st.session_state:
    st.session_state.observations = load_observations()
if "scores" not in st.session_state:
    st.session_state.scores = None
if "fog_threshold" not in st.session_state:
    st.session_state.fog_threshold = 5
if "selected_csv" not in st.session_state:
    st.session_state.selected_csv = None
_saved_settings = load_settings()

if "grid_center_lat" not in st.session_state:
    st.session_state.grid_center_lat = _saved_settings.get("grid_center_lat", 40.31)
if "grid_center_lon" not in st.session_state:
    st.session_state.grid_center_lon = _saved_settings.get("grid_center_lon", -75.13)
if "grid_radius_mi" not in st.session_state:
    st.session_state.grid_radius_mi = _saved_settings.get("grid_radius_mi", 40.0)
if "map_transparency" not in st.session_state:
    st.session_state.map_transparency = 20
_plants_data = load_plants()
if "selected_plants" not in st.session_state:
    st.session_state.selected_plants = _plants_data.get("selected_plants", [])
if "plant_obs_dict" not in st.session_state:
    st.session_state.plant_obs_dict = _plants_data.get("plant_obs_dict", {})
if "plant_influence_radius" not in st.session_state:
    st.session_state.plant_influence_radius = 3.0
if "plant_match_mode" not in st.session_state:
    st.session_state.plant_match_mode = "Any"
if "show_plant_pins" not in st.session_state:
    st.session_state.show_plant_pins = True
if "map_color_by" not in st.session_state:
    st.session_state.map_color_by = "Combined Score"
if "cloud_obs_list" not in st.session_state:
    st.session_state.cloud_obs_list = load_cloud_observations()
if "contrast_obs_list" not in st.session_state:
    st.session_state.contrast_obs_list = load_contrast_observations()
# Combined Score component weights
if "weight_fog" not in st.session_state:
    st.session_state.weight_fog = 10
if "weight_plant" not in st.session_state:
    st.session_state.weight_plant = 10
if "weight_cloud" not in st.session_state:
    st.session_state.weight_cloud = 10
if "weight_contrast" not in st.session_state:
    st.session_state.weight_contrast = 10


# ── Combined Score recompute helper ───────────────────────────────────────────
def recompute_combined(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild CombinedScore from individual score columns using the current
    session state toggles (combine_plant, combine_cloud, combine_contrast).
    Always uses MatchRate as the base. Individual score columns are unchanged.
    """
    if scores_df is None or scores_df.empty:
        return scores_df

    df = scores_df.copy()
    # Compute min-max normalized MatchRate for the base
    mr = df.get("MatchRate", pd.Series(1.0, index=df.index)).fillna(0.0)
    if mr.max() > mr.min():
        mr_scaled = (mr - mr.min()) / (mr.max() - mr.min())
    else:
        mr_scaled = mr.copy()

    total_weight = float(st.session_state.weight_fog)
    combined_raw = mr_scaled * st.session_state.weight_fog

    if "PlantScore" in df.columns:
        combined_raw += df["PlantScore"].fillna(0.0) * st.session_state.weight_plant
        total_weight += st.session_state.weight_plant
    if "CloudScore" in df.columns:
        combined_raw += df["CloudScore"].fillna(0.0) * st.session_state.weight_cloud
        total_weight += st.session_state.weight_cloud
    if "ContrastScore" in df.columns:
        combined_raw += df["ContrastScore"].fillna(0.0) * st.session_state.weight_contrast
        total_weight += st.session_state.weight_contrast

    if total_weight > 0:
        combined = combined_raw / total_weight
    else:
        combined = combined_raw * 0.0

    df["CombinedScore"] = combined
    return df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)


# ── Fog data loader (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_fog_data(file_path: str, mtime: float) -> pd.DataFrame:
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, parse_dates=["Timestamp"])
    
    if "Timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def get_fog_df() -> pd.DataFrame | None:
    if st.session_state.selected_csv and os.path.exists(st.session_state.selected_csv):
        mtime = os.path.getmtime(st.session_state.selected_csv)
        return load_fog_data(st.session_state.selected_csv, mtime)
    return None


# ── Score computation (lazy, cached in session_state) ─────────────────────────
def get_scores() -> pd.DataFrame | None:
    fog_df = get_fog_df()
    if st.session_state.scores is None:
        if st.session_state.observations:
            with st.spinner("Computing location scores for the current grid..."):
                current_grid = list(set(generate_grid(
                    center_lat=st.session_state.grid_center_lat,
                    center_lon=st.session_state.grid_center_lon,
                    radius_km=st.session_state.grid_radius_mi * 1.60934
                )))
                
                # Create the full grid DataFrame
                grid_df = pd.DataFrame(current_grid, columns=["Lat_g", "Lon_g"])
                grid_df["Lat"] = grid_df["Lat_g"]
                grid_df["Lon"] = grid_df["Lon_g"]
                
                base_scores = pd.DataFrame()
                
                if fog_df is not None and not fog_df.empty:
                    fog_df_rounded = fog_df.copy()
                    fog_df_rounded["Lat_g"] = fog_df_rounded["Lat"].round(4)
                    fog_df_rounded["Lon_g"] = fog_df_rounded["Lon"].round(4)
                    
                    # Score only points with data
                    filtered_df = pd.merge(fog_df_rounded, grid_df, on=["Lat_g", "Lon_g"], how="inner")
                    # Clean up the merge artifacts so score_all_observations sees exact Lat/Lon
                    filtered_df = filtered_df.drop(columns=["Lat_g", "Lon_g"])
                    # Some merges create Lat_x, Lon_x if there's overlap. Let's explicitly ensure Lat, Lon are from fog_df
                    if "Lat_x" in filtered_df.columns:
                        filtered_df = filtered_df.rename(columns={"Lat_x": "Lat", "Lon_x": "Lon"})
                    
                    raw_fog_scores = score_all_observations(
                        st.session_state.observations,
                        filtered_df,
                        fog_threshold=st.session_state.fog_threshold,
                    )
                    
                    if not raw_fog_scores.empty:
                        raw_fog_scores["Lat_g"] = raw_fog_scores["Lat"].round(4)
                        raw_fog_scores["Lon_g"] = raw_fog_scores["Lon"].round(4)
                        raw_fog_scores = raw_fog_scores.drop(columns=["Lat", "Lon"])
                        
                        base_scores = pd.merge(grid_df, raw_fog_scores, on=["Lat_g", "Lon_g"], how="left")
                        base_scores = base_scores.drop(columns=["Lat_g", "Lon_g"])
                    else:
                        base_scores = grid_df.drop(columns=["Lat_g", "Lon_g"])
                else:
                    base_scores = grid_df.drop(columns=["Lat_g", "Lon_g"])
                
                # Ensure missing columns are populated
                if "MatchRate" not in base_scores.columns:
                    base_scores["MatchRate"] = np.nan
                    base_scores["Confidence_Z"] = np.nan
                    base_scores["Matches"] = 0
                    base_scores["ObsCount"] = len(st.session_state.observations)
                    
                if "Elevation_m" not in base_scores.columns:
                    base_scores["Elevation_m"] = np.nan
                    base_scores["IsValley"] = False
                
                st.session_state.scores = compute_plant_scores(
                    base_scores,
                    st.session_state.plant_obs_dict,
                    influence_radius_mi=st.session_state.plant_influence_radius,
                    match_mode=st.session_state.plant_match_mode
                )
                
                # Score Cloud Differential
                st.session_state.scores = scorer.score_cloud_observations(
                    st.session_state.scores,
                    st.session_state.cloud_obs_list,
                    fog_df
                )

                # Score High-Contrast Cloud Days
                st.session_state.scores = score_contrast_observations(
                    st.session_state.scores,
                    st.session_state.contrast_obs_list,
                    fog_df
                )
    return st.session_state.scores


def get_hunts() -> list[str]:
    os.makedirs(HUNTS_DIR, exist_ok=True)
    return [d for d in os.listdir(HUNTS_DIR) if os.path.isdir(os.path.join(HUNTS_DIR, d)) and not d.startswith(".")]

def switch_hunt(new_hunt: str):
    st.session_state.current_hunt = new_hunt
    st.session_state.observations = load_observations()
    st.session_state.cloud_obs_list = load_cloud_observations()
    st.session_state.contrast_obs_list = load_contrast_observations()
    _new_settings = load_settings()
    st.session_state.grid_center_lat = _new_settings.get("grid_center_lat", 40.31)
    st.session_state.grid_center_lon = _new_settings.get("grid_center_lon", -75.13)
    st.session_state.grid_radius_mi = _new_settings.get("grid_radius_mi", 40.0)
    st.session_state.selected_csv = get_master_parquet() if os.path.exists(get_master_parquet()) else None
    invalidate_scores()

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Veil Finder Mobile")
    st.caption("Read-only field viewer")
    # ── Hunt Selector ──
    st.markdown('<p class="section-header">Active Hunt / Project</p>', unsafe_allow_html=True)
    hunts = get_hunts()
    if st.session_state.current_hunt not in hunts:
        hunts.append(st.session_state.current_hunt)
        
    selected_hunt = st.selectbox(
        "Hunt", hunts, index=hunts.index(st.session_state.current_hunt), label_visibility="collapsed"
    )
    if selected_hunt != st.session_state.current_hunt:
        switch_hunt(selected_hunt)
        st.rerun()
        
    st.markdown("---")
    with st.expander("⚖️ Combined Score Weights", expanded=False):
        st.caption(
            "Adjust how much influence each layer has on the **Combined Score**. "
            "Set a weight to 0 to completely exclude that layer."
        )

        _changed = False
        
        # Fog Match Rate is always available
        _new_fog = st.slider("🌫️ Fog Match Rate", 0, 10, st.session_state.weight_fog, key="sl_weight_fog")
        if _new_fog != st.session_state.weight_fog:
            st.session_state.weight_fog = _new_fog
            _changed = True

        _has_plant    = bool(st.session_state.selected_plants)
        _has_cloud    = bool(st.session_state.cloud_obs_list)
        _has_contrast = bool(st.session_state.contrast_obs_list)

        if _has_plant:
            _new_plant = st.slider("🌿 Plant Proximity Score", 0, 10, st.session_state.weight_plant, key="sl_weight_plant")
            if _new_plant != st.session_state.weight_plant:
                st.session_state.weight_plant = _new_plant
                _changed = True
        if _has_cloud:
            _new_cloud = st.slider("⛅ Cloud Differential Score", 0, 10, st.session_state.weight_cloud, key="sl_weight_cloud")
            if _new_cloud != st.session_state.weight_cloud:
                st.session_state.weight_cloud = _new_cloud
                _changed = True
        if _has_contrast:
            _new_contrast = st.slider("⚡ High-Contrast Cloud Score", 0, 10, st.session_state.weight_contrast, key="sl_weight_contrast")
            if _new_contrast != st.session_state.weight_contrast:
                st.session_state.weight_contrast = _new_contrast
                _changed = True
                
        if _changed:
            st.rerun()

    with st.expander("🖼️ Map Image Overlays", expanded=False):
        if "map_overlays" not in st.session_state:
            st.session_state.map_overlays = _saved_settings.get("map_overlays", [])
        for idx, overlay in enumerate(st.session_state.map_overlays):
            st.markdown(f"**{overlay['name']}**")
            vis = st.checkbox("Visible", value=overlay.get("visible", True), key=f"ov_vis_{idx}")
            if vis != overlay.get("visible", True):
                overlay["visible"] = vis
                st.session_state.settings_cache = load_settings()
                st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                save_settings(st.session_state.settings_cache)
                st.rerun()
            if vis:
                n_op = st.slider("Opacity", 0.0, 1.0, value=float(overlay["opacity"]), step=0.05, key=f"ov_op_{idx}")
                if n_op != overlay["opacity"]:
                    overlay["opacity"] = n_op
                    st.session_state.settings_cache = load_settings()
                    st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                    save_settings(st.session_state.settings_cache)
                    st.rerun()
            st.markdown("---")
    with st.expander("🗺️ Map Settings", expanded=False):
        new_trans = st.slider("Point Transparency (%)", min_value=0, max_value=95, value=int(st.session_state.map_transparency), step=5, help="Make grid points semi-transparent to view roads and landmarks under them.")
        if new_trans != st.session_state.map_transparency:
            st.session_state.map_transparency = new_trans
            st.rerun()

# ─── Map View ──────────────────────────────────────────────────────────────
st.header("Location Scoring Map")

# Auto-select master parquet if not set
if st.session_state.selected_csv is None:
    if os.path.exists(get_master_parquet()):
        st.session_state.selected_csv = get_master_parquet()

fog_df = get_fog_df()
if fog_df is None:
    st.warning("No satellite data found for this hunt.")
elif not st.session_state.observations:
    st.info("No observations loaded for this hunt.")
else:
    scores = get_scores()
    if scores is not None and not scores.empty:
        scores = recompute_combined(scores)

        # Summary metrics
        top1 = scores.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        if st.session_state.selected_plants:
            m1.metric("Top Suitability Score", f"{top1['CombinedScore']:.2f}")
            m2.metric("Top Plant Score", f"{top1['PlantScore']:.2f}")
        else:
            m1.metric("Top Candidate Match Rate",  f"{top1['MatchRate']:.1%}")
            m2.metric("Top Candidate Confidence",   f"{top1['Confidence_Z']:+.2f}sigma")
        m3.metric("Grid Points Scored",         f"{len(scores):,}")
        m4.metric("Observations Used",          f"{len(st.session_state.observations)}")

        # Map
        color_col = "CombinedScore"
        color_title = "Combined Suitability"
        color_scale = "Plotly3"
        
        _max_score = float(scores["CombinedScore"].max()) if not scores["CombinedScore"].empty and not pd.isna(scores["CombinedScore"].max()) else 1.0
        range_color = [0.0, _max_score if _max_score > 0 else 1.0]

        map_df = scores.copy()
        # Clip very small values so zero-match points still render (tiny dot)
        map_df["_size"] = (map_df[color_col].fillna(0) * 15).clip(lower=1.5)
        map_df["MatchRate_pct"] = (map_df["MatchRate"] * 100).round(1)

        # Dynamic map center and zoom level based on user's active grid settings
        map_center = {
            "lat": st.session_state.grid_center_lat,
            "lon": st.session_state.grid_center_lon
        }
        # Clamped auto-zoom calculation: 13.5 - log2(radius_mi)
        auto_zoom = float(np.clip(13.5 - np.log2(st.session_state.grid_radius_mi), 3.0, 15.0))

        custom_data_cols = ["Lat", "Lon", "Elevation_m", "IsValley",
                            "MatchRate_pct", "Confidence_Z", "Matches", "ObsCount"]
        if "PlantScore" in map_df.columns:
            custom_data_cols += ["PlantScore"]       # index 8
        else:
            map_df["PlantScore"] = 1.0
            custom_data_cols += ["PlantScore"]

        if "CloudScore" in map_df.columns:
            custom_data_cols += ["CloudScore"]       # index 9
        else:
            map_df["CloudScore"] = 1.0
            custom_data_cols += ["CloudScore"]

        if "ContrastScore" in map_df.columns:
            custom_data_cols += ["ContrastScore"]    # index 10
        else:
            map_df["ContrastScore"] = 1.0
            custom_data_cols += ["ContrastScore"]

        if "CombinedScore" in map_df.columns:
            custom_data_cols += ["CombinedScore"]    # index 11
        else:
            map_df["CombinedScore"] = map_df["MatchRate"]
            custom_data_cols += ["CombinedScore"]

        fig = px.scatter_mapbox(
            map_df,
            lat="Lat",
            lon="Lon",
            color=color_col,
            size="_size",
            size_max=18,
            color_continuous_scale=color_scale,
            range_color=range_color,
            opacity=(100.0 - st.session_state.map_transparency) / 100.0,
            custom_data=custom_data_cols,
            mapbox_style="carto-darkmatter",
            zoom=auto_zoom,
            center=map_center,
            height=570,
        )

        # Generate hover template
        hover_template_str = (
            "<b>Match Rate: %{customdata[4]:.1f}%</b><br>"
            "Confidence: %{customdata[5]:+.2f}sigma<br>"
            "Matches: %{customdata[6]} / %{customdata[7]} obs<br>"
            "Lat: %{customdata[0]:.4f}  Lon: %{customdata[1]:.4f}<br>"
            "Elevation: %{customdata[2]:.0f} m  Valley: %{customdata[3]}"
        )
        hover_template_str += (
            "<br>Plant Proximity Score: %{customdata[8]:.2f}"
            "<br>Cloud Differential: %{customdata[9]:.2f}"
            "<br>High-Contrast Cloud Score: %{customdata[10]:.2f}"
            "<br>Combined Suitability: %{customdata[11]:.2f}"
        )
        hover_template_str += "<extra></extra>"

        fig.update_traces(
            hovertemplate=hover_template_str
        )

        # Map Overlays
        mapbox_layers = []
        if "map_overlays" in st.session_state:
            for overlay in st.session_state.map_overlays:
                _ov_dir = os.path.join(get_hunt_dir(), "overlays")
                _resolved_path = os.path.join(_ov_dir, overlay["name"])
                
                if overlay.get("visible", True) and os.path.exists(_resolved_path):
                    with open(_resolved_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    
                    ext = os.path.splitext(_resolved_path)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    b64_url = f"data:{mime};base64,{b64}"
                    
                    corners = get_rotated_corners(
                        center_lat=float(overlay["lat"]),
                        center_lon=float(overlay["lon"]),
                        width_km=float(overlay["width"]),
                        height_km=float(overlay["height"]),
                        rotation_deg=float(overlay["rotation"]),
                        anchor_x=float(overlay.get("anchor_x", 0.5)),
                        anchor_y=float(overlay.get("anchor_y", 0.5))
                    )
                    
                    mapbox_layers.append({
                        "sourcetype": "image",
                        "source": b64_url,
                        "coordinates": corners,
                        "opacity": float(overlay["opacity"]),
                        "below": "traces"
                    })

        if mapbox_layers:
            fig.update_layout(mapbox_layers=mapbox_layers)

        # Extract coordinates clicked/selected on the map
        clicked_lat = None
        clicked_lon = None
        clicked_coords = None
        if "scoring_map" in st.session_state and st.session_state.scoring_map:
            points = st.session_state.scoring_map.get("selection", {}).get("points", [])
            if points:
                pt = points[0]
                lat = pt.get("lat") or pt.get("y")
                lon = pt.get("lon") or pt.get("x")
                if lat is not None and lon is not None:
                    clicked_lat = float(lat)
                    clicked_lon = float(lon)
                    clicked_coords = f"{clicked_lat:.6f}, {clicked_lon:.6f}"

        # iNaturalist pins trace: ONLY show if a grid point is clicked
        if st.session_state.plant_obs_dict and clicked_lat is not None:
            colors = px.colors.qualitative.Plotly
            species_list = list(st.session_state.plant_obs_dict.keys())
            
            for idx, species_name in enumerate(species_list):
                obs_list = st.session_state.plant_obs_dict[species_name]
                species_color = colors[idx % len(colors)]
                
                filtered_obs = []
                for o in obs_list:
                    # Only include pins within influence radius of the CLICKED point
                    dist = scorer.haversine_distance(clicked_lat, clicked_lon, o["lat"], o["lon"])
                    if dist <= st.session_state.plant_influence_radius:
                        filtered_obs.append({
                            "Lat": o["lat"],
                            "Lon": o["lon"],
                            "Species": species_name,
                            "User": o["user"],
                            "Observed On": o["observed_on"]
                        })
                
                if filtered_obs:
                    obs_df = pd.DataFrame(filtered_obs)
                    fig.add_trace(go.Scattermapbox(
                        lat=obs_df["Lat"],
                        lon=obs_df["Lon"],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=species_color,
                            symbol="circle"
                        ),
                        customdata=np.stack([
                            obs_df["Species"],
                            obs_df["User"],
                            obs_df["Observed On"],
                            obs_df["Lat"],
                            obs_df["Lon"]
                        ], axis=-1),
                        hovertemplate=(
                            "<b>🌿 iNaturalist Observation</b><br>"
                            "Species: %{customdata[0]}<br>"
                            "Observer: @%{customdata[1]}<br>"
                            "Observed: %{customdata[2]}<br>"
                            "Location: %{customdata[3]:.4f}, %{customdata[4]:.4f}"
                            "<extra></extra>"
                        ),
                        name=f"{species_name} (in radius)"
                    ))

        # Custom center marker
        fig.add_trace(go.Scattermapbox(
            lat=[st.session_state.grid_center_lat],
            lon=[st.session_state.grid_center_lon],
            mode="markers+text",
            marker=dict(size=14, color="white", symbol="star"),
            text=["Grid Center"],
            textposition="top right",
            textfont=dict(color="white", size=11),
            hoverinfo="text",
            hovertext=f"Grid Center ({st.session_state.grid_center_lat}, {st.session_state.grid_center_lon})",
            name="Center",
            showlegend=False,
        ))

        # Preserve manual zoom/pan state unless grid parameters change
        grid_rev_key = f"{st.session_state.grid_center_lat}_{st.session_state.grid_center_lon}_{st.session_state.grid_radius_mi}"

        fig.update_layout(
            uirevision=grid_rev_key,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title=dict(
                    text="Match Rate",
                    font=dict(color="#e6edf3"),
                ),
                tickformat=".0%",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                len=0.55,
                bgcolor="#161b22",
                bordercolor="#30363d",
                borderwidth=1,
                tickfont=dict(color="#e6edf3"),
            ),
        )

        # Enable interactive click and selection directly on the map
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True},
            on_select="rerun",
            key="scoring_map"
        )
        st.caption("💡 **Tip**: Click any point on the map above to load its coordinates instantly into the **Coordinate Copier** below!")

        col_tbl, col_cpy = st.columns([2.7, 1.3])

        # coordinates were already extracted above

        with col_tbl:
            st.subheader("Top 10 Candidate Locations")
            tbl_cols = ["Lat", "Lon", "Elevation_m", "IsValley", "MatchRate", "Confidence_Z", "Matches", "ObsCount"]
            if "PlantScore" in scores.columns:
                tbl_cols += ["PlantScore"]
            if "CloudScore" in scores.columns:
                tbl_cols += ["CloudScore"]
            if "ContrastScore" in scores.columns:
                tbl_cols += ["ContrastScore"]
            if "CombinedScore" in scores.columns:
                tbl_cols += ["CombinedScore"]

            top10 = scores.head(10)[tbl_cols].copy()
            top10.insert(0, "Rank", range(1, len(top10) + 1))

            # Add formatted copy-paste coordinates column
            top10["Google Maps Coordinates"] = top10.apply(lambda r: f"{float(r['Lat']):.6f}, {float(r['Lon']):.6f}", axis=1)

            top10["MatchRate"]    = top10["MatchRate"].map("{:.1%}".format)
            top10["Confidence_Z"] = top10["Confidence_Z"].map("{:+.2f}sigma".format)
            top10["Elevation_m"]  = top10["Elevation_m"].map("{:.0f} m".format)

            col_names = ["Rank", "Lat", "Lon", "Elevation", "Valley?",
                         "Match Rate", "Confidence", "Matches", "Obs Used"]
            if "PlantScore" in scores.columns:
                top10["PlantScore"] = top10["PlantScore"].map("{:.2f}".format)
                col_names += ["Plant Score"]
            if "CloudScore" in scores.columns:
                top10["CloudScore"] = top10["CloudScore"].map("{:.2f}".format)
                col_names += ["Cloud Diff"]
            if "ContrastScore" in scores.columns:
                top10["ContrastScore"] = top10["ContrastScore"].map("{:.2f}".format)
                col_names += ["Contrast"]
            if "CombinedScore" in scores.columns:
                top10["CombinedScore"] = top10["CombinedScore"].map("{:.2f}".format)
                col_names += ["Combined Score"]

            col_names += ["Google Maps Coordinates"]
            top10.columns = col_names
            st.dataframe(top10, use_container_width=True, hide_index=True)

        with col_cpy:
            st.subheader("📍 Coordinate Copier")
            
            if clicked_coords:
                st.success("🎯 **Map Point Selected!**")
                st.markdown("**Google Maps & Sheets Coordinates**:")
                st.code(clicked_coords, language="text")
                
                # Direct search link for google maps
                gmaps_url = f"https://www.google.com/maps/search/?api=1&query={clicked_coords.replace(' ', '')}"
                st.markdown(f"[🔗 View on Google Maps]({gmaps_url})")
                
                if st.button("Reset Selection", use_container_width=True, help="Clear map selection and go back to dropdown"):
                    st.session_state.scoring_map = None
                    st.rerun()
            else:
                st.caption("Click any point on the map, or select a location from the dropdown below to copy coordinates:")
                # Dropdown of top 20 candidates
                if "CombinedScore" in scores.columns:
                    copy_opts = [
                        f"Rank #{i+1}: {row['Lat']:.6f}, {row['Lon']:.6f} (Suitability: {row['CombinedScore']:.2f})"
                        for i, row in scores.head(20).iterrows()
                    ]
                else:
                    copy_opts = [
                        f"Rank #{i+1}: {row['Lat']:.6f}, {row['Lon']:.6f} ({row['MatchRate']:.1%} match)"
                        for i, row in scores.head(20).iterrows()
                    ]
                    
                selected_opt = st.selectbox(
                    "Select location to copy",
                    copy_opts,
                    label_visibility="visible",
                    help="Select a location to easily copy its coordinates for pasting into Google Maps, Google Sheets, etc."
                )
                if selected_opt:
                    # Extract the lat, lon
                    coords_str = selected_opt.split(": ")[1].split(" (")[0]
                    
                    st.markdown("**Copy/Paste Coordinates**:")
                    st.code(coords_str, language="text")
                    st.caption("💡 Hover and click the **Copy icon** in the top-right of the box above to copy!")
                    
                    # Direct search link for google maps
                    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={coords_str.replace(' ', '')}"
                    st.markdown(f"[🔗 View on Google Maps]({gmaps_url})")


