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
import base64
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from datetime import date, time, datetime

# ── Backend path ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))
import scorer
from scorer  import (
    score_all_observations, get_point_diagnostics, scores_to_geojson,
    compute_plant_scores, detect_high_contrast_days, score_contrast_observations,
)
import osm_client
from osm_client import fetch_parks_osm
import inaturalist_client
import fetcher
from fetcher import smart_fetch, get_missing_data_points, load_master_csv
from grid_utils import generate_grid
import geometry_utils
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
            st.warning("settings.json could not be parsed — using defaults.")
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
    st.session_state.fog_threshold = 10
if "selected_csv" not in st.session_state:
    _mp = get_master_parquet()
    st.session_state.selected_csv = _mp if os.path.exists(_mp) else None
_saved_settings = load_settings()

if "grid_center_lat" not in st.session_state:
    st.session_state.grid_center_lat = _saved_settings.get("grid_center_lat", 40.31)
if "grid_center_lon" not in st.session_state:
    st.session_state.grid_center_lon = _saved_settings.get("grid_center_lon", -75.13)
if "grid_radius_mi" not in st.session_state:
    st.session_state.grid_radius_mi = _saved_settings.get("grid_radius_mi", 40.0)
if "map_opacity" not in st.session_state:
    st.session_state.map_opacity = 80
_plants_data = load_plants()
if "selected_plants" not in st.session_state:
    st.session_state.selected_plants = _plants_data.get("selected_plants", [])
if "plant_obs_dict" not in st.session_state:
    st.session_state.plant_obs_dict = _plants_data.get("plant_obs_dict", {})
if "plant_influence_radius" not in st.session_state:
    st.session_state.plant_influence_radius = 3.0
if "plant_match_mode" not in st.session_state:
    st.session_state.plant_match_mode = "All"
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

    df["CombinedScore"] = combined.fillna(0.0)
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
                    # Optimize memory: round Lat/Lon on the fly without duplicating the massive dataframe
                    temp_grid = grid_df.copy()
                    
                    # Instead of an inner merge that duplicates data, we filter fog_df by checking if rounded coordinates are in the grid
                    grid_set = set(zip(temp_grid["Lat_g"], temp_grid["Lon_g"]))
                    
                    fog_mask = pd.Series(list(zip(fog_df["Lat"].round(4), fog_df["Lon"].round(4)))).isin(grid_set)
                    filtered_df = fog_df[fog_mask.values].copy()
                    
                    # Ensure Lat_g and Lon_g exist for the downstream logic
                    filtered_df["Lat_g"] = filtered_df["Lat"].round(4)
                    filtered_df["Lon_g"] = filtered_df["Lon"].round(4)
                    
                    # Clean up the merge artifacts so score_all_observations sees exact Lat/Lon
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
        
    with st.expander("➕ Create New Hunt", expanded=False):
        new_hunt_name = st.text_input("New Hunt Name", placeholder="e.g. veil_nine")
        if st.button("Create", use_container_width=True) and new_hunt_name:
            # Basic sanitization
            new_hunt_clean = "".join(c for c in new_hunt_name if c.isalnum() or c in ("_", "-"))
            if new_hunt_clean and new_hunt_clean not in hunts:
                os.makedirs(os.path.join(HUNTS_DIR, new_hunt_clean), exist_ok=True)
                switch_hunt(new_hunt_clean)
                st.rerun()

    st.markdown("---")
    with st.expander("⚖️ Combined Score Weights & Settings", expanded=False):
        st.caption("Adjust detection thresholds and how much influence each layer has on the map.")

        _changed = False
        
        # Fog Threshold Control
        st.markdown("**Fog Detection Threshold**")
        _new_fog_thresh = st.radio(
            "fog_thresh",
            ["Moderate  (Score >= 5)", "Confirmed  (Score >= 10)"],
            index=0 if st.session_state.fog_threshold == 5 else 1,
            label_visibility="collapsed"
        )
        parsed_fog_thresh = 5 if "Moderate" in _new_fog_thresh else 10
        if parsed_fog_thresh != st.session_state.fog_threshold:
            st.session_state.fog_threshold = parsed_fog_thresh
            _changed = True
            
        st.markdown("---")
        
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
                
            # Plant Match Strategy Control
            if len(st.session_state.selected_plants) > 1:
                st.markdown("**Plant Match Strategy**")
                _new_match_mode = st.selectbox(
                    "match_mode",
                    ["Match Any (Either plant is nearby)", "Match All (All plants must be nearby)"],
                    index=0 if st.session_state.plant_match_mode == "Any" else 1,
                    label_visibility="collapsed"
                )
                parsed_mode = "Any" if "Any" in _new_match_mode else "All"
                if parsed_mode != st.session_state.plant_match_mode:
                    st.session_state.plant_match_mode = parsed_mode
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
            invalidate_scores()
            st.rerun()


    with st.expander("🗺️ Map Settings", expanded=False):
        if "map_opacity" not in st.session_state:
            st.session_state.map_opacity = 80
        _nt = st.slider(
            "Point Opacity (%)", 5, 100,
            int(st.session_state.map_opacity), step=5,
            key="fa_trans",
            help="Higher = solid grid points. Lower = more see-through. Image overlay opacity is controlled separately."
        )
        if _nt != st.session_state.map_opacity:
            st.session_state.map_opacity = _nt
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
                n_op_pct = st.slider("Overlay Opacity (%)", 0, 100, int(round(float(overlay["opacity"]) * 100)), step=5, key=f"ov_op_{idx}")
                n_op = n_op_pct / 100.0
                if n_op != overlay["opacity"]:
                    overlay["opacity"] = n_op
                    st.session_state.settings_cache = load_settings()
                    st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                    save_settings(st.session_state.settings_cache)
                    st.rerun()
            st.markdown("---")

# ─── Map View ──────────────────────────────────────────────────────────────
st.header("Location Scoring Map")

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

        # ── Score range filter ─────────────────────────────────────────────
        _score_min = float(map_df["CombinedScore"].min()) if "CombinedScore" in map_df.columns else 0.0
        _score_max = float(map_df["CombinedScore"].max()) if "CombinedScore" in map_df.columns else 1.0
        _score_max = _score_max if _score_max > _score_min else _score_min + 0.001

        _filter_range = st.slider(
            "Filter by Combined Score",
            min_value=round(_score_min, 3),
            max_value=round(_score_max, 3),
            value=(round(_score_min, 3), round(_score_max, 3)),
            step=round((_score_max - _score_min) / 200, 4) or 0.001,
            help="Hide grid points outside this score range. Drag either end to focus the map.",
            key="score_range_filter",
        )
        map_df = map_df[
            (map_df["CombinedScore"] >= _filter_range[0]) &
            (map_df["CombinedScore"] <= _filter_range[1])
        ]
        if map_df.empty:
            st.warning("No points match the current score filter — try widening the range.")
            st.stop()

        # ── Candidate picker (placed above map so star renders on next rerun) ───
        if "main_pin_idx" not in st.session_state:
            st.session_state.main_pin_idx = 0

        _main_pin_opts = [
            f"#{i+1}  {row['Lat']:.5f}, {row['Lon']:.5f}  (score {row.get('CombinedScore', row.get('MatchRate', 0)):.3f})"
            for i, row in scores.head(20).iterrows()
        ]
        _main_pin_sel = st.selectbox(
            "📌 Highlight a top candidate on the map",
            options=_main_pin_opts,
            index=min(st.session_state.main_pin_idx, len(_main_pin_opts) - 1),
            key="main_pin_select",
            help="Places a cyan star on the map at the selected candidate location.",
        )
        _main_pin_new_idx = _main_pin_opts.index(_main_pin_sel)
        if _main_pin_new_idx != st.session_state.main_pin_idx:
            st.session_state.main_pin_idx = _main_pin_new_idx
            st.rerun()
        _main_pinned_row = scores.iloc[st.session_state.main_pin_idx]
        _main_pin_lat   = float(_main_pinned_row["Lat"])
        _main_pin_lon   = float(_main_pinned_row["Lon"])

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


        # Preserve user's manual pan/zoom unless they changed the grid center
        map_center_lat = map_center["lat"]
        map_center_lon = map_center["lon"]
        map_zoom = auto_zoom
        
        if "last_map_grid" not in st.session_state:
            st.session_state.last_map_grid = (st.session_state.grid_center_lat, st.session_state.grid_center_lon)
        
        grid_changed = st.session_state.last_map_grid != (st.session_state.grid_center_lat, st.session_state.grid_center_lon)
        if grid_changed:
            st.session_state.last_map_grid = (st.session_state.grid_center_lat, st.session_state.grid_center_lon)
        else:
            if "scoring_map" in st.session_state and st.session_state.scoring_map:
                _sm = st.session_state.scoring_map
                if _sm.get("center"):
                    map_center_lat = _sm["center"]["lat"]
                    map_center_lon = _sm["center"]["lng"]
                if _sm.get("zoom"):
                    map_zoom = _sm["zoom"]

        # Build Folium Map
        m = folium.Map(
            location=[map_center_lat, map_center_lon], 
            zoom_start=map_zoom, 
            tiles="cartodbdark_matter",
            control_scale=True
        )
        
        cmin, cmax = range_color
        if cmax <= cmin:
            cmax = cmin + 0.01
        cmap = cm.LinearColormap(
            colors=["#000004", "#51127c", "#b63679", "#fb8861", "#fcffa4"], 
            vmin=cmin, 
            vmax=cmax,
            caption=color_title
        )
        m.add_child(cmap)

        for _, row in map_df.iterrows():
            val = row.get(color_col, 0)
            if pd.isna(val):
                continue
            color = cmap(val)
            
            html = f"<b>Match Rate: {row['MatchRate_pct']:.1f}%</b><br>"
            html += f"Confidence: {row['Confidence_Z']:+.2f}sigma<br>"
            html += f"Matches: {row['Matches']} / {row['ObsCount']} obs<br>"
            html += f"Lat: {row['Lat']:.4f}  Lon: {row['Lon']:.4f}<br>"
            html += f"Elevation: {row['Elevation_m']:.0f} m  Valley: {row['IsValley']}<br>"
            if "PlantScore" in map_df.columns:
                html += f"Plant Proximity Score: {row['PlantScore']:.2f}<br>"
            if "CloudScore" in map_df.columns:
                html += f"Cloud Differential: {row['CloudScore']:.2f}<br>"
            if "ContrastScore" in map_df.columns:
                html += f"High-Contrast Cloud Score: {row['ContrastScore']:.2f}<br>"
            if "CombinedScore" in map_df.columns:
                html += f"Combined Suitability: {row['CombinedScore']:.2f}<br>"
                
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=max(2, row["_size"] / 2),
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=st.session_state.map_opacity / 100.0,
                tooltip=folium.Tooltip(html)
            ).add_to(m)

        # --- Global Plant Layers ---
        if st.session_state.get("show_plant_pins", True):
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
            for p_idx, p in enumerate(st.session_state.selected_plants):
                tid = p["id"]
                if tid in st.session_state.plant_obs_dict:
                    obs_list = st.session_state.plant_obs_dict[tid]
                    if obs_list:
                        c = colors[p_idx % len(colors)]
                        for o in obs_list:
                            folium.Marker(
                                location=[o["lat"], o["lon"]],
                                tooltip=p["common"],
                                icon=folium.Icon(color=c, icon="leaf")
                            ).add_to(m)

        clicked_lat = None
        clicked_lon = None
        clicked_coords = None

        if "scoring_map" in st.session_state and st.session_state.scoring_map:
            last_clicked = st.session_state.scoring_map.get("last_clicked")
            if last_clicked:
                clicked_lat = float(last_clicked["lat"])
                clicked_lon = float(last_clicked["lng"])
                clicked_coords = f"{clicked_lat:.6f}, {clicked_lon:.6f}"

        # Map Overlays
        if "map_overlays" in st.session_state:
            for overlay in st.session_state.map_overlays:
                _ov_dir = os.path.join(get_hunt_dir(), "overlays")
                _resolved_path = os.path.join(_ov_dir, overlay["name"])
                
                if overlay.get("visible", True) and os.path.exists(_resolved_path):
                    ax = overlay.get("anchor_x", 0.5)
                    ay = overlay.get("anchor_y", 0.5)
                    lat_span = overlay["height"] / 111.0
                    lon_span = overlay["width"] / (111.0 * np.cos(np.radians(overlay["lat"])))
                    
                    sw = [overlay["lat"] - lat_span * (1.0 - ay), overlay["lon"] - lon_span * ax]
                    ne = [overlay["lat"] + lat_span * ay, overlay["lon"] + lon_span * (1.0 - ax)]
                    
                    folium.raster_layers.ImageOverlay(
                        image=_resolved_path,
                        bounds=[sw, ne],
                        opacity=float(overlay["opacity"]),
                        interactive=True,
                        cross_origin=False,
                        zindex=1
                    ).add_to(m)

        # Dynamic local OSM traces (Trails/Streams and Park Boundaries) for the active point
        active_lat = clicked_lat if clicked_lat is not None else _main_pin_lat
        active_lon = clicked_lon if clicked_lon is not None else _main_pin_lon
        
        _parks = []
        if active_lat is not None and active_lon is not None:
            r_lat = round(active_lat, 2)
            r_lon = round(active_lon, 2)
            _local_features = osm_client.fetch_osm_features(
                r_lat - 0.02, r_lon - 0.02, 
                r_lat + 0.02, r_lon + 0.02
            )
            
            # Draw local Polygons (Parks/Forests)
            for poly in _local_features.get("polygons", []):
                coords = poly["coords"]
                
                # Calculate center for haversine plant count
                c_lat = sum([c[0] for c in coords]) / len(coords)
                c_lon = sum([c[1] for c in coords]) / len(coords)
                
                _count = 0
                if st.session_state.plant_obs_dict:
                    for _species, _obs_list in st.session_state.plant_obs_dict.items():
                        for _o in _obs_list:
                            if scorer.haversine_distance(c_lat, c_lon, _o["lat"], _o["lon"]) <= 0.2:
                                _count += 1
                                
                _parks.append({
                    "osm_id": poly.get("id", "0"),
                    "type": poly.get("type", "unknown"),
                    "name": poly.get("name", "Unnamed Park"),
                    "lat": c_lat,
                    "lon": c_lon,
                    "plants_count": _count
                })
                                
                html = f"<b>🌳 {poly['name']}</b><br>Plant Observations (0.2mi): {_count}<br>Center: {c_lat:.5f}, {c_lon:.5f}"
                
                color = "#228b22"  # green
                fill_opacity = 0.3
                if "water" in poly["tags"].get("natural", ""):
                    color = "#1e90ff" # blue
                    fill_opacity = 0.4
                    
                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    tooltip=folium.Tooltip(html)
                ).add_to(m)
            
            # Draw local Lines (Trails/Streams)
            for line in _local_features.get("lines", []):
                coords = line["coords"]
                
                color = "orange"
                width = 3
                if "waterway" in line["tags"]:
                    color = "cyan"
                    width = 4
                elif "power" in line["tags"]:
                    color = "yellow"
                    width = 2
                    
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=width,
                    tooltip=line["name"]
                ).add_to(m)

        folium.Marker(
            location=[_main_pin_lat, _main_pin_lon],
            tooltip=f"📌 Selected: {_main_pin_lat:.6f}, {_main_pin_lon:.6f}",
            icon=folium.Icon(color="cyan", icon="star")
        ).add_to(m)

        folium.Marker(
            location=[st.session_state.grid_center_lat, st.session_state.grid_center_lon],
            tooltip=f"Grid Center ({st.session_state.grid_center_lat}, {st.session_state.grid_center_lon})",
            icon=folium.Icon(color="white", icon="star")
        ).add_to(m)

        # Add custom Right-Click (contextmenu) Google Maps link
        from branca.element import MacroElement
        from jinja2 import Template

        class RightClickGoogleMaps(MacroElement):
            _template = Template(u"""
                {% macro script(this, kwargs) %}
                    var {{ this.get_name() }} = function(e) {
                        var lat = e.latlng.lat.toFixed(5);
                        var lng = e.latlng.lng.toFixed(5);
                        var gmaps_url = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
                        var content = `<div style="text-align: center;">
                            <b>Coordinates:</b> ${lat}, ${lng}<br><br>
                            <a href="${gmaps_url}" target="_blank" style="background-color: #4CAF50; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; display: inline-block;">Open in Google Maps</a>
                        </div>`;
                        
                        L.popup()
                            .setLatLng(e.latlng)
                            .setContent(content)
                            .openOn({{ this._parent.get_name() }});
                    };
                    {{ this._parent.get_name() }}.on('contextmenu', {{ this.get_name() }});
                {% endmacro %}
            """)
            def __init__(self):
                super(RightClickGoogleMaps, self).__init__()
                self._name = 'RightClickGoogleMaps'

        m.add_child(RightClickGoogleMaps())

        folium.LatLngPopup().add_to(m)

        st.caption("💡 **Tip**: Click anywhere on the map to see its coordinates in a popup and load it into the **Coordinate Copier**!")

        st_folium(
            m,
            width=1200,
            height=650,
            returned_objects=["last_clicked"],
            key="scoring_map"
        )

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

            # Build full URLs so LinkColumn renders them as clickable
            top10["Google Maps Coordinates"] = top10.apply(
                lambda r: f"https://www.google.com/maps/search/?api=1&query={float(r['Lat']):.6f},{float(r['Lon']):.6f}",
                axis=1,
            )

            # Format numeric columns before renaming
            top10["MatchRate"]    = top10["MatchRate"].map("{:.1%}".format)
            top10["Confidence_Z"] = top10["Confidence_Z"].map("{:+.2f}sigma".format)
            top10["Elevation_m"]  = top10["Elevation_m"].map("{:.0f} m".format)
            if "PlantScore" in top10.columns:
                top10["PlantScore"] = top10["PlantScore"].map("{:.2f}".format)
            if "CloudScore" in top10.columns:
                top10["CloudScore"] = top10["CloudScore"].map("{:.2f}".format)
            if "ContrastScore" in top10.columns:
                top10["ContrastScore"] = top10["ContrastScore"].map("{:.2f}".format)
            if "CombinedScore" in top10.columns:
                top10["CombinedScore"] = top10["CombinedScore"].map("{:.2f}".format)

            # Rename using a dict — safe against column count mismatches
            top10 = top10.rename(columns={
                "Elevation_m":   "Elevation",
                "IsValley":      "Valley?",
                "MatchRate":     "Match Rate",
                "Confidence_Z":  "Confidence",
                "Matches":       "Matches",
                "ObsCount":      "Obs Used",
                "PlantScore":    "Plant Score",
                "CloudScore":    "Cloud Diff",
                "ContrastScore": "Contrast",
                "CombinedScore": "Combined Score",
            })
            st.dataframe(
                top10,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Google Maps Coordinates": st.column_config.LinkColumn(
                        "Google Maps",
                        display_text="🔗 Open",
                    )
                },
            )

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
                    index=st.session_state.main_pin_idx,
                    label_visibility="visible",
                    help="Select a location to easily copy its coordinates for pasting into Google Maps, Google Sheets, etc.",
                    key="main_copy_sel",
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

        # ── Nearby plant summary (Park-Centric) ─────────────────────────────────────────
        st.markdown("---")
        if st.session_state.plant_obs_dict:
            # 1. Collect all nearby plant observations within 3.0 miles of selected coordinate
            _nearby_plants = []
            for _sp, _obs_list in st.session_state.plant_obs_dict.items():
                for _o in _obs_list:
                    _dist = scorer.haversine_distance(active_lat, active_lon, _o["lat"], _o["lon"])
                    if _dist <= 3.0:
                        _o_copy = _o.copy()
                        _o_copy["species"] = _sp
                        _o_copy["dist_to_center"] = _dist
                        _nearby_plants.append(_o_copy)
                        
            if _nearby_plants or _parks:
                st.markdown("**🌳 Nearby Parks & Plant Observations:**")
                
                # Group plants by closest park if within 0.2 miles
                _park_plants = {p["osm_id"]: [] for p in _parks}
                _other_plants = []
                
                for _plant in _nearby_plants:
                    _closest_park = None
                    _min_dist = float("inf")
                    for _p in _parks:
                        _p_dist = scorer.haversine_distance(_plant["lat"], _plant["lon"], _p["lat"], _p["lon"])
                        if _p_dist < _min_dist:
                            _min_dist = _p_dist
                            _closest_park = _p
                            
                    if _closest_park is not None and _min_dist <= 0.2:
                        _park_plants[_closest_park["osm_id"]].append(_plant)
                    else:
                        _other_plants.append(_plant)
                        
                # Sort parks: count of plants descending, then distance ascending
                _parks_sorted = []
                for _p in _parks:
                    _p_plants = _park_plants[_p["osm_id"]]
                    _dist_to_center = scorer.haversine_distance(active_lat, active_lon, _p["lat"], _p["lon"])
                    _parks_sorted.append({
                        **_p,
                        "dist_to_center": _dist_to_center,
                        "plants_count": len(_p_plants),
                        "plants": _p_plants
                    })
                _parks_sorted.sort(key=lambda x: (-x["plants_count"], x["dist_to_center"]))
                
                # Render park expanders
                for _p in _parks_sorted:
                    _label = f"🌳 {_p['name']} ({_p['plants_count']})"
                    with st.expander(_label):
                        st.markdown(f"**Distance from selected center:** {_p['dist_to_center']:.2f} miles")
                        st.markdown(f"**OSM Type/ID:** `{_p['type']}/{_p['osm_id']}`")
                        _gmap_park = f"https://www.google.com/maps/search/?api=1&query={_p['lat']:.6f},{_p['lon']:.6f}"
                        st.markdown(f"[🔗 Open Park in Google Maps]({_gmap_park})")
                        st.markdown("---")
                        
                        if _p["plants"]:
                            # Sort plants inside park by distance to park center
                            _p["plants"].sort(key=lambda x: scorer.haversine_distance(_p["lat"], _p["lon"], x["lat"], x["lon"]))
                            for _plant in _p["plants"]:
                                _gmap_plant = f"https://www.google.com/maps/search/?api=1&query={_plant['lat']:.6f},{_plant['lon']:.6f}"
                                _dist_to_park = scorer.haversine_distance(_p["lat"], _p["lon"], _plant["lat"], _plant["lon"])
                                st.markdown(
                                    f"- **{_plant['species']}**: [{_plant['lat']:.5f}, {_plant['lon']:.5f}]({_gmap_plant}) "
                                    f"({_dist_to_park:.2f} mi from park center) — Observer: @{_plant.get('user', 'unknown')}, Date: {_plant.get('observed_on', 'unknown')}"
                                )
                        else:
                            st.write("No nearby plant observations recorded in this park.")
                            
                # Render fallback expander for plants not inside any park
                if _other_plants:
                    _other_plants.sort(key=lambda x: x["dist_to_center"])
                    with st.expander(f"🌿 Other ({len(_other_plants)})"):
                        st.markdown("*These observations are within 3 miles of the selected coordinates, but not within 0.2 miles of a known park.*")
                        st.markdown("---")
                        for _plant in _other_plants:
                            _gmap_plant = f"https://www.google.com/maps/search/?api=1&query={_plant['lat']:.6f},{_plant['lon']:.6f}"
                            st.markdown(
                                f"- **{_plant['species']}**: [{_plant['lat']:.5f}, {_plant['lon']:.5f}]({_gmap_plant}) "
                                f"({_plant['dist_to_center']:.2f} mi away) — Observer: @{_plant.get('user', 'unknown')}, Date: {_plant.get('observed_on', 'unknown')}"
                            )


