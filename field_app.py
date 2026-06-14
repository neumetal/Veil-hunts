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
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Veil Finder")
    st.caption("Trailcam location inference via fog pattern matching")
    
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

    # Data selector — prefer fog_master.parquet, fall back to legacy fog_data_*.csv
    st.markdown('<p class="section-header">Satellite Data</p>', unsafe_allow_html=True)
    os.makedirs(get_export_dir(), exist_ok=True)

    # Auto-select master Parquet on first load
    if st.session_state.selected_csv is None:
        if os.path.exists(_MASTER_PARQUET):
            st.session_state.selected_csv = _MASTER_PARQUET

    # Build dropdown: master first, then any legacy CSVs
    all_files = []
    if os.path.exists(_MASTER_PARQUET):
        all_files.append(_MASTER_PARQUET)
    all_files += sorted(
        [f for f in glob.glob(os.path.join(get_export_dir(), "fog_data_*.csv"))],
        reverse=True,
    )

    if all_files:
        csv_names   = [os.path.basename(c) for c in all_files]
        default_idx = 0  # fog_master.parquet is always first when it exists
        sel_name = st.selectbox("Active fog data", csv_names,
                                index=default_idx, label_visibility="collapsed")
        new_csv = os.path.join(get_export_dir(), sel_name)
        if new_csv != st.session_state.selected_csv:
            st.session_state.selected_csv = new_csv
            invalidate_scores()
    else:
        st.info("No satellite data yet. Log observations then use the **Data Status** tab to fetch.")

    # Fog threshold
    st.markdown('<p class="section-header">Fog Detection Threshold</p>', unsafe_allow_html=True)
    threshold_choice = st.radio(
        "threshold",
        ["Moderate  (Score >= 5)", "Confirmed  (Score >= 10)"],
        label_visibility="collapsed",
    )
    new_threshold = 5 if "5" in threshold_choice else 10
    if new_threshold != st.session_state.fog_threshold:
        st.session_state.fog_threshold = new_threshold
        invalidate_scores()

    # Stats
    st.markdown("---")
    obs  = st.session_state.observations
    n    = len(obs)
    n_fog   = sum(1 for o in obs if o.get("fog_observed"))
    n_clear = n - n_fog

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", n)
    c2.metric("Fog",   n_fog)
    c3.metric("Clear", n_clear)

    # Quick rescore button
    if st.button("Re-score Now", use_container_width=True, type="secondary"):
        invalidate_scores()
        st.rerun()

    st.markdown("---")
    with st.expander("Grid Settings", expanded=False):
        new_lat = st.number_input("Center Latitude", value=st.session_state.grid_center_lat, format="%.6f", step=0.1)
        new_lon = st.number_input("Center Longitude", value=st.session_state.grid_center_lon, format="%.6f", step=0.1)
        new_rad = st.number_input("Radius (miles)", value=float(st.session_state.grid_radius_mi), min_value=1.0, max_value=250.0, step=5.0)
        
        st.markdown("---")
        new_trans = st.slider("Point Transparency (%)", min_value=0, max_value=95, value=st.session_state.map_transparency, step=5, help="Make grid points semi-transparent to view roads and landmarks under them.")
        
        if new_trans != st.session_state.map_transparency:
            st.session_state.map_transparency = new_trans
            st.rerun()
            
        if (new_lat != st.session_state.grid_center_lat or
            new_lon != st.session_state.grid_center_lon or
            new_rad != st.session_state.grid_radius_mi):
            st.session_state.grid_center_lat = new_lat
            st.session_state.grid_center_lon = new_lon
            st.session_state.grid_radius_mi = new_rad
            save_settings({
                "grid_center_lat": new_lat,
                "grid_center_lon": new_lon,
                "grid_radius_mi": new_rad
            })
            invalidate_scores()
            st.rerun()

    with st.expander("🌿 iNaturalist Plant Settings", expanded=False):
        # 1. Search and Add Plant
        st.caption("Search plant name to fetch local observations:")
        add_col_text, add_col_btn = st.columns([3, 1])
        with add_col_text:
            plant_q = st.text_input("Search Species", placeholder="e.g. Japanese Stiltgrass", label_visibility="collapsed")
        with add_col_btn:
            add_clicked = st.button("Add", use_container_width=True)

        if add_clicked and plant_q.strip():
            # Check if species already added
            already_added = any(
                p["sci"].lower() == plant_q.lower() or p["common"].lower() == plant_q.lower()
                for p in st.session_state.selected_plants
            )
            if already_added:
                st.error("Species already in your list.")
            else:
                with st.spinner("Searching taxon..."):
                    tid, sci_name, comm_name = inaturalist_client.resolve_taxon_id(plant_q)
                
                if tid is None:
                    st.error("No plant species found matching that name on iNaturalist.")
                else:
                    # Let's fetch observations immediately inside our active grid bounds!
                    current_pts = generate_grid(
                        center_lat=st.session_state.grid_center_lat,
                        center_lon=st.session_state.grid_center_lon,
                        radius_km=st.session_state.grid_radius_mi * 1.60934
                    )
                    lats = [p[0] for p in current_pts]
                    lons = [p[1] for p in current_pts]
                    swlat, nelat = min(lats) - 0.05, max(lats) + 0.05
                    swlon, nelon = min(lons) - 0.05, max(lons) + 0.05
                    
                    with st.spinner(f"Fetching verified observations for {comm_name}..."):
                        obs_list = inaturalist_client.fetch_plant_observations(
                            taxon_id=tid,
                            swlat=swlat, swlon=swlon,
                            nelat=nelat, nelon=nelon,
                            max_results=2000,
                            min_distance_mi=0.5
                        )
                    
                    st.session_state.selected_plants.append({
                        "id": tid,
                        "sci": sci_name,
                        "common": comm_name
                    })
                    st.session_state.plant_obs_dict[comm_name] = obs_list
                    save_plants(st.session_state.selected_plants, st.session_state.plant_obs_dict)
                    invalidate_scores()
                    st.success(f"Added **{comm_name}** ({len(obs_list)} local observations found)!")
                    st.rerun()

        # Display currently selected plants with "Remove" buttons
        if st.session_state.selected_plants:
            st.markdown("<p class='section-header'>Active Species</p>", unsafe_allow_html=True)
            for idx, plant in enumerate(st.session_state.selected_plants):
                p_col1, p_col2 = st.columns([3.2, 0.8])
                obs_count = len(st.session_state.plant_obs_dict.get(plant["common"], []))
                p_col1.caption(f"🌱 **{plant['common']}**\n*{plant['sci']}* ({obs_count} local)")
                if p_col2.button("✖", key=f"del_plant_{idx}", use_container_width=True):
                    st.session_state.selected_plants.pop(idx)
                    st.session_state.plant_obs_dict.pop(plant["common"], None)
                    save_plants(st.session_state.selected_plants, st.session_state.plant_obs_dict)
                    invalidate_scores()
                    st.rerun()

        st.markdown("---")
        
        # 2. Influence radius
        new_plant_rad = st.slider(
            "Plant Influence (mi)",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.plant_influence_radius),
            step=0.5,
            help="Proximity range where plant observations affect grid scoring."
        )
        if new_plant_rad != st.session_state.plant_influence_radius:
            st.session_state.plant_influence_radius = new_plant_rad
            invalidate_scores()
            st.rerun()

        # 3. Match Mode
        if len(st.session_state.selected_plants) > 1:
            new_match_mode = st.radio(
                "Match Strategy",
                ["Match Any (Either plant is nearby)", "Match All (All plants must be nearby)"],
                index=0 if st.session_state.plant_match_mode == "Any" else 1,
                help="How to score points when multiple plants are added."
            )
            parsed_mode = "Any" if "Any" in new_match_mode else "All"
            if parsed_mode != st.session_state.plant_match_mode:
                st.session_state.plant_match_mode = parsed_mode
                invalidate_scores()
                st.rerun()

        # 4. Map Instructions
        st.info("💡 **Tip**: Click any grid point on the map to see the specific nearby plant observations that contributed to its score!")

    st.caption(f"Center: {st.session_state.grid_center_lat}N, {st.session_state.grid_center_lon}W")
    
    current_grid_sz = len(generate_grid(
        center_lat=st.session_state.grid_center_lat, 
        center_lon=st.session_state.grid_center_lon, 
        radius_km=st.session_state.grid_radius_mi * 1.60934
    ))
    st.caption(f"Grid: {current_grid_sz} pts  |  Resolution: 0.05 deg")

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
        st.caption("Upload images (like old maps or clue diagrams) to overlay on the main grid.")
        
        if "map_overlays" not in st.session_state:
            st.session_state.map_overlays = _saved_settings.get("map_overlays", [])
            
        overlay_dir = os.path.join(get_hunt_dir(), "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
        
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file is not None:
            fpath = os.path.join(overlay_dir, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if not any(o["name"] == uploaded_file.name for o in st.session_state.map_overlays):
                st.session_state.map_overlays.append({
                    "name": uploaded_file.name,
                    "path": fpath,
                    "lat": st.session_state.grid_center_lat,
                    "lon": st.session_state.grid_center_lon,
                    "width": 10.0,
                    "height": 10.0,
                    "rotation": 0.0,
                    "opacity": 0.5,
                    "anchor_x": 0.5,
                    "anchor_y": 0.5,
                    "visible": True
                })
                # save to settings
                st.session_state.settings_cache = load_settings()
                st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                save_settings(st.session_state.settings_cache)
                st.success(f"Added {uploaded_file.name}!")
                st.rerun()

        for idx, overlay in enumerate(st.session_state.map_overlays):
            st.markdown(f"**{overlay['name']}**")
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                vis = st.checkbox("Visible", value=overlay.get("visible", True), key=f"ov_vis_{idx}")
            with col_v2:
                del_btn = st.button("Delete", key=f"ov_del_{idx}")

            if del_btn:
                st.session_state.map_overlays.pop(idx)
                st.session_state.settings_cache = load_settings()
                st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                save_settings(st.session_state.settings_cache)
                st.rerun()
                break
                
            _ov_changed = False
            if vis != overlay.get("visible", True):
                overlay["visible"] = vis
                _ov_changed = True
                
            if vis:
                n_lat = st.number_input("Anchor Lat", value=overlay["lat"], step=0.01, format="%.4f", key=f"ov_lat_{idx}")
                n_lon = st.number_input("Anchor Lon", value=overlay["lon"], step=0.01, format="%.4f", key=f"ov_lon_{idx}")
                
                _ov_dir = os.path.join(get_hunt_dir(), "overlays")
                _resolved_path = os.path.join(_ov_dir, overlay["name"])
                if not os.path.exists(_resolved_path):
                    st.warning(f"Image not found: {_resolved_path}")
                    continue
                img = Image.open(_resolved_path)
                img_width, img_height = img.size
                aspect_ratio = img_width / max(img_height, 1)
                
                n_ax = overlay.get("anchor_x", 0.5)
                n_ay = overlay.get("anchor_y", 0.5)
                
                st.caption("🎯 Click the image below to move the Anchor Crosshair")
                
                preview_img = img.copy()
                preview_img.thumbnail((350, 350)) # resize to fixed max width/height to avoid CSS scaling bugs
                pw, ph = preview_img.size
                
                draw = ImageDraw.Draw(preview_img)
                cx, cy = int(n_ax * pw), int(n_ay * ph)
                
                # Draw crosshair
                draw.line((cx - 10, cy, cx + 10, cy), fill="red", width=3)
                draw.line((cx, cy - 10, cx, cy + 10), fill="red", width=3)
                
                # Without use_column_width=True, the image is rendered at true pixel size (pw, ph)
                coords = streamlit_image_coordinates(preview_img, key=f"ov_img_{idx}")
                if coords is not None:
                    c_ax = coords["x"] / pw
                    c_ay = coords["y"] / ph
                    if abs(c_ax - n_ax) > 0.005 or abs(c_ay - n_ay) > 0.005:
                        overlay["anchor_x"] = c_ax
                        overlay["anchor_y"] = c_ay
                        _ov_changed = True
                        # Immediately save and rerun to update the crosshair and map
                        st.session_state.settings_cache = load_settings()
                        st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                        save_settings(st.session_state.settings_cache)
                        st.rerun()
                    
                n_w = st.slider("Scale (Width km)", 0.1, 500.0, float(overlay["width"]), step=0.5, key=f"ov_scale_{idx}")
                n_h = n_w / aspect_ratio
                    
                n_rot = st.slider("Rotation (°)", 0.0, 360.0, value=float(overlay["rotation"]), step=1.0, key=f"ov_rot_{idx}")
                n_op = st.slider("Opacity", 0.0, 1.0, value=float(overlay["opacity"]), step=0.05, key=f"ov_op_{idx}")
                
                if (n_lat != overlay["lat"] or n_lon != overlay["lon"] or n_w != overlay["width"] or 
                    n_rot != overlay["rotation"] or n_op != overlay["opacity"] or
                    n_ax != overlay.get("anchor_x", 0.5) or n_ay != overlay.get("anchor_y", 0.5)):
                    overlay["lat"] = n_lat
                    overlay["lon"] = n_lon
                    overlay["width"] = n_w
                    overlay["height"] = n_h
                    overlay["rotation"] = n_rot
                    overlay["opacity"] = n_op
                    overlay["anchor_x"] = n_ax
                    overlay["anchor_y"] = n_ay
                    _ov_changed = True
                    
            if _ov_changed:
                st.session_state.settings_cache = load_settings()
                st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays
                save_settings(st.session_state.settings_cache)
                st.rerun()
            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Log Observation",
    "Observation Log",
    "Scoring Map",
    "Diagnostics",
    "Data Status",
    "Export",
])


# ─── TAB 1: Log Observation ───────────────────────────────────────────────────
with tab1:
    st.header("Log a New Observation")
    st.caption(
        "Enter what you saw in your trail cam photo. "
        "Use the time window that covers when the photo was taken (e.g. 3:00 AM - 6:00 AM)."
    )

    obs_type = st.radio(
        "Observation Type",
        ["Fog Detection", "Cloud Differential", "High-Contrast Cloud"],
        horizontal=True,
    )

    # ── High-Contrast: detect windows BEFORE the form ────────────────────────
    if obs_type == "High-Contrast Cloud":
        _fog_for_contrast = get_fog_df()
        if _fog_for_contrast is None or "CloudCover_Pct" not in (_fog_for_contrast.columns if _fog_for_contrast is not None else []):
            st.warning("No satellite data with cloud cover loaded. Fetch data first in the **Data Status** tab.")
            _contrast_windows = []
        else:
            _contrast_windows = detect_high_contrast_days(_fog_for_contrast)

        if not _contrast_windows:
            st.info(
                "No high-contrast windows detected in the current satellite data. "
                "Either the data has no days with extreme clear/cloudy variation, "
                "or satellite data hasn't been fetched yet."
            )

    with st.form("obs_form", clear_on_submit=True):

        # Time inputs only shown for Fog and Cloud Differential
        if obs_type in ("Fog Detection", "Cloud Differential"):
            row1_col1, row1_col2, row1_col3 = st.columns([2, 1.5, 1.5])
            with row1_col1:
                obs_date = st.date_input("Date of Photo", value=date.today())
            with row1_col2:
                time_start = st.time_input("Window Start (EDT)", value=time(3, 0), step=1800)
            with row1_col3:
                time_end = st.time_input("Window End (EDT)", value=time(6, 0), step=1800)
        else:
            # High-Contrast: date/window come from the dropdown, not free-form times
            obs_date = date.today()    # placeholder; overridden below
            time_start = time(9, 0)   # placeholder
            time_end   = time(12, 0)  # placeholder

        if obs_type == "Fog Detection":
            row2_col1, row2_col2 = st.columns([1, 2])
            with row2_col1:
                fog_observed = st.toggle(
                    "Fog Visible in Photo?",
                    value=True,
                    help="ON = fog present. OFF = clear / no fog.",
                )
                st.markdown(
                    '<span class="obs-badge-fog">FOG</span>' if fog_observed
                    else '<span class="obs-badge-clear">CLEAR</span>',
                    unsafe_allow_html=True,
                )
            with row2_col2:
                photo_filename = st.text_input(
                    "Photo Filename (optional)",
                    placeholder="20251115_trail_0312.jpg",
                    help="Just the filename — no need for the full path.",
                )
            notes = st.text_area(
                "Notes (optional)",
                placeholder="Dense fog, frame mostly white. Fog appears to be moving west...",
                height=90,
            )

        elif obs_type == "Cloud Differential":
            cloud_relation = st.radio(
                "During this window...",
                ["Trailcam was Cloudier than Home", "Trailcam was Sunnier than Home"],
            )

        else:  # High-Contrast Cloud
            if _contrast_windows:
                # Build dropdown labels
                win_options = [
                    f"{w['date']} | {w['window']} — spread {w['spread']:.0f}%  "
                    f"({w['pct_clear']:.0f}% clear / {w['pct_cloudy']:.0f}% cloudy)"
                    for w in _contrast_windows
                ]
                selected_win_label = st.selectbox(
                    "Select a High-Contrast Window",
                    win_options,
                    help="These are date/timeframe pairs where the satellite data shows some areas were mostly clear and others mostly cloudy.",
                )
                selected_win_idx = win_options.index(selected_win_label)
                selected_win = _contrast_windows[selected_win_idx]

                contrast_condition = st.radio(
                    "What did your trailcam look like during this window?",
                    ["Cloudy", "Sunny"],
                    horizontal=True,
                    help="'Cloudy' = overcast / no shadows visible. 'Sunny' = bright, shadows present.",
                )
                contrast_notes = st.text_area("Notes (optional)", height=70)
            else:
                selected_win = None
                contrast_condition = "Cloudy"
                contrast_notes = ""

        submitted = st.form_submit_button(
            "Add Observation", use_container_width=True, type="primary"
        )

        if submitted:
            if obs_type == "Fog Detection":
                date_str    = str(obs_date)
                t_start_str = time_start.strftime("%H:%M")
                t_end_str   = time_end.strftime("%H:%M")

                if t_start_str == t_end_str:
                    st.error("Start and end times cannot be the same.")
                else:
                    is_duplicate = any(
                        o["date"] == date_str and o["time_start"] == t_start_str and o["time_end"] == t_end_str
                        for o in st.session_state.observations
                    )
                    if is_duplicate:
                        st.error(
                            f"A fog observation for **{date_str}** from **{t_start_str}** to **{t_end_str}** already exists. "
                            "Edit or delete it in the Observation Log tab."
                        )
                    else:
                        new_obs = {
                            "id":             str(uuid.uuid4()),
                            "date":           date_str,
                            "time_start":     t_start_str,
                            "time_end":       t_end_str,
                            "fog_observed":   bool(fog_observed),
                            "notes":          notes.strip(),
                            "photo_filename": photo_filename.strip(),
                            "created_at":     datetime.now().isoformat(),
                        }
                        st.session_state.observations.append(new_obs)
                        st.session_state.observations.sort(key=lambda x: x["date"])
                        save_observations(st.session_state.observations)
                        invalidate_scores()
                        st.success(f"Fog observation added for **{date_str}** ({t_start_str}–{t_end_str}, fog={'Yes' if fog_observed else 'No'}).")
                        st.rerun()

            elif obs_type == "Cloud Differential":
                date_str    = str(obs_date)
                t_start_str = time_start.strftime("%H:%M")
                t_end_str   = time_end.strftime("%H:%M")

                if t_start_str == t_end_str:
                    st.error("Start and end times cannot be the same.")
                else:
                    is_duplicate = any(
                        o["date"] == date_str and o["time_start"] == t_start_str and o["time_end"] == t_end_str
                        for o in st.session_state.cloud_obs_list
                    )
                    if is_duplicate:
                        st.error(
                            f"A cloud observation for **{date_str}** from **{t_start_str}** to **{t_end_str}** already exists. "
                            "Delete it in the Observation Log tab to replace it."
                        )
                    else:
                        new_obs = {
                            "id":         str(uuid.uuid4()),
                            "date":       date_str,
                            "time_start": t_start_str,
                            "time_end":   t_end_str,
                            "relation":   cloud_relation,
                        }
                        st.session_state.cloud_obs_list.append(new_obs)
                        st.session_state.cloud_obs_list.sort(key=lambda x: x["date"])
                        save_cloud_observations(st.session_state.cloud_obs_list)
                        invalidate_scores()
                        st.success(f"Cloud observation added for **{date_str}**.")
                        st.rerun()

            else:  # High-Contrast Cloud
                if selected_win is None:
                    st.error("No high-contrast windows available to log.")
                else:
                    obs_date_str = selected_win["date"]
                    obs_window   = selected_win["window"]
                    is_duplicate = any(
                        o["date"] == obs_date_str and o["window"] == obs_window
                        for o in st.session_state.contrast_obs_list
                    )
                    if is_duplicate:
                        st.error(
                            f"A High-Contrast observation for **{obs_date_str} ({obs_window})** already exists. "
                            "Delete it in the Observation Log tab to replace it."
                        )
                    else:
                        new_obs = {
                            "id":                 str(uuid.uuid4()),
                            "date":               obs_date_str,
                            "window":             obs_window,
                            "trailcam_condition": contrast_condition,
                            "notes":              contrast_notes.strip(),
                            "spread":             selected_win["spread"],
                            "pct_clear":          selected_win["pct_clear"],
                            "pct_cloudy":         selected_win["pct_cloudy"],
                            "created_at":         datetime.now().isoformat(),
                        }
                        st.session_state.contrast_obs_list.append(new_obs)
                        st.session_state.contrast_obs_list.sort(key=lambda x: x["date"])
                        save_contrast_observations(st.session_state.contrast_obs_list)
                        invalidate_scores()
                        st.success(
                            f"High-Contrast observation added: **{obs_date_str} ({obs_window})** — "
                            f"trailcam was **{contrast_condition}**."
                        )
                        st.rerun()

    # Tip
    if len(st.session_state.observations) > 0:
        st.info(
            f"You have **{len(st.session_state.observations)} observation(s)** logged. "
            "Head to the **Scoring Map** tab to see which grid points match your data."
        )


# ─── TAB 2: Observation Log ───────────────────────────────────────────────────
with tab2:
    st.header("Observation Log")

    if not st.session_state.observations:
        st.info("No observations yet. Add them in the **Log Observation** tab.")
    else:
        obs_df = pd.DataFrame(st.session_state.observations)

        # Show editable table — only fog/notes/photo are editable
        edit_df = obs_df[["date", "time_start", "time_end", "fog_observed", "photo_filename", "notes"]].copy()
        edit_df.columns = ["Date", "Start", "End", "Fog?", "Photo", "Notes"]

        st.caption(
            f"{len(obs_df)} observations — "
            f"Fog: {obs_df['fog_observed'].sum()}  |  "
            f"Clear: {(~obs_df['fog_observed']).sum()}  |  "
            f"Date range: {obs_df['date'].min()} to {obs_df['date'].max()}"
        )

        edited_df = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Date":  st.column_config.TextColumn("Date",  disabled=True, width="small"),
                "Start": st.column_config.TextColumn("Start", disabled=True, width="small"),
                "End":   st.column_config.TextColumn("End",   disabled=True, width="small"),
                "Fog?":  st.column_config.CheckboxColumn("Fog?", width="small"),
                "Photo": st.column_config.TextColumn("Photo Filename", width="medium"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
        )

        col_save, col_gap = st.columns([1, 4])
        with col_save:
            if st.button("Save Changes", type="primary", use_container_width=True):
                changed = False
                for i, row in edited_df.iterrows():
                    orig = st.session_state.observations[i]
                    new_fog   = bool(row["Fog?"])
                    new_photo = str(row["Photo"]).strip() if row["Photo"] else ""
                    new_notes = str(row["Notes"]).strip() if row["Notes"] else ""

                    if (orig["fog_observed"] != new_fog or
                            orig["photo_filename"] != new_photo or
                            orig["notes"] != new_notes):
                        orig["fog_observed"]   = new_fog
                        orig["photo_filename"] = new_photo
                        orig["notes"]          = new_notes
                        changed = True

                if changed:
                    save_observations(st.session_state.observations)
                    invalidate_scores()
                    st.success("Changes saved. Scores will be recomputed.")
                    st.rerun()
                else:
                    st.info("No changes detected.")

        # Delete section
        st.markdown("---")
        st.subheader("Delete Fog Observations")
        all_dates = [o["date"] for o in st.session_state.observations]
        to_delete = st.multiselect(
            "Select dates to remove",
            options=all_dates,
            help="Observations with these dates will be permanently removed.",
        )
        if to_delete:
            if st.button(
                f"Delete {len(to_delete)} observation(s)",
                type="secondary",
                use_container_width=False,
            ):
                st.session_state.observations = [
                    o for o in st.session_state.observations if o["date"] not in to_delete
                ]
                save_observations(st.session_state.observations)
                invalidate_scores()
                st.success(f"Deleted {len(to_delete)} observation(s).")
                st.rerun()

    st.markdown("---")
    st.subheader("Cloud Observations")
    if not st.session_state.cloud_obs_list:
        st.info("No cloud observations yet. Add them in the **Log Observation** tab.")
    else:
        cloud_df = pd.DataFrame(st.session_state.cloud_obs_list)
        st.dataframe(cloud_df[["date", "time_start", "time_end", "relation"]], use_container_width=True, hide_index=True)
        
        all_cloud_dates = [o["date"] for o in st.session_state.cloud_obs_list]
        to_delete_cloud = st.multiselect(
            "Select cloud observation dates to remove",
            options=all_cloud_dates,
            help="Cloud observations with these dates will be permanently removed.",
            key="delete_cloud"
        )
        if to_delete_cloud:
            if st.button(
                f"Delete {len(to_delete_cloud)} cloud observation(s)",
                type="secondary",
                use_container_width=False,
            ):
                st.session_state.cloud_obs_list = [
                    o for o in st.session_state.cloud_obs_list if o["date"] not in to_delete_cloud
                ]
                save_cloud_observations(st.session_state.cloud_obs_list)
                invalidate_scores()
                st.success(f"Deleted {len(to_delete_cloud)} cloud observation(s).")
                st.rerun()

    st.markdown("---")
    st.subheader("⚡ High-Contrast Cloud Observations")
    if not st.session_state.contrast_obs_list:
        st.info("No high-contrast cloud observations yet. Add them in the **Log Observation** tab by selecting 'High-Contrast Cloud'.")
    else:
        contrast_df = pd.DataFrame(st.session_state.contrast_obs_list)
        display_cols = ["date", "window", "trailcam_condition", "spread", "pct_clear", "pct_cloudy"]
        display_cols = [c for c in display_cols if c in contrast_df.columns]
        st.dataframe(
            contrast_df[display_cols].rename(columns={
                "date": "Date", "window": "Window", "trailcam_condition": "Trailcam",
                "spread": "Spread %", "pct_clear": "% Clear", "pct_cloudy": "% Cloudy",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Use date+window combo as the delete key so each row is uniquely addressable
        contrast_options = [
            f"{o['date']} | {o['window']} ({o.get('trailcam_condition', '?')})"
            for o in st.session_state.contrast_obs_list
        ]
        to_delete_contrast = st.multiselect(
            "Select high-contrast observations to remove",
            options=contrast_options,
            help="High-Contrast observations with these date/window combinations will be permanently removed.",
            key="delete_contrast",
        )
        if to_delete_contrast:
            if st.button(
                f"Delete {len(to_delete_contrast)} high-contrast observation(s)",
                type="secondary",
                use_container_width=False,
                key="btn_delete_contrast",
            ):
                # Identify indices to keep
                to_delete_set = set(to_delete_contrast)
                st.session_state.contrast_obs_list = [
                    o for o, label in zip(st.session_state.contrast_obs_list, contrast_options)
                    if label not in to_delete_set
                ]
                save_contrast_observations(st.session_state.contrast_obs_list)
                invalidate_scores()
                st.success(f"Deleted {len(to_delete_contrast)} high-contrast observation(s).")
                st.rerun()

# ─── TAB 3: Scoring Map ───────────────────────────────────────────────────────
with tab3:
    st.header("Location Scoring Map")

    fog_df = get_fog_df()

    if fog_df is None:
        st.warning("Select a satellite data CSV in the sidebar.")
    elif not st.session_state.observations:
        st.info("Add observations in the Log tab to populate the scoring map.")
    else:
        scores = get_scores()
        # Re-apply the Combined Score formula based on current component toggles.
        # Individual scores (PlantScore, CloudScore, ContrastScore) stay cached;
        # only CombinedScore is recalculated here so toggling is instant.
        if scores is not None and not scores.empty:
            scores = recompute_combined(scores)

        if scores is None or scores.empty:
            st.warning("No scores available. Check that your observation dates overlap with the satellite data.")
        else:
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


# ─── TAB 4: Diagnostics ──────────────────────────────────────────────────────
with tab4:
    st.header("Point Diagnostics")
    st.caption(
        "Select a specific grid point and see exactly how it matched "
        "or missed each of your observations."
    )

    fog_df = get_fog_df()

    if fog_df is None:
        st.warning("Select a satellite data CSV in the sidebar.")
    elif not st.session_state.observations:
        st.info("Add observations first.")
    else:
        scores = get_scores()
        if scores is not None and not scores.empty:
            scores = recompute_combined(scores)

        col_a, col_b, col_c = st.columns([1.5, 1.5, 2])

        # Pre-fill with top candidate
        default_lat = float(scores.iloc[0]["Lat"]) if (scores is not None and not scores.empty) else 40.31
        default_lon = float(scores.iloc[0]["Lon"]) if (scores is not None and not scores.empty) else -75.13

        with col_a:
            diag_lat = st.number_input("Latitude",  value=default_lat, format="%.4f", step=0.05)
        with col_b:
            diag_lon = st.number_input("Longitude", value=default_lon, format="%.4f", step=0.05)
        with col_c:
            st.write("")
            st.write("")
            run_btn = st.button("Run Diagnostics", type="primary")

        # Quick-pick from top candidates
        if scores is not None and not scores.empty:
            with st.expander("Quick-pick from top candidates"):
                top_opts = [
                    f"#{i+1}  ({r['Lat']:.4f}, {r['Lon']:.4f})  -  {r['MatchRate']:.1%} match"
                    for i, r in scores.head(10).iterrows()
                ]
                picked = st.selectbox("Pick a candidate", top_opts, label_visibility="collapsed")
                if picked and st.button("Load This Point"):
                    idx = int(picked.split("#")[1].split(" ")[0]) - 1
                    diag_lat = float(scores.iloc[idx]["Lat"])
                    diag_lon = float(scores.iloc[idx]["Lon"])
                    run_btn  = True

        if run_btn:
            diag_df = get_point_diagnostics(
                diag_lat, diag_lon,
                st.session_state.observations,
                fog_df,
                fog_threshold=st.session_state.fog_threshold,
            )

            if diag_df.empty:
                st.warning("No data returned. Check that the lat/lon matches a grid point.")
            else:
                valid     = diag_df[diag_df["Result"] != "No Data"]
                n_match   = (valid["Result"] == "Match").sum()
                n_total   = len(valid)
                match_rate = n_match / n_total if n_total > 0 else 0.0

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Observations",  n_total)
                d2.metric("Matches",       n_match)
                d3.metric("Misses",        n_total - n_match)
                d4.metric("Match Rate",    f"{match_rate:.1%}")

                st.markdown("---")

                # Style the Result column
                def _style_result(val: str) -> str:
                    if val == "Match":
                        return "background-color: #0d2a0d; color: #4ade80; font-weight:600"
                    elif val == "Miss":
                        return "background-color: #2a0d0d; color: #f87171; font-weight:600"
                    elif val == "No Data":
                        return "color: #555"
                    return ""

                def _style_fog_obs(val: str) -> str:
                    if val == "Yes":
                        return "color: #00e5ff; font-weight:600"
                    elif val == "No":
                        return "color: #4ade80; font-weight:600"
                    return ""

                styled = (
                    diag_df.style
                    .map(_style_result,  subset=["Result"])
                    .map(_style_fog_obs, subset=["Fog_Observed"])
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Mini match timeline
                valid_copy = valid.copy()
                valid_copy["match_val"] = valid_copy["Result"].map({"Match": 1, "Miss": 0})
                valid_copy["date_dt"]   = pd.to_datetime(valid_copy["Date"])

                fig_timeline = px.bar(
                    valid_copy,
                    x="date_dt",
                    y="match_val",
                    color="Result",
                    color_discrete_map={"Match": "#4ade80", "Miss": "#f87171"},
                    labels={"date_dt": "Date", "match_val": "Match"},
                    title=f"Match History for ({diag_lat:.4f}, {diag_lon:.4f})",
                    height=250,
                )
                fig_timeline.update_layout(
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    font_color="#e6edf3",
                    yaxis=dict(tickvals=[0, 1], ticktext=["Miss", "Match"], range=[-0.1, 1.3]),
                    showlegend=True,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)


# ─── TAB 5: Data Status ───────────────────────────────────────────────────────
with tab5:
    st.header("Data Status")

    # ── Coverage overview ─────────────────────────────────────────────────────
    fog_df_status = get_fog_df()
    obs_dates_all = {
        o["date"]
        for o in (
            st.session_state.observations
            + st.session_state.cloud_obs_list
            + st.session_state.contrast_obs_list
        )
    }

    fog_dates_all: set[str] = set()
    if fog_df_status is not None and not fog_df_status.empty:
        if "CloudCover_Pct" not in fog_df_status.columns:
            st.error("⚠️ Your satellite data is outdated and missing cloud cover metrics. A full re-fetch is required.")
        else:
            fog_dates_all = {str(d) for d in fog_df_status["Timestamp"].dt.date}

    covered = obs_dates_all & fog_dates_all
    missing_from_csv = sorted(obs_dates_all - fog_dates_all)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Satellite Days in CSV",        len(fog_dates_all))
    m2.metric("Observations Logged",           len(obs_dates_all))
    m3.metric("Observations with Data",        len(covered))
    m4.metric("Missing from CSV",              len(missing_from_csv),
              delta=f"-{len(missing_from_csv)}" if missing_from_csv else None,
              delta_color="inverse")

    if fog_dates_all:
        sorted_fog = sorted(fog_dates_all)
        c_a, c_b = st.columns(2)
        c_a.info(f"Data starts: **{sorted_fog[0]}**")
        c_b.info(f"Data ends  : **{sorted_fog[-1]}**")

    if missing_from_csv:
        st.warning(
            f"**{len(missing_from_csv)} observation date(s) are not yet in the satellite data.** "
            "Use the button below to fetch them."
        )
        st.write("Missing: " + ", ".join(missing_from_csv))
    elif obs_dates_all:
        st.success("All observation dates have satellite data. You're good to score!")

    if fog_dates_all:
        with st.expander(f"View all {len(fog_dates_all)} dates in CSV"):
            n_cols = 5
            sorted_fog_list = sorted(fog_dates_all)
            chunks = [sorted_fog_list[i:i+n_cols] for i in range(0, len(sorted_fog_list), n_cols)]
            st.dataframe(
                pd.DataFrame(chunks), use_container_width=True, hide_index=True,
                column_config={str(i): st.column_config.TextColumn("") for i in range(n_cols)},
            )

    # ── High-Contrast Days Panel ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Detected High-Contrast Days")
    st.caption(
        "These are date/timeframe windows in your satellite data where some areas were mostly clear "
        "while others were mostly cloudy simultaneously — ideal days to check your trailcam!"
    )

    if fog_df_status is None or fog_df_status.empty or "CloudCover_Pct" not in fog_df_status.columns:
        st.info("No satellite data with cloud cover available. Fetch data first.")
    else:
        _hc_days = detect_high_contrast_days(fog_df_status)
        if not _hc_days:
            st.info(
                "No high-contrast days found in current satellite data. "
                "The data may not have enough spatial variation, or the thresholds (30% clear / 70% cloudy) "
                "were not met for any date/window combination."
            )
        else:
            hc_df = pd.DataFrame(_hc_days)
            # Already logged indicators
            logged_keys = {
                (o["date"], o["window"]) for o in st.session_state.contrast_obs_list
            }
            hc_df["Logged?"] = hc_df.apply(
                lambda r: "✅ Yes" if (r["date"], r["window"]) in logged_keys else "—",
                axis=1,
            )
            st.dataframe(
                hc_df.rename(columns={
                    "date": "Date", "window": "Window",
                    "spread": "Spread %", "pct_clear": "% Clear",
                    "pct_cloudy": "% Cloudy", "min_cloud": "Min Cloud %",
                    "max_cloud": "Max Cloud %", "n_points": "Grid Points",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Logged?": st.column_config.TextColumn("Logged?", width="small"),
                },
            )
            st.caption(
                f"Found **{len(_hc_days)} high-contrast window(s)**. "
                f"**{sum(1 for h in _hc_days if (h['date'], h['window']) in logged_keys)}** already logged. "
                "Switch to **Log Observation > High-Contrast Cloud** to record your trailcam findings for any of these."
            )

    # ── Fetch Date Range for High-Contrast Scanning ───────────────────────────
    st.markdown("---")
    st.subheader("📅 Fetch Date Range (High-Contrast Scan)")
    st.caption(
        "Pull satellite data for every day in a date range — no observations required. "
        "This fills your CSV so the ⚡ detector above can surface high-contrast windows "
        "across the full range."
    )

    _hc_col1, _hc_col2 = st.columns(2)
    with _hc_col1:
        _range_start = st.date_input(
            "Start Date",
            value=date(2026, 5, 27),
            key="hc_range_start",
        )
    with _hc_col2:
        _range_end = st.date_input(
            "End Date",
            value=date.today(),
            key="hc_range_end",
        )

    if _range_start > _range_end:
        st.error("Start date must be before end date.")
    else:
        # Count how many of those days are already fully in the CSV
        _range_dates = pd.date_range(_range_start, _range_end, freq="D")
        _range_date_strs = [str(d.date()) for d in _range_dates]
        _already_have = fog_dates_all & set(_range_date_strs)
        _need = set(_range_date_strs) - _already_have
        _n_days = len(_range_dates)

        rcol1, rcol2 = st.columns(2)
        rcol1.metric("Days in Range", _n_days)
        rcol2.metric("Already Fetched", len(_already_have), delta=f"{len(_need)} missing", delta_color="inverse" if _need else "off")

        if not _need:
            st.success(f"All {_n_days} days already in satellite data — the ⚡ detector above has full coverage!")
        else:
            _grid_for_range = generate_grid(
                center_lat=st.session_state.grid_center_lat,
                center_lon=st.session_state.grid_center_lon,
                radius_km=st.session_state.grid_radius_mi * 1.60934,
            )
            _est_min = max(1, round(len(_grid_for_range) * 0.13 / 60))
            st.caption(
                f"**{len(_need)} day(s)** missing from CSV across **{len(_grid_for_range)} grid points**. "
                f"Estimated fetch time: **~{_est_min} min**."
            )

            if st.button("Fetch Date Range", type="primary", key="btn_fetch_range"):
                # Build synthetic obs dicts — just need the date field for smart_fetch
                _synthetic_obs = [{"date": d} for d in _range_date_strs]

                _existing_for_range = get_fog_df()
                _, _missing_pts = get_missing_data_points(
                    _synthetic_obs, _existing_for_range, _grid_for_range
                )

                if not _missing_pts:
                    st.success("Nothing to fetch — all dates already present for the full grid!")
                else:
                    _prog_bar   = st.progress(0, text="Starting...")
                    _log_holder = st.empty()
                    _range_log_lines: list[str] = []

                    def _rlog(msg: str) -> None:
                        _range_log_lines.append(msg)
                        _log_holder.markdown("\n".join(f"- {m}" for m in _range_log_lines[-6:]))

                    def _ron_point(done: int, total: int) -> None:
                        _prog_bar.progress(done / total, text=f"Fetching: {done} / {total} points")

                    _new_df, _n_fetched, _result_msg = smart_fetch(
                        _synthetic_obs,
                        center_lat=st.session_state.grid_center_lat,
                        center_lon=st.session_state.grid_center_lon,
                        radius_km=st.session_state.grid_radius_mi * 1.60934,
                        export_dir=get_export_dir(),
                        on_point_progress=_ron_point,
                        log_fn=_rlog,
                    )

                    if _n_fetched > 0:
                        _prog_bar.progress(1.0, text="Done!")
                        st.success(_result_msg)
                        st.cache_data.clear()
                        st.session_state.selected_csv = _MASTER_CSV
                        invalidate_scores()
                        st.rerun()
                    else:
                        st.info(_result_msg)

    # ── Fetch Missing Data (observation-driven) ────────────────────────────────
    st.markdown("---")
    st.subheader("Fetch Missing Data")

    if not st.session_state.observations and not st.session_state.cloud_obs_list:
        st.info("Log some observations first, then fetch their satellite data here.")
    else:
        current_grid_for_fetch = generate_grid(
            center_lat=st.session_state.grid_center_lat,
            center_lon=st.session_state.grid_center_lon,
            radius_km=st.session_state.grid_radius_mi * 1.60934
        )
        existing_df_for_fetch = get_fog_df()
        all_dates, missing_points = get_missing_data_points(
            st.session_state.observations + st.session_state.cloud_obs_list,
            existing_df_for_fetch,
            current_grid_for_fetch
        )

        if not missing_points:
            st.success("Nothing to fetch — all observation dates already have satellite data for the current grid.")
        else:
            n_pts = len(missing_points)
            est_min = max(1, round(n_pts * 0.13 / 60))
            st.caption(
                f"Your grid requires **{len(current_grid_for_fetch)} total points**.\n\n"
                f"We found **{n_pts} point(s)** missing data for your observation dates.\n\n"
                f"Estimated time to fetch: **~{est_min} min**."
            )

            if st.button("Fetch Missing Data for Current Grid", type="primary",
                         use_container_width=False):

                progress_bar   = st.progress(0, text="Starting...")
                log_placeholder = st.empty()
                _log_lines: list[str] = []

                def _log(msg: str) -> None:
                    _log_lines.append(msg)
                    # Show last 6 messages
                    log_placeholder.markdown(
                        "\n".join(f"- {m}" for m in _log_lines[-6:])
                    )

                def _on_point(done: int, total: int) -> None:
                    progress_bar.progress(
                        done / total,
                        text=f"Fetching grid points: {done} / {total}",
                    )

                new_fog_df, n_fetched, result_msg = smart_fetch(
                    st.session_state.observations + st.session_state.cloud_obs_list,
                    center_lat=st.session_state.grid_center_lat,
                    center_lon=st.session_state.grid_center_lon,
                    radius_km=st.session_state.grid_radius_mi * 1.60934,
                    export_dir=get_export_dir(),
                    on_point_progress=_on_point,
                    log_fn=_log,
                )

                if n_fetched > 0:
                    progress_bar.progress(1.0, text="Fetch complete!")
                    st.success(result_msg)
                    # Reload the master CSV and reset everything
                    st.cache_data.clear()
                    st.session_state.selected_csv = _MASTER_CSV
                    invalidate_scores()
                    st.rerun()
                else:
                    st.info(result_msg)


# ─── TAB 6: Export ────────────────────────────────────────────────────────────
with tab6:
    st.header("Export")

    scores_ex = get_scores()
    if scores_ex is not None and not scores_ex.empty:
        scores_ex = recompute_combined(scores_ex)

    # Scored grid points
    st.subheader("Scored Grid Points")
    if scores_ex is not None and not scores_ex.empty:
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("**CSV** — import as spreadsheet or delimited text layer in QGIS")
            csv_bytes = scores_ex.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download  scored_points.csv",
                data=csv_bytes,
                file_name="veil_scored_points.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_e2:
            st.markdown("**GeoJSON** — drag into QGIS, ArcGIS, JOSM, Google Earth, etc.")
            geojson_dict  = scores_to_geojson(scores_ex)
            geojson_bytes = json.dumps(geojson_dict, indent=2).encode("utf-8")
            st.download_button(
                "Download  scored_points.geojson",
                data=geojson_bytes,
                file_name="veil_scored_points.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

        st.caption(
            "QGIS tip: Layer -> Add Layer -> Add Vector Layer, "
            "or simply drag the .geojson file onto the QGIS canvas. "
            "Style by the 'MatchRate' field using a graduated renderer."
        )
    else:
        st.info(
            "Scores not yet computed. Visit the **Scoring Map** tab first, "
            "or click **Re-score Now** in the sidebar."
        )

    st.markdown("---")

    # Observations
    st.subheader("Observation Log")
    if st.session_state.observations:
        col_o1, col_o2 = st.columns(2)
        obs_df_ex = pd.DataFrame(st.session_state.observations).drop(
            columns=["id", "created_at"], errors="ignore"
        )

        with col_o1:
            st.markdown("**CSV** — open in Excel / import elsewhere")
            st.download_button(
                "Download  observations.csv",
                data=obs_df_ex.to_csv(index=False).encode("utf-8"),
                file_name="veil_observations.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_o2:
            st.markdown("**JSON** — re-import or back up your observation log")
            st.download_button(
                "Download  observations.json",
                data=json.dumps(st.session_state.observations, indent=2).encode("utf-8"),
                file_name="veil_observations.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Cloud Sync & Backups")
    st.caption("Backup all your hunts to your remote GitHub repository.")
    
    import subprocess
    
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        remote_url = st.text_input("GitHub Remote URL", value="https://github.com/neumetal/Veil-hunts.git", key="git_remote")
    with col_g2:
        st.write("")
        st.write("")
        if st.button("Save Remote", use_container_width=True):
            try:
                res = subprocess.run(["git", "remote", "-v"], cwd=_HERE, capture_output=True, text=True)
                if "origin" in res.stdout:
                    subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=_HERE)
                else:
                    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=_HERE)
                st.toast("Remote updated!")
            except Exception as e:
                st.error(f"Failed to set remote: {e}")
                
    if st.button("🚀 Backup All Hunts to GitHub", type="primary"):
        with st.spinner("Pushing to GitHub..."):
            try:
                subprocess.run(["git", "add", "."], cwd=_HERE, check=True)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # git commit returns non-zero if there are no changes, ignore errors
                subprocess.run(["git", "commit", "-m", f"Backup: {ts}"], cwd=_HERE)
                
                # We need to branch and push properly. Force pushing isn't needed, but it helps avoid merge conflicts if they wipe it.
                res = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=_HERE, capture_output=True, text=True)
                
                # If "main" doesn't match local branch name, try pushing head
                if "error: src refspec main does not match any" in res.stderr:
                    res = subprocess.run(["git", "push", "-u", "origin", "HEAD:main"], cwd=_HERE, capture_output=True, text=True)
                    
                if res.returncode == 0 or "Everything up-to-date" in res.stderr:
                    st.success("Successfully backed up to GitHub!")
                else:
                    st.error(f"Push failed: {res.stderr}")
            except Exception as e:
                st.error(f"Backup error: {e}")

    try:
        log_res = subprocess.run(["git", "log", "--pretty=format:%h|%s", "-n", "10"], cwd=_HERE, capture_output=True, text=True)
        if log_res.returncode == 0 and log_res.stdout:
            commits = log_res.stdout.strip().split("\n")
            if commits:
                st.markdown("#### Restore Previous Backup")
                col_r1, col_r2 = st.columns([3, 1])
                with col_r1:
                    selected_commit = st.selectbox("Select a past backup", commits, label_visibility="collapsed")
                with col_r2:
                    if st.button("Restore", use_container_width=True):
                        commit_hash = selected_commit.split("|")[0]
                        with st.spinner(f"Restoring to {commit_hash}..."):
                            subprocess.run(["git", "checkout", commit_hash, "--", "hunts/"], cwd=_HERE, check=True)
                            switch_hunt(st.session_state.current_hunt)  # Reload state
                            st.success("Restored successfully!")
                            st.rerun()
    except Exception:
        pass
    else:
        st.info("No observations to export yet.")

# ─── Map View ──────────────────────────────────────────────────────────────
st.header("Location Scoring Map")

fog_df = get_fog_df()
if fog_df is None:
    st.warning("Select a satellite data CSV in the sidebar.")
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
# ─── TAB 4: Diagnostics ──────────────────────────────────────────────────────
