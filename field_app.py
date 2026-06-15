"""
field_app.py  –  Veil Finder Field Viewer (read-only, cloud-safe)

Streamlit app for use in the field. Shows the scoring map, lets you adjust
weights and overlay visibility, and lets you copy coordinates to Google Maps.

Run:
    streamlit run field_app.py
"""
import os
import sys
import json
import glob
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Backend imports (no importlib.reload – breaks Streamlit Cloud cold starts) ─
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "backend"))

from scorer import (
    score_all_observations,
    compute_plant_scores,
    score_cloud_observations,
    score_contrast_observations,
    haversine_distance,
)
from fetcher import load_master_csv
from grid_utils import generate_grid
from geometry_utils import get_rotated_corners

# ── Constants ──────────────────────────────────────────────────────────────────
HUNTS_DIR = os.path.join(_HERE, "hunts")

SCORE_COLORSCALE = [
    [0.00, "rgba(60,60,70,0.35)"],
    [0.40, "rgba(60,60,70,0.35)"],
    [0.55, "#e8c830"],
    [0.72, "#ff7b00"],
    [0.88, "#ff2800"],
    [1.00, "#ff0055"],
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veil Finder Field",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
  div[data-testid="stMetric"] {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 8px; padding: 0.6rem 1rem;
  }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
  div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
  .stAlert { border-radius: 8px; }
  hr { border-color: #30363d; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Path helpers ───────────────────────────────────────────────────────────────
def get_hunt_dir() -> str:
    return os.path.join(HUNTS_DIR, st.session_state.get("current_hunt", "veil_eight"))

def get_settings_file() -> str:
    return os.path.join(get_hunt_dir(), "settings.json")

def get_export_dir() -> str:
    return os.path.join(get_hunt_dir(), "scans_export")

def get_master_parquet() -> str:
    return os.path.join(get_export_dir(), "fog_master.parquet")

def get_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "observations.json")

def get_cloud_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "cloud_observations.json")

def get_contrast_obs_file() -> str:
    return os.path.join(get_hunt_dir(), "contrast_observations.json")

def get_plants_file() -> str:
    return os.path.join(get_hunt_dir(), "plants.json")

# ── I/O helpers ────────────────────────────────────────────────────────────────
def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def load_settings() -> dict:
    return load_json(get_settings_file(), {})

def save_settings(d: dict) -> None:
    fpath = get_settings_file()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def get_hunts() -> list:
    os.makedirs(HUNTS_DIR, exist_ok=True)
    return [d for d in os.listdir(HUNTS_DIR)
            if os.path.isdir(os.path.join(HUNTS_DIR, d)) and not d.startswith(".")]

def invalidate_scores():
    st.session_state.scores = None

# ── Session state init ─────────────────────────────────────────────────────────
if "current_hunt" not in st.session_state:
    st.session_state.current_hunt = get_hunts()[0] if get_hunts() else "veil_eight"

_settings = load_settings()

if "grid_center_lat" not in st.session_state:
    st.session_state.grid_center_lat = _settings.get("grid_center_lat", 40.31)
if "grid_center_lon" not in st.session_state:
    st.session_state.grid_center_lon = _settings.get("grid_center_lon", -75.13)
if "grid_radius_mi" not in st.session_state:
    st.session_state.grid_radius_mi  = _settings.get("grid_radius_mi", 40.0)
if "map_transparency" not in st.session_state:
    st.session_state.map_transparency = 20
if "weight_fog" not in st.session_state:
    st.session_state.weight_fog = 10
if "weight_plant" not in st.session_state:
    st.session_state.weight_plant = 10
if "weight_cloud" not in st.session_state:
    st.session_state.weight_cloud = 10
if "weight_contrast" not in st.session_state:
    st.session_state.weight_contrast = 10
if "fog_threshold" not in st.session_state:
    st.session_state.fog_threshold = 5
if "scores" not in st.session_state:
    st.session_state.scores = None
if "map_overlays" not in st.session_state:
    st.session_state.map_overlays = _settings.get("map_overlays", [])

# Load observations + plant data
_obs = load_json(get_obs_file(), [])
if "observations" not in st.session_state:
    st.session_state.observations = _obs

_cloud_obs = load_json(get_cloud_obs_file(), [])
if "cloud_obs_list" not in st.session_state:
    st.session_state.cloud_obs_list = _cloud_obs

_contrast_obs = load_json(get_contrast_obs_file(), [])
if "contrast_obs_list" not in st.session_state:
    st.session_state.contrast_obs_list = _contrast_obs

_plants_data = load_json(get_plants_file(), {})
if "plant_obs_dict" not in st.session_state:
    st.session_state.plant_obs_dict = _plants_data.get("plant_obs_dict", {})

# Auto-select master parquet
if "selected_csv" not in st.session_state:
    mp = get_master_parquet()
    st.session_state.selected_csv = mp if os.path.exists(mp) else None

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_fog_data(path: str, mtime: float) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "Timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def get_fog_df():
    p = st.session_state.selected_csv
    if p and os.path.exists(p):
        return load_fog_data(p, os.path.getmtime(p))
    return None

# ── Scoring ────────────────────────────────────────────────────────────────────
def recompute_combined(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    mr = df.get("MatchRate", pd.Series(0.0, index=df.index)).fillna(0.0)
    mr_scaled = (mr - mr.min()) / (mr.max() - mr.min()) if mr.max() > mr.min() else mr.copy()

    total_w = float(st.session_state.weight_fog)
    raw = mr_scaled * total_w

    if "PlantScore" in df.columns:
        raw += df["PlantScore"].fillna(0.0) * st.session_state.weight_plant
        total_w += st.session_state.weight_plant
    if "CloudScore" in df.columns:
        raw += df["CloudScore"].fillna(0.0) * st.session_state.weight_cloud
        total_w += st.session_state.weight_cloud
    if "ContrastScore" in df.columns:
        raw += df["ContrastScore"].fillna(0.0) * st.session_state.weight_contrast
        total_w += st.session_state.weight_contrast

    df["CombinedScore"] = (raw / total_w if total_w > 0 else raw * 0.0).fillna(0.0)
    return df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)

def get_scores():
    if st.session_state.scores is not None:
        return st.session_state.scores

    fog_df = get_fog_df()
    if not st.session_state.observations:
        return None

    with st.spinner("Computing scores…"):
        current_grid = list(set(generate_grid(
            center_lat=st.session_state.grid_center_lat,
            center_lon=st.session_state.grid_center_lon,
            radius_km=st.session_state.grid_radius_mi * 1.60934,
        )))
        grid_df = pd.DataFrame(current_grid, columns=["Lat_g", "Lon_g"])
        grid_df["Lat"] = grid_df["Lat_g"]
        grid_df["Lon"] = grid_df["Lon_g"]

        base_scores = pd.DataFrame()

        if fog_df is not None and not fog_df.empty:
            grid_set = set(zip(grid_df["Lat_g"], grid_df["Lon_g"]))
            fog_mask = pd.Series(
                list(zip(fog_df["Lat"].round(4), fog_df["Lon"].round(4)))
            ).isin(grid_set)
            filtered = fog_df[fog_mask.values].copy()
            filtered["Lat_g"] = filtered["Lat"].round(4)
            filtered["Lon_g"] = filtered["Lon"].round(4)

            raw = score_all_observations(
                st.session_state.observations,
                filtered,
                fog_threshold=st.session_state.fog_threshold,
            )
            if not raw.empty:
                raw["Lat_g"] = raw["Lat"].round(4)
                raw["Lon_g"] = raw["Lon"].round(4)
                raw = raw.drop(columns=["Lat", "Lon"])
                base_scores = pd.merge(grid_df, raw, on=["Lat_g", "Lon_g"], how="left")
                base_scores = base_scores.drop(columns=["Lat_g", "Lon_g"])
            else:
                base_scores = grid_df.drop(columns=["Lat_g", "Lon_g"])
        else:
            base_scores = grid_df.drop(columns=["Lat_g", "Lon_g"])

        if "MatchRate" not in base_scores.columns:
            base_scores["MatchRate"]    = np.nan
            base_scores["Confidence_Z"] = np.nan
            base_scores["Matches"]      = 0
            base_scores["ObsCount"]     = len(st.session_state.observations)
        if "Elevation_m" not in base_scores.columns:
            base_scores["Elevation_m"] = np.nan
            base_scores["IsValley"]    = False

        base_scores = compute_plant_scores(
            base_scores,
            st.session_state.plant_obs_dict,
            influence_radius_mi=3.0,
            match_mode="Any",
        )
        base_scores = score_cloud_observations(
            base_scores, st.session_state.cloud_obs_list, fog_df
        )
        base_scores = score_contrast_observations(
            base_scores, st.session_state.contrast_obs_list, fog_df
        )

    st.session_state.scores = base_scores
    return st.session_state.scores

# ── Switch hunt helper ─────────────────────────────────────────────────────────
def switch_hunt(name: str):
    st.session_state.current_hunt   = name
    st.session_state.observations   = load_json(get_obs_file(), [])
    st.session_state.cloud_obs_list = load_json(get_cloud_obs_file(), [])
    st.session_state.contrast_obs_list = load_json(get_contrast_obs_file(), [])
    _s = load_settings()
    st.session_state.grid_center_lat = _s.get("grid_center_lat", 40.31)
    st.session_state.grid_center_lon = _s.get("grid_center_lon", -75.13)
    st.session_state.grid_radius_mi  = _s.get("grid_radius_mi", 40.0)
    st.session_state.map_overlays    = _s.get("map_overlays", [])
    mp = get_master_parquet()
    st.session_state.selected_csv    = mp if os.path.exists(mp) else None
    invalidate_scores()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🌫️ Veil Finder")
    st.caption("Field viewer — read only")
    st.markdown("---")

    # ── Hunt selector ──────────────────────────────────────────────────────────
    hunts = get_hunts()
    if st.session_state.current_hunt not in hunts:
        hunts.append(st.session_state.current_hunt)
    sel = st.selectbox("Active Hunt", hunts,
                       index=hunts.index(st.session_state.current_hunt))
    if sel != st.session_state.current_hunt:
        switch_hunt(sel)
        st.rerun()

    st.markdown("---")

    # ── Score weights ──────────────────────────────────────────────────────────
    with st.expander("⚖️ Score Weights", expanded=True):
        st.caption("Drag sliders to rebalance which layers matter most.")
        _changed = False

        _nf = st.slider("🌫️ Fog Match Rate", 0, 10,
                        int(st.session_state.weight_fog), key="fa_wfog")
        if _nf != st.session_state.weight_fog:
            st.session_state.weight_fog = _nf
            _changed = True

        if st.session_state.plant_obs_dict:
            _np = st.slider("🌿 Plant Proximity", 0, 10,
                            int(st.session_state.weight_plant), key="fa_wplant")
            if _np != st.session_state.weight_plant:
                st.session_state.weight_plant = _np
                _changed = True

        if st.session_state.cloud_obs_list:
            _nc = st.slider("⛅ Cloud Differential", 0, 10,
                            int(st.session_state.weight_cloud), key="fa_wcloud")
            if _nc != st.session_state.weight_cloud:
                st.session_state.weight_cloud = _nc
                _changed = True

        if st.session_state.contrast_obs_list:
            _nk = st.slider("⚡ Contrast Score", 0, 10,
                            int(st.session_state.weight_contrast), key="fa_wcontrast")
            if _nk != st.session_state.weight_contrast:
                st.session_state.weight_contrast = _nk
                _changed = True

        if _changed:
            st.rerun()

    # ── Map display settings ───────────────────────────────────────────────────
    with st.expander("🗺️ Map Settings", expanded=False):
        _nt = st.slider("Point Transparency (%)", 0, 95,
                        int(st.session_state.map_transparency), step=5,
                        key="fa_trans")
        if _nt != st.session_state.map_transparency:
            st.session_state.map_transparency = _nt
            st.rerun()

    # ── Overlays (visibility + opacity only) ──────────────────────────────────
    if st.session_state.map_overlays:
        with st.expander("🖼️ Map Overlays", expanded=False):
            for idx, ov in enumerate(st.session_state.map_overlays):
                st.markdown(f"**{ov.get('name', f'Overlay {idx+1}')}**")
                vis = st.checkbox("Visible", value=ov.get("visible", True),
                                  key=f"fa_ovis_{idx}")
                if vis != ov.get("visible", True):
                    st.session_state.map_overlays[idx]["visible"] = vis
                    _s2 = load_settings()
                    _s2["map_overlays"] = st.session_state.map_overlays
                    save_settings(_s2)
                    st.rerun()
                if vis:
                    n_op = st.slider("Opacity", 0.0, 1.0,
                                     float(ov.get("opacity", 0.5)), step=0.05,
                                     key=f"fa_oop_{idx}")
                    if n_op != ov.get("opacity", 0.5):
                        st.session_state.map_overlays[idx]["opacity"] = n_op
                        _s2 = load_settings()
                        _s2["map_overlays"] = st.session_state.map_overlays
                        save_settings(_s2)
                        st.rerun()
                st.markdown("---")

    # ── Re-score button ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Re-score", use_container_width=True):
        invalidate_scores()
        st.rerun()

    # ── Data status ────────────────────────────────────────────────────────────
    fog_df_check = get_fog_df()
    if fog_df_check is not None:
        st.success(f"✅ Satellite data loaded  ({len(fog_df_check):,} rows)")
    else:
        st.warning("⚠️ No satellite data for this hunt")
    n_obs = len(st.session_state.observations)
    st.caption(f"{n_obs} fog observations loaded")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — MAP VIEW
# ══════════════════════════════════════════════════════════════════════════════
st.header("📍 Location Scoring Map")

fog_df = get_fog_df()

if fog_df is None:
    st.warning("No satellite data found for this hunt. "
               "Make sure the master parquet has been synced from the main app.")
    st.stop()

if not st.session_state.observations:
    st.info("No observations loaded for this hunt. Add fog/clear observations in the main app first.")
    st.stop()

scores_raw = get_scores()
if scores_raw is None or scores_raw.empty:
    st.info("Scores not yet computed. Hit **Re-score** in the sidebar.")
    st.stop()

scores = recompute_combined(scores_raw)

# ── Summary metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Grid Points", f"{len(scores):,}")
top1 = scores.iloc[0]
m2.metric("Top Combined Score", f"{top1.get('CombinedScore', 0.0):.2f}")
m3.metric("Top Match Rate",     f"{float(top1.get('MatchRate', 0)):.1%}")
n_fog_obs = sum(1 for o in st.session_state.observations if o.get("fog_observed"))
m4.metric("Fog Observations", n_fog_obs)

st.markdown("---")

# ── Candidate picker (above map so selection is reflected on next rerun) ────────
if "fa_pin_idx" not in st.session_state:
    st.session_state.fa_pin_idx = 0

_pin_opts = [
    f"#{i+1}  {row['Lat']:.5f}, {row['Lon']:.5f}  (score {row.get('CombinedScore', row.get('MatchRate', 0)):.3f})"
    for i, row in scores.head(20).iterrows()
]
_pin_sel = st.selectbox(
    "📌 Highlight a candidate on the map",
    options=_pin_opts,
    index=min(st.session_state.fa_pin_idx, len(_pin_opts) - 1),
    key="fa_pin_select",
    help="Selecting a location here places a star marker on the map and loads its coordinates below.",
)
_pin_new_idx = _pin_opts.index(_pin_sel)
if _pin_new_idx != st.session_state.fa_pin_idx:
    st.session_state.fa_pin_idx = _pin_new_idx
    st.rerun()
_pinned_row = scores.iloc[st.session_state.fa_pin_idx]
_pinned_lat = float(_pinned_row["Lat"])
_pinned_lon = float(_pinned_row["Lon"])
_pinned_coords = f"{_pinned_lat:.6f}, {_pinned_lon:.6f}"

# ── Build the map figure ────────────────────────────────────────────────────────
alpha = 1.0 - (int(st.session_state.map_transparency) / 100.0)
map_df = scores.copy()
map_df["MatchRate"] = map_df["MatchRate"].fillna(0.0)

color_col = "CombinedScore" if "CombinedScore" in map_df.columns else "MatchRate"
_max_score = float(map_df[color_col].max()) if map_df[color_col].notna().any() else 1.0

# ── Score range filter ─────────────────────────────────────────────────────────
_fmin = float(map_df["CombinedScore"].min()) if "CombinedScore" in map_df.columns else 0.0
_fmax = float(map_df["CombinedScore"].max()) if "CombinedScore" in map_df.columns else 1.0
_fmax = _fmax if _fmax > _fmin else _fmin + 0.001

_frange = st.slider(
    "Filter by Combined Score",
    min_value=round(_fmin, 3),
    max_value=round(_fmax, 3),
    value=(round(_fmin, 3), round(_fmax, 3)),
    step=round((_fmax - _fmin) / 200, 4) or 0.001,
    help="Hide grid points outside this score range. Drag either end to focus the map.",
    key="fa_score_filter",
)
map_df = map_df[
    (map_df["CombinedScore"] >= _frange[0]) &
    (map_df["CombinedScore"] <= _frange[1])
]
if map_df.empty:
    st.warning("No points match the current score filter — try widening the range.")
    st.stop()

# Build hover text
def make_hover(row):
    parts = [
        f"<b>Lat:</b> {row['Lat']:.5f}",
        f"<b>Lon:</b> {row['Lon']:.5f}",
        f"<b>Match Rate:</b> {float(row.get('MatchRate', 0)):.1%}",
    ]
    if "Elevation_m" in row and not pd.isna(row.get("Elevation_m")):
        parts.append(f"<b>Elevation:</b> {row['Elevation_m']:.0f} m")
    if "CombinedScore" in row:
        parts.append(f"<b>Combined Score:</b> {row['CombinedScore']:.3f}")
    return "<br>".join(parts)

map_df["hover"] = map_df.apply(make_hover, axis=1)

auto_zoom = float(np.clip(13.5 - np.log2(st.session_state.grid_radius_mi), 3.0, 15.0))

fig = px.scatter_mapbox(
    map_df,
    lat="Lat",
    lon="Lon",
    color=color_col,
    color_continuous_scale=SCORE_COLORSCALE,
    range_color=[0, max(_max_score, 0.01)],
    zoom=auto_zoom,
    center={"lat": st.session_state.grid_center_lat,
            "lon": st.session_state.grid_center_lon},
    mapbox_style="carto-darkmatter",
    custom_data=["hover"],
    opacity=alpha,
    size_max=8,
)

fig.update_traces(
    hovertemplate="%{customdata[0]}<extra></extra>",
    marker=dict(size=7),
)

# ── Use persisted click state (separate key from the widget) ──────────────────
# Streamlit forbids writing to st.session_state[widget_key] directly.
# Instead we store the last clicked point under non-widget keys.
if "fa_clicked_lat" not in st.session_state:
    st.session_state.fa_clicked_lat = None
if "fa_clicked_lon" not in st.session_state:
    st.session_state.fa_clicked_lon = None

_clicked_lat = st.session_state.fa_clicked_lat
_clicked_lon = st.session_state.fa_clicked_lon

# Use whichever is set: map click > pinned candidate
_active_lat = _clicked_lat if _clicked_lat is not None else _pinned_lat
_active_lon = _clicked_lon if _clicked_lon is not None else _pinned_lon

# ── Plant observation pins ─────────────────────────────────────────────────────
if st.session_state.plant_obs_dict:
    _plant_colors = px.colors.qualitative.Plotly
    for _pidx, (_species, _obs_list) in enumerate(st.session_state.plant_obs_dict.items()):
        _sp_color = _plant_colors[_pidx % len(_plant_colors)]
        _nearby = [
            o for o in _obs_list
            if haversine_distance(_active_lat, _active_lon, o["lat"], o["lon"]) <= 3.0
        ]
        if _nearby:
            _odf = pd.DataFrame(_nearby)
            _gmaps_links = _odf.apply(
                lambda r: f"https://www.google.com/maps/search/?api=1&query={r['lat']:.6f},{r['lon']:.6f}",
                axis=1,
            )
            fig.add_trace(go.Scattermapbox(
                lat=_odf["lat"],
                lon=_odf["lon"],
                mode="markers",
                marker=dict(size=11, color=_sp_color, symbol="circle"),
                customdata=np.stack([
                    _odf["user"],
                    _odf.get("observed_on", pd.Series([""] * len(_odf))),
                    _odf["lat"].map("{:.5f}".format),
                    _odf["lon"].map("{:.5f}".format),
                    _gmaps_links,
                ], axis=-1),
                hovertemplate=(
                    f"<b>🌿 {_species}</b><br>"
                    "Observer: @%{customdata[0]}<br>"
                    "Date: %{customdata[1]}<br>"
                    "Location: %{customdata[2]}, %{customdata[3]}<br>"
                    "<a href='%{customdata[4]}'>🔗 Google Maps</a>"
                    "<extra></extra>"
                ),
                name=f"{_species} (nearby)",
            ))

# ── Star marker for selected/pinned candidate ──────────────────────────────────
fig.add_trace(go.Scattermapbox(
    lat=[_active_lat],
    lon=[_active_lon],
    mode="markers",
    marker=dict(size=18, color="#00e5ff", symbol="star"),
    hovertext=f"📌 Selected: {_active_lat:.6f}, {_active_lon:.6f}",
    hoverinfo="text",
    name="Selected",
    showlegend=False,
))

# ── Grid center marker ─────────────────────────────────────────────────────────
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

# ── Overlays ────────────────────────────────────────────────────────────────────
mapbox_layers = []
for ov in st.session_state.map_overlays:
    if not ov.get("visible", True):
        continue
    ov_dir   = os.path.join(get_hunt_dir(), "overlays")
    ov_path  = os.path.join(ov_dir, os.path.basename(ov.get("name", "")))
    if not os.path.exists(ov_path):
        continue
    with open(ov_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext  = os.path.splitext(ov_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    corners = get_rotated_corners(
        center_lat=float(ov["lat"]),
        center_lon=float(ov["lon"]),
        width_km=float(ov["width"]),
        height_km=float(ov["height"]),
        rotation_deg=float(ov["rotation"]),
        anchor_x=float(ov.get("anchor_x", 0.5)),
        anchor_y=float(ov.get("anchor_y", 0.5)),
    )
    mapbox_layers.append({
        "sourcetype": "image",
        "source": f"data:{mime};base64,{b64}",
        "coordinates": corners,
        "opacity": float(ov.get("opacity", 0.5)),
        "below": "traces",
    })

if mapbox_layers:
    fig.update_layout(mapbox_layers=mapbox_layers)

# ── Layout ─────────────────────────────────────────────────────────────────────
grid_rev = f"{st.session_state.grid_center_lat}_{st.session_state.grid_center_lon}_{st.session_state.grid_radius_mi}"
fig.update_layout(
    uirevision=grid_rev,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(
        title=dict(text="Score", font=dict(color="#e6edf3")),
        tickformat=".2f",
        len=0.55,
        bgcolor="#161b22",
        bordercolor="#30363d",
        borderwidth=1,
        tickfont=dict(color="#e6edf3"),
    ),
)

# ── Render map ────────────────────────────────────────────────────────────────
map_event = st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True},
    on_select="rerun",
    key="field_map_v2",
)
st.caption("💡 Click any grid point to show nearby plant pins and load coordinates. The 🔵 star marks your selected candidate.")

# ── Extract click from event and persist in non-widget state ───────────────────
clicked_coords = None
clicked_lat = None
clicked_lon = None
if map_event and hasattr(map_event, "selection"):
    _sel = map_event.selection if isinstance(map_event.selection, dict) else {}
    _pts = _sel.get("points", [])
    if _pts:
        _pt = _pts[0]
        _lat = _pt.get("lat") or _pt.get("y")
        _lon = _pt.get("lon") or _pt.get("x")
        if _lat is not None and _lon is not None:
            clicked_lat = float(_lat)
            clicked_lon = float(_lon)
            clicked_coords = f"{clicked_lat:.6f}, {clicked_lon:.6f}"
            # Persist so next rerun can draw plant pins before map renders
            if (clicked_lat != st.session_state.fa_clicked_lat or
                    clicked_lon != st.session_state.fa_clicked_lon):
                st.session_state.fa_clicked_lat = clicked_lat
                st.session_state.fa_clicked_lon = clicked_lon
                st.rerun()

st.markdown("---")
col_tbl, col_cpy = st.columns([2.5, 1.5])

with col_tbl:
    st.subheader("Top 20 Candidate Locations")
    show_cols = ["Lat", "Lon"]
    if "Elevation_m" in scores.columns:
        show_cols += ["Elevation_m"]
    if "IsValley" in scores.columns:
        show_cols += ["IsValley"]
    show_cols += ["MatchRate"]
    if "CombinedScore" in scores.columns:
        show_cols += ["CombinedScore"]

    top20 = scores.head(20)[show_cols].copy()
    top20.insert(0, "Rank", range(1, len(top20) + 1))
    # Build full Google Maps URLs so LinkColumn renders them as clickable links
    top20["Google Maps"] = top20.apply(
        lambda r: f"https://www.google.com/maps/search/?api=1&query={float(r['Lat']):.6f},{float(r['Lon']):.6f}",
        axis=1,
    )
    top20["MatchRate"] = top20["MatchRate"].map("{:.1%}".format)
    if "Elevation_m" in top20.columns:
        top20["Elevation_m"] = top20["Elevation_m"].map("{:.0f} m".format)
    if "CombinedScore" in top20.columns:
        top20["CombinedScore"] = top20["CombinedScore"].map("{:.3f}".format)
    st.dataframe(
        top20,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Google Maps": st.column_config.LinkColumn(
                "Google Maps",
                display_text="🔗 Open",
            )
        },
    )

with col_cpy:
    st.subheader("📍 Coordinate Copier")
    # Show clicked map point if available, otherwise show pinned candidate
    display_coords = clicked_coords if clicked_coords else _pinned_coords
    display_label  = "🎯 Map Point Selected!" if clicked_coords else f"📌 Candidate #{st.session_state.fa_pin_idx + 1}"

    st.success(display_label)
    st.code(display_coords, language="text")
    st.caption("Paste directly into Google Maps or Sheets.")
    _gmaps = f"https://www.google.com/maps/search/?api=1&query={display_coords.replace(' ', '')}"
    st.markdown(f"[🔗 Open in Google Maps]({_gmaps})")

    if clicked_coords and st.button("Clear map selection"):
        st.session_state.fa_clicked_lat = None
        st.session_state.fa_clicked_lon = None
        st.rerun()

    if st.session_state.plant_obs_dict:
        st.markdown("---")
        st.caption("🌿 **Nearby plants** (within 3 mi of selected point):")
        for _sp, _obs in st.session_state.plant_obs_dict.items():
            _nearby_count = sum(
                1 for o in _obs
                if haversine_distance(_active_lat, _active_lon, o["lat"], o["lon"]) <= 3.0
            )
            if _nearby_count:
                st.markdown(f"- **{_sp}**: {_nearby_count} observations")
