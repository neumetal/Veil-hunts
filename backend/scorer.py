"""
scorer.py
Location inference engine for Veil Finder.

Core concept
------------
For each observation O = (date, time_start, time_end, fog_observed):
  - Pull all satellite fog data for every grid point within that window.
  - Compute a "coverage fraction" per grid point:
        coverage = (hours with Fog_Score >= threshold) / (total hours in window)
  - If fog_observed == True:  match_score = coverage
    (a point that was foggy all window scores 1.0; clear all window scores 0.0)
  - If fog_observed == False: match_score = 1 - coverage
    (a point that was clear all window scores 1.0)
  - Narrow windows are more geographically informative, so each observation is
    weighted by:  weight = 1 / window_hours

After all observations:
  MatchRate   = sum(weight * match_score) / sum(weight)   per grid point
  Confidence  = (MatchRate - grid_mean) / grid_std        (z-score)

High MatchRate + high Confidence = strong trailcam candidate.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------
def score_all_observations(
    observations: list[dict],
    fog_df: pd.DataFrame,
    fog_threshold: int = 5,
) -> pd.DataFrame:
    """
    Score every grid point against all logged observations.

    Parameters
    ----------
    observations  : list of observation dicts (from observations.json)
    fog_df        : master fog DataFrame (from run_analysis.py CSV)
    fog_threshold : Fog_Score threshold above which an hour counts as "foggy"

    Returns
    -------
    DataFrame sorted by MatchRate descending, with columns:
        Lat, Lon, Elevation_m, IsValley,
        MatchRate, TotalScore, Confidence_Z, Matches, ObsCount
    """
    if not observations or fog_df is None or fog_df.empty:
        return pd.DataFrame()

    # Unique grid points (with metadata)
    grid_pts = (
        fog_df[["Lat", "Lon", "Elevation_m", "IsValley"]]
        .drop_duplicates(subset=["Lat", "Lon"])
        .set_index(["Lat", "Lon"])
        .copy()
    )
    grid_pts["_w_score"]   = 0.0
    grid_pts["_w_total"]   = 0.0
    grid_pts["_n_obs"]     = 0
    grid_pts["_n_matches"] = 0

    for obs in observations:
        dt_start = pd.Timestamp(f"{obs['date']} {obs['time_start']}")
        dt_end   = pd.Timestamp(f"{obs['date']} {obs['time_end']}")

        # Handle windows that cross midnight (e.g. 23:00 - 02:00)
        if dt_end <= dt_start:
            dt_end += pd.Timedelta(days=1)

        window_hours = (dt_end - dt_start).total_seconds() / 3600.0
        weight = 1.0 / max(window_hours, 0.25)   # floor at 15 min to avoid div-by-zero

        # Satellite rows inside the time window
        window_df = fog_df[
            (fog_df["Timestamp"] >= dt_start) &
            (fog_df["Timestamp"] <= dt_end)
        ]
        if window_df.empty:
            continue

        total_hours_in_window = window_df["Timestamp"].nunique()

        # Count fog hours per grid point
        fog_hour_counts = (
            window_df[window_df["Fog_Score"] >= fog_threshold]
            .groupby(["Lat", "Lon"])["Timestamp"]
            .nunique()
            .rename("fog_h")
        )

        # Reindex to all grid points (missing = 0 fog hours)
        fog_hours = fog_hour_counts.reindex(grid_pts.index, fill_value=0)
        coverage  = fog_hours / total_hours_in_window   # Series indexed by (Lat, Lon)

        # Match scores:
        #   fog observed  -> reward fog coverage   (more fog hours = closer to 1.0)
        #   clear observed -> reward low coverage  (fewer fog hours = closer to 1.0)
        if obs["fog_observed"]:
            match_scores = coverage
        else:
            match_scores = 1.0 - coverage

        grid_pts["_w_score"]   += weight * match_scores
        grid_pts["_w_total"]   += weight
        grid_pts["_n_obs"]     += 1
        grid_pts["_n_matches"] += (match_scores >= 0.5).astype(int)

    # Reset index and compute final metrics
    result = grid_pts.reset_index()

    has_data = result["_w_total"] > 0
    result["MatchRate"] = np.where(
        has_data,
        result["_w_score"] / result["_w_total"],
        np.nan,
    )
    result["TotalScore"] = result["_w_score"]
    result["ObsCount"]   = result["_n_obs"]
    result["Matches"]    = result["_n_matches"]

    # Z-score confidence
    mr_vals = result["MatchRate"].dropna()
    if len(mr_vals) > 1:
        mr_mean = mr_vals.mean()
        mr_std  = mr_vals.std()
        result["Confidence_Z"] = np.where(
            has_data,
            (result["MatchRate"] - mr_mean) / (mr_std if mr_std > 0 else 1.0),
            np.nan,
        )
    else:
        result["Confidence_Z"] = 0.0

    # Drop temp columns
    result = result.drop(columns=["_w_score", "_w_total", "_n_obs", "_n_matches"])

    return result.sort_values("MatchRate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-point diagnostics
# ---------------------------------------------------------------------------
def get_point_diagnostics(
    lat: float,
    lon: float,
    observations: list[dict],
    fog_df: pd.DataFrame,
    fog_threshold: int = 5,
) -> pd.DataFrame:
    """
    Return a per-observation breakdown for a specific grid point.
    Shows how the point fared against each logged observation.

    Returns DataFrame with one row per observation, columns:
        Date, Window, Fog_Observed, Fog_Hours_in_Window, Coverage_Pct,
        Match_Score, Result
    """
    rows = []

    for obs in observations:
        dt_start = pd.Timestamp(f"{obs['date']} {obs['time_start']}")
        dt_end   = pd.Timestamp(f"{obs['date']} {obs['time_end']}")
        if dt_end <= dt_start:
            dt_end += pd.Timedelta(days=1)

        window_hours = (dt_end - dt_start).total_seconds() / 3600.0

        pt_mask = (
            (fog_df["Timestamp"] >= dt_start) &
            (fog_df["Timestamp"] <= dt_end) &
            (np.isclose(fog_df["Lat"].values, lat)) &
            (np.isclose(fog_df["Lon"].values, lon))
        )
        pt_df = fog_df[pt_mask]

        if pt_df.empty:
            rows.append({
                "Date":             obs["date"],
                "Window":           f"{obs['time_start']}-{obs['time_end']}",
                "Fog_Observed":     "Yes" if obs["fog_observed"] else "No",
                "Fog_Hours":        "N/A",
                "Coverage":         "N/A",
                "Match_Score":      "N/A",
                "Result":           "No Data",
                "Photo":            obs.get("photo_filename", ""),
                "Notes":            obs.get("notes", ""),
            })
            continue

        total_h   = pt_df["Timestamp"].nunique()
        fog_h     = (pt_df["Fog_Score"] >= fog_threshold).sum()
        coverage  = fog_h / total_h if total_h > 0 else 0.0

        match_score = coverage if obs["fog_observed"] else (1.0 - coverage)
        result = "Match" if match_score >= 0.5 else "Miss"

        rows.append({
            "Date":         obs["date"],
            "Window":       f"{obs['time_start']}-{obs['time_end']}",
            "Fog_Observed": "Yes" if obs["fog_observed"] else "No",
            "Fog_Hours":    f"{int(fog_h)}/{total_h}",
            "Coverage":     f"{coverage * 100:.0f}%",
            "Match_Score":  round(match_score, 3),
            "Result":       result,
            "Photo":        obs.get("photo_filename", ""),
            "Notes":        obs.get("notes", ""),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GeoJSON export helper
# ---------------------------------------------------------------------------
def scores_to_geojson(scores_df: pd.DataFrame) -> dict:
    """
    Convert a scored grid DataFrame to a GeoJSON FeatureCollection.
    Compatible with QGIS, ArcGIS, JOSM, Google Earth, etc.
    Coordinates are [lon, lat] as per the GeoJSON spec (RFC 7946).
    """
    features = []
    for _, row in scores_df.iterrows():
        props = {}
        for col in scores_df.columns:
            val = row[col]
            # JSON cannot serialise NaN / numpy types
            if isinstance(val, float) and np.isnan(val):
                props[col] = None
            elif hasattr(val, "item"):        # numpy scalar
                props[col] = val.item()
            else:
                props[col] = val

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(row["Lon"], 6), round(row["Lat"], 6)],
            },
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Plant scoring & dynamic decay functions
# ---------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate the great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def compute_plant_scores(
    scores_df: pd.DataFrame,
    plant_obs_dict: dict[str, list[dict]],
    influence_radius_mi: float = 3.0,
    match_mode: str = "Any"
) -> pd.DataFrame:
    """
    Score each grid point based on spatial proximity to iNaturalist plant observations.
    
    Parameters
    ----------
    scores_df           : DataFrame containing scored grid points with columns 'Lat', 'Lon'
    plant_obs_dict      : Dict mapping species name -> list of observation dicts (with 'lat', 'lon')
    influence_radius_mi : Radius in miles. Score decays to ~0.05 at this distance.
    match_mode          : "Any" (max of selected species scores) or "All" (product of scores)
    
    Returns
    -------
    DataFrame with plant-specific scoring columns appended.
    """
    if scores_df.empty:
        return scores_df

    res_df = scores_df.copy()
    
    if not plant_obs_dict:
        # No plants selected, return with neutral 1.0 plant scores
        res_df["PlantScore"] = 1.0
        res_df["CombinedScore"] = res_df["MatchRate"].fillna(0.0)
        return res_df

    species_scores = {}
    
    for species_name, obs_list in plant_obs_dict.items():
        if not obs_list:
            # If a species has no local observations, its score is 0.0 everywhere
            species_scores[species_name] = np.zeros(len(res_df))
            res_df[f"PlantScore_{species_name}"] = 0.0
            continue
            
        obs_lats = np.array([o["lat"] for o in obs_list])
        obs_lons = np.array([o["lon"] for o in obs_list])
        
        scores = []
        for idx, row in res_df.iterrows():
            grid_lat = row["Lat"]
            grid_lon = row["Lon"]
            
            # Distance from this grid point to all observations of this species
            dists = haversine_distance(grid_lat, grid_lon, obs_lats, obs_lons)
            min_dist = np.min(dists)
            
            # Gaussian decay: e^(-3 * (d/R)^2)
            # Clamps to 0 if outside influence radius to preserve spatial boundary
            if min_dist <= influence_radius_mi:
                score = np.exp(-3.0 * (min_dist / max(influence_radius_mi, 0.1))**2)
            else:
                score = 0.0
                
            scores.append(score)
            
        species_scores[species_name] = np.array(scores)
        res_df[f"PlantScore_{species_name}"] = scores

    # Combine individual species scores
    if len(species_scores) == 1:
        combined_plant = list(species_scores.values())[0]
    else:
        if match_mode == "All":
            # Product of all species scores (requires ALL plants to be present nearby)
            combined_plant = np.prod(list(species_scores.values()), axis=0)
        else:
            # Maximum of all species scores (requires ANY plant to be present nearby)
            combined_plant = np.max(list(species_scores.values()), axis=0)

    res_df["PlantScore"] = combined_plant
    
    # Scale both scores to [0, 1] for better combined contrast
    mr = res_df["MatchRate"].fillna(0.0)
    if mr.max() > mr.min():
        mr_scaled = (mr - mr.min()) / (mr.max() - mr.min())
    else:
        mr_scaled = mr
        
    ps = combined_plant
    if ps.max() > ps.min():
        ps_scaled = (ps - ps.min()) / (ps.max() - ps.min())
    else:
        ps_scaled = ps
        
    res_df["CombinedScore"] = mr_scaled * ps_scaled
    
    # Sort by CombinedScore descending
    return res_df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cloud differential scoring
# ---------------------------------------------------------------------------
def score_cloud_observations(
    scores_df: pd.DataFrame,
    cloud_obs: list[dict],
    fog_df: pd.DataFrame,
    home_lat: float = 39.9248,
    home_lon: float = -75.1715,
) -> pd.DataFrame:
    """
    Score grid points based on cloud cover differential compared to home location.
    Appends 'CloudScore' to scores_df and updates 'CombinedScore'.
    """
    if scores_df.empty:
        return scores_df

    res_df = scores_df.copy()

    if not cloud_obs or fog_df is None or fog_df.empty or "CloudCover_Pct" not in fog_df.columns:
        res_df["CloudScore"] = 1.0
        return res_df

    from backend.api_client import fetch_weather

    res_df["CloudScore_Raw"] = 0.0

    for obs in cloud_obs:
        date = obs["date"]
        # Fetch home data for this date
        home_df = fetch_weather(home_lat, home_lon, date, date)
        if home_df is None or home_df.empty or "cloudcover" not in home_df.columns:
            continue

        dt_start = pd.Timestamp(f"{obs['date']} {obs['time_start']}")
        dt_end   = pd.Timestamp(f"{obs['date']} {obs['time_end']}")
        if dt_end <= dt_start:
            dt_end += pd.Timedelta(days=1)

        home_mask = (home_df["time"] >= dt_start) & (home_df["time"] <= dt_end)
        home_cloud = home_df.loc[home_mask, "cloudcover"].mean()
        
        if pd.isna(home_cloud):
            continue

        pt_mask = (fog_df["Timestamp"] >= dt_start) & (fog_df["Timestamp"] <= dt_end)
        pt_df = fog_df[pt_mask]

        if pt_df.empty:
            continue

        # Group by Lat/Lon to get average cloud cover
        grid_clouds = pt_df.groupby(["Lat", "Lon"])["CloudCover_Pct"].mean().reset_index()

        # Round grid_clouds to 4 decimal places to avoid float precision mismatch
        grid_clouds["Lat_g"] = grid_clouds["Lat"].round(4)
        grid_clouds["Lon_g"] = grid_clouds["Lon"].round(4)
        grid_clouds = grid_clouds.groupby(["Lat_g", "Lon_g"])["CloudCover_Pct"].mean().reset_index()

        relation = obs.get("relation", "Trailcam was Cloudier than Home")

        if "Cloudier" in relation:
            grid_clouds["diff"] = np.maximum(0, grid_clouds["CloudCover_Pct"] - home_cloud)
        else:
            grid_clouds["diff"] = np.maximum(0, home_cloud - grid_clouds["CloudCover_Pct"])

        # Create rounded coords in res_df for merging
        res_df["Lat_g"] = res_df["Lat"].round(4)
        res_df["Lon_g"] = res_df["Lon"].round(4)

        # Merge diff back to res_df
        merged = pd.merge(res_df[["Lat_g", "Lon_g"]], grid_clouds[["Lat_g", "Lon_g", "diff"]], on=["Lat_g", "Lon_g"], how="left")
        merged["diff"] = merged["diff"].fillna(0)
        res_df["CloudScore_Raw"] += merged["diff"]
        res_df = res_df.drop(columns=["Lat_g", "Lon_g"])

    # Normalize CloudScore_Raw to [0, 1]
    cs_raw = res_df["CloudScore_Raw"]
    if cs_raw.max() > cs_raw.min():
        res_df["CloudScore"] = (cs_raw - cs_raw.min()) / (cs_raw.max() - cs_raw.min())
    else:
        # If all 0, or all equal, CloudScore is 1.0 to not penalize
        res_df["CloudScore"] = 1.0 if cs_raw.max() >= 0 else 0.0

    res_df = res_df.drop(columns=["CloudScore_Raw"])

    # Update CombinedScore
    if "CombinedScore" in res_df.columns:
        res_df["CombinedScore"] = res_df["CombinedScore"] * res_df["CloudScore"]
    else:
        # Fallback if CombinedScore missing
        mr = res_df.get("MatchRate", pd.Series(1.0, index=res_df.index)).fillna(0.0)
        if mr.max() > mr.min():
            mr_scaled = (mr - mr.min()) / (mr.max() - mr.min())
        else:
            mr_scaled = mr
        res_df["CombinedScore"] = mr_scaled * res_df["CloudScore"]

    return res_df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-Contrast Cloud Day Detection
# ---------------------------------------------------------------------------

# The three trailcam timeframe windows (hour ranges, inclusive start, exclusive end)
CONTRAST_WINDOWS = [
    ("9AM-12PM",  9,  12),
    ("12PM-3PM", 12,  15),
    ("3PM-6PM",  15,  18),
]


def detect_high_contrast_days(
    fog_df: pd.DataFrame,
    clear_threshold: float = 30.0,
    cloudy_threshold: float = 70.0,
    min_zone_pct: float = 5.0,
) -> list[dict]:
    """
    Scan satellite data for date/window combinations where there is
    high spatial contrast in cloud cover: some grid points are mostly
    clear while others are mostly cloudy at the same time.

    Parameters
    ----------
    fog_df            : master fog DataFrame (must have CloudCover_Pct column)
    clear_threshold   : max cloud % to count a point as 'mostly clear'  (default 30)
    cloudy_threshold  : min cloud % to count a point as 'mostly cloudy' (default 70)
    min_zone_pct      : minimum % of points that must fall in each zone  (default 5%)

    Returns
    -------
    List of dicts (sorted newest-first, then by spread):
        date, window, spread, pct_clear, pct_cloudy, min_cloud, max_cloud, n_points
    """
    if fog_df is None or fog_df.empty or "CloudCover_Pct" not in fog_df.columns:
        return []

    df = fog_df.copy()
    df["_date"] = df["Timestamp"].dt.date
    df["_hour"] = df["Timestamp"].dt.hour

    results = []

    for date_val, date_group in df.groupby("_date"):
        for win_label, h_start, h_end in CONTRAST_WINDOWS:
            win_df = date_group[
                (date_group["_hour"] >= h_start) & (date_group["_hour"] < h_end)
            ]
            if win_df.empty:
                continue

            # Average cloud cover per grid point over the full window
            pt_avg = win_df.groupby(["Lat", "Lon"])["CloudCover_Pct"].mean()
            if len(pt_avg) < 2:
                continue

            n_pts    = len(pt_avg)
            n_clear  = (pt_avg <= clear_threshold).sum()
            n_cloudy = (pt_avg >= cloudy_threshold).sum()

            pct_clear  = 100.0 * n_clear  / n_pts
            pct_cloudy = 100.0 * n_cloudy / n_pts

            # Both zones must have meaningful representation
            if pct_clear < min_zone_pct or pct_cloudy < min_zone_pct:
                continue

            min_c  = float(pt_avg.min())
            max_c  = float(pt_avg.max())
            spread = max_c - min_c

            results.append({
                "date":       str(date_val),
                "window":     win_label,
                "spread":     round(spread, 1),
                "pct_clear":  round(pct_clear, 1),
                "pct_cloudy": round(pct_cloudy, 1),
                "min_cloud":  round(min_c, 1),
                "max_cloud":  round(max_c, 1),
                "n_points":   n_pts,
            })

    # Sort: newest date first, then within a date by spread (most contrast first)
    results.sort(key=lambda x: (x["date"], x["spread"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# High-Contrast Cloud scoring
# ---------------------------------------------------------------------------
def score_contrast_observations(
    scores_df: pd.DataFrame,
    contrast_obs: list[dict],
    fog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score grid points using High-Contrast Cloud Day observations.

    For each logged contrast observation the engine:
      1. Looks up the satellite cloud cover per grid point for that date/window.
      2. Converts cloud cover to a score contribution:
           Cloudy observation: contribution = avg_cloud_cover / 100
           Sunny  observation: contribution = (100 - avg_cloud_cover) / 100
      3. Averages contributions across all observations, then min-max normalises
         to [0, 1] to produce ContrastScore.
      4. Multiplies ContrastScore into CombinedScore.

    Points with no satellite data for an observed window receive a neutral 0.5.
    If no contrast observations have been logged, ContrastScore defaults to 1.0
    (no impact on CombinedScore).
    """
    if scores_df.empty:
        return scores_df

    res_df = scores_df.copy()

    if not contrast_obs or fog_df is None or fog_df.empty or "CloudCover_Pct" not in fog_df.columns:
        res_df["ContrastScore"] = 1.0
        return res_df

    df = fog_df.copy()
    df["_date"] = df["Timestamp"].dt.date.astype(str)
    df["_hour"] = df["Timestamp"].dt.hour

    # Build window -> (h_start, h_end) lookup
    win_hours = {lbl: (h_start, h_end) for lbl, h_start, h_end in CONTRAST_WINDOWS}

    res_df["Lat_g"] = res_df["Lat"].round(4)
    res_df["Lon_g"] = res_df["Lon"].round(4)
    res_df["_raw"] = 0.0
    res_df["_cnt"] = 0

    for obs in contrast_obs:
        obs_date   = obs["date"]
        obs_window = obs["window"]
        obs_cond   = obs["trailcam_condition"]   # "Cloudy" or "Sunny"

        if obs_window not in win_hours:
            continue

        h_start, h_end = win_hours[obs_window]

        win_df = df[
            (df["_date"] == obs_date) &
            (df["_hour"] >= h_start) &
            (df["_hour"] < h_end)
        ]
        if win_df.empty:
            continue

        # Average cloud cover per grid point within the window
        pt_avg = (
            win_df.groupby(["Lat", "Lon"])["CloudCover_Pct"]
            .mean()
            .reset_index()
        )
        pt_avg["Lat_g"] = pt_avg["Lat"].round(4)
        pt_avg["Lon_g"] = pt_avg["Lon"].round(4)

        if obs_cond == "Cloudy":
            pt_avg["_contrib"] = pt_avg["CloudCover_Pct"] / 100.0
        else:  # Sunny
            pt_avg["_contrib"] = (100.0 - pt_avg["CloudCover_Pct"]) / 100.0

        merged = pd.merge(
            res_df[["Lat_g", "Lon_g"]],
            pt_avg[["Lat_g", "Lon_g", "_contrib"]],
            on=["Lat_g", "Lon_g"],
            how="left",
        )
        # Points with no satellite data for this window get a neutral 0.5
        merged["_contrib"] = merged["_contrib"].fillna(0.5)

        res_df["_raw"] += merged["_contrib"].values
        res_df["_cnt"] += 1

    # Average across all observations and normalize to [0, 1]
    has_data = res_df["_cnt"] > 0
    raw = np.where(has_data, res_df["_raw"] / res_df["_cnt"].clip(lower=1), 0.5)

    raw_s = pd.Series(raw)
    if raw_s.max() > raw_s.min():
        contrast_score = (raw_s - raw_s.min()) / (raw_s.max() - raw_s.min())
    else:
        contrast_score = pd.Series(1.0, index=raw_s.index)

    res_df["ContrastScore"] = contrast_score.values
    res_df = res_df.drop(columns=["Lat_g", "Lon_g", "_raw", "_cnt"])

    # Fold into CombinedScore
    if "CombinedScore" in res_df.columns:
        res_df["CombinedScore"] = res_df["CombinedScore"] * res_df["ContrastScore"]
    else:
        mr = res_df.get("MatchRate", pd.Series(1.0, index=res_df.index)).fillna(0.0)
        if mr.max() > mr.min():
            mr_scaled = (mr - mr.min()) / (mr.max() - mr.min())
        else:
            mr_scaled = mr
        res_df["CombinedScore"] = mr_scaled * res_df["ContrastScore"]

    return res_df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)
