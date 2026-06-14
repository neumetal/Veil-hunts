"""
fetcher.py
Smart observation-driven data fetcher for Veil Finder.

Key behaviors
-------------
- Reads observations.json to know which dates need satellite data.
- Compares against the existing fog_master.csv to find missing dates.
- Fetches only the date range that covers missing dates (one API call
  per grid point, regardless of how many dates fall in that range).
- Caches elevation data after first fetch to avoid redundant calls.
- Appends new data to fog_master.csv and deduplicates.

One API call per grid point covers the full date range, so fetching
Nov 1-Dec 31 costs the same as fetching Nov 1-Nov 2: 549 calls total.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib

_BACKEND = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_BACKEND)
sys.path.insert(0, _BACKEND)

from grid_utils  import generate_grid, fetch_elevations, classify_valley_points
import api_client
importlib.reload(api_client)
from api_client  import fetch_all_weather

import veil_logic
importlib.reload(veil_logic)
from veil_logic  import build_master_dataframe

# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_master_csv(export_dir: str) -> pd.DataFrame | None:
    """Load fog_master.parquet if it exists, else return None.
    If CloudCover_Pct column is missing (old schema), return None to force re-fetch.
    """
    master_file = os.path.join(export_dir, "fog_master.parquet")
    if os.path.exists(master_file):
        try:
            df = pd.read_parquet(master_file)
            if "CloudCover_Pct" not in df.columns:
                print("WARNING: Old master Parquet detected without CloudCover_Pct. Forcing re-fetch.")
                return None
            if "Timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            return df
        except Exception as e:
            print(f"WARNING: Could not load master Parquet: {e}")
    return None


def get_missing_data_points(
    observations: list[dict],
    fog_df: pd.DataFrame | None,
    current_grid: list[tuple[float, float]],
) -> tuple[list[str], list[tuple[float, float]]]:
    """
    Returns (all_obs_dates, points_to_fetch).
    A point needs fetching if it is missing ANY observation date in fog_df.
    """
    obs_dates = sorted({o["date"] for o in observations if o.get("date")})
    if not obs_dates:
        return [], []

    if fog_df is None or fog_df.empty or "CloudCover_Pct" not in fog_df.columns:
        return obs_dates, current_grid

    # Create a fast lookup of dates available for each (lat, lon)
    # fog_df["date_str"] = fog_df["Timestamp"].dt.strftime("%Y-%m-%d")
    date_strs = fog_df["Timestamp"].dt.strftime("%Y-%m-%d")
    available_data = fog_df.groupby(["Lat", "Lon"]).apply(lambda x: set(date_strs[x.index])).to_dict()

    points_to_fetch = []
    obs_dates_set = set(obs_dates)

    for pt in current_grid:
        # Floating point grouping can be tricky, we round to 4 decimals
        lat, lon = pt
        
        # Try to find exactly or very closely matching point
        # Since points are generated exactly the same way, direct lookup usually works
        pt_dates = available_data.get(pt, set())
        
        if not pt_dates:
            # Fallback to near match in case of float weirdness
            for (df_lat, df_lon), dts in available_data.items():
                if np.isclose(df_lat, lat, atol=1e-4) and np.isclose(df_lon, lon, atol=1e-4):
                    pt_dates = dts
                    break
        
        if not obs_dates_set.issubset(pt_dates):
            points_to_fetch.append(pt)

    return obs_dates, points_to_fetch


# ─────────────────────────────────────────────────────────────────────────────
# Elevation cache
# ─────────────────────────────────────────────────────────────────────────────
def load_or_fetch_elevations(
    points: list[tuple[float, float]],
    export_dir: str,
    log_fn=print,
) -> dict[tuple[float, float], float | None]:
    """
    Load elevation data from local cache, or fetch from Open-Meteo and cache it.
    The cache is stored at <export_dir>/elevation_cache.json.
    After the first run this is essentially instant.
    """
    elevation_cache = os.path.join(export_dir, "elevation_cache.json")
    os.makedirs(os.path.dirname(elevation_cache), exist_ok=True)

    elevations = {}
    if os.path.exists(elevation_cache):
        try:
            with open(elevation_cache, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Keys are stored as "lat,lon" strings
            elevations = {
                (round(float(k.split(",")[0]), 4), round(float(k.split(",")[1]), 4)): v
                for k, v in raw.items()
            }
            log_fn(f"Elevation cache loaded ({len(elevations)} points).")
        except Exception as e:
            log_fn(f"Cache load failed ({e}). Re-fetching...")

    # Which points are missing from cache?
    missing_points = []
    for pt in points:
        lat, lon = pt
        # find matching point
        found = False
        for c_pt in elevations.keys():
            if np.isclose(c_pt[0], lat, atol=1e-4) and np.isclose(c_pt[1], lon, atol=1e-4):
                found = True
                break
        if not found:
            missing_points.append(pt)

    if missing_points:
        log_fn(f"Fetching elevation data for {len(missing_points)} new point(s)...")
        new_elevs = fetch_elevations(missing_points)
        elevations.update(new_elevs)

        # Persist cache
        serializable = {f"{lat},{lon}": v for (lat, lon), v in elevations.items()}
        with open(elevation_cache, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
        log_fn(f"Elevation cache updated and saved.")
    else:
        log_fn("All required elevations were found in cache.")

    # Return only the requested points
    res = {}
    for pt in points:
        lat, lon = pt
        for c_pt, v in elevations.items():
            if np.isclose(c_pt[0], lat, atol=1e-4) and np.isclose(c_pt[1], lon, atol=1e-4):
                res[pt] = v
                break
        if pt not in res:
             res[pt] = None
             
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main smart fetch
# ─────────────────────────────────────────────────────────────────────────────
def smart_fetch(
    observations: list[dict],
    center_lat: float,
    center_lon: float,
    radius_km: float,
    export_dir: str,
    on_point_progress=None,
    log_fn=print,
) -> tuple[pd.DataFrame | None, int, str]:
    """
    Fetch satellite fog data only for observation dates not already in the master CSV,
    specifically checking if the required data exists for all points in the dynamic grid.
    """
    os.makedirs(export_dir, exist_ok=True)

    if not observations:
        return None, 0, "No observations logged yet — nothing to fetch."

    # ── 1. Grid ────────────────────────────────────────────────────────────
    log_fn("Generating dynamic spatial grid...")
    grid_points = generate_grid(center_lat=center_lat, center_lon=center_lon, radius_km=radius_km)
    log_fn(f"Grid contains {len(grid_points)} points.")

    # 2. Find missing dates for our grid
    log_fn("Checking local fog_master.parquet for missing data...")
    existing_df = load_master_csv(export_dir)
    all_dates, missing_points = get_missing_data_points(observations, existing_df, grid_points)

    if not all_dates:
        return existing_df, 0, "No observation dates to fetch."

    if not missing_points:
        return existing_df, 0, "All observation dates already have satellite data for this grid. Nothing to fetch."

    start_date = all_dates[0]
    end_date   = all_dates[-1]

    log_fn(f"Observation dates: {all_dates}")
    log_fn(f"Missing data for : {len(missing_points)} point(s) out of {len(grid_points)}")
    log_fn(f"Fetch range      : {start_date} -> {end_date}")

    # ── 3. Elevations (cached) ─────────────────────────────────────────────
    # 4. We have new data. Get elevations and categorize valleys
    elevations = load_or_fetch_elevations(missing_points, export_dir, log_fn=log_fn)
    valley_cls, mean_e, std_e, thresh = classify_valley_points(elevations)

    # ── 4. Weather fetch ───────────────────────────────────────────────────
    log_fn(f"Fetching weather data for {len(missing_points)} points...")
    weather_data = fetch_all_weather(
        missing_points, start_date, end_date,
        on_progress=on_point_progress,
    )

    if not weather_data:
        return existing_df, 0, "ERROR: Weather API returned no data. Check your internet connection."

    # ── 5. Score ───────────────────────────────────────────────────────────
    log_fn("Calculating fog scores...")
    new_df = build_master_dataframe(weather_data, elevations, valley_cls)

    # Keep only the dates we actually needed (discard extras in the range)
    new_df = new_df[
        new_df["Timestamp"].dt.strftime("%Y-%m-%d").isin(set(all_dates))
    ].copy()

    # ── 6. Merge + save ────────────────────────────────────────────────────
    if existing_df is not None and not existing_df.empty:
        master = pd.concat([existing_df, new_df], ignore_index=True)
        master = master.drop_duplicates(subset=["Timestamp", "Lat", "Lon"])
    else:
        master = new_df

    master = master.sort_values(["Timestamp", "Lat", "Lon"]).reset_index(drop=True)
    # 6. Save back to master
    master_file = os.path.join(export_dir, "fog_master.parquet")
    master.to_parquet(master_file, engine="pyarrow", index=False)

    n_new_pts = len(missing_points)
    msg = (
        f"Fetched data for {n_new_pts} point(s). "
        f"Master data now contains {len(master)} total rows."
    )
    log_fn(msg)
    return master, n_new_pts, msg


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    obs_path = os.path.join(_ROOT, "observations.json")
    if not os.path.exists(obs_path):
        print("No observations.json found. Nothing to fetch.")
        raise SystemExit(0)

    with open(obs_path, encoding="utf-8") as f:
        obs = _json.load(f)

    print(f"Loaded {len(obs)} observations from {obs_path}")
    fog_df, n_fetched, msg = smart_fetch(obs)
    print(msg)
