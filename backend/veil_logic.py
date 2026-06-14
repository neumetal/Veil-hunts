"""
veil_logic.py
Core fog probability scoring logic for the Valley-Adjusted Fog Grid.

Scoring Rules (temperature values in degC):
  Dew_Point_Spread = T_adjusted - T_dewpoint

  Spread ≤ 1.1degC  (≤ 2degF)           -> Fog_Score = 10  (High)
  1.1degC < Spread ≤ 2.8degC  (2–5degF)   -> Fog_Score = 5   (Moderate)
  Spread > 2.8degC  (> 5degF)            -> Fog_Score = 0   (Clear)

  +2 bonus if WMO weather_code ∈ {45, 48}  (Fog / Depositing rime fog)

Valley Correction:
  Points with elevation ≤ (grid_mean - 1.5sigma) are "Valley Points".
  T_adjusted = T_measured - 2.78degC  (-5degF) for these points.
"""
import pandas as pd
import numpy as np

# -- Scoring thresholds (Celsius) ---------------------------------------------
VALLEY_CORRECTION_C = -2.78   # -5 degF

SPREAD_HIGH_THRESH_C = 1.1    # ≤ this -> score 10
SPREAD_MED_THRESH_C  = 2.8    # ≤ this -> score 5; > this -> score 0

# WMO fog weather codes
FOG_CODES   = frozenset({45, 48})
FOG_BONUS   = 2


# -----------------------------------------------------------------------------
# Per-row scoring
# -----------------------------------------------------------------------------
def _score_row(spread_c: float, weather_code) -> int:
    """Compute fog probability score for a single hour/point."""
    if spread_c <= SPREAD_HIGH_THRESH_C:
        score = 10
    elif spread_c <= SPREAD_MED_THRESH_C:
        score = 5
    else:
        score = 0

    # Weather code fog bonus
    try:
        if int(weather_code) in FOG_CODES:
            score += FOG_BONUS
    except (TypeError, ValueError):
        pass

    return score


# -----------------------------------------------------------------------------
# Master DataFrame assembly
# -----------------------------------------------------------------------------
def build_master_dataframe(
    weather_data: dict,
    elevations: dict,
    valley_classifications: dict,
) -> pd.DataFrame:
    """
    Combine weather data, elevation, and valley corrections into the master
    analysis DataFrame.

    Args:
        weather_data          : {(lat, lon): DataFrame}  from api_client
        elevations            : {(lat, lon): float|None} from grid_utils
        valley_classifications: {(lat, lon): bool}       from grid_utils

    Returns:
        master DataFrame sorted by [Timestamp, Lat, Lon] with columns:
            Timestamp, Lat, Lon, Elevation_m, IsValley,
            T_Measured_C, T_Adjusted_C, DewPoint_C,
            DewPoint_Spread_C, RH_Pct, Weather_Code, Fog_Score
    """
    chunks = []

    for pt, df in weather_data.items():
        lat, lon = pt
        elevation = elevations.get(pt)
        is_valley = valley_classifications.get(pt, False)

        w = df.copy()
        w["elevation"] = elevation
        w["is_valley"] = is_valley

        # Valley temperature correction
        w["T_adjusted"] = w["temperature_2m"].copy()
        if is_valley:
            w["T_adjusted"] = w["T_adjusted"] + VALLEY_CORRECTION_C

        # Spread and score
        w["dew_point_spread"] = w["T_adjusted"] - w["dewpoint_2m"]
        w["fog_score"] = w.apply(
            lambda row: _score_row(row["dew_point_spread"], row.get("weather_code")),
            axis=1,
        )

        chunks.append(w)

    if not chunks:
        raise ValueError("No weather data available — cannot build DataFrame.")

    master = pd.concat(chunks, ignore_index=True)

    # Rename to clean public column names
    master = master.rename(columns={
        "time":                 "Timestamp",
        "lat":                  "Lat",
        "lon":                  "Lon",
        "elevation":            "Elevation_m",
        "is_valley":            "IsValley",
        "temperature_2m":       "T_Measured_C",
        "T_adjusted":           "T_Adjusted_C",
        "dewpoint_2m":          "DewPoint_C",
        "dew_point_spread":     "DewPoint_Spread_C",
        "relative_humidity_2m": "RH_Pct",
        "weather_code":         "Weather_Code",
        "cloudcover":           "CloudCover_Pct",
        "fog_score":            "Fog_Score",
    })

    final_cols = [
        "Timestamp", "Lat", "Lon", "Elevation_m", "IsValley",
        "T_Measured_C", "T_Adjusted_C", "DewPoint_C",
        "DewPoint_Spread_C", "RH_Pct", "Weather_Code", "CloudCover_Pct", "Fog_Score",
    ]

    return (
        master[final_cols]
        .sort_values(["Timestamp", "Lat", "Lon"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Verification + summary stats
# -----------------------------------------------------------------------------
def print_verification_stats(master_df: pd.DataFrame) -> list:
    """
    Print verification stats for the full analysis run and return a list of
    peak fog timestamps (highest avg Fog_Score) for auto-heatmap generation.

    Verification covers:
      - Average measured temperature across all grid points
      - Average adjusted temperature (reflects valley correction impact)
      - Average grid elevation
      - Average fog score across the date range
      - Per-day fog score summary
      - Top 5 peak fog hours
    """
    print("\n================== VERIFICATION STATS ==================")

    avg_temp_c   = master_df["T_Measured_C"].mean()
    avg_adj_c    = master_df["T_Adjusted_C"].mean()
    avg_elev_m   = master_df["Elevation_m"].dropna().mean()
    avg_score    = master_df["Fog_Score"].mean()

    n_unique_pts = master_df[["Lat", "Lon"]].drop_duplicates().shape[0]
    n_valley_pts = (
        master_df[["Lat", "Lon", "IsValley"]]
        .drop_duplicates()
        .query("IsValley == True")
        .shape[0]
    )

    print(f"  Grid Points Processed  : {n_unique_pts}")
    print(f"  Valley Points Applied  : {n_valley_pts}  (-2.78degC correction)")
    print(f"  Avg Measured Temp      : {avg_temp_c:>7.2f} degC  ({avg_temp_c*9/5+32:.1f} degF)")
    print(f"  Avg Adjusted Temp      : {avg_adj_c:>7.2f} degC  (valley-corrected)")
    print(f"  Avg Grid Elevation     : {avg_elev_m:>7.1f} m")
    print(f"  Avg Fog Score          : {avg_score:>7.2f}  (0=clear, 10=fog, 12=confirmed)")

    # Per-day summary
    master_df = master_df.copy()
    master_df["Date"] = master_df["Timestamp"].dt.date
    daily = master_df.groupby("Date")["Fog_Score"].agg(["mean", "max", "count"])
    print("\n  --- Daily Fog Score Summary ---")
    for date, row in daily.iterrows():
        bar = "#" * int(row["mean"] * 2)
        print(f"  {date}  avg={row['mean']:>5.2f}  max={row['max']:>2.0f}  {bar}")

    # Top 5 peak fog hours
    hourly_avg = (
        master_df.groupby("Timestamp")["Fog_Score"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n  --- Top 5 Peak Fog Hours (grid avg) ---")
    for ts, score in hourly_avg.head(5).items():
        pct_fog = 100 * (master_df[master_df["Timestamp"] == ts]["Fog_Score"] >= 10).mean()
        print(f"  {ts}   avg={score:.2f}   fog_coverage={pct_fog:.0f}%")

    print("========================================================\n")

    return hourly_avg.head(5).index.tolist()