"""
api_client.py
Handles all Open-Meteo Historical Weather API calls with batching and retry logic.

API Reference: https://open-meteo.com/en/docs/historical-weather-api
"""
import requests
import pandas as pd
import time

# -- Open-Meteo Archive API ---------------------------------------------------
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS     = "temperature_2m,dewpoint_2m,relative_humidity_2m,weather_code,cloudcover"
TIMEZONE        = "America/New_York"

# Rate limiting
BATCH_SIZE  = 50    # Number of points before a longer pause
BATCH_DELAY = 1.5   # Seconds to wait between batches
POINT_DELAY = 0.12  # Seconds between individual point calls


# -----------------------------------------------------------------------------
# Single-point fetch
# -----------------------------------------------------------------------------
def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Fetch hourly weather data for a single lat/lon from Open-Meteo Archive API.

    Args:
        lat, lon    : Coordinates
        start_date  : 'YYYY-MM-DD'
        end_date    : 'YYYY-MM-DD'
        retries     : Number of retry attempts on failure

    Returns:
        DataFrame with columns [time, temperature_2m, dewpoint_2m,
                                 relative_humidity_2m, weather_code, cloudcover, lat, lon]
        Returns None if all retries fail.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          HOURLY_VARS,
        "timezone":        TIMEZONE,
        "temperature_unit": "celsius",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(WEATHER_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data   = resp.json()
            hourly = data.get("hourly", {})

            if not hourly or "time" not in hourly:
                raise ValueError("Empty hourly payload from API.")

            df = pd.DataFrame(hourly)
            df["lat"] = lat
            df["lon"] = lon
            df["time"] = pd.to_datetime(df["time"])
            return df

        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                # All retries exhausted — caller will skip this point
                return None


# -----------------------------------------------------------------------------
# Bulk fetch for all grid points
# -----------------------------------------------------------------------------
def fetch_all_weather(
    points: list[tuple[float, float]],
    start_date: str,
    end_date: str,
    on_progress=None,
) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Fetch weather data for every point in the grid.
    Applies per-point delays and batch pauses to respect Open-Meteo rate limits.

    Args:
        points     : List of (lat, lon) tuples
        start_date : 'YYYY-MM-DD'
        end_date   : 'YYYY-MM-DD'
        on_progress: Optional callable(points_done, total_points) called after each point

    Returns:
        Dict mapping (lat, lon) -> DataFrame. Skipped points are omitted.
    """
    results: dict = {}
    total   = len(points)
    skipped = 0

    print(f"\n[Weather] Fetching {total} grid points  ({start_date} -> {end_date})")
    print(f"          ~{total * POINT_DELAY / 60:.1f} min estimated (excl. batch pauses)\n")

    for i, pt in enumerate(points):
        lat, lon = pt
        df = fetch_weather(lat, lon, start_date, end_date)

        if df is not None:
            results[pt] = df
        else:
            skipped += 1
            print(f"  SKIP  ({lat:>8.4f}, {lon:>9.4f})  -- all retries failed.")

        # Fire progress callback
        if on_progress is not None:
            on_progress(i + 1, total)

        # Progress update every 50 points
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1:>4}/{total}  {100*(i+1)/total:>5.1f}%]  fetched={len(results)}  skipped={skipped}")

        # Batch pause
        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < total:
            print(f"  [Batch pause {BATCH_DELAY}s ...]")
            time.sleep(BATCH_DELAY)
        else:
            time.sleep(POINT_DELAY)

    print(f"\n[Weather] Done -- {len(results)}/{total} points OK, {skipped} skipped.\n")
    return results
