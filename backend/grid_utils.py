"""
grid_utils.py
Generates the spatial analysis grid and fetches elevation data from Open-Meteo.

Grid: 0.05-degree resolution (~5km) within a 40-mile (64.37km) radius
centered on Doylestown, PA (40.31N, 75.13W).
"""
import numpy as np
import requests
import time

# -- Grid Configuration ------------------------------------------------------
CENTER_LAT = 40.31       # Doylestown, PA
CENTER_LON = -75.13
RADIUS_KM  = 64.37       # 40 statute miles
RESOLUTION_DEG = 0.05    # ~5.5 km per step at this latitude

# -- Open-Meteo Elevation API -------------------------------------------------
ELEVATION_API_URL  = "https://api.open-meteo.com/v1/elevation"
ELEVATION_BATCH_SZ = 100   # Max coordinates per API call


# -----------------------------------------------------------------------------
# Haversine distance
# -----------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# -----------------------------------------------------------------------------
# Grid generation
# -----------------------------------------------------------------------------
def generate_grid(
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    radius_km:  float = RADIUS_KM,
    resolution_deg: float = RESOLUTION_DEG,
) -> list[tuple[float, float]]:
    """
    Return a list of (lat, lon) tuples covering a circle of `radius_km`
    around (center_lat, center_lon) at `resolution_deg` spacing.
    Points outside the circle are filtered by haversine distance.
    """
    lat_span = radius_km / 111.0
    lon_span = radius_km / (111.0 * np.cos(np.radians(center_lat)))

    lats = np.arange(center_lat - lat_span, center_lat + lat_span + resolution_deg, resolution_deg)
    lons = np.arange(center_lon - lon_span, center_lon + lon_span + resolution_deg, resolution_deg)

    points = []
    for lat in lats:
        for lon in lons:
            if haversine(center_lat, center_lon, lat, lon) <= radius_km:
                points.append((round(float(lat), 4), round(float(lon), 4)))

    return points


# -----------------------------------------------------------------------------
# Elevation fetch
# -----------------------------------------------------------------------------
def fetch_elevations(
    points: list[tuple[float, float]],
    batch_size: int = ELEVATION_BATCH_SZ,
) -> dict[tuple[float, float], float | None]:
    """
    Fetch elevation (metres) for every grid point via Open-Meteo.
    Batches up to `batch_size` coordinates per request.
    Returns {(lat, lon): elevation_m}. Failed points map to None.
    """
    elevations: dict = {}
    total  = len(points)
    n_batches = (total - 1) // batch_size + 1

    print(f"\n[Elevation] Fetching {total} points in {n_batches} batch(es)...")

    for b_idx in range(0, total, batch_size):
        batch  = points[b_idx : b_idx + batch_size]
        lats   = [p[0] for p in batch]
        lons   = [p[1] for p in batch]
        params = {
            "latitude":  ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
        }

        success = False
        for attempt in range(3):
            try:
                resp = requests.get(ELEVATION_API_URL, params=params, timeout=30)
                resp.raise_for_status()
                elev_list = resp.json().get("elevation", [])
                for j, pt in enumerate(batch):
                    elevations[pt] = float(elev_list[j]) if j < len(elev_list) else None
                batch_num = b_idx // batch_size + 1
                print(f"  Batch {batch_num}/{n_batches}: {len(batch)} points OK.")
                success = True
                break
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < 2:
                    print(f"  Batch {b_idx // batch_size + 1} attempt {attempt+1} failed: {exc}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  WARNING: Batch {b_idx // batch_size + 1} permanently failed: {exc}")

        if not success:
            for pt in batch:
                elevations[pt] = None

        # Small pause between batches
        if b_idx + batch_size < total:
            time.sleep(0.4)

    fetched = sum(1 for v in elevations.values() if v is not None)
    print(f"  Elevation fetch complete: {fetched}/{total} points.")
    return elevations


# -----------------------------------------------------------------------------
# Valley classification
# -----------------------------------------------------------------------------
def classify_valley_points(
    elevations: dict[tuple[float, float], float | None],
) -> tuple[dict, float, float, float]:
    """
    Classify grid points as Valley Points using the 1.5-sigma rule.

    A point is a Valley Point if:
        elevation <= (grid_mean - 1.5 * grid_std)

    Returns:
        classifications : {(lat, lon): is_valley (bool)}
        mean_elev       : float
        std_elev        : float
        threshold       : float
    """
    valid_elevs = [v for v in elevations.values() if v is not None]
    elev_arr = np.array(valid_elevs)

    mean_elev = float(np.mean(elev_arr))
    std_elev  = float(np.std(elev_arr))
    threshold = mean_elev - 1.5 * std_elev

    print("\n------- VERIFICATION: Elevation & Valley Logic -------")
    print(f"  Grid Mean Elevation : {mean_elev:>8.1f} m")
    print(f"  Std Dev             : {std_elev:>8.1f} m")
    print(f"  Valley Threshold    : {threshold:>8.1f} m  (mean - 1.5sigma)")

    classifications: dict = {}
    valley_count = 0
    for pt, elev in elevations.items():
        is_valley = (elev is not None) and (elev <= threshold)
        classifications[pt] = is_valley
        if is_valley:
            valley_count += 1

    pct = 100 * valley_count / len(elevations) if elevations else 0
    print(f"  Valley Points Found : {valley_count} / {len(elevations)}  ({pct:.1f}%)")
    print("------------------------------------------------------\n")

    return classifications, mean_elev, std_elev, threshold
