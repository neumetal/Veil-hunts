import math

def get_rotated_corners(
    center_lat: float, 
    center_lon: float, 
    width_km: float, 
    height_km: float, 
    rotation_deg: float,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5
) -> list[list[float]]:
    """
    Calculate the 4 corners of an image for Plotly Mapbox overlay.
    Plotly expects corners in this order: [top-left, top-right, bottom-right, bottom-left].
    Each corner is [lon, lat].
    
    anchor_x: 0.0 is left edge, 1.0 is right edge.
    anchor_y: 0.0 is top edge, 1.0 is bottom edge.
    The point defined by (anchor_x, anchor_y) on the image will be placed exactly at (center_lat, center_lon)
    and the image will rotate around this anchor.
    """
    # 1 degree of latitude is roughly 111.32 km
    lat_factor = 111.32
    # 1 degree of longitude depends on the latitude
    lon_factor = 111.32 * math.cos(math.radians(center_lat))

    if lon_factor == 0:
        lon_factor = 0.0001 # avoid division by zero at poles

    # Distances from the anchor to the edges in km
    left_km = - (anchor_x * width_km)
    right_km = (1.0 - anchor_x) * width_km
    
    # Note: image Y goes down (0 is top), but map Y goes up (positive is North).
    # So top edge is positive North distance, bottom is negative North distance.
    top_km = (anchor_y * height_km)
    bottom_km = - ((1.0 - anchor_y) * height_km)

    # Unrotated corners relative to anchor (dx_km, dy_km)
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    corners_km = [
        (left_km, top_km),
        (right_km, top_km),
        (right_km, bottom_km),
        (left_km, bottom_km)
    ]

    angle_rad = math.radians(-rotation_deg) # Negative because we rotate clockwise
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated_coords = []
    for x_km, y_km in corners_km:
        # Rotate around the anchor (0,0)
        rx_km = x_km * cos_a - y_km * sin_a
        ry_km = x_km * sin_a + y_km * cos_a
        
        # Convert rotated offsets back to degrees
        rx_deg = rx_km / lon_factor
        ry_deg = ry_km / lat_factor
        
        rotated_coords.append([center_lon + rx_deg, center_lat + ry_deg])

    return rotated_coords

import pandas as pd
try:
    from shapely.geometry import Point, LineString, MultiLineString
    from shapely.ops import nearest_points
    import pyproj
    geod = pyproj.Geod(ellps="WGS84")
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

def distance_to_line_meters(lat: float, lon: float, line_coords: list) -> float:
    """
    Calculate the shortest distance in meters from a point to a line segment.
    line_coords is a list of (lat, lon) tuples.
    """
    if not _SHAPELY_AVAILABLE:
        return 999999.0
        
    pt = Point(lon, lat)
    line = LineString([(c[1], c[0]) for c in line_coords])
    
    _, nearest_pt = nearest_points(pt, line)
    _, _, dist = geod.inv(lon, lat, nearest_pt.x, nearest_pt.y)
    return dist

def filter_points_by_lines(df: pd.DataFrame, features_lines: list, max_distance_meters: float = 91.44) -> pd.Series:
    """
    Given a dataframe of scores and a list of line features (trails/streams),
    return a boolean mask of which points are within max_distance_meters of ANY line.
    Defaults to 91.44 meters (300 feet).
    """
    if not _SHAPELY_AVAILABLE or not features_lines:
        return pd.Series(False, index=df.index)
        
    all_lines = []
    for f in features_lines:
        all_lines.append(LineString([(c[1], c[0]) for c in f["coords"]]))
        
    multi_line = MultiLineString(all_lines)
    
    mask = []
    for _, row in df.iterrows():
        pt = Point(row["Lon"], row["Lat"])
        _, nearest_pt = nearest_points(pt, multi_line)
        _, _, dist = geod.inv(row["Lon"], row["Lat"], nearest_pt.x, nearest_pt.y)
        mask.append(dist <= max_distance_meters)
        
    return pd.Series(mask, index=df.index)
