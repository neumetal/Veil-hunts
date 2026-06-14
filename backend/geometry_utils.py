import math

def get_rotated_corners(center_lat: float, center_lon: float, width_km: float, height_km: float, rotation_deg: float) -> list[list[float]]:
    """
    Calculate the 4 corners of an image for Plotly Mapbox overlay.
    Plotly expects corners in this order: [top-left, top-right, bottom-right, bottom-left].
    Each corner is [lon, lat].
    """
    # 1 degree of latitude is roughly 111.32 km
    lat_factor = 111.32
    # 1 degree of longitude depends on the latitude
    lon_factor = 111.32 * math.cos(math.radians(center_lat))

    if lon_factor == 0:
        lon_factor = 0.0001 # avoid division by zero at poles

    # Half width/height in degrees
    half_w_deg = (width_km / 2.0) / lon_factor
    half_h_deg = (height_km / 2.0) / lat_factor

    # Unrotated corners relative to center (dx, dy in degrees)
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    corners = [
        (-half_w_deg, half_h_deg),
        (half_w_deg, half_h_deg),
        (half_w_deg, -half_h_deg),
        (-half_w_deg, -half_h_deg)
    ]

    angle_rad = math.radians(-rotation_deg) # Negative because we rotate clockwise
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated_coords = []
    for dx, dy in corners:
        # Scale dx back to km for rotation, then back to deg? 
        # Actually, it's better to do the rotation in mercator/km space, then convert to degrees.
        x_km = dx * lon_factor
        y_km = dy * lat_factor
        
        rx_km = x_km * cos_a - y_km * sin_a
        ry_km = x_km * sin_a + y_km * cos_a
        
        rx_deg = rx_km / lon_factor
        ry_deg = ry_km / lat_factor
        
        rotated_coords.append([center_lon + rx_deg, center_lat + ry_deg])

    return rotated_coords
