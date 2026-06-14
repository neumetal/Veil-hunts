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
