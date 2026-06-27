import requests
import streamlit as st

@st.cache_data(ttl=3600, max_entries=100, show_spinner="Fetching nearby parks from OSM...")
def fetch_parks_osm(lat: float, lon: float) -> list:
    """
    Fetch public parks within 5 km of (lat, lon) using Overpass API mirrors.
    """
    endpoints = [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter"
    ]
    
    query = f"""[out:json][timeout:15];
(
  nwr["leisure"="park"](around:5000, {lat}, {lon});
  nwr["boundary"="national_park"](around:5000, {lat}, {lon});
  nwr["boundary"="protected_area"](around:5000, {lat}, {lon});
  nwr["leisure"="nature_reserve"](around:5000, {lat}, {lon});
);
out center;"""

    headers = {
        "User-Agent": "VeilFinderBot/2.0 (contact: wesley@example.com)"
    }
    
    for url in endpoints:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                parks = []
                for elem in elements:
                    tags = elem.get("tags", {})
                    name = tags.get("name") or tags.get("official_name") or "Unnamed Park/Protected Area"
                    
                    # Get center/centroid
                    el_lat = elem.get("lat") or elem.get("center", {}).get("lat")
                    el_lon = elem.get("lon") or elem.get("center", {}).get("lon")
                    
                    if el_lat is not None and el_lon is not None:
                        parks.append({
                            "name": name,
                            "lat": float(el_lat),
                            "lon": float(el_lon),
                            "osm_id": elem.get("id"),
                            "type": elem.get("type"),
                        })
                return parks
        except Exception:
            continue
            
    return []
