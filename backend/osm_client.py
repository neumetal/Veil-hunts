import requests
import streamlit as st
import json

@st.cache_data(ttl=3600, max_entries=50, show_spinner="Fetching detailed map features...")
def fetch_osm_features(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> dict:
    """
    Fetch polygons (parks, forests, water bodies) and lines (trails, streams, powerlines)
    within a bounding box. Returns detailed geometry for distance math and plotting.
    """
    endpoints = [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter"
    ]
    
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    
    query = f"""[out:json][timeout:25];
(
  way["highway"~"path|footway|track"]({bbox});
  way["waterway"~"stream|river"]({bbox});
  way["power"="line"]({bbox});
  nwr["leisure"="park"]({bbox});
  nwr["boundary"="national_park"]({bbox});
  nwr["boundary"="protected_area"]({bbox});
  nwr["leisure"="nature_reserve"]({bbox});
  nwr["landuse"="forest"]({bbox});
  nwr["natural"="wood"]({bbox});
  nwr["natural"="water"]({bbox});
);
out geom;"""

    headers = {
        "User-Agent": "VeilFinderBot/3.0 (contact: wesley@example.com)"
    }
    
    results = {
        "polygons": [],
        "lines": []
    }
    
    for url in endpoints:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                for el in data.get("elements", []):
                    tags = el.get("tags", {})
                    name = tags.get("name") or tags.get("official_name") or "Unnamed Feature"
                    geom = el.get("geometry", [])
                    if not geom:
                        continue
                        
                    coords = [(g["lat"], g["lon"]) for g in geom]
                    
                    feature = {
                        "osm_id": el.get("id"),
                        "name": name,
                        "type": el.get("type"),
                        "tags": tags,
                        "coords": coords
                    }
                    
                    # Classify as polygon or line based on tags
                    is_polygon = False
                    if "leisure" in tags or "boundary" in tags or "landuse" in tags or "natural" in tags:
                        is_polygon = True
                        
                    if is_polygon:
                        results["polygons"].append(feature)
                    else:
                        results["lines"].append(feature)
                        
                return results
        except Exception:
            continue
            
    return results

@st.cache_data(ttl=3600, max_entries=100, show_spinner="Fetching nearby parks from OSM...")
def fetch_parks_osm(lat: float, lon: float) -> list:
    """
    Legacy fetcher for single point (used in original app)
    """
    # ... existing code for backwards compatibility if needed, but we will mostly use the new one
    bbox_radius = 0.05  # roughly 5km degrees
    res = fetch_osm_features(lat - bbox_radius, lon - bbox_radius, lat + bbox_radius, lon + bbox_radius)
    parks = []
    for p in res["polygons"]:
        if "leisure" in p["tags"] or "boundary" in p["tags"]:
            # calculate centroid roughly
            lats = [c[0] for c in p["coords"]]
            lons = [c[1] for c in p["coords"]]
            parks.append({
                "name": p["name"],
                "lat": sum(lats)/len(lats),
                "lon": sum(lons)/len(lons),
                "osm_id": p["osm_id"],
                "type": p["type"]
            })
    return parks

