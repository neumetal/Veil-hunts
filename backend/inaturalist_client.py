"""
inaturalist_client.py
Robust client to fetch verified plant observations from iNaturalist.
"""

import urllib.parse
import requests
import time
import math

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

BASE_URL = "https://api.inaturalist.org/v1"


def resolve_taxon_id(query: str) -> tuple[int | None, str | None, str | None]:
    """
    Search for a scientific/common plant name on iNaturalist and return (taxon_id, scientific_name, common_name).
    Filters results to the kingdom 'Plantae'.
    """
    if not query.strip():
        return None, None, None

    url = f"{BASE_URL}/taxa"
    params = {
        "q": query.strip(),
        "iconic_taxa": "Plantae",
        "per_page": 5
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                # Find best matching taxon
                best = results[0]
                taxon_id = best.get("id")
                sci_name = best.get("name")
                comm_name = best.get("preferred_common_name") or best.get("name")
                return taxon_id, sci_name, comm_name
    except Exception as e:
        print(f"ERROR: Failed to resolve iNaturalist taxon '{query}': {e}")
    
    return None, None, None


def fetch_plant_observations(
    taxon_id: int,
    swlat: float,
    swlon: float,
    nelat: float,
    nelon: float,
    max_results: int = 300,
    min_distance_mi: float = 0.0
) -> list[dict]:
    """
    Fetch plant observations for a specific taxon ID within a geographic bounding box.
    If min_distance_mi > 0, filters observations to ensure they are spaced out.
    Returns a list of dicts: [{'lat': float, 'lon': float, 'observed_on': str, 'user': str, 'url': str}]
    """
    url = f"{BASE_URL}/observations"
    
    observations = []
    page = 1
    per_page = 200  # API max is 200 per page

    # Bounding box coordinates must be strings/floats
    params = {
        "taxon_id": taxon_id,
        "swlat": swlat,
        "swlon": swlon,
        "nelat": nelat,
        "nelon": nelon,
        "quality_grade": "research",  # High quality, community vetted data
        "per_page": per_page,
        "page": page
    }

    while len(observations) < max_results:
        params["page"] = page
        try:
            # Respect API etiquette with a tiny delay
            if page > 1:
                time.sleep(0.2)
                
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"WARNING: iNaturalist API returned status code {response.status_code}")
                break
                
            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                loc = obs.get("location")
                if loc:
                    try:
                        # Location is returned as 'lat,lon' string
                        lat_str, lon_str = loc.split(",")
                        lat = float(lat_str)
                        lon = float(lon_str)
                        
                        if min_distance_mi > 0:
                            too_close = False
                            for acc in observations:
                                if haversine_distance(lat, lon, acc["lat"], acc["lon"]) < min_distance_mi:
                                    too_close = True
                                    break
                            if too_close:
                                continue
                        
                        observations.append({
                            "lat": lat,
                            "lon": lon,
                            "observed_on": obs.get("observed_on_string") or obs.get("observed_on") or "Unknown",
                            "user": obs.get("user", {}).get("login") or "Anonymous",
                            "url": obs.get("uri") or f"https://www.inaturalist.org/observations/{obs.get('id')}",
                            "id": obs.get("id"),
                            "photo_url": obs.get("photos", [{}])[0].get("url") if obs.get("photos") else None
                        })
                    except ValueError:
                        continue

            # Check if we have fetched all records
            total_results = data.get("total_results", 0)
            if len(observations) >= total_results or len(results) < per_page:
                break
                
            page += 1
        except Exception as e:
            print(f"ERROR: iNaturalist observations fetch failed: {e}")
            break

    return observations[:max_results]


# Simple manual test block
if __name__ == "__main__":
    print("Testing taxon resolution...")
    tid, sci, comm = resolve_taxon_id("Japanese Stiltgrass")
    print(f"Resolved: ID={tid}, Sci={sci}, Common={comm}")
    
    if tid:
        print("\nTesting observations fetch...")
        # Bounding box around eastern Pennsylvania
        obs = fetch_plant_observations(
            taxon_id=tid,
            swlat=40.0,
            swlon=-75.5,
            nelat=40.5,
            nelon=-75.0,
            max_results=5
        )
        print(f"Fetched {len(obs)} research observations:")
        for o in obs:
            print(f" - ID: {o['id']} at ({o['lat']}, {o['lon']}) by @{o['user']}")
