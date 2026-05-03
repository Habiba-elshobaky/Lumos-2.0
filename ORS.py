"""
ORS.py — Navigation via OpenRouteService + Nominatim geocoding.
Accepts place names, not just raw coordinates.
"""

from routingpy import ORS
from geopy.geocoders import Nominatim
import json
import os
from keys import ORS_API_KEY

client     = ORS(api_key=ORS_API_KEY)
geocoder   = Nominatim(user_agent="lumos_assistive_ai")
CACHE_FILE = "route_cache.json"


def geocode(place_name: str) -> tuple | None:
    """Converts a place name to (lon, lat). Returns None if not found."""
    try:
        loc = geocoder.geocode(place_name, timeout=5)
        if loc:
            return (loc.longitude, loc.latitude)
    except Exception as e:
        print(f">>> [GEOCODE ERROR]: {e}")
    return None


def reverse_geocode(lon: float, lat: float) -> str:
    """Converts coordinates to a human-readable address."""
    try:
        loc = geocoder.reverse((lat, lon), timeout=5)
        return loc.address if loc else f"{lat:.4f}, {lon:.4f}"
    except:
        return f"{lat:.4f}, {lon:.4f}"


def get_navigation_steps(start_coords: tuple, end_coords: tuple) -> list:
    """Fetches walking directions from ORS. Falls back to cache if API fails."""
    try:
        route = client.directions(
            locations=[start_coords, end_coords],
            profile="foot-walking"
        )
        steps = route.raw["features"][0]["properties"]["segments"][0]["steps"]
        with open(CACHE_FILE, "w") as f:
            json.dump(steps, f)
        return steps
    except Exception as e:
        print(f">>> [ORS ERROR]: {e}")
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        return []


def steps_to_speech(steps: list) -> list[str]:
    """Converts raw ORS step dicts into clean spoken sentences."""
    sentences = []
    for step in steps:
        instruction = step.get("instruction", "")
        distance    = int(step.get("distance", 0))
        if instruction:
            if distance > 0:
                sentences.append(f"In {distance} metres, {instruction.lower()}.")
            else:
                sentences.append(instruction)
    return sentences


def navigate_to_place(place_name: str, current_coords: tuple) -> list[str]:
    """High-level: name → geocode → route → spoken steps."""
    dest = geocode(place_name)
    if not dest:
        return [f"I could not find {place_name} on the map. Please try a different name."]
    steps = get_navigation_steps(current_coords, dest)
    if not steps:
        return ["I could not get directions right now. Please try again."]
    return steps_to_speech(steps)