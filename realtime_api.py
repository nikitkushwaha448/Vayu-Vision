"""
Real-time Air Quality API client for Air-Pulse
Supports OpenAQ and WAQI with automatic fallback
"""

import requests
import numpy as np
from typing import Dict, Optional, List
import config

def compute_aqi_from_pm25(pm25: float) -> int:
    """
    Compute AQI from PM2.5 using US EPA breakpoints
    
    Args:
        pm25: PM2.5 concentration in µg/m³
        
    Returns:
        AQI value
    """
    if pm25 <= 12.0:
        return int((pm25 / 12.0) * 50)
    elif pm25 <= 35.4:
        return int(((pm25 - 12.1) / (35.4 - 12.1)) * 50 + 50)
    elif pm25 <= 55.4:
        return int(((pm25 - 35.5) / (55.4 - 35.5)) * 50 + 100)
    elif pm25 <= 150.4:
        return int(((pm25 - 55.5) / (150.4 - 55.5)) * 50 + 150)
    elif pm25 <= 250.4:
        return int(((pm25 - 150.5) / (250.4 - 150.5)) * 50 + 200)
    else:
        return int(((pm25 - 250.5) / 500) * 99 + 250)


def _normalize_pollutants(raw: Dict) -> Dict:
    """Normalize various pollutant key formats to the lowercase keys used
    across the app: 'pm25','pm10','o3','no2','so2','co'.
    """
    if not raw:
        return {}

    mapping = {}
    for k, v in raw.items():
        if v is None:
            continue
        key = str(k).lower()
        # Normalize common variants
        if key in ('pm2.5', 'pm25', 'pm_2_5'):
            mapping['pm25'] = float(v)
        elif key in ('pm10', 'pm_10'):
            mapping['pm10'] = float(v)
        elif key in ('o3', 'o_3'):
            mapping['o3'] = float(v)
        elif key in ('no2', 'nox', 'no_2'):
            mapping['no2'] = float(v)
        elif key in ('so2', 'so_2'):
            mapping['so2'] = float(v)
        elif key in ('co',):
            mapping['co'] = float(v)
        else:
            # ignore unknown keys
            continue

    return mapping

def fetch_from_openaq_direct(city: str, country: str = "India") -> Optional[Dict]:
    """
    Fetch air quality data directly from OpenAQ by city name
    
    Args:
        city: City name
        country: Country name
        
    Returns:
        Dict with pollutant data or None if not found
    """
    try:
        url = f"{config.get_openaq_url()}/latest"
        params = {
            "city": city,
            "country": country,
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            pollutants = {}

            for measurement in result.get("measurements", []):
                parameter = measurement.get("parameter", "")
                value = measurement.get("value")
                if value is not None:
                    pollutants[str(parameter)] = value

            if pollutants:
                return _normalize_pollutants(pollutants)
        return None
    except Exception as e:
        print(f"OpenAQ direct fetch error: {e}")
        return None

def fetch_nearest_by_city(city: str, radius_m: int = 100000) -> Optional[Dict]:
    """
    Find nearest OpenAQ stations and fetch their data
    
    Args:
        city: City name
        radius_m: Search radius in meters
        
    Returns:
        Dict with aggregated pollutant data or None
    """
    try:
        # Approximate geocoding for major Indian cities
        city_coords = {
            "delhi": (28.7041, 77.1025),
            "mumbai": (19.0760, 72.8777),
            "bangalore": (12.9716, 77.5946),
            "hyderabad": (17.3850, 78.4867),
            "pune": (18.5204, 73.8567),
            "ahmedabad": (23.0225, 72.5714),
            "gurgaon": (28.4595, 77.0266),
            "noida": (28.5355, 77.3910),
            "ghaziabad": (28.6692, 77.4538),
            "lucknow": (26.8467, 80.9462),
            "Chennai": (13.0827, 80.2707),
            "kerala": (10.8505, 76.2711),
            "nagaland": (26.1584, 94.5624),
        }
        
        city_lower = city.lower()
        coords = city_coords.get(city_lower)
        
        if not coords:
            return None
        
        lat, lon = coords
        
        # Query nearest stations
        url = f"{config.get_openaq_url()}/locations"
        params = {
            "coordinates": f"{lat},{lon}",
            "nearest": True,
            "limit": 5
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data.get("results"):
            location_id = data["results"][0].get("id")

            # Get latest measurements from this location
            url = f"{config.get_openaq_url()}/latest"
            params = {"location_id": location_id}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            latest_data = response.json()
            if latest_data.get("results"):
                result = latest_data["results"][0]
                pollutants = {}

                for measurement in result.get("measurements", []):
                    parameter = measurement.get("parameter", "")
                    value = measurement.get("value")
                    if value is not None:
                        pollutants[str(parameter)] = value

                if pollutants:
                    return _normalize_pollutants(pollutants)

        return None
    except Exception as e:
        print(f"OpenAQ nearest fetch error: {e}")
        return None

def fetch_from_waqi(city: str, token: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch air quality data from WAQI API
    
    Args:
        city: City name
        token: WAQI API token (uses config if not provided)
        
    Returns:
        Dict with pollutant data or None if not found
    """
    try:
        if not token:
            token = config.get_waqi_token()
        
        if not token:
            return None
        
        url = f"{config.get_waqi_url()}/feed/{city}"
        params = {"token": token}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") == "ok" and data.get("data"):
            pollutants = {}
            data_obj = data["data"]

            # Extract pollutants
            pollutant_map = {
                "pm25": "PM2.5",
                "pm10": "PM10",
                "o3": "O3",
                "no2": "NO2",
                "so2": "SO2",
                "co": "CO"
            }

            # WAQI sometimes nests pollutant values differently
            iaqi = data_obj.get("iaqi") or {}
            for key, param_name in pollutant_map.items():
                # Prefer iaqi entries
                if key in iaqi and isinstance(iaqi[key], dict) and iaqi[key].get("v") is not None:
                    pollutants[param_name] = iaqi[key]["v"]
                elif key in data_obj and isinstance(data_obj[key], dict) and data_obj[key].get("v") is not None:
                    pollutants[param_name] = data_obj[key]["v"]

            # Fallback: if top-level keys like 'pm25' exist with numbers
            for key, param_name in pollutant_map.items():
                if param_name not in pollutants and key in data_obj and isinstance(data_obj[key], (int, float)):
                    pollutants[param_name] = data_obj[key]

            if pollutants:
                return _normalize_pollutants(pollutants)

        return None
    except Exception as e:
        print(f"WAQI fetch error: {e}")
        return None

def fetch_aqi_data(city: str) -> Optional[Dict]:
    """
    Fetch AQI data with automatic fallback
    
    Try OpenAQ direct → Nearest stations → WAQI fallback
    
    Args:
        city: City name
        
    Returns:
        Dict with air quality data or None
    """
    # Try 1: OpenAQ direct lookup
    result = fetch_from_openaq_direct(city)
    if result:
        return result

    # Try 2: OpenAQ nearest stations
    result = fetch_nearest_by_city(city)
    if result:
        return result

    # Try 3: WAQI fallback
    result = fetch_from_waqi(city)
    if result:
        return result

    return None


def fetch_latest_by_city(city: str) -> Optional[Dict]:
    """Compatibility wrapper expected by app.py.

    Returns a simple mapping of pollutant names to numeric values or None.
    """
    return fetch_aqi_data(city)

def get_aqi_category(aqi: int) -> str:
    """
    Get AQI category string
    
    Args:
        aqi: AQI value
        
    Returns:
        Category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
