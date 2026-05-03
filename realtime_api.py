"""
Simple OpenAQ client for fetching latest pollutant measurements by city.

Provides a helper to retrieve common pollutant values (pm25, pm10, o3, no2, so2, co)
and returns averaged values across returned locations when multiple measurements
are available.
"""
import requests
from collections import defaultdict

OPENAQ_BASE = "https://api.openaq.org/v2/latest"
OPENAQ_MEASUREMENTS = "https://api.openaq.org/v2/measurements"

WAQI_BASE = "https://api.waqi.info/feed"

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"


def geocode_city(city_name, timeout=10):
    """Resolve a city name to (lat, lon) using Nominatim. Returns (lat, lon) or None."""
    try:
        params = {
            'q': city_name,
            'format': 'json',
            'limit': 1,
        }
        resp = requests.get(NOMINATIM_SEARCH, params=params, timeout=timeout, headers={'User-Agent': 'air-pulse/1.0'})
        resp.raise_for_status()
        if not resp.text:
            return None
        data = resp.json()
        if not data:
            return None
        lat = float(data[0]['lat'])
        lon = float(data[0]['lon'])
        return lat, lon
    except Exception:
        return None


def fetch_nearest_by_city(city_name, radius_m=50000, limit=100, timeout=10):
    """Geocode the city and query OpenAQ latest measurements near those coordinates.

    Returns same mapping as `fetch_latest_by_city` or None.
    """
    coords = geocode_city(city_name, timeout=timeout)
    if coords is None:
        return None
    lat, lon = coords
    params = {
        'coordinates': f"{lat},{lon}",
        'radius': radius_m,
        'limit': limit,
    }
    try:
        # Use measurements endpoint to get parameter/value pairs near coordinates
        resp = requests.get(OPENAQ_MEASUREMENTS, params=params, timeout=timeout)
        resp.raise_for_status()
        if not resp.text:
            return None
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return None

        # Convert measurements list into a results-like structure for aggregation
        pseudo_results = []
        for r in results:
            pseudo_results.append({'measurements': [{'parameter': r.get('parameter'), 'value': r.get('value')}]})

        agg = _aggregate_results(pseudo_results)
        mapped = {
            'pm25': agg.get('pm25') or agg.get('pm2.5'),
            'pm10': agg.get('pm10'),
            'o3': agg.get('o3'),
            'no2': agg.get('no2'),
            'so2': agg.get('so2'),
            'co': agg.get('co'),
        }
        if not any(v is not None for v in mapped.values()):
            return None
        return mapped
    except Exception:
        return None


def _aggregate_results(results):
    # Aggregate values by parameter name and compute mean
    sums = defaultdict(float)
    counts = defaultdict(int)

    for item in results:
        measurements = item.get('measurements') or item.get('parameters') or []
        for m in measurements:
            param = m.get('parameter') or m.get('parameter')
            # OpenAQ v2 uses 'value'
            value = m.get('value') if 'value' in m else m.get('lastValue')
            try:
                val = float(value)
            except Exception:
                continue
            sums[param.lower()] += val
            counts[param.lower()] += 1

    aggregated = {}
    for param, total in sums.items():
        aggregated[param] = total / max(1, counts[param])

    return aggregated


def fetch_latest_by_city(city_name, limit=100, timeout=10):
    """Fetch latest measurements for a given city name from OpenAQ.

    Returns a dict with keys: pm25, pm10, o3, no2, so2, co (values are floats or None).
    """
    params = {
        'city': city_name,
        'limit': limit,
    }
    try:
        resp = requests.get(OPENAQ_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
        if not resp.text:
            return None
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return None

        agg = _aggregate_results(results)

        # Map common parameter names to expected keys
        mapped = {
            'pm25': agg.get('pm25') or agg.get('pm2.5'),
            'pm10': agg.get('pm10'),
            'o3': agg.get('o3'),
            'no2': agg.get('no2'),
            'so2': agg.get('so2'),
            'co': agg.get('co'),
        }

        # If nothing found, return None
        if not any(v is not None for v in mapped.values()):
            return None

        return mapped
    except Exception:
        return None


def compute_aqi_from_pm25(pm25_value):
    """Estimate US-EPA AQI from PM2.5 concentration using standard breakpoints.

    Returns estimated AQI as float. If pm25_value is None, returns None.
    """
    if pm25_value is None:
        return None
    try:
        c = float(pm25_value)
    except Exception:
        return None

    # Breakpoints for PM2.5 (µg/m3) - US EPA
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi, 1)

    # If above known range
    return 500.0


def fetch_from_waqi(city_name, token, timeout=10):
    """Fetch pollutant snapshot from WAQI (AQICN) API as a fallback.

    Returns a dict with keys similar to fetch_latest_by_city (pm25, pm10, o3, no2, so2, co)
    and may include 'aqi' when provided by the API. Returns None on failure.
    """
    if not token:
        return None
    try:
        url = f"{WAQI_BASE}/{city_name}/?token={token}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        if not resp.text:
            return None
        j = resp.json()
        if j.get('status') != 'ok' or not isinstance(j.get('data'), dict):
            return None

        data = j['data']
        aqi_val = data.get('aqi')
        iaqi = data.get('iaqi', {})

        mapped = {
            'pm25': (iaqi.get('pm25') or {}).get('v'),
            'pm10': (iaqi.get('pm10') or {}).get('v'),
            'o3': (iaqi.get('o3') or {}).get('v'),
            'no2': (iaqi.get('no2') or {}).get('v'),
            'so2': (iaqi.get('so2') or {}).get('v'),
            'co': (iaqi.get('co') or {}).get('v'),
        }
        # include reported AQI when present
        if aqi_val is not None:
            mapped['aqi'] = aqi_val

        return mapped
    except Exception:
        return None

