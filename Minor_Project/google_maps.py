"""
Google Maps Routes API integration for the Ambulance Route Optimization System.

Provides traffic-aware route retrieval via Google Maps Routes API (v2 computeRoutes),
polyline decoding, haversine-based offline travel estimates when the API is unavailable,
and structured comparisons between ACO-optimized routes and Google Maps routes.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import config
except ImportError:
    config = None

def load_env_file(env_path: Optional[str] = None) -> bool:
    """Load key=value pairs from .env into os.environ (uses python-dotenv if installed)."""
    try:
        from dotenv import load_dotenv
        return bool(load_dotenv(env_path)) if env_path else bool(load_dotenv())
    except ImportError:
        path = env_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.isfile(path):
            return False
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, val)
        return True


load_env_file()

logger = logging.getLogger(__name__)

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
EARTH_RADIUS_METERS = 6371000.0
OFFLINE_URBAN_SPEED_KMH = 25.0
OFFLINE_ROAD_WINDING_FACTOR = 1.4
DEFAULT_TIMEOUT_SECONDS = 15

_PLACEHOLDER_KEYS = {"", "your_api_key_here", "none", "null"}


def _resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve Google Maps API key from argument, environment, or config module."""
    candidates = [
        api_key,
        os.environ.get("GOOGLE_MAPS_API_KEY"),
        getattr(config, "GOOGLE_MAPS_API_KEY", None) if config is not None else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped and stripped.lower() not in _PLACEHOLDER_KEYS:
                return stripped
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance in meters between two WGS84 points.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance between the two points in meters.
    """
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """
    Decode a Google encoded polyline string into a list of (lat, lon) tuples.

    Implements Google's standard Encoded Polyline Algorithm Format (1e-5 precision).

    Args:
        encoded: Encoded polyline string.

    Returns:
        List of (latitude, longitude) tuples in decimal degrees.
    """
    if not encoded:
        return []

    coordinates: List[Tuple[float, float]] = []
    index = 0
    length = len(encoded)
    lat = 0
    lng = 0

    while index < length:
        for is_lng in (False, True):
            shift = 0
            result = 0
            while True:
                if index >= length:
                    raise ValueError("Malformed encoded polyline: unexpected end of string")
                byte_val = ord(encoded[index]) - 63
                index += 1
                result |= (byte_val & 0x1F) << shift
                shift += 5
                if byte_val < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lng:
                lng += delta
            else:
                lat += delta

        coordinates.append((round(lat / 1e5, 5), round(lng / 1e5, 5)))

    return coordinates


def get_offline_estimate(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> Dict[str, Any]:
    """
    Estimate route distance and travel time offline using haversine distance.

    Uses a 1.4x road winding factor and an average urban speed of 25 km/h.
    Results are explicitly marked as ESTIMATES and not real Google Maps data.

    Args:
        origin_lat: Origin latitude in decimal degrees.
        origin_lon: Origin longitude in decimal degrees.
        dest_lat: Destination latitude in decimal degrees.
        dest_lon: Destination longitude in decimal degrees.

    Returns:
        Route dictionary matching the Google Maps route response schema with
        'source': 'offline_estimate'.
    """
    o_lat, o_lon = float(origin_lat), float(origin_lon)
    d_lat, d_lon = float(dest_lat), float(dest_lon)

    straight_line_meters = haversine_distance(o_lat, o_lon, d_lat, d_lon)
    estimated_distance_meters = straight_line_meters * OFFLINE_ROAD_WINDING_FACTOR
    estimated_duration_seconds = (
        (estimated_distance_meters / 1000.0) / OFFLINE_URBAN_SPEED_KMH
    ) * 3600.0

    return {
        "duration_seconds": round(estimated_duration_seconds, 2),
        "distance_meters": round(estimated_distance_meters, 2),
        "time_min": round(estimated_duration_seconds / 60.0, 2),
        "distance_km": round(estimated_distance_meters / 1000.0, 3),
        "straight_line_distance_meters": round(straight_line_meters, 2),
        "polyline": "",
        "route_points": [(o_lat, o_lon), (d_lat, d_lon)],
        "coordinates": [(o_lat, o_lon), (d_lat, d_lon)],
        "traffic_conditions": {
            "status": "unavailable",
            "note": "ESTIMATE ONLY: No live traffic data available offline.",
        },
        "status": "offline_estimate",
        "source": "offline_estimate",
        "is_estimate": True,
        "disclaimer": (
            "ESTIMATE ONLY: Calculated offline from straight-line haversine distance "
            f"with a {OFFLINE_ROAD_WINDING_FACTOR}x road winding factor at "
            f"{OFFLINE_URBAN_SPEED_KMH} km/h average urban speed. Not real Google Maps data."
        ),
    }


def _parse_duration_seconds(duration_value: Any) -> Optional[float]:
    """Parse a Routes API duration value (e.g., '642s' or numeric) into seconds."""
    if duration_value is None:
        return None
    if isinstance(duration_value, (int, float)):
        return float(duration_value)
    if isinstance(duration_value, str):
        text = duration_value.strip()
        if text.endswith("s"):
            text = text[:-1]
        return float(text)
    raise ValueError(f"Unsupported duration format: {duration_value!r}")


def _extract_traffic_conditions(
    route: Dict[str, Any],
    duration_seconds: Optional[float],
    static_duration_seconds: Optional[float],
) -> Dict[str, Any]:
    """Summarize traffic conditions from Routes API travelAdvisory and durations."""
    advisory = route.get("travelAdvisory") or {}
    intervals = advisory.get("speedReadingIntervals") or []

    speed_counts: Dict[str, int] = {}
    for interval in intervals:
        speed = str(interval.get("speed", "UNKNOWN"))
        speed_counts[speed] = speed_counts.get(speed, 0) + 1

    traffic_delay_seconds: Optional[float] = None
    if duration_seconds is not None and static_duration_seconds is not None:
        traffic_delay_seconds = round(duration_seconds - static_duration_seconds, 2)

    if speed_counts.get("TRAFFIC_JAM", 0) > 0:
        congestion_level = "HIGH"
    elif speed_counts.get("SLOW", 0) > 0:
        congestion_level = "MODERATE"
    elif speed_counts.get("NORMAL", 0) > 0:
        congestion_level = "LOW"
    elif traffic_delay_seconds is not None and static_duration_seconds and static_duration_seconds > 0:
        ratio = traffic_delay_seconds / static_duration_seconds
        if ratio >= 0.3:
            congestion_level = "HIGH"
        elif ratio >= 0.1:
            congestion_level = "MODERATE"
        else:
            congestion_level = "LOW"
    else:
        congestion_level = "UNKNOWN"

    return {
        "congestion_level": congestion_level,
        "traffic_delay_seconds": traffic_delay_seconds,
        "static_duration_seconds": static_duration_seconds,
        "speed_reading_intervals": intervals,
        "speed_summary": speed_counts,
    }


def _fallback_with_error(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    status_label: str,
    error_message: str,
) -> Dict[str, Any]:
    """Return an offline estimate annotated with the reason Google Maps was not used."""
    estimate = get_offline_estimate(origin_lat, origin_lon, dest_lat, dest_lon)
    estimate["status"] = status_label
    estimate["error"] = error_message
    return estimate


def get_google_maps_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch a traffic-aware driving route from the Google Maps Routes API.

    Uses the Routes API v2 computeRoutes endpoint with TRAFFIC_AWARE routing.
    Falls back to get_offline_estimate() if no API key is configured or if the
    API call fails.

    Args:
        origin_lat: Origin latitude in decimal degrees.
        origin_lon: Origin longitude in decimal degrees.
        dest_lat: Destination latitude in decimal degrees.
        dest_lon: Destination longitude in decimal degrees.
        api_key: Optional Google Maps API key. If None, checks
            os.environ.get('GOOGLE_MAPS_API_KEY') and config.GOOGLE_MAPS_API_KEY.

    Returns:
        Dictionary containing duration_seconds, distance_meters, polyline,
        route_points, traffic_conditions, status, and source.
    """
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        msg = "No Google Maps API key found; using offline estimate instead."
        logger.warning(msg)
        return _fallback_with_error(
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            status_label="fallback_no_api_key: API key missing; returning offline estimate",
            error_message=msg,
        )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": resolved_key,
        "X-Goog-FieldMask": (
            "routes.duration,"
            "routes.staticDuration,"
            "routes.distanceMeters,"
            "routes.polyline.encodedPolyline,"
            "routes.travelAdvisory"
        ),
    }

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(origin_lat),
                    "longitude": float(origin_lon),
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(dest_lat),
                    "longitude": float(dest_lon),
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "extraComputations": ["TRAFFIC_ON_POLYLINE"],
        "computeAlternativeRoutes": False,
    }

    try:
        response = requests.post(
            ROUTES_API_URL,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        msg = f"Google Maps API request failed: {exc}"
        logger.error(msg)
        return _fallback_with_error(
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            status_label=f"fallback_api_error: {exc}",
            error_message=msg,
        )

    if response.status_code != 200:
        detail = response.text[:300]
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                detail = err_json["error"].get("message", detail)
        except ValueError:
            pass
        msg = f"Google Maps API returned HTTP {response.status_code}: {detail}"
        logger.error(msg)
        return _fallback_with_error(
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            status_label=f"fallback_api_error: HTTP {response.status_code} - {detail}",
            error_message=msg,
        )

    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected non-object JSON response from Google Maps API")
        if "error" in data:
            err_msg = data["error"].get("message", str(data["error"]))
            raise ValueError(f"API error payload: {err_msg}")

        routes = data.get("routes") or []
        if not routes:
            raise ValueError("Google Maps API returned no routes for the given coordinates")

        route = routes[0]
        duration_seconds = _parse_duration_seconds(route.get("duration"))
        static_duration_seconds = _parse_duration_seconds(route.get("staticDuration"))
        distance_meters = route.get("distanceMeters")

        if duration_seconds is None or distance_meters is None:
            raise ValueError("Google Maps API response missing duration or distanceMeters")

        encoded_polyline = (route.get("polyline") or {}).get("encodedPolyline", "")
        route_points = decode_polyline(encoded_polyline) if encoded_polyline else []
        traffic_conditions = _extract_traffic_conditions(
            route, duration_seconds, static_duration_seconds
        )

        return {
            "duration_seconds": round(float(duration_seconds), 2),
            "distance_meters": float(distance_meters),
            "time_min": round(float(duration_seconds) / 60.0, 2),
            "distance_km": round(float(distance_meters) / 1000.0, 3),
            "polyline": encoded_polyline,
            "route_points": route_points,
            "coordinates": route_points,
            "traffic_conditions": traffic_conditions,
            "status": "success",
            "source": "google_maps_api",
            "is_estimate": False,
        }
    except (ValueError, KeyError, TypeError) as exc:
        msg = f"Failed to parse Google Maps API response: {exc}"
        logger.error(msg)
        return _fallback_with_error(
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            status_label=f"fallback_parse_error: {exc}",
            error_message=msg,
        )


def _extract_duration_seconds(result: Dict[str, Any], label: str) -> float:
    """Extract duration in seconds from an ACO or Google Maps result dictionary."""
    seconds_keys = (
        "duration_seconds",
        "travel_time_seconds",
        "eta_seconds",
        "time_seconds",
        "duration",
        "travel_time",
    )
    for key in seconds_keys:
        if key in result and result[key] is not None:
            return float(result[key])

    minutes_keys = ("duration_minutes", "travel_time_minutes", "eta_minutes")
    for key in minutes_keys:
        if key in result and result[key] is not None:
            return float(result[key]) * 60.0

    raise KeyError(f"Could not find duration field in {label} result: {list(result.keys())}")


def _extract_distance_meters(result: Dict[str, Any], label: str) -> float:
    """Extract distance in meters from an ACO or Google Maps result dictionary."""
    meters_keys = (
        "distance_meters",
        "total_distance_meters",
        "length_meters",
        "distance",
        "total_distance",
        "length",
    )
    for key in meters_keys:
        if key in result and result[key] is not None:
            return float(result[key])

    km_keys = ("distance_km", "total_distance_km")
    for key in km_keys:
        if key in result and result[key] is not None:
            return float(result[key]) * 1000.0

    raise KeyError(f"Could not find distance field in {label} result: {list(result.keys())}")


def format_comparison(
    aco_result: Dict[str, Any],
    gmaps_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Format a structured comparison between an ACO route result and a Google Maps result.

    Args:
        aco_result: Result dict from ACO optimization (containing duration/distance).
        gmaps_result: Result dict from get_google_maps_route() or get_offline_estimate().

    Returns:
        Dictionary containing both durations, both distances, time and distance
        differences, percentage differences, source metadata, and a formatted
        human-readable summary string in 'formatted_text'.
    """
    aco_duration = _extract_duration_seconds(aco_result, "ACO")
    aco_distance = _extract_distance_meters(aco_result, "ACO")
    gmaps_duration = _extract_duration_seconds(gmaps_result, "Google Maps")
    gmaps_distance = _extract_distance_meters(gmaps_result, "Google Maps")

    gmaps_source = gmaps_result.get("source", "unknown")
    is_estimate = gmaps_source != "google_maps_api" or bool(gmaps_result.get("is_estimate"))
    gmaps_label = (
        "Offline Estimate (Fallback - NOT Google Maps)"
        if is_estimate
        else "Google Maps Routes API"
    )

    time_diff_seconds = aco_duration - gmaps_duration
    distance_diff_meters = aco_distance - gmaps_distance

    time_diff_percent = (
        (time_diff_seconds / gmaps_duration) * 100.0 if gmaps_duration > 0 else 0.0
    )
    distance_diff_percent = (
        (distance_diff_meters / gmaps_distance) * 100.0 if gmaps_distance > 0 else 0.0
    )

    if abs(time_diff_seconds) < 1e-6:
        faster_route = "Equal"
    elif time_diff_seconds < 0:
        faster_route = "ACO"
    else:
        faster_route = gmaps_label

    sign_t = "+" if time_diff_seconds >= 0 else "-"
    sign_d = "+" if distance_diff_meters >= 0 else "-"

    lines = [
        "=== Route Comparison: ACO vs Google Maps ===",
        f"ACO Optimized Route      : {aco_duration:.1f} s ({aco_duration / 60.0:.2f} min) | "
        f"{aco_distance:.1f} m ({aco_distance / 1000.0:.2f} km)",
        f"{gmaps_label:25s}: {gmaps_duration:.1f} s ({gmaps_duration / 60.0:.2f} min) | "
        f"{gmaps_distance:.1f} m ({gmaps_distance / 1000.0:.2f} km)",
        f"Time Difference (ACO-GM) : {sign_t}{abs(time_diff_seconds):.1f} s "
        f"({sign_t}{abs(time_diff_percent):.2f}%)",
        f"Dist Difference (ACO-GM) : {sign_d}{abs(distance_diff_meters):.1f} m "
        f"({sign_d}{abs(distance_diff_percent):.2f}%)",
        f"Faster Route             : {faster_route}",
    ]

    if is_estimate:
        disclaimer = gmaps_result.get(
            "disclaimer",
            "Note: Baseline values are offline estimates, not real Google Maps API data.",
        )
        lines.append(f"Disclaimer               : {disclaimer}")
    elif gmaps_result.get("traffic_conditions"):
        congestion = gmaps_result["traffic_conditions"].get("congestion_level", "UNKNOWN")
        lines.append(f"Google Maps Traffic      : {congestion}")

    formatted_text = "\n".join(lines)

    return {
        "aco": {
            "label": "ACO Optimized Route",
            "duration_seconds": round(aco_duration, 2),
            "distance_meters": round(aco_distance, 2),
        },
        "google_maps": {
            "label": gmaps_label,
            "source": gmaps_source,
            "is_estimate": is_estimate,
            "duration_seconds": round(gmaps_duration, 2),
            "distance_meters": round(gmaps_distance, 2),
        },
        "aco_duration_seconds": round(aco_duration, 2),
        "aco_distance_meters": round(aco_distance, 2),
        "gmaps_duration_seconds": round(gmaps_duration, 2),
        "gmaps_distance_meters": round(gmaps_distance, 2),
        "time_difference_seconds": round(time_diff_seconds, 2),
        "time_difference_percent": round(time_diff_percent, 2),
        "distance_difference_meters": round(distance_diff_meters, 2),
        "distance_difference_percent": round(distance_diff_percent, 2),
        "faster_route": faster_route,
        "gmaps_source": gmaps_source,
        "is_gmaps_estimate": is_estimate,
        "formatted_text": formatted_text,
    }


def extract_traffic_multiplier(gmaps_result: Optional[Dict[str, Any]]) -> Optional[float]:
    """Derive a traffic congestion multiplier (duration / staticDuration) from a Routes API result."""
    if not isinstance(gmaps_result, dict) or gmaps_result.get("source") != "google_maps_api":
        return None
    duration = gmaps_result.get("duration_seconds")
    traffic = gmaps_result.get("traffic_conditions") or {}
    static_dur = traffic.get("static_duration_seconds")
    if duration is not None and static_dur is not None and float(static_dur) > 0:
        return max(1.0, float(duration) / float(static_dur))
    return None


def get_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper accepting (lat, lon) tuples for origin and destination."""
    return get_google_maps_route(
        float(origin[0]),
        float(origin[1]),
        float(destination[0]),
        float(destination[1]),
        api_key=api_key,
    )
