"""Road network loading and processing module for the Ambulance Route Optimization System.

Handles downloading, caching, querying, and inspecting the Ranchi, Jharkhand, India
drivable road network using OSMnx and NetworkX.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import osmnx as ox

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

# Base paths and default configuration fallbacks (used if config.py is absent or omits keys)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DEFAULT_DATA_DIR, exist_ok=True)

DATA_DIR: str = getattr(config, "DATA_DIR", _DEFAULT_DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

NETWORK_CACHE_PATH: str = getattr(
    config,
    "NETWORK_CACHE_PATH",
    getattr(
        config,
        "RANCHI_NETWORK_PATH",
        os.path.join(DATA_DIR, "ranchi_network.graphml"),
    ),
)

PLACE_NAME: str = getattr(config, "PLACE_NAME", "Ranchi, Jharkhand, India")
NETWORK_TYPE: str = getattr(config, "NETWORK_TYPE", "drive")

DEFAULT_SPEED_LIMITS: dict[str, int] = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 30,
    "residential": 20,
    "unclassified": 20,
    "service": 15,
    "living_street": 10,
    "default": 25,
}

SPEED_LIMITS: dict[str, int] = dict(DEFAULT_SPEED_LIMITS)
if config is not None and isinstance(getattr(config, "SPEED_LIMITS", None), dict):
    SPEED_LIMITS.update(config.SPEED_LIMITS)


def load_or_download_network(
    cache_path: Optional[str] = None,
    place_name: Optional[str] = None,
    network_type: Optional[str] = None,
) -> nx.MultiDiGraph:
    """Load the cached Ranchi road network or download it via OSMnx.

    Attempts to load a GraphML file from ``cache_path`` (defaults to
    ``data/ranchi_network.graphml`` or the path specified in ``config.py``).
    If the cache does not exist or cannot be read, downloads the drivable road
    network for Ranchi, Jharkhand, India from OpenStreetMap, saves it to the
    cache file, and returns the graph.

    Args:
        cache_path: Optional override path for the GraphML cache file.
        place_name: Optional override place query string for OSMnx.
        network_type: Optional override OSMnx network type (default ``"drive"``).

    Returns:
        nx.MultiDiGraph: The drivable road network graph.

    Raises:
        RuntimeError: If the cached network cannot be loaded and downloading
            from OpenStreetMap fails (e.g. due to lack of network connectivity).
    """
    target_path = cache_path or NETWORK_CACHE_PATH
    target_place = place_name or PLACE_NAME
    target_network_type = network_type or NETWORK_TYPE

    cache_dir = os.path.dirname(os.path.abspath(target_path))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(target_path):
        try:
            logger.info("Loading cached road network from %s", target_path)
            G = ox.load_graphml(target_path)
            if isinstance(G, nx.MultiDiGraph) and G.number_of_nodes() > 0:
                return G
            logger.warning("Cached graph at %s is empty or invalid; re-downloading.", target_path)
        except Exception as exc:
            logger.warning("Failed to load cached network from %s (%s); re-downloading.", target_path, exc)

    overpass_endpoints = [
        "https://overpass.private.coffee/api",
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
    ]
    bounds = getattr(config, "RANCHI_BOUNDS", None) if config is not None else None
    last_exc: Optional[Exception] = None

    for endpoint in overpass_endpoints:
        try:
            ox.settings.overpass_url = endpoint
            ox.settings.requests_timeout = 45
            ox.settings.overpass_rate_limit = False
            if isinstance(bounds, dict) and all(k in bounds for k in ("west", "south", "east", "north")):
                bbox = (
                    float(bounds["west"]),
                    float(bounds["south"]),
                    float(bounds["east"]),
                    float(bounds["north"]),
                )
                logger.info(
                    "Downloading road network for Ranchi bbox %s via %s (network_type=%r)",
                    bbox,
                    endpoint,
                    target_network_type,
                )
                G = ox.graph_from_bbox(bbox, network_type=target_network_type)
            else:
                logger.info(
                    "Downloading road network for %r via %s (network_type=%r)",
                    target_place,
                    endpoint,
                    target_network_type,
                )
                G = ox.graph_from_place(target_place, network_type=target_network_type)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Overpass endpoint %s failed (%s); trying next mirror.", endpoint, exc)
    else:
        raise RuntimeError(
            f"Unable to load cached network at '{target_path}' and failed to download "
            f"road network for '{target_place}': {last_exc}"
        ) from last_exc

    try:
        ox.save_graphml(G, filepath=target_path)
        logger.info("Saved road network cache to %s", target_path)
    except Exception as exc:
        logger.warning("Downloaded network successfully, but failed to save cache to %s: %s", target_path, exc)

    return G


def get_speed_limit(highway_type: Any) -> int:
    """Return the realistic Indian road speed limit in km/h for a highway type.

    Args:
        highway_type: OSM ``highway`` tag value (string, list of strings, or None).

    Returns:
        int: Speed limit in km/h.
    """
    if isinstance(highway_type, (list, tuple)):
        for item in highway_type:
            if item is not None:
                highway_type = item
                break
        else:
            highway_type = None

    if not highway_type or not isinstance(highway_type, str):
        return int(SPEED_LIMITS.get("default", 25))

    normalized = highway_type.strip().lower()
    if normalized in SPEED_LIMITS:
        return int(SPEED_LIMITS[normalized])

    if normalized.endswith("_link"):
        base_type = normalized[:-5]
        if base_type in SPEED_LIMITS:
            return int(SPEED_LIMITS[base_type])

    return int(SPEED_LIMITS.get("default", 25))


def _parse_maxspeed(raw_maxspeed: Any, highway_type: Any) -> float:
    """Parse an OSM ``maxspeed`` attribute into km/h, falling back to highway speed."""
    if raw_maxspeed is None:
        return float(get_speed_limit(highway_type))

    if isinstance(raw_maxspeed, (list, tuple)):
        parsed_values = [
            _parse_single_maxspeed(val) for val in raw_maxspeed if val is not None
        ]
        valid = [v for v in parsed_values if v is not None and v > 0]
        if valid:
            return float(valid[0])
        return float(get_speed_limit(highway_type))

    parsed = _parse_single_maxspeed(raw_maxspeed)
    if parsed is not None and parsed > 0:
        return parsed
    return float(get_speed_limit(highway_type))


def _parse_single_maxspeed(value: Any) -> Optional[float]:
    """Parse a single maxspeed value (numeric or OSM string) into km/h."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    speed = float(match.group(1))
    if "mph" in text:
        speed *= 1.60934
    return round(speed, 2)


def _parse_oneway(raw_oneway: Any) -> bool:
    """Normalize an OSM ``oneway`` attribute to a boolean."""
    if isinstance(raw_oneway, bool):
        return raw_oneway
    if isinstance(raw_oneway, str):
        return raw_oneway.strip().lower() in {"true", "yes", "1", "t", "y"}
    if isinstance(raw_oneway, (int, float)):
        return bool(raw_oneway)
    return False


def _normalize_text_attr(value: Any, default: str) -> str:
    """Normalize a string or list-of-strings OSM attribute."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ", ".join(items) if items else default
    text = str(value).strip()
    return text if text else default


def get_road_info(G: nx.MultiDiGraph, u: int, v: int, key: int = 0) -> dict[str, Any]:
    """Extract normalized road segment properties for edge ``(u, v, key)``.

    Args:
        G: Road network graph.
        u: Source node ID.
        v: Target node ID.
        key: MultiDiGraph edge key (default ``0``).

    Returns:
        dict[str, Any]: Dictionary containing ``length`` (meters), ``highway``,
            ``name``, ``maxspeed`` (km/h), and ``oneway`` (bool).

    Raises:
        KeyError: If the specified edge ``(u, v, key)`` does not exist in ``G``.
    """
    edge_data = G.get_edge_data(u, v, key)
    if edge_data is None:
        raise KeyError(f"Edge ({u}, {v}, {key}) not found in road network graph.")

    raw_highway = edge_data.get("highway")
    highway_str = _normalize_text_attr(
        raw_highway[0] if isinstance(raw_highway, (list, tuple)) and raw_highway else raw_highway,
        default="unclassified",
    )

    try:
        length = float(edge_data.get("length", 0.0))
    except (TypeError, ValueError):
        length = 0.0

    return {
        "length": length,
        "highway": highway_str,
        "name": _normalize_text_attr(edge_data.get("name"), default="Unnamed Road"),
        "maxspeed": _parse_maxspeed(edge_data.get("maxspeed"), raw_highway),
        "oneway": _parse_oneway(edge_data.get("oneway", False)),
    }


def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Find the nearest graph node ID to the given latitude and longitude.

    Args:
        G: Road network graph.
        lat: Latitude in decimal degrees (Y coordinate).
        lon: Longitude in decimal degrees (X coordinate).

    Returns:
        int: Nearest node ID in ``G``.

    Raises:
        ValueError: If the graph is empty or coordinates are invalid.
    """
    if G is None or G.number_of_nodes() == 0:
        raise ValueError("Road network graph is empty.")

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid coordinates: lat={lat!r}, lon={lon!r}") from exc

    node_id = ox.distance.nearest_nodes(G, X=lon_f, Y=lat_f)
    return int(node_id)


def count_traffic_signals(G: nx.MultiDiGraph, path: Optional[Iterable[int]]) -> int:
    """Count nodes along ``path`` that have ``highway == 'traffic_signals'``.

    Args:
        G: Road network graph.
        path: Sequence of node IDs representing a route.

    Returns:
        int: Number of traffic signal nodes on the path.
    """
    if not path:
        return 0

    count = 0
    for node_id in path:
        if node_id not in G.nodes:
            continue
        highway_attr = G.nodes[node_id].get("highway")
        if isinstance(highway_attr, (list, tuple)):
            if any(str(val).strip().lower() == "traffic_signals" for val in highway_attr):
                count += 1
        elif highway_attr is not None and str(highway_attr).strip().lower() == "traffic_signals":
            count += 1
    return count


def get_path_edges(G: nx.MultiDiGraph, path: Optional[list[int]]) -> list[dict[str, Any]]:
    """Return a list of edge attribute dicts for each consecutive node pair in ``path``.

    For parallel edges between ``u`` and ``v`` in a MultiDiGraph, selects the edge
    with the smallest ``length`` attribute.

    Args:
        G: Road network graph.
        path: Ordered list of node IDs.

    Returns:
        list[dict[str, Any]]: Edge attribute dictionaries along the path.

    Raises:
        KeyError: If any consecutive node pair in ``path`` is not connected in ``G``.
    """
    if not path or len(path) < 2:
        return []

    edges: list[dict[str, Any]] = []
    for u, v in zip(path[:-1], path[1:]):
        edge_dict = G.get_edge_data(u, v)
        if not edge_dict:
            raise KeyError(f"No edge found between nodes {u} and {v} in graph.")

        best_edge = min(
            edge_dict.values(),
            key=lambda attrs: float(attrs.get("length", float("inf"))),
        )
        edges.append(dict(best_edge))

    return edges


def get_path_distance(G: nx.MultiDiGraph, path: Optional[list[int]]) -> float:
    """Calculate the total distance in meters along a node path.

    Args:
        G: Road network graph.
        path: Ordered list of node IDs.

    Returns:
        float: Total path distance in meters.
    """
    edges = get_path_edges(G, path)
    return float(sum(float(edge.get("length", 0.0)) for edge in edges))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def find_nearest_hospital(
    lat: float,
    lon: float,
    emergency_only: bool = True,
    hospitals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Find the nearest Ranchi hospital to the given coordinates.

    Args:
        lat: Reference latitude (e.g., patient location).
        lon: Reference longitude (e.g., patient location).
        emergency_only: If True, filter to hospitals with emergency services.
        hospitals: Optional list of hospital dicts; defaults to ``config.HOSPITALS``.

    Returns:
        Copy of the nearest hospital dictionary with ``distance_km`` added.
    """
    hospital_list = hospitals if hospitals is not None else getattr(config, "HOSPITALS", [])
    if not hospital_list:
        raise ValueError("No hospitals configured.")

    candidates = [
        h for h in hospital_list if not emergency_only or bool(h.get("emergency", True))
    ]
    if not candidates:
        candidates = list(hospital_list)

    best_hospital: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for h in candidates:
        dist = _haversine_km(lat, lon, float(h["lat"]), float(h["lon"]))
        if dist < best_dist:
            best_dist = dist
            best_hospital = dict(h)
            best_hospital["distance_km"] = round(dist, 3)

    if best_hospital is None:
        raise ValueError("Unable to select a nearest hospital.")
    return best_hospital


def get_hospital_by_identifier(
    identifier: str,
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    hospitals: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve a hospital by 0-based index or case-insensitive name substring.

    Args:
        identifier: Hospital index string (e.g., ``'0'``) or partial name (e.g., ``'RIMS'``).
        ref_lat: Optional reference latitude for calculating ``distance_km``.
        ref_lon: Optional reference longitude for calculating ``distance_km``.
        hospitals: Optional list of hospital dicts; defaults to ``config.HOSPITALS``.

    Returns:
        Copy of the matched hospital dictionary.
    """
    hospital_list = hospitals if hospitals is not None else getattr(config, "HOSPITALS", [])
    if not hospital_list:
        raise ValueError("No hospitals configured.")

    selected: Optional[Dict[str, Any]] = None
    ident_str = str(identifier).strip()

    if ident_str.isdigit():
        idx = int(ident_str)
        if 0 <= idx < len(hospital_list):
            selected = dict(hospital_list[idx])
        else:
            raise ValueError(
                f"Hospital index {idx} out of range [0, {len(hospital_list) - 1}]."
            )
    else:
        query = ident_str.lower()
        for h in hospital_list:
            if query in h["name"].lower():
                selected = dict(h)
                break
        if selected is None:
            available = ", ".join(h["name"] for h in hospital_list)
            raise ValueError(
                f"Hospital matching {identifier!r} not found. Available: {available}"
            )

    if ref_lat is not None and ref_lon is not None:
        selected["distance_km"] = round(
            _haversine_km(ref_lat, ref_lon, float(selected["lat"]), float(selected["lon"])),
            3,
        )
    return selected


# Convenience alias
load_road_network = load_or_download_network
