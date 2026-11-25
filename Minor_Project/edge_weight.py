"""Dynamic edge weight calculation module for the Ambulance Route Optimization System.

This module computes estimated travel times (in seconds) for road segments in
a road network graph, accounting for road geometry, speed limits, dynamic
traffic congestion, weather severity impacts, traffic signal delays, and
emergency ambulance priority.

Mathematical Formula
--------------------
    Edge Weight (seconds) = (Base_Travel_Time * Traffic_Multiplier * Weather_Multiplier
                             + Signal_Delay + Road_Delay) / AMBULANCE_SPEED_MULTIPLIER

Where:
    Base_Travel_Time = (road_length_meters / 1000) / speed_limit_kmh * 3600
                     = road_length_meters / (speed_limit_kmh / 3.6)

    Traffic_Multiplier = Google Maps duration / free-flow duration (when Routes API is available)
                       = time-of-day estimated congestion factor (when API is unavailable)

    Weather_Multiplier = factor from trained Random Forest weather severity model:
                         clear: 1.0, light_rain: 1.15, moderate_rain: 1.3,
                         heavy_rain: 1.5, extreme: 1.8

    Signal_Delay = 30 seconds per traffic signal on the edge
                   (configured via config.TRAFFIC_SIGNAL_DELAY)

    Road_Delay = road-specific operational delay in seconds:
                 +10s for roundabouts/rotaries, +5s for narrow/service/living streets,
                 +3s for bridges or tunnels (bottlenecks).

    For ambulances (is_ambulance=True):
        Final weight is divided by AMBULANCE_SPEED_MULTIPLIER (1.3)
        since emergency ambulances travel faster using sirens and traffic priority.

Assumptions
-----------
1. Free-flow base travel time assumes constant speed equal to the road segment's
   posted or inferred speed limit (from OSM ``maxspeed`` or ``config.SPEED_LIMITS``).
2. Traffic congestion and adverse weather compound multiplicatively on the
   moving travel time (``Base_Travel_Time * Traffic_Multiplier * Weather_Multiplier``),
   while signal stop delays are additive (``+ Signal_Delay``).
3. When live Google Maps traffic data is unavailable, ``estimate_traffic_multiplier``
   provides a time-of-day heuristic with +/-5% stochastic variation, clamped at
   a minimum of 1.0 (free-flow).
4. In OSMnx graphs where traffic signals are annotated on intersection nodes
   rather than edges, signals are counted on the target node ``v`` of a directed
   edge ``(u, v)`` (when edge-level signal counts are absent) to avoid double
   counting intersections along a path.
5. Ambulance emergency priority yields an overall travel-time reduction by
   dividing the final edge weight by ``config.AMBULANCE_SPEED_MULTIPLIER``.
"""

from datetime import datetime
import re
from typing import Any, Dict, Mapping, Optional, Union

import networkx as nx
import numpy as np

from config import (
    AMBULANCE_SPEED_MULTIPLIER,
    SPEED_LIMITS,
    TRAFFIC_SIGNAL_DELAY,
    WEATHER_SEVERITY_THRESHOLDS,
)

_DEFAULT_SPEED_KMH: float = float(SPEED_LIMITS.get("unclassified", 20))
_DEFAULT_EDGE_LENGTH_M: float = 100.0
_MPH_TO_KMH: float = 1.60934


def calculate_base_travel_time(length_m: float, speed_kmh: float) -> float:
    """Calculate free-flow travel time in seconds for a road segment.

    Applies the formula:
        Base_Travel_Time = (length_m / 1000) / speed_kmh * 3600
                         = length_m / (speed_kmh / 3.6)

    Args:
        length_m: Road segment length in meters (must be non-negative).
        speed_kmh: Speed limit in kilometers per hour (must be positive).

    Returns:
        Estimated free-flow travel time in seconds.

    Raises:
        ValueError: If ``length_m < 0`` or ``speed_kmh <= 0``.
    """
    length_val = np.float64(length_m)
    speed_val = np.float64(speed_kmh)

    if length_val < 0:
        raise ValueError(f"length_m must be non-negative, got {length_m}.")
    if speed_val <= 0:
        raise ValueError(f"speed_kmh must be positive, got {speed_kmh}.")

    speed_mps = np.divide(speed_val, 3.6)
    travel_time_s = np.divide(length_val, speed_mps)
    return float(travel_time_s)


def estimate_traffic_multiplier(hour: Optional[Union[int, float]] = None) -> float:
    """Estimate a traffic congestion multiplier based on the hour of the day.

    NOTE: This is an OFFLINE ESTIMATE used only when live Google Maps API
    traffic data is unavailable. It models typical urban diurnal congestion
    patterns with +/-5% random variation for realism.

    Time-of-day bands (24-hour clock):
        - Peak hours (08:00-10:00, 17:00-19:00): 1.5 - 1.8
        - Moderate hours (07:00-08:00, 10:00-12:00, 14:00-17:00, 19:00-21:00): 1.2 - 1.4
        - Off-peak hours (all remaining hours): 1.0 - 1.1

    Args:
        hour: Hour of the day in [0, 24). If ``None``, the current local system
            hour is used.

    Returns:
        Estimated congestion multiplier (>= 1.0), where 1.0 represents
        free-flow traffic.

    Raises:
        ValueError: If ``hour`` is outside the valid range [0, 24).
    """
    if hour is None:
        current_hour: float = float(datetime.now().hour)
    else:
        current_hour = float(hour)

    if current_hour < 0 or current_hour >= 24:
        raise ValueError(f"hour must be in [0, 24), got {hour}.")

    if (8 <= current_hour < 10) or (17 <= current_hour < 19):
        low, high = 1.5, 1.8
    elif (
        (7 <= current_hour < 8)
        or (10 <= current_hour < 12)
        or (14 <= current_hour < 17)
        or (19 <= current_hour < 21)
    ):
        low, high = 1.2, 1.4
    else:
        low, high = 1.0, 1.1

    base_multiplier = np.random.uniform(low, high)
    variation = np.random.uniform(0.95, 1.05)  # +/-5% stochastic variation
    estimated = np.maximum(1.0, np.multiply(base_multiplier, variation))
    return float(estimated)


def get_weather_multiplier(weather_severity: Union[str, int, float] = "clear") -> float:
    """Resolve the travel-time multiplier for a given weather severity.

    Args:
        weather_severity: Weather severity label (``'clear'``, ``'light_rain'``,
            ``'moderate_rain'``, ``'heavy_rain'``, ``'extreme'``) or an explicit
            numeric multiplier (>= 1.0).

    Returns:
        Weather impact multiplier from ``config.WEATHER_SEVERITY_THRESHOLDS``.

    Raises:
        ValueError: If ``weather_severity`` is an unrecognized label or a
            non-positive number.
    """
    if isinstance(weather_severity, (int, float, np.number)):
        multiplier = float(weather_severity)
        if multiplier <= 0:
            raise ValueError(
                f"Numeric weather_severity must be positive, got {weather_severity}."
            )
        return multiplier

    normalized = str(weather_severity).strip().lower().replace(" ", "_")
    if normalized not in WEATHER_SEVERITY_THRESHOLDS:
        valid = ", ".join(WEATHER_SEVERITY_THRESHOLDS.keys())
        raise ValueError(
            f"Unknown weather_severity '{weather_severity}'. Expected one of: {valid}."
        )
    return float(WEATHER_SEVERITY_THRESHOLDS[normalized])


def calculate_road_specific_delay(edge_data: Mapping[str, Any]) -> float:
    """Estimate road-specific delay in seconds from OSM edge attributes.

    Models bottlenecks independent of traffic signals:
        - Roundabouts / junctions: +10.0s
        - Bridges / tunnels (lane bottlenecks): +3.0s
        - Narrow / local streets (service, living_street, track): +5.0s
    """
    delay = 0.0
    junction = str(edge_data.get("junction", "")).strip().lower()
    if junction in {"roundabout", "circular", "jughandle"}:
        delay += 10.0

    bridge = str(edge_data.get("bridge", "")).strip().lower()
    tunnel = str(edge_data.get("tunnel", "")).strip().lower()
    if bridge in {"yes", "true", "viaduct"} or tunnel in {"yes", "true"}:
        delay += 3.0

    highway = edge_data.get("highway")
    hw_str = str(highway[0] if isinstance(highway, (list, tuple)) and highway else highway).lower()
    if hw_str in {"service", "living_street", "track"}:
        delay += 5.0

    return delay


def calculate_edge_weight(
    length_m: float,
    speed_kmh: float,
    traffic_multiplier: float = 1.0,
    weather_multiplier: float = 1.0,
    signal_delay: float = 0.0,
    road_delay: float = 0.0,
    is_ambulance: bool = True,
) -> float:
    """Calculate the dynamic edge weight (estimated travel time in seconds).

    Applies the full edge weight formula:
        raw_time = Base_Travel_Time * Traffic_Multiplier * Weather_Multiplier + Signal_Delay + Road_Delay
    And for emergency ambulances (``is_ambulance=True``):
        weight = raw_time / AMBULANCE_SPEED_MULTIPLIER

    Args:
        length_m: Road segment length in meters.
        speed_kmh: Road speed limit in km/h.
        traffic_multiplier: Congestion factor (>= 1.0 for normal/congested flow).
        weather_multiplier: Weather slowdown factor (from weather model).
        signal_delay: Total traffic signal delay on the edge in seconds.
        road_delay: Road-specific operational delay in seconds (roundabouts, narrow roads, etc.).
        is_ambulance: Whether to apply emergency ambulance priority speed
            advantage (dividing final weight by ``config.AMBULANCE_SPEED_MULTIPLIER``).

    Returns:
        Estimated travel time in seconds as a float.
    """
    traffic_val = np.float64(traffic_multiplier)
    weather_val = np.float64(weather_multiplier)
    delay_val = np.float64(signal_delay)
    road_delay_val = np.float64(road_delay)

    if traffic_val <= 0:
        raise ValueError(
            f"traffic_multiplier must be positive, got {traffic_multiplier}."
        )
    if weather_val <= 0:
        raise ValueError(
            f"weather_multiplier must be positive, got {weather_multiplier}."
        )
    if delay_val < 0:
        raise ValueError(
            f"signal_delay must be non-negative, got {signal_delay}."
        )
    if road_delay_val < 0:
        raise ValueError(
            f"road_delay must be non-negative, got {road_delay}."
        )

    base_time = np.float64(calculate_base_travel_time(length_m, speed_kmh))
    moving_time = np.multiply(np.multiply(base_time, traffic_val), weather_val)
    total_weight = moving_time + delay_val + road_delay_val

    if is_ambulance:
        total_weight = np.divide(total_weight, np.float64(AMBULANCE_SPEED_MULTIPLIER))

    return float(total_weight)


def _parse_speed_value(raw_speed: Any) -> Optional[float]:
    """Parse an OSM or numeric speed limit value into km/h."""
    if raw_speed is None:
        return None

    if isinstance(raw_speed, (list, tuple, set)):
        parsed_values = [
            val for item in raw_speed if (val := _parse_speed_value(item)) is not None
        ]
        return float(np.mean(parsed_values)) if parsed_values else None

    if isinstance(raw_speed, (int, float, np.number)):
        val = float(raw_speed)
        return val if val > 0 else None

    text = str(raw_speed).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    val = float(match.group(1))
    if val <= 0:
        return None
    if "mph" in text:
        val = float(np.multiply(val, _MPH_TO_KMH))
    return val


def _extract_edge_speed_kmh(edge_data: Mapping[str, Any]) -> float:
    """Resolve edge speed limit (km/h) from edge attributes or OSM highway type."""
    for attr in ("speed_kph", "speed_kmh", "speed_limit", "maxspeed"):
        if attr in edge_data:
            parsed = _parse_speed_value(edge_data[attr])
            if parsed is not None:
                return parsed

    highway = edge_data.get("highway")
    if isinstance(highway, (list, tuple)):
        candidates = [
            float(SPEED_LIMITS[str(h).lower()])
            for h in highway
            if str(h).lower() in SPEED_LIMITS
        ]
        if candidates:
            return float(np.max(candidates))
    elif highway is not None:
        hw_key = str(highway).lower()
        if hw_key in SPEED_LIMITS:
            return float(SPEED_LIMITS[hw_key])

    return _DEFAULT_SPEED_KMH


def _count_edge_signals(
    G: nx.Graph,
    u: Any,
    v: Any,
    edge_data: Mapping[str, Any],
) -> int:
    """Count traffic signals associated with an edge or its destination intersection."""
    for attr in ("signal_count", "traffic_signals", "signals"):
        if attr in edge_data:
            val = edge_data[attr]
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float, np.number)):
                return max(0, int(val))
            if isinstance(val, (list, tuple, set)):
                return len(val)
            if str(val).strip().lower() in ("yes", "true", "traffic_signals"):
                return 1

    edge_highway = edge_data.get("highway")
    if isinstance(edge_highway, (list, tuple)):
        if any(str(h).lower() == "traffic_signals" for h in edge_highway):
            return 1
    elif str(edge_highway).lower() == "traffic_signals":
        return 1

    # Check destination node v in OSMnx graphs to avoid double-counting intersections.
    if v in G.nodes:
        node_data = G.nodes[v]
        node_hw = node_data.get("highway")
        if isinstance(node_hw, (list, tuple)):
            if any(str(h).lower() == "traffic_signals" for h in node_hw):
                return 1
        elif str(node_hw).lower() == "traffic_signals":
            return 1
        if node_data.get("has_signal") or node_data.get("traffic_signal"):
            return 1

    return 0


def _extract_multiplier_from_entry(entry: Any) -> Optional[float]:
    """Normalize a traffic_data entry (float or dict) into a multiplier."""
    if isinstance(entry, (int, float, np.number)):
        val = float(entry)
        return val if val > 0 else None

    if isinstance(entry, Mapping):
        for key in ("traffic_multiplier", "multiplier", "congestion_factor"):
            if key in entry and entry[key] is not None:
                val = float(entry[key])
                return val if val > 0 else None

        duration = entry.get("duration", entry.get("duration_seconds"))
        free_flow = entry.get(
            "free_flow_duration",
            entry.get("static_duration", entry.get("free_flow_seconds")),
        )
        if duration is not None and free_flow is not None and float(free_flow) > 0:
            ratio = np.divide(np.float64(duration), np.float64(free_flow))
            return float(np.maximum(1.0, ratio))

    return None


def _resolve_edge_traffic_multiplier(
    u: Any,
    v: Any,
    key: Optional[Any],
    edge_data: Mapping[str, Any],
    traffic_data: Optional[Mapping[Any, Any]],
    hour: Optional[Union[int, float]],
) -> float:
    """Look up an edge's traffic multiplier or fall back to time-of-day estimation."""
    if traffic_data:
        lookup_keys = []
        if key is not None:
            lookup_keys.append((u, v, key))
        lookup_keys.extend([(u, v), f"{u}_{v}", f"{u}->{v}"])

        osmid = edge_data.get("osmid")
        if isinstance(osmid, (int, str)):
            lookup_keys.append(osmid)
        elif isinstance(osmid, list):
            lookup_keys.extend(item for item in osmid if isinstance(item, (int, str)))

        for k in lookup_keys:
            if k in traffic_data:
                multiplier = _extract_multiplier_from_entry(traffic_data[k])
                if multiplier is not None:
                    return multiplier

    return estimate_traffic_multiplier(hour=hour)


def assign_edge_weights(
    G: nx.Graph,
    weather_severity: Union[str, int, float] = "clear",
    traffic_data: Optional[Mapping[Any, Any]] = None,
    hour: Optional[Union[int, float]] = None,
    is_ambulance: bool = True,
    global_traffic_multiplier: Optional[float] = None,
    weather_info: Optional[Mapping[str, Any]] = None,
) -> nx.Graph:
    """Compute and assign dynamic ``'weight'`` attributes to all edges in graph ``G``.

    Supports ``weather_severity`` directly or via ``weather_info`` dict, and
    ``global_traffic_multiplier`` (e.g., from Google Maps Routes API
    ``duration / staticDuration``) when edge-specific ``traffic_data`` is absent.
    """
    if weather_info is not None and isinstance(weather_info, Mapping):
        if "impact_multiplier" in weather_info:
            weather_multiplier = float(weather_info["impact_multiplier"])
        elif "severity" in weather_info:
            weather_multiplier = get_weather_multiplier(weather_info["severity"])
        else:
            weather_multiplier = get_weather_multiplier(weather_severity)
    else:
        weather_multiplier = get_weather_multiplier(weather_severity)

    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, data) for u, v, data in G.edges(data=True))

    for u, v, key, data in edge_iter:
        raw_length = data.get("length", data.get("length_m", _DEFAULT_EDGE_LENGTH_M))
        length_m = float(raw_length) if raw_length is not None else _DEFAULT_EDGE_LENGTH_M
        if length_m < 0:
            length_m = 0.0

        speed_kmh = _extract_edge_speed_kmh(data)
        base_time = calculate_base_travel_time(length_m, speed_kmh)

        if traffic_data:
            traffic_multiplier = _resolve_edge_traffic_multiplier(
                u=u,
                v=v,
                key=key,
                edge_data=data,
                traffic_data=traffic_data,
                hour=hour,
            )
        elif global_traffic_multiplier is not None and float(global_traffic_multiplier) > 0:
            # Apply Google Maps corridor congestion factor with minor per-road-class adjustment
            hw = data.get("highway")
            hw_str = str(hw[0] if isinstance(hw, (list, tuple)) and hw else hw).lower()
            road_congestion_bias = 1.05 if hw_str in {"primary", "trunk", "secondary"} else 0.95
            traffic_multiplier = float(np.maximum(1.0, float(global_traffic_multiplier) * road_congestion_bias))
        else:
            traffic_multiplier = estimate_traffic_multiplier(hour=hour)

        signal_count = _count_edge_signals(G, u, v, data)
        signal_delay = float(np.multiply(signal_count, TRAFFIC_SIGNAL_DELAY))
        road_delay = calculate_road_specific_delay(data)

        weight = calculate_edge_weight(
            length_m=length_m,
            speed_kmh=speed_kmh,
            traffic_multiplier=traffic_multiplier,
            weather_multiplier=weather_multiplier,
            signal_delay=signal_delay,
            road_delay=road_delay,
            is_ambulance=is_ambulance,
        )

        data["base_travel_time"] = base_time
        data["traffic_multiplier"] = traffic_multiplier
        data["weather_multiplier"] = weather_multiplier
        data["signal_delay"] = signal_delay
        data["road_delay"] = road_delay
        data["weight"] = weight

    return G
