"""Visualization module for the Ambulance Route Optimization System.

Creates interactive Folium route maps, ACO convergence plots,
comparison bar charts, formatted comparison tables, and hospital markers.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import folium


BASE_DIR = r"c:\Users\archi\Desktop\Minor_Project"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RANCHI_CENTER: Tuple[float, float] = (23.3441, 85.3096)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _ensure_parent_dir(filepath: str) -> None:
    """Ensure the output directory and parent directory of filepath exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _extract_numeric(value: Any, default: float = 0.0) -> float:
    """Extract a numeric value from a scalar or nested dict (e.g. {'value': 120})."""
    if value is None:
        return default
    if isinstance(value, dict):
        if "value" in value:
            return _extract_numeric(value["value"], default)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_travel_time(result: Optional[Dict[str, Any]]) -> float:
    """Extract travel time in seconds from a route result dictionary."""
    if not isinstance(result, dict):
        return 0.0
    for key in (
        "best_cost",
        "travel_time",
        "duration",
        "duration_seconds",
        "duration_in_traffic",
        "time",
        "total_time",
        "best_time",
        "eta",
        "cost",
        "travel_time_sec",
    ):
        if key in result and result[key] is not None:
            return _extract_numeric(result[key])
    if "time_min" in result and result["time_min"] is not None:
        return _extract_numeric(result["time_min"]) * 60.0
    return 0.0


def _get_distance(result: Optional[Dict[str, Any]]) -> float:
    """Extract distance in meters from a route result dictionary."""
    if not isinstance(result, dict):
        return 0.0
    for key in (
        "distance",
        "total_distance",
        "distance_meters",
        "distance_m",
        "length",
    ):
        if key in result and result[key] is not None:
            return _extract_numeric(result[key])
    if "distance_km" in result and result["distance_km"] is not None:
        return _extract_numeric(result["distance_km"]) * 1000.0
    return 0.0


def _node_to_latlon(G: Any, node: Any) -> Optional[Tuple[float, float]]:
    """Convert a graph node ID (or coordinate pair) to (lat, lon)."""
    if isinstance(node, (tuple, list)) and len(node) >= 2:
        return (float(node[0]), float(node[1]))

    if G is not None and hasattr(G, "nodes") and node in G.nodes:
        attrs = G.nodes[node]
        if "y" in attrs and "x" in attrs:
            return (float(attrs["y"]), float(attrs["x"]))
        if "lat" in attrs and "lon" in attrs:
            return (float(attrs["lat"]), float(attrs["lon"]))
        if "lat" in attrs and "lng" in attrs:
            return (float(attrs["lat"]), float(attrs["lng"]))
        if "latitude" in attrs and "longitude" in attrs:
            return (float(attrs["latitude"]), float(attrs["longitude"]))
    return None


def _extract_coords_and_meta(
    loc: Any, G: Any = None
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    """Normalize a location input into ((lat, lon), metadata_dict)."""
    if loc is None:
        return None, {}

    if isinstance(loc, dict):
        lat = loc.get("lat", loc.get("latitude", loc.get("y")))
        lon = loc.get("lon", loc.get("lng", loc.get("longitude", loc.get("x"))))
        if lat is not None and lon is not None:
            return (float(lat), float(lon)), loc
        for nested_key in ("coords", "location", "point", "position"):
            if nested_key in loc and loc[nested_key] is not None:
                coords, _ = _extract_coords_and_meta(loc[nested_key], G=G)
                if coords is not None:
                    return coords, loc
        if "node" in loc:
            return _node_to_latlon(G, loc["node"]), loc
        return None, loc

    if isinstance(loc, (tuple, list)) and len(loc) >= 2:
        if isinstance(loc[0], (int, float, np.number)) and isinstance(
            loc[1], (int, float, np.number)
        ):
            return (float(loc[0]), float(loc[1])), {}

    return _node_to_latlon(G, loc), {}


def _normalize_points(points: Optional[Iterable[Any]]) -> List[Tuple[float, float]]:
    """Convert a sequence of coordinate representations to [(lat, lon), ...]."""
    if not points:
        return []
    normalized: List[Tuple[float, float]] = []
    for pt in points:
        coords, _ = _extract_coords_and_meta(pt)
        if coords is not None:
            normalized.append(coords)
    return normalized


def _format_duration(seconds: float) -> str:
    """Format duration in seconds into a readable string."""
    mins = seconds / 60.0
    return f"{seconds:.1f}s ({mins:.1f} min)"


def _format_distance(meters: float) -> str:
    """Format distance in meters into a readable string."""
    km = meters / 1000.0
    return f"{meters:.0f}m ({km:.2f} km)"


def create_route_map(
    G: Any,
    aco_path: Sequence[Any],
    gmaps_points: Sequence[Any],
    ambulance_loc: Any,
    patient_loc: Any,
    hospital_loc: Any,
    aco_result: Optional[Dict[str, Any]],
    gmaps_result: Optional[Dict[str, Any]],
    filename: str = os.path.join(OUTPUT_DIR, "route_map.html"),
) -> folium.Map:
    """Create an interactive Folium map comparing ACO and Google Maps routes.

    Args:
        G: Road network graph containing node coordinates.
        aco_path: Sequence of graph node IDs representing the ACO route.
        gmaps_points: Sequence of (lat, lon) points for the Google Maps route.
        ambulance_loc: Ambulance starting location (coords, dict, or node ID).
        patient_loc: Patient location (coords, dict, or node ID).
        hospital_loc: Destination hospital location (coords, dict, or node ID).
        aco_result: Dictionary with ACO travel_time and distance metrics.
        gmaps_result: Dictionary with Google Maps travel_time and distance metrics.
        filename: Output HTML file path.

    Returns:
        The generated folium.Map object.
    """
    _ensure_parent_dir(filename)

    route_map = folium.Map(location=list(RANCHI_CENTER), zoom_start=13)

    aco_time = _get_travel_time(aco_result)
    aco_dist = _get_distance(aco_result)
    gmaps_time = _get_travel_time(gmaps_result)
    gmaps_dist = _get_distance(gmaps_result)
    is_gmaps_est = bool((gmaps_result or {}).get("is_estimate", False)) or (
        (gmaps_result or {}).get("source") != "google_maps_api"
    )
    gmaps_label = "Offline Baseline Estimate" if is_gmaps_est else "Google Maps Traffic-Aware ETA"

    # Convert ACO node IDs to (lat, lon) coordinates and draw solid green line
    aco_coords: List[Tuple[float, float]] = []
    if aco_path:
        for node in aco_path:
            coord = _node_to_latlon(G, node)
            if coord is not None:
                aco_coords.append(coord)

    if len(aco_coords) >= 2:
        aco_info = (
            f"<b>ACO Route</b><br>"
            f"Travel Time: {_format_duration(aco_time)}<br>"
            f"Distance: {_format_distance(aco_dist)}"
        )
        folium.PolyLine(
            locations=aco_coords,
            color="green",
            weight=5,
            opacity=0.8,
            tooltip=f"ACO Route: {_format_duration(aco_time)}, {_format_distance(aco_dist)}",
            popup=folium.Popup(aco_info, max_width=260),
        ).add_to(route_map)

    # Draw Google Maps route as dashed blue line
    gmaps_coords = _normalize_points(gmaps_points)
    if len(gmaps_coords) >= 2:
        gmaps_info = (
            f"<b>{gmaps_label}</b><br>"
            f"Normal Traffic ETA: {_format_duration(gmaps_time)}<br>"
            f"Distance: {_format_distance(gmaps_dist)}<br>"
            f"<i>Note: Represents regular car traffic, not emergency ambulance priority.</i>"
        )
        folium.PolyLine(
            locations=gmaps_coords,
            color="blue",
            weight=4,
            opacity=0.7,
            dash_array="10 10",
            tooltip=f"{gmaps_label}: {_format_duration(gmaps_time)}, {_format_distance(gmaps_dist)}",
            popup=folium.Popup(gmaps_info, max_width=280),
        ).add_to(route_map)

    # Ambulance marker (red icon with 'ambulance' prefix)
    amb_coords, amb_meta = _extract_coords_and_meta(ambulance_loc, G=G)
    if amb_coords is not None:
        amb_label = amb_meta.get("name", amb_meta.get("id", "Ambulance"))
        amb_popup = (
            f"<b>{amb_label}</b><br>"
            f"Status: Dispatched<br>"
            f"Location: ({amb_coords[0]:.5f}, {amb_coords[1]:.5f})<br>"
            f"ACO ETA: {_format_duration(aco_time)}"
        )
        folium.Marker(
            location=list(amb_coords),
            popup=folium.Popup(amb_popup, max_width=260),
            tooltip="Ambulance Location",
            icon=folium.Icon(color="red", icon="ambulance", prefix="fa"),
        ).add_to(route_map)

    # Patient marker (orange icon with 'user' prefix)
    pat_coords, pat_meta = _extract_coords_and_meta(patient_loc, G=G)
    if pat_coords is not None:
        pat_label = pat_meta.get("name", "Patient")
        pat_popup = (
            f"<b>{pat_label}</b><br>"
            f"Location: ({pat_coords[0]:.5f}, {pat_coords[1]:.5f})"
        )
        folium.Marker(
            location=list(pat_coords),
            popup=folium.Popup(pat_popup, max_width=250),
            tooltip="Patient Location",
            icon=folium.Icon(color="orange", icon="user", prefix="fa"),
        ).add_to(route_map)

    # Hospital marker (blue icon with 'plus-sign' prefix)
    hosp_coords, hosp_meta = _extract_coords_and_meta(hospital_loc, G=G)
    if hosp_coords is not None:
        hosp_name = hosp_meta.get(
            "name",
            hosp_meta.get(
                "hospital_name",
                (aco_result or {}).get(
                    "hospital_name",
                    (gmaps_result or {}).get("hospital_name", "Destination Hospital"),
                ),
            ),
        )
        hosp_popup = (
            f"<b>{hosp_name}</b><br>"
            f"Location: ({hosp_coords[0]:.5f}, {hosp_coords[1]:.5f})"
        )
        folium.Marker(
            location=list(hosp_coords),
            popup=folium.Popup(hosp_popup, max_width=260),
            tooltip=hosp_name,
            icon=folium.Icon(color="blue", icon="plus-sign", prefix="glyphicon"),
        ).add_to(route_map)

    # Interactive HTML legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #555;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        <div style="font-weight: bold; margin-bottom: 6px;">Ranchi Ambulance Route Legend</div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 28px; height: 0; border-top: 4px solid green; margin-right: 8px;"></span>
            <span>ACO Ambulance Est.: {_format_duration(aco_time)}, {_format_distance(aco_dist)}</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 28px; height: 0; border-top: 4px dashed blue; margin-right: 8px;"></span>
            <span>{gmaps_label}: {_format_duration(gmaps_time)}, {_format_distance(gmaps_dist)}</span>
        </div>
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(legend_html))

    # Fit bounds around route & active markers if available
    bound_points = list(aco_coords) + list(gmaps_coords)
    for pt in (amb_coords, pat_coords, hosp_coords):
        if pt is not None:
            bound_points.append(pt)
    if len(bound_points) >= 2:
        route_map.fit_bounds(bound_points, padding=(30, 30))

    route_map.save(filename)
    return route_map


def plot_convergence(
    convergence_data: Union[Sequence[float], Sequence[Tuple[int, float]], Dict[int, float]],
    filename: str = os.path.join(OUTPUT_DIR, "convergence.png"),
) -> None:
    """Plot the ACO convergence curve (best travel time per iteration) and save as PNG.

    Args:
        convergence_data: Sequence of best travel times per iteration, sequence of
            (iteration, best_time) tuples, or mapping of iteration -> best_time.
        filename: Output PNG file path.
    """
    _ensure_parent_dir(filename)

    if isinstance(convergence_data, dict):
        iterations = sorted(convergence_data.keys())
        costs = [float(convergence_data[k]) for k in iterations]
    elif convergence_data and isinstance(convergence_data[0], (tuple, list)):
        iterations = [int(item[0]) for item in convergence_data]
        costs = [float(item[1]) for item in convergence_data]
    else:
        costs = [float(x) for x in (convergence_data or [])]
        iterations = list(range(1, len(costs) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    if costs:
        ax.plot(
            iterations,
            costs,
            color="green",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Best Travel Time",
        )
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "No convergence data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Travel Time (seconds)", fontsize=11)
    ax.set_title("ACO Convergence - Best Travel Time per Iteration", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_comparison_chart(
    aco_result: Optional[Dict[str, Any]],
    gmaps_result: Optional[Dict[str, Any]],
    filename: str = os.path.join(OUTPUT_DIR, "comparison.png"),
) -> None:
    """Create a side-by-side bar chart comparing ACO and Google Maps routes.

    Args:
        aco_result: Dictionary with ACO travel_time and distance metrics.
        gmaps_result: Dictionary with Google Maps travel_time and distance metrics.
        filename: Output PNG file path.
    """
    _ensure_parent_dir(filename)

    aco_time = _get_travel_time(aco_result)
    gmaps_time = _get_travel_time(gmaps_result)
    aco_dist = _get_distance(aco_result)
    gmaps_dist = _get_distance(gmaps_result)

    labels = [
        "ACO\n(Ambulance Est.)",
        "Offline Baseline\n(Est.)"
        if (
            bool((gmaps_result or {}).get("is_estimate", False))
            or (gmaps_result or {}).get("source") != "google_maps_api"
        )
        else "Google Maps\n(Traffic ETA)",
    ]
    colors = ["green", "blue"]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Travel Time (seconds)
    time_values = [aco_time, gmaps_time]
    bars1 = ax1.bar(x, time_values, color=colors, width=0.5, alpha=0.85, edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Travel Time (seconds)", fontsize=11)
    ax1.set_title("Travel Time Comparison", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    max_time = max(time_values) if max(time_values) > 0 else 1.0
    ax1.set_ylim(0, max_time * 1.18)
    for bar, val in zip(bars1, time_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_time * 0.02,
            f"{val:.1f} s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Subplot 2: Distance (meters)
    dist_values = [aco_dist, gmaps_dist]
    bars2 = ax2.bar(x, dist_values, color=colors, width=0.5, alpha=0.85, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Distance (meters)", fontsize=11)
    ax2.set_title("Route Distance Comparison", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    max_dist = max(dist_values) if max(dist_values) > 0 else 1.0
    ax2.set_ylim(0, max_dist * 1.18)
    for bar, val in zip(bars2, dist_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_dist * 0.02,
            f"{val:.1f} m",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("ACO vs Google Maps Route Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def create_comparison_table(
    aco_result: Optional[Dict[str, Any]],
    gmaps_result: Optional[Dict[str, Any]],
    aco_execution_time: float,
) -> str:
    """Create a formatted string table comparing ACO and Google Maps route metrics.

    Args:
        aco_result: Dictionary with ACO travel_time and distance metrics.
        gmaps_result: Dictionary with Google Maps travel_time and distance metrics.
        aco_execution_time: Wall-clock execution time of ACO in seconds.

    Returns:
        Formatted comparison table string.
    """
    aco_time = _get_travel_time(aco_result)
    gmaps_time = _get_travel_time(gmaps_result)
    aco_dist = _get_distance(aco_result)
    gmaps_dist = _get_distance(gmaps_result)
    exec_time = _extract_numeric(aco_execution_time)

    time_diff_sec = aco_time - gmaps_time
    if gmaps_time != 0:
        time_diff_pct = (time_diff_sec / gmaps_time) * 100.0
        pct_str = f"{time_diff_pct:+.2f}%"
    else:
        pct_str = "N/A"

    is_gmaps_est = bool((gmaps_result or {}).get("is_estimate", False)) or (
        (gmaps_result or {}).get("source") != "google_maps_api"
    )
    gmaps_eta_label = (
        "Offline Baseline ETA (Estimate)"
        if is_gmaps_est
        else "Google Maps Traffic-Aware ETA (Car)"
    )
    gmaps_dist_label = (
        "Offline Baseline Distance (Estimate)"
        if is_gmaps_est
        else "Google Maps Route Distance"
    )

    rows = [
        ("ACO Estimated Ambulance Travel Time", f"{aco_time:.2f} s ({aco_time / 60.0:.2f} min)"),
        (gmaps_eta_label, f"{gmaps_time:.2f} s ({gmaps_time / 60.0:.2f} min)"),
        ("Time Difference (ACO - Reference)", f"{time_diff_sec:+.2f} s ({pct_str})"),
        ("ACO Route Distance", f"{aco_dist:.2f} m ({aco_dist / 1000.0:.2f} km)"),
        (gmaps_dist_label, f"{gmaps_dist:.2f} m ({gmaps_dist / 1000.0:.2f} km)"),
        ("ACO Execution Time", f"{exec_time:.4f} s"),
    ]

    metric_col_width = max(len("Metric"), *(len(r[0]) for r in rows)) + 2
    value_col_width = max(len("Value"), *(len(r[1]) for r in rows)) + 2

    separator = f"+{'-' * (metric_col_width + 2)}+{'-' * (value_col_width + 2)}+"
    header = f"| {'Metric':<{metric_col_width}} | {'Value':<{value_col_width}} |"
    title = "Route Comparison Summary (ACO vs Google Maps)"

    lines = [title, separator, header, separator]
    for metric, value in rows:
        lines.append(f"| {metric:<{metric_col_width}} | {value:<{value_col_width}} |")
    lines.append(separator)

    return "\n".join(lines)


def add_hospital_markers(
    map_obj: folium.Map,
    hospitals: Union[Sequence[Any], Dict[Any, Any]],
) -> folium.Map:
    """Add markers for all hospitals in Ranchi onto an existing Folium map.

    Uses purple icons to stay visually distinct from route markers (red, orange, blue).

    Args:
        map_obj: Existing folium.Map object to annotate.
        hospitals: Collection of hospitals (list of dicts, dict of name->coords/dict,
            or list of (name, lat, lon) / (lat, lon, name) tuples).

    Returns:
        The updated folium.Map object.
    """
    if not hospitals:
        return map_obj

    entries: List[Tuple[str, Tuple[float, float], Dict[str, Any]]] = []

    if isinstance(hospitals, dict):
        for name, data in hospitals.items():
            coords, meta = _extract_coords_and_meta(data)
            if coords is not None:
                hosp_name = str(meta.get("name", meta.get("hospital_name", name)))
                entries.append((hosp_name, coords, meta))
    else:
        for item in hospitals:
            if isinstance(item, dict):
                coords, meta = _extract_coords_and_meta(item)
                if coords is not None:
                    hosp_name = str(
                        meta.get("name", meta.get("hospital_name", "Hospital"))
                    )
                    entries.append((hosp_name, coords, meta))
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                if isinstance(item[0], str):
                    hosp_name = item[0]
                    coords = (float(item[1]), float(item[2]))
                elif isinstance(item[2], str):
                    hosp_name = item[2]
                    coords = (float(item[0]), float(item[1]))
                else:
                    continue
                entries.append((hosp_name, coords, {}))

    for hosp_name, coords, meta in entries:
        extra_info = ""
        if "address" in meta:
            extra_info += f"<br>{meta['address']}"
        if "phone" in meta:
            extra_info += f"<br>Phone: {meta['phone']}"

        popup_html = (
            f"<b>{hosp_name}</b>"
            f"{extra_info}"
            f"<br>Coords: ({coords[0]:.5f}, {coords[1]:.5f})"
        )
        folium.Marker(
            location=list(coords),
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=hosp_name,
            icon=folium.Icon(color="purple", icon="hospital-o", prefix="fa"),
        ).add_to(map_obj)

    return map_obj
