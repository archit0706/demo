"""
Main entry point for the Ambulance Route Optimization System.

Orchestrates road network loading, weather severity prediction, dynamic edge
weighting, Ant Colony Optimization (ACO), Dijkstra baseline comparison,
Google Maps (or offline) route estimation, evaluation, and visualization.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aco
import config
import edge_weight
import google_maps
import road_network
import visualization
import weather_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the route optimization pipeline."""
    now = datetime.now()
    parser = argparse.ArgumentParser(
        description="Ambulance Route Optimization System using ACO and Dynamic Weather/Traffic Weights"
    )
    parser.add_argument(
        "--ambulance-lat",
        type=float,
        default=23.3500,
        help="Ambulance starting latitude (default: 23.3500)",
    )
    parser.add_argument(
        "--ambulance-lon",
        type=float,
        default=85.3100,
        help="Ambulance starting longitude (default: 85.3100)",
    )
    parser.add_argument(
        "--patient-lat",
        type=float,
        default=23.3600,
        help="Patient pickup latitude (default: 23.3600)",
    )
    parser.add_argument(
        "--patient-lon",
        type=float,
        default=85.3200,
        help="Patient pickup longitude (default: 85.3200)",
    )
    parser.add_argument(
        "--hospital",
        type=str,
        default=None,
        help="Target hospital name or 0-based index (default: auto-select nearest with emergency services)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=now.month,
        choices=range(1, 13),
        metavar="1-12",
        help=f"Month for weather prediction (1-12, default: {now.month})",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=now.year,
        help=f"Year for weather prediction (default: {now.year})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (skip Google Maps API calls)",
    )
    parser.add_argument(
        "--train-weather",
        action="store_true",
        help="Retrain the weather prediction model before running",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip interactive HTML map generation",
    )
    return parser.parse_args()


def resolve_offline_mode(offline_flag: bool) -> Tuple[bool, Optional[str]]:
    """Determine whether to run in offline mode based on flag and API key."""
    google_maps.load_env_file()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or getattr(
        config, "GOOGLE_MAPS_API_KEY", None
    )

    if offline_flag:
        return True, None

    if not api_key or api_key.strip() in {"", "your_api_key_here"}:
        print(
            "[WARN] GOOGLE_MAPS_API_KEY is not set or still placeholder. "
            "Automatically switching to offline mode."
        )
        return True, None

    return False, api_key.strip()


def print_banner(args: argparse.Namespace, offline_mode: bool) -> None:
    """Display the startup banner and run configuration."""
    width = 68
    print("=" * width)
    print("AMBULANCE ROUTE OPTIMIZATION SYSTEM".center(width))
    print("ACO + Dynamic Weather & Traffic Weighting".center(width))
    print("=" * width)
    print(f"  Ambulance Location : ({args.ambulance_lat:.4f}, {args.ambulance_lon:.4f})")
    print(f"  Patient Location   : ({args.patient_lat:.4f}, {args.patient_lon:.4f})")
    print(f"  Prediction Period  : {args.month:02d}/{args.year}")
    print(f"  Routing Mode       : {'Offline (Estimated)' if offline_mode else 'Online (Google Maps API)'}")
    print(f"  Retrain Weather    : {'Yes' if args.train_weather else 'No'}")
    print(f"  Generate Map       : {'No' if args.no_map else 'Yes'}")
    print("=" * width)


def initialize_components(train_weather: bool) -> Tuple[Any, Any]:
    """Load the road network graph and load or train the weather model."""
    print("\n[STEP 1/7] Initializing System Components...")

    try:
        print("  -> Loading road network...")
        graph = road_network.load_road_network()
        print(
            f"  [OK] Road network loaded: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges."
        )
    except Exception as exc:
        print(f"  [ERROR] Failed to load road network: {exc}")
        raise

    try:
        if train_weather:
            print("  -> Training weather prediction model (--train-weather)...")
            model = weather_model.train_weather_model()
            print("  [OK] Weather model trained and saved.")
        else:
            print("  -> Loading weather prediction model...")
            model = weather_model.load_or_train_model()
            print("  [OK] Weather model ready.")
    except Exception as exc:
        print(f"  [ERROR] Failed to initialize weather model: {exc}")
        raise

    return graph, model


def select_target_hospital(
    hospital_arg: Optional[str], patient_lat: float, patient_lon: float
) -> Dict[str, Any]:
    """Select the specified hospital or auto-select the nearest emergency hospital."""
    print("\n[STEP 2/7] Selecting Target Hospital...")

    try:
        if hospital_arg is not None:
            hospital = road_network.get_hospital_by_identifier(
                hospital_arg, patient_lat, patient_lon
            )
        else:
            hospital = road_network.find_nearest_hospital(
                patient_lat, patient_lon, emergency_only=True
            )

        print(f"  [OK] Selected Hospital : {hospital['name']}")
        print(
            f"       Coordinates       : ({hospital['lat']:.4f}, {hospital['lon']:.4f})"
        )
        if "distance_km" in hospital:
            print(f"       Straight Distance : {hospital['distance_km']:.2f} km")
        if "emergency" in hospital:
            print(
                f"       Emergency Service : {'Available' if hospital['emergency'] else 'Unavailable'}"
            )
        return hospital
    except Exception as exc:
        print(f"  [ERROR] Could not select a target hospital: {exc}")
        raise


def predict_and_display_weather(
    _model: Any, month: int, year: int
) -> Dict[str, Any]:
    """Predict weather conditions and impact multiplier for the target month/year."""
    print("\n[STEP 3/7] Predicting Weather Conditions...")

    try:
        weather_info = weather_model.predict_weather_details(month=month, year=year)
        severity = weather_info.get("severity", "Unknown")
        multiplier = float(weather_info.get("impact_multiplier", 1.0))

        print(f"  [OK] Predicted Severity : {severity}")
        if "lst_c" in weather_info:
            print(f"       Monthly Avg LST    : {weather_info['lst_c']:.2f} C")
        if "ndvi" in weather_info:
            print(f"       Monthly Avg NDVI   : {weather_info['ndvi']:.3f}")
        print(f"       Impact Multiplier  : {multiplier:.3f}x")
        return weather_info
    except Exception as exc:
        print(
            f"  [WARN] Weather prediction failed ({exc}). "
            "Falling back to clear weather conditions (1.0x)."
        )
        return {
            "severity": "clear",
            "impact_multiplier": 1.0,
        }


def apply_dynamic_weights(
    graph: Any,
    weather_info: Dict[str, Any],
    global_traffic_multiplier: Optional[float] = None,
) -> Any:
    """Assign dynamic edge weights incorporating weather and traffic factors."""
    print("\n[STEP 4/7] Assigning Dynamic Edge Weights...")

    try:
        weighted_graph = edge_weight.assign_edge_weights(
            graph,
            weather_info=weather_info,
            global_traffic_multiplier=global_traffic_multiplier,
        )
        source_note = (
            f"Google Maps live congestion factor ({global_traffic_multiplier:.2f}x)"
            if global_traffic_multiplier
            else "offline time-of-day traffic congestion estimate"
        )
        print(f"  [OK] Dynamic edge weights applied ({source_note}).")
        return weighted_graph
    except Exception as exc:
        print(f"  [ERROR] Failed to assign dynamic edge weights: {exc}")
        raise


def fetch_reference_route(
    origin_loc: Tuple[float, float],
    dest_loc: Tuple[float, float],
    offline_mode: bool,
    api_key: Optional[str],
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Query Google Maps Routes API (or offline estimate) and extract traffic multiplier."""
    try:
        if offline_mode:
            print("  -> Computing offline reference route estimate...")
            gmaps_result = google_maps.get_offline_estimate(
                origin_loc[0], origin_loc[1], dest_loc[0], dest_loc[1]
            )
            print(
                f"  [OK] Offline Estimate: {gmaps_result['distance_km']:.2f} km, "
                f"{gmaps_result['time_min']:.2f} min"
            )
            return gmaps_result, None

        print("  -> Querying Google Maps Routes API...")
        gmaps_result = google_maps.get_google_maps_route(
            origin_loc[0], origin_loc[1], dest_loc[0], dest_loc[1], api_key=api_key
        )
        print(
            f"  [OK] Google Maps Route ({gmaps_result.get('source')}): "
            f"{gmaps_result['distance_km']:.2f} km, {gmaps_result['time_min']:.2f} min"
        )
        traffic_mult = google_maps.extract_traffic_multiplier(gmaps_result)
        return gmaps_result, traffic_mult
    except Exception as exc:
        print(
            f"  [WARN] Google Maps lookup failed ({exc}). "
            "Falling back to offline estimate."
        )
        gmaps_result = google_maps.get_offline_estimate(
            origin_loc[0], origin_loc[1], dest_loc[0], dest_loc[1]
        )
        return gmaps_result, None


def compute_routes(
    graph: Any,
    ambulance_loc: Tuple[float, float],
    patient_loc: Tuple[float, float],
    hospital: Dict[str, Any],
    gmaps_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Find nearest graph nodes and compute ACO and Dijkstra routes.

    The full dispatch spans ambulance -> patient -> hospital; optimization targets
    the critical patient -> hospital segment while also estimating the pickup leg.
    """
    print("\n[STEP 5/7] Computing Optimal Routes...")

    hospital_loc = (hospital["lat"], hospital["lon"])

    try:
        ambulance_node = road_network.find_nearest_node(
            graph, ambulance_loc[0], ambulance_loc[1]
        )
        patient_node = road_network.find_nearest_node(
            graph, patient_loc[0], patient_loc[1]
        )
        hospital_node = road_network.find_nearest_node(
            graph, hospital_loc[0], hospital_loc[1]
        )
        print(
            f"  [OK] Nearest Nodes -> Ambulance: {ambulance_node}, "
            f"Patient: {patient_node}, Hospital: {hospital_node}"
        )
    except Exception as exc:
        print(f"  [ERROR] Failed to snap locations to road network nodes: {exc}")
        raise

    # Pickup leg context (ambulance -> patient)
    pickup_info: Dict[str, Any] = {
        "ambulance_node": ambulance_node,
        "patient_node": patient_node,
        "hospital_node": hospital_node,
    }
    try:
        pickup_res = aco.run_dijkstra(graph, ambulance_node, patient_node)
        pickup_info["distance_km"] = pickup_res.get("distance_km", 0.0)
        pickup_info["time_min"] = pickup_res.get("time_min", 0.0)
        pickup_info["route"] = pickup_res.get("route", [])
        print(
            f"  [INFO] Pickup Leg (Ambulance -> Patient): "
            f"{pickup_info['distance_km']:.2f} km, {pickup_info['time_min']:.2f} min"
        )
    except Exception as exc:
        print(f"  [WARN] Could not estimate ambulance -> patient leg: {exc}")

    # Critical segment: Patient -> Hospital via ACO
    try:
        print("  -> Running Ant Colony Optimization (Patient -> Hospital)...")
        aco_result = aco.run_aco(graph, patient_node, hospital_node)
        print(
            f"  [OK] ACO Route: {aco_result['distance_km']:.2f} km, "
            f"{aco_result['time_min']:.2f} min "
            f"(computed in {aco_result.get('execution_time', 0.0):.3f}s)"
        )
    except Exception as exc:
        print(f"  [ERROR] ACO optimization failed: {exc}")
        raise

    # Baseline: Dijkstra
    try:
        print("  -> Running Dijkstra shortest-path comparison...")
        dijkstra_result = aco.run_dijkstra(graph, patient_node, hospital_node)
        print(
            f"  [OK] Dijkstra Route: {dijkstra_result['distance_km']:.2f} km, "
            f"{dijkstra_result['time_min']:.2f} min"
        )
    except Exception as exc:
        print(f"  [WARN] Dijkstra comparison failed ({exc}).")
        dijkstra_result = {
            "route": [],
            "distance_km": 0.0,
            "time_min": 0.0,
            "error": str(exc),
        }

    return pickup_info, aco_result, dijkstra_result, gmaps_result


def _pct_diff(reference: float, candidate: float) -> Optional[float]:
    """Return percentage difference ((candidate - reference) / reference * 100)."""
    if reference <= 0:
        return None
    return ((candidate - reference) / reference) * 100.0


def evaluate_and_compare(
    aco_result: Dict[str, Any],
    dijkstra_result: Dict[str, Any],
    gmaps_result: Dict[str, Any],
    offline_mode: bool,
) -> Dict[str, Any]:
    """Evaluate route metrics, print comparison table, and return delta statistics."""
    print("\n[STEP 6/7] Evaluating & Comparing Routes...")

    gmaps_label = (
        "Offline Estimate"
        if offline_mode or "Offline" in str(gmaps_result.get("source", ""))
        else "Google Maps API"
    )

    rows = [
        ("ACO (Dynamic)", aco_result["distance_km"], aco_result["time_min"]),
        ("Dijkstra (Baseline)", dijkstra_result["distance_km"], dijkstra_result["time_min"]),
        (gmaps_label, gmaps_result["distance_km"], gmaps_result["time_min"]),
    ]

    print("  +----------------------+---------------+---------------+-------------+")
    print("  | Method               | Distance (km) | Time (min)    | ACO vs (%)  |")
    print("  +----------------------+---------------+---------------+-------------+")

    aco_time = float(aco_result.get("time_min", 0.0))
    for name, dist_km, time_min in rows:
        if name.startswith("ACO"):
            diff_str = "baseline"
        else:
            pct = _pct_diff(time_min, aco_time)
            diff_str = f"{pct:+.2f}%" if pct is not None else "N/A"
        print(
            f"  | {name:<20} | {dist_km:>13.2f} | {time_min:>13.2f} | {diff_str:>11} |"
        )
    print("  +----------------------+---------------+---------------+-------------+")

    dijkstra_time = float(dijkstra_result.get("time_min", 0.0))
    gmaps_time = float(gmaps_result.get("time_min", 0.0))

    metrics = {
        "aco_vs_dijkstra_diff_min": aco_time - dijkstra_time,
        "aco_vs_dijkstra_pct": _pct_diff(dijkstra_time, aco_time),
        "aco_vs_gmaps_diff_min": aco_time - gmaps_time,
        "aco_vs_gmaps_pct": _pct_diff(gmaps_time, aco_time),
        "gmaps_label": gmaps_label,
    }

    if dijkstra_time > 0:
        print(
            f"  * ACO vs Dijkstra   : {metrics['aco_vs_dijkstra_diff_min']:+.2f} min "
            f"({metrics['aco_vs_dijkstra_pct']:+.2f}%)"
        )
    if gmaps_time > 0:
        print(
            f"  * ACO vs {gmaps_label:<11}: {metrics['aco_vs_gmaps_diff_min']:+.2f} min "
            f"({metrics['aco_vs_gmaps_pct']:+.2f}%)"
        )

    return metrics


def generate_visualizations(
    graph: Any,
    aco_result: Dict[str, Any],
    dijkstra_result: Dict[str, Any],
    gmaps_result: Dict[str, Any],
    ambulance_loc: Tuple[float, float],
    patient_loc: Tuple[float, float],
    hospital: Dict[str, Any],
    skip_map: bool,
) -> Dict[str, Optional[str]]:
    """Generate interactive route map, ACO convergence curve, and comparison chart."""
    print("\n[STEP 7/7] Generating Visualizations...")
    outputs: Dict[str, Optional[str]] = {
        "map_path": None,
        "convergence_path": None,
        "comparison_path": None,
    }

    output_dir = Path(getattr(config, "OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_map:
        print("  [SKIP] Interactive map generation skipped (--no-map).")
    else:
        try:
            map_path = str(output_dir / "route_map.html")
            route_map = visualization.create_route_map(
                G=graph,
                aco_path=aco_result.get("route", []),
                gmaps_points=gmaps_result.get("route_points", []),
                ambulance_loc=ambulance_loc,
                patient_loc=patient_loc,
                hospital_loc=hospital,
                aco_result=aco_result,
                gmaps_result=gmaps_result,
                filename=map_path,
            )
            visualization.add_hospital_markers(route_map, config.HOSPITALS)
            route_map.save(map_path)
            outputs["map_path"] = map_path
            print(f"  [OK] Interactive map saved to: {map_path}")
        except Exception as exc:
            print(f"  [WARN] Failed to generate interactive map: {exc}")

    try:
        history = aco_result.get("convergence_history", [])
        if history:
            conv_path = str(output_dir / "convergence.png")
            visualization.plot_convergence(history, filename=conv_path)
            outputs["convergence_path"] = conv_path
            print(f"  [OK] Convergence curve saved to: {conv_path}")
        else:
            print("  [WARN] No convergence history available; skipping convergence plot.")
    except Exception as exc:
        print(f"  [WARN] Failed to plot convergence curve: {exc}")

    try:
        comp_path = str(output_dir / "comparison.png")
        visualization.plot_comparison_chart(
            aco_result=aco_result,
            gmaps_result=gmaps_result,
            filename=comp_path,
        )
        outputs["comparison_path"] = comp_path
        print(f"  [OK] Comparison chart saved to: {comp_path}")
    except Exception as exc:
        print(f"  [WARN] Failed to plot comparison chart: {exc}")

    try:
        table_text = visualization.create_comparison_table(
            aco_result=aco_result,
            gmaps_result=gmaps_result,
            aco_execution_time=aco_result.get("execution_time", 0.0),
        )
        print("\n" + table_text)
        report_path = str(output_dir / "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(table_text + "\n")
        outputs["report_path"] = report_path
    except Exception as exc:
        print(f"  [WARN] Failed to generate comparison table: {exc}")

    return outputs


def print_final_summary(
    hospital: Dict[str, Any],
    weather_info: Dict[str, Any],
    pickup_info: Dict[str, Any],
    aco_result: Dict[str, Any],
    dijkstra_result: Dict[str, Any],
    gmaps_result: Dict[str, Any],
    comparison_metrics: Dict[str, Any],
    outputs: Dict[str, Optional[str]],
) -> None:
    """Print final execution summary with key metrics and output file paths."""
    width = 68
    print("\n" + "=" * width)
    print("FINAL OPTIMIZATION SUMMARY".center(width))
    print("=" * width)
    print(f"  Target Hospital      : {hospital.get('name', 'Unknown')}")
    print(
        f"  Weather Condition    : {weather_info.get('severity', 'Unknown')} "
        f"({weather_info.get('impact_multiplier', 1.0):.3f}x)"
    )
    if "distance_km" in pickup_info:
        print(
            f"  Ambulance -> Patient : {pickup_info['distance_km']:.2f} km "
            f"({pickup_info['time_min']:.2f} min)"
        )
    print("  Critical Segment (Patient -> Hospital):")
    print(
        f"    - ACO Optimal      : {aco_result.get('distance_km', 0.0):.2f} km | "
        f"{aco_result.get('time_min', 0.0):.2f} min"
    )
    print(
        f"    - Dijkstra         : {dijkstra_result.get('distance_km', 0.0):.2f} km | "
        f"{dijkstra_result.get('time_min', 0.0):.2f} min"
    )
    print(
        f"    - {comparison_metrics.get('gmaps_label', 'Reference'):<16} : "
        f"{gmaps_result.get('distance_km', 0.0):.2f} km | "
        f"{gmaps_result.get('time_min', 0.0):.2f} min"
    )
    if comparison_metrics.get("aco_vs_dijkstra_pct") is not None:
        print(
            f"    - ACO vs Dijkstra  : {comparison_metrics['aco_vs_dijkstra_diff_min']:+.2f} min "
            f"({comparison_metrics['aco_vs_dijkstra_pct']:+.2f}%)"
        )
    if comparison_metrics.get("aco_vs_gmaps_pct") is not None:
        print(
            f"    - ACO vs Reference : {comparison_metrics['aco_vs_gmaps_diff_min']:+.2f} min "
            f"({comparison_metrics['aco_vs_gmaps_pct']:+.2f}%)"
        )

    print("\n  Output Files:")
    print(f"    - Interactive Map   : {outputs.get('map_path') or 'Not generated'}")
    print(f"    - Convergence Plot  : {outputs.get('convergence_path') or 'Not generated'}")
    print(f"    - Comparison Chart  : {outputs.get('comparison_path') or 'Not generated'}")
    print("=" * width)


def main() -> None:
    """Run the end-to-end Ambulance Route Optimization workflow."""
    args = parse_args()
    offline_mode, api_key = resolve_offline_mode(args.offline)
    print_banner(args, offline_mode)

    ambulance_loc = (args.ambulance_lat, args.ambulance_lon)
    patient_loc = (args.patient_lat, args.patient_lon)

    try:
        graph, model = initialize_components(args.train_weather)
        hospital = select_target_hospital(args.hospital, patient_loc[0], patient_loc[1])
        weather_info = predict_and_display_weather(model, args.month, args.year)
        hospital_loc = (hospital["lat"], hospital["lon"])
        gmaps_result, global_traffic_mult = fetch_reference_route(
            patient_loc, hospital_loc, offline_mode, api_key
        )
        weighted_graph = apply_dynamic_weights(
            graph, weather_info, global_traffic_multiplier=global_traffic_mult
        )
        pickup_info, aco_result, dijkstra_result, gmaps_result = compute_routes(
            weighted_graph,
            ambulance_loc,
            patient_loc,
            hospital,
            gmaps_result,
        )
        comparison_metrics = evaluate_and_compare(
            aco_result, dijkstra_result, gmaps_result, offline_mode
        )
        outputs = generate_visualizations(
            weighted_graph,
            aco_result,
            dijkstra_result,
            gmaps_result,
            ambulance_loc,
            patient_loc,
            hospital,
            args.no_map,
        )
        print_final_summary(
            hospital,
            weather_info,
            pickup_info,
            aco_result,
            dijkstra_result,
            gmaps_result,
            comparison_metrics,
            outputs,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Execution interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] Optimization pipeline aborted: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
