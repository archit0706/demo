"""Interactive Web Frontend for the Ranchi Ambulance Route Optimization System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure local modules resolve cleanly regardless of launch working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import aco
import config
import edge_weight
import google_maps
import road_network
import visualization
import weather_model

st.set_page_config(
    page_title="Ranchi Ambulance Route Optimizer",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANCHI_LANDMARKS: Dict[str, Tuple[float, float]] = {
    "Albert Ekka Chowk (City Center)": (23.3441, 85.3096),
    "Lalpur Chowk": (23.3736, 85.3358),
    "Kanke Road (CMPDI)": (23.4045, 85.3186),
    "Ranchi Railway Station": (23.3463, 85.3204),
    "Hinoo Chowk": (23.3228, 85.3145),
    "Harmu Housing Colony": (23.3612, 85.2980),
    "Ratu Road (Piska More)": (23.3812, 85.2947),
    "Namkum Industrial Area": (23.3472, 85.3745),
    "Dhurwa (HEC Township)": (23.3118, 85.2845),
    "Custom Coordinates": (23.3600, 85.3200),
}

AMBULANCE_BASES: Dict[str, Tuple[float, float]] = {
    "Main Road Dispatch (Default)": (23.3500, 85.3100),
    "Sadar EMS Hub": (23.3685, 85.3260),
    "Bariatu EMS Station": (23.3895, 85.3440),
    "Dhurwa EMS Point": (23.3180, 85.2810),
    "Namkum EMS Point": (23.3510, 85.3760),
    "Custom Coordinates": (23.3500, 85.3100),
}


@st.cache_resource(show_spinner="Loading Ranchi road network (36,571 nodes)...")
def get_cached_network():
    """Load the cached Ranchi drivable road network once per session."""
    return road_network.load_or_download_network()


@st.cache_resource(show_spinner="Loading trained Ranchi weather model...")
def get_cached_weather_model():
    """Load or train the Random Forest weather severity classifier."""
    return weather_model.load_or_train_model()


def _has_valid_api_key() -> Tuple[bool, Optional[str]]:
    """Check whether a non-placeholder GOOGLE_MAPS_API_KEY is available."""
    google_maps.load_env_file()
    key = google_maps._resolve_api_key()
    return (key is not None), key


def _summarize_route_components(G: Any, path: List[Any]) -> Dict[str, float]:
    """Sum dynamic edge-weight components along a node path."""
    if not path or len(path) < 2:
        return {
            "base_time_s": 0.0,
            "avg_traffic_mult": 1.0,
            "weather_mult": 1.0,
            "signal_delay_s": 0.0,
            "road_delay_s": 0.0,
            "edges": 0,
        }
    base_s = 0.0
    sig_s = 0.0
    road_s = 0.0
    traffic_vals: List[float] = []
    weather_vals: List[float] = []
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            continue
        edata = G[u][v]
        if G.is_multigraph():
            best_edge = min(edata.values(), key=lambda d: float(d.get("weight", 1.0)))
        else:
            best_edge = edata
        base_s += float(best_edge.get("base_travel_time", 0.0))
        sig_s += float(best_edge.get("signal_delay", 0.0))
        road_s += float(best_edge.get("road_delay", 0.0))
        traffic_vals.append(float(best_edge.get("traffic_multiplier", 1.0)))
        weather_vals.append(float(best_edge.get("weather_multiplier", 1.0)))

    return {
        "base_time_s": base_s,
        "avg_traffic_mult": float(np.mean(traffic_vals)) if traffic_vals else 1.0,
        "weather_mult": float(np.mean(weather_vals)) if weather_vals else 1.0,
        "signal_delay_s": sig_s,
        "road_delay_s": road_s,
        "edges": len(traffic_vals),
    }


def _build_interactive_map(
    G: Any,
    aco_path: List[Any],
    pickup_path: List[Any],
    gmaps_points: List[Tuple[float, float]],
    ambulance_loc: Tuple[float, float],
    patient_loc: Tuple[float, float],
    hospital: Dict[str, Any],
    aco_result: Dict[str, Any],
    gmaps_result: Dict[str, Any],
    pickup_info: Dict[str, Any],
) -> folium.Map:
    """Create the interactive Folium map with routes, pickup leg, and all Ranchi hospitals."""
    map_path = os.path.join(config.OUTPUT_DIR, "route_map.html")
    route_map = visualization.create_route_map(
        G=G,
        aco_path=aco_path,
        gmaps_points=gmaps_points,
        ambulance_loc=ambulance_loc,
        patient_loc=patient_loc,
        hospital_loc=hospital,
        aco_result=aco_result,
        gmaps_result=gmaps_result,
        filename=map_path,
    )

    # Optionally overlay Ambulance -> Patient dispatch leg as dotted orange line
    pickup_coords = [
        (float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
        for n in (pickup_path or [])
        if n in G.nodes and "y" in G.nodes[n] and "x" in G.nodes[n]
    ]
    if len(pickup_coords) >= 2:
        folium.PolyLine(
            locations=pickup_coords,
            color="#e67e22",
            weight=3,
            opacity=0.85,
            dash_array="4 8",
            tooltip=(
                f"Ambulance Pickup Leg: {pickup_info.get('time_min', 0.0):.2f} min, "
                f"{pickup_info.get('distance_km', 0.0):.2f} km"
            ),
        ).add_to(route_map)

    # Add all other Ranchi hospitals (excluding selected hospital to avoid duplicate icon)
    other_hospitals = [
        h for h in config.HOSPITALS if h["name"] != hospital.get("name")
    ]
    visualization.add_hospital_markers(route_map, other_hospitals)
    route_map.save(map_path)
    return route_map


def _plot_convergence_fig(convergence_data: List[float], dijkstra_s: float) -> plt.Figure:
    """Create a Matplotlib convergence figure with Dijkstra reference line."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    iterations = list(range(1, len(convergence_data) + 1))
    ax.plot(
        iterations,
        convergence_data,
        color="#1e8449",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="ACO Best Travel Time",
    )
    if np.isfinite(dijkstra_s) and dijkstra_s > 0:
        ax.axhline(
            dijkstra_s,
            color="#555555",
            linestyle="--",
            linewidth=1.4,
            label=f"Dijkstra Exact Optimum ({dijkstra_s:.1f}s)",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Travel Time (seconds)")
    ax.set_title("ACO Convergence — Best Travel Time per Iteration", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_comparison_fig(aco_result: Dict[str, Any], gmaps_result: Dict[str, Any]) -> plt.Figure:
    """Create a side-by-side comparison bar chart figure."""
    aco_time = float(aco_result.get("best_cost", 0.0))
    aco_dist = float(aco_result.get("distance_km", 0.0)) * 1000.0
    gmaps_time = float(gmaps_result.get("duration_seconds", 0.0))
    gmaps_dist = float(gmaps_result.get("distance_meters", 0.0))

    is_est = bool(gmaps_result.get("is_estimate", False)) or (
        gmaps_result.get("source") != "google_maps_api"
    )
    ref_label = "Offline Baseline\n(Est.)" if is_est else "Google Maps\n(Traffic ETA)"
    labels = ["ACO\n(Ambulance Est.)", ref_label]
    colors = ["#27ae60", "#2980b9"]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    times = [aco_time, gmaps_time]
    bars1 = ax1.bar(x, times, color=colors, width=0.48, edgecolor="black", alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Travel Time (seconds)")
    ax1.set_title("Travel Time Comparison")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    max_t = max(times) if max(times) > 0 else 1.0
    ax1.set_ylim(0, max_t * 1.18)
    for bar, val in zip(bars1, times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_t * 0.02,
            f"{val:.1f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    dists = [aco_dist, gmaps_dist]
    bars2 = ax2.bar(x, dists, color=colors, width=0.48, edgecolor="black", alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Distance (meters)")
    ax2.set_title("Route Distance Comparison")
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    max_d = max(dists) if max(dists) > 0 else 1.0
    ax2.set_ylim(0, max_d * 1.18)
    for bar, val in zip(bars2, dists):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_d * 0.02,
            f"{val:.0f} m",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.suptitle("ACO Ambulance Route vs Reference Route", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main() -> None:
    st.title("🚑 Ranchi Ambulance Route Optimization System")
    st.caption(
        "Dynamic Ant Colony Optimization (ACO) on OpenStreetMap Ranchi road network "
        "with ML Weather Severity Prediction & Google Maps Routes API comparison."
    )

    G = get_cached_network()
    _ = get_cached_weather_model()
    api_available, api_key = _has_valid_api_key()

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("Dispatch Configuration")

        st.subheader("1. Ambulance Location")
        amb_preset = st.selectbox(
            "Ambulance Preset", list(AMBULANCE_BASES.keys()), index=0
        )
        default_amb = AMBULANCE_BASES[amb_preset]
        c1, c2 = st.columns(2)
        amb_lat = c1.number_input(
            "Amb Lat", value=float(default_amb[0]), format="%.4f", step=0.0020
        )
        amb_lon = c2.number_input(
            "Amb Lon", value=float(default_amb[1]), format="%.4f", step=0.0020
        )

        st.subheader("2. Patient Location")
        pat_preset = st.selectbox(
            "Patient Preset", list(RANCHI_LANDMARKS.keys()), index=0
        )
        default_pat = RANCHI_LANDMARKS[pat_preset]
        c3, c4 = st.columns(2)
        pat_lat = c3.number_input(
            "Pat Lat", value=float(default_pat[0]), format="%.4f", step=0.0020
        )
        pat_lon = c4.number_input(
            "Pat Lon", value=float(default_pat[1]), format="%.4f", step=0.0020
        )

        st.subheader("3. Target Hospital")
        hospital_names = ["Auto-select Nearest Emergency Hospital"] + [
            h["name"] for h in config.HOSPITALS
        ]
        selected_hosp_option = st.selectbox("Hospital", hospital_names, index=0)

        st.subheader("4. Weather & Traffic")
        now = datetime.now()
        c5, c6 = st.columns(2)
        month = c5.selectbox("Month", list(range(1, 13)), index=now.month - 1)
        year = c6.number_input("Year", min_value=2010, max_value=2030, value=now.year)

        use_online_api = st.toggle(
            "Use Google Maps Routes API (Live Traffic)",
            value=api_available,
            disabled=not api_available,
            help=(
                "Queries Google Maps Routes API v2 for traffic-aware ETA and route geometry."
                if api_available
                else "Set GOOGLE_MAPS_API_KEY in .env or environment variables to enable."
            ),
        )
        if api_available:
            st.success("Google Maps API key detected (.env / environment).")
        else:
            st.info("No API key found — running in Offline Estimate mode.")

        with st.expander("ACO Hyperparameters", expanded=False):
            n_ants = st.slider("Number of Ants", 10, 60, int(config.ACO_PARAMS["n_ants"]), 5)
            n_iterations = st.slider("Iterations", 20, 150, int(config.ACO_PARAMS["n_iterations"]), 10)
            alpha = st.slider("Alpha (Pheromone)", 0.5, 3.0, float(config.ACO_PARAMS["alpha"]), 0.1)
            beta = st.slider("Beta (Heuristic)", 1.0, 5.0, float(config.ACO_PARAMS["beta"]), 0.5)
            rho = st.slider("Rho (Evaporation)", 0.05, 0.5, float(config.ACO_PARAMS["rho"]), 0.05)

        run_clicked = st.button("🚀 Optimize Ambulance Route", type="primary", use_container_width=True)

    # ---------------- Run Optimization ----------------
    if run_clicked or "opt_state" not in st.session_state:
        ambulance_loc = (float(amb_lat), float(amb_lon))
        patient_loc = (float(pat_lat), float(pat_lon))

        if selected_hosp_option == "Auto-select Nearest Emergency Hospital":
            hospital = road_network.find_nearest_hospital(
                patient_loc[0], patient_loc[1], emergency_only=True
            )
        else:
            hospital = road_network.get_hospital_by_identifier(
                selected_hosp_option, patient_loc[0], patient_loc[1]
            )
        hospital_loc = (float(hospital["lat"]), float(hospital["lon"]))

        with st.spinner("Predicting weather, weighting road network, and running ACO..."):
            weather_info = weather_model.predict_weather_details(month=int(month), year=int(year))

            if use_online_api and api_available:
                gmaps_result = google_maps.get_google_maps_route(
                    patient_loc[0],
                    patient_loc[1],
                    hospital_loc[0],
                    hospital_loc[1],
                    api_key=api_key,
                )
            else:
                gmaps_result = google_maps.get_offline_estimate(
                    patient_loc[0],
                    patient_loc[1],
                    hospital_loc[0],
                    hospital_loc[1],
                )

            global_traffic_mult = google_maps.extract_traffic_multiplier(gmaps_result)
            weighted_graph = edge_weight.assign_edge_weights(
                G,
                weather_info=weather_info,
                global_traffic_multiplier=global_traffic_mult,
            )

            amb_node = road_network.find_nearest_node(weighted_graph, ambulance_loc[0], ambulance_loc[1])
            pat_node = road_network.find_nearest_node(weighted_graph, patient_loc[0], patient_loc[1])
            hosp_node = road_network.find_nearest_node(weighted_graph, hospital_loc[0], hospital_loc[1])

            pickup_res = aco.run_dijkstra(weighted_graph, amb_node, pat_node)
            aco_params = {
                "n_ants": int(n_ants),
                "n_iterations": int(n_iterations),
                "alpha": float(alpha),
                "beta": float(beta),
                "rho": float(rho),
                "Q": float(config.ACO_PARAMS["Q"]),
                "initial_pheromone": float(config.ACO_PARAMS["initial_pheromone"]),
            }
            aco_result = aco.run_aco(weighted_graph, pat_node, hosp_node, params=aco_params)
            dijkstra_result = aco.run_dijkstra(weighted_graph, pat_node, hosp_node)

            route_map = _build_interactive_map(
                G=weighted_graph,
                aco_path=aco_result.get("route", []),
                pickup_path=pickup_res.get("route", []),
                gmaps_points=gmaps_result.get("route_points", []),
                ambulance_loc=ambulance_loc,
                patient_loc=patient_loc,
                hospital=hospital,
                aco_result=aco_result,
                gmaps_result=gmaps_result,
                pickup_info=pickup_res,
            )
            route_breakdown = _summarize_route_components(
                weighted_graph, aco_result.get("route", [])
            )

            st.session_state["opt_state"] = {
                "ambulance_loc": ambulance_loc,
                "patient_loc": patient_loc,
                "hospital": hospital,
                "weather_info": weather_info,
                "gmaps_result": gmaps_result,
                "global_traffic_mult": global_traffic_mult,
                "pickup_res": pickup_res,
                "aco_result": aco_result,
                "dijkstra_result": dijkstra_result,
                "map_html": route_map.get_root().render(),
                "route_breakdown": route_breakdown,
            }

    state = st.session_state["opt_state"]
    hospital = state["hospital"]
    weather_info = state["weather_info"]
    gmaps_result = state["gmaps_result"]
    aco_result = state["aco_result"]
    dijkstra_result = state["dijkstra_result"]
    pickup_res = state["pickup_res"]
    breakdown = state["route_breakdown"]

    is_est = bool(gmaps_result.get("is_estimate", False)) or (
        gmaps_result.get("source") != "google_maps_api"
    )
    gmaps_label = "Offline Estimate (Not Google Maps)" if is_est else "Google Maps Traffic-Aware ETA (Car)"

    aco_s = float(aco_result.get("best_cost", 0.0))
    gmaps_s = float(gmaps_result.get("duration_seconds", 0.0))
    diff_s = aco_s - gmaps_s
    diff_pct = (diff_s / gmaps_s * 100.0) if gmaps_s > 0 else 0.0

    # ---------------- Status Banner ----------------
    if is_est:
        st.warning(
            f"**Offline / Fallback Reference Mode**: {gmaps_result.get('status', 'offline_estimate')}. "
            "Baseline ETA & distance are haversine estimates, not live Google Maps data."
        )
    else:
        traffic_lvl = (gmaps_result.get("traffic_conditions") or {}).get("congestion_level", "UNKNOWN")
        st.success(
            f"**Live Google Maps Routes API Active** — Traffic Congestion Level: **{traffic_lvl}** "
            f"(Applied congestion factor: `{(state['global_traffic_mult'] or 1.0):.2f}x`)."
        )

    # ---------------- Top Metrics ----------------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Selected Hospital",
        hospital["name"].split("(")[0].strip()[:26],
        f"{hospital.get('distance_km', 0.0):.2f} km straight-line",
    )
    m2.metric(
        "Predicted Weather",
        weather_info["severity"].replace("_", " ").title(),
        f"{weather_info['impact_multiplier']:.2f}x travel multiplier",
    )
    m3.metric(
        "ACO Ambulance Time",
        f"{aco_result['time_min']:.2f} min ({aco_s:.0f}s)",
        f"{aco_result['distance_km']:.2f} km | {aco_result.get('execution_time', 0.0):.2f}s solve",
    )
    m4.metric(
        "Reference ETA (Car)" if not is_est else "Offline Baseline ETA",
        f"{gmaps_result['time_min']:.2f} min ({gmaps_s:.0f}s)",
        f"{gmaps_result['distance_km']:.2f} km",
    )
    m5.metric(
        "Time Difference (ACO - Ref)",
        f"{diff_s:+.1f} s",
        f"{diff_pct:+.2f}%",
        delta_color="inverse",
    )

    # ---------------- Interactive Map ----------------
    st.subheader("Interactive Ranchi Route Map")
    st.markdown(
        "**Legend:** 🟢 **Solid Green** = ACO Ambulance Route (`Patient → Hospital`) &nbsp;|&nbsp; "
        "🔵 **Dashed Blue** = Google Maps / Reference Route &nbsp;|&nbsp; "
        "🟠 **Dotted Orange** = Ambulance Pickup Leg (`Ambulance → Patient`: "
        f"`{pickup_res.get('distance_km', 0.0):.2f} km`, `{pickup_res.get('time_min', 0.0):.2f} min`) &nbsp;|&nbsp; "
        "🏥 **Purple Markers** = All 18 Ranchi Hospitals"
    )
    components.html(state["map_html"], height=600, scrolling=False)

    # ---------------- Evaluation & Analysis Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Evaluation & Comparison",
            "📈 ACO Convergence",
            "🧮 Edge Weight & Weather ML",
            "🏥 Ranchi Hospitals Dataset",
        ]
    )

    with tab1:
        col_tbl, col_chart = st.columns([1.05, 1.25])
        with col_tbl:
            st.subheader("Route Evaluation Table")
            eval_df = pd.DataFrame(
                [
                    {
                        "Metric": "ACO Estimated Ambulance Travel Time",
                        "Value": f"{aco_s:.2f} s ({aco_result['time_min']:.2f} min)",
                    },
                    {
                        "Metric": gmaps_label,
                        "Value": f"{gmaps_s:.2f} s ({gmaps_result['time_min']:.2f} min)",
                    },
                    {
                        "Metric": "Dijkstra Exact Optimum (Same Weights)",
                        "Value": f"{dijkstra_result['best_cost']:.2f} s ({dijkstra_result['time_min']:.2f} min)",
                    },
                    {
                        "Metric": "Travel-Time Difference (ACO - Reference)",
                        "Value": f"{diff_s:+.2f} s ({diff_pct:+.2f}%)",
                    },
                    {
                        "Metric": "ACO Route Distance",
                        "Value": f"{aco_result['distance_km'] * 1000:.1f} m ({aco_result['distance_km']:.2f} km)",
                    },
                    {
                        "Metric": "Reference Route Distance",
                        "Value": f"{gmaps_result['distance_meters']:.1f} m ({gmaps_result['distance_km']:.2f} km)",
                    },
                    {
                        "Metric": "ACO Execution Time",
                        "Value": f"{aco_result.get('execution_time', 0.0):.4f} s",
                    },
                    {
                        "Metric": "Ants Reaching Destination",
                        "Value": f"{aco_result.get('ants_succeeded', 0)} / {int(n_ants) * int(n_iterations)}",
                    },
                ]
            )
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
            st.info(
                "**ETA Distinction**: Google Maps traffic-aware ETA estimates travel time for standard passenger vehicles, "
                "whereas ACO estimates emergency ambulance travel time incorporating weather severity, traffic signal "
                "delays, road-specific bottlenecks, and the `1.3x` ambulance priority multiplier."
            )

        with col_chart:
            st.subheader("Travel Time & Distance Comparison")
            comp_fig = _plot_comparison_fig(aco_result, gmaps_result)
            st.pyplot(comp_fig, use_container_width=True)
            plt.close(comp_fig)

    with tab2:
        st.subheader("ACO Convergence Curve")
        conv_hist = aco_result.get("convergence_history", [])
        if conv_hist:
            conv_fig = _plot_convergence_fig(conv_hist, float(dijkstra_result.get("best_cost", 0.0)))
            st.pyplot(conv_fig, use_container_width=True)
            plt.close(conv_fig)
        else:
            st.warning("No convergence history available.")

    with tab3:
        st.subheader("Dynamic Edge Weight Formula")
        st.latex(
            r"W(e) = \frac{T_{\text{base}}(e) \times M_{\text{traffic}}(e) \times M_{\text{weather}} + D_{\text{signal}}(e) + D_{\text{road}}(e)}{M_{\text{ambulance}}}"
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Road Segments", f"{breakdown['edges']}")
        b2.metric("Base Free-Flow Sum", f"{breakdown['base_time_s']:.1f} s")
        b3.metric("Avg Traffic Factor", f"{breakdown['avg_traffic_mult']:.2f}x")
        b4.metric("Weather Factor", f"{breakdown['weather_mult']:.2f}x")
        b5.metric("Signal + Road Delays", f"{breakdown['signal_delay_s'] + breakdown['road_delay_s']:.0f} s")

        st.markdown(
            f"""
            **Weather Prediction Details (`RandomForestClassifier` on Ranchi District Climate Data)**
            - **Period**: `{int(month):02d}/{int(year)}` | **Predicted Severity**: `{weather_info['severity']}` (`{weather_info['impact_multiplier']:.2f}x`)
            - **Monthly Historical Averages Used**: `LST_C = {weather_info['lst_c']:.2f} °C`, `NDVI = {weather_info['ndvi']:.3f}`
            - **Scope Note**: The weather ML model predicts rainfall-derived weather severity only and does **not** directly predict traffic congestion or travel time.
            """
        )

    with tab4:
        st.subheader("Verified Ranchi Hospitals Dataset (18 Facilities)")
        hosp_df = pd.DataFrame(config.HOSPITALS)
        st.dataframe(hosp_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

