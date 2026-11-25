"""Ant Colony Optimization (ACO) for Ambulance Route Optimization.

Implements AntColonyOptimizer and a NetworkX Dijkstra baseline to compute the
fastest ambulance route (minimum total travel time in seconds) on a weighted
road network graph.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

try:
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

# Fallback default ACO hyperparameters used if config.py is absent or incomplete
DEFAULT_ACO_PARAMS: Dict[str, Any] = {
    "n_ants": 30,
    "n_iterations": 100,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.1,
    "Q": 10.0,
    "initial_pheromone": 0.1,
    "elitist_weight": 1.0,
}

_MIN_WEIGHT = 1e-9
_MIN_PHEROMONE = 1e-12


def _load_default_params() -> Dict[str, Any]:
    """Load default ACO parameters from config if available, else fallback."""
    defaults = dict(DEFAULT_ACO_PARAMS)
    if config is None:
        return defaults

    # Check common dict-style configs in config.py
    for attr_name in ("ACO_PARAMS", "ACO_CONFIG", "DEFAULT_ACO_PARAMS", "ACO_DEFAULTS"):
        cfg_dict = getattr(config, attr_name, None)
        if isinstance(cfg_dict, dict):
            for key in defaults:
                if key in cfg_dict and cfg_dict[key] is not None:
                    defaults[key] = cfg_dict[key]

    # Check common scalar-style constants in config.py
    scalar_aliases = {
        "n_ants": ("N_ANTS", "ACO_N_ANTS", "n_ants"),
        "n_iterations": ("N_ITERATIONS", "ACO_N_ITERATIONS", "n_iterations"),
        "alpha": ("ALPHA", "ACO_ALPHA", "alpha"),
        "beta": ("BETA", "ACO_BETA", "beta"),
        "rho": ("RHO", "ACO_RHO", "EVAPORATION_RATE", "rho"),
        "Q": ("Q", "ACO_Q", "PHEROMONE_Q"),
        "initial_pheromone": (
            "INITIAL_PHEROMONE",
            "ACO_INITIAL_PHEROMONE",
            "initial_pheromone",
        ),
        "elitist_weight": ("ELITIST_WEIGHT", "ACO_ELITIST_WEIGHT", "elitist_weight"),
    }
    for key, aliases in scalar_aliases.items():
        for alias in aliases:
            if hasattr(config, alias):
                val = getattr(config, alias)
                if val is not None:
                    defaults[key] = val
                    break

    return defaults


def _node_haversine_m(G: nx.Graph, u: Any, v: Any) -> float:
    """Return great-circle distance in meters between nodes u and v if coords exist."""
    nu = G.nodes.get(u, {})
    nv = G.nodes.get(v, {})
    lat1, lon1 = nu.get("y", nu.get("lat")), nu.get("x", nu.get("lon"))
    lat2, lon2 = nv.get("y", nv.get("lat")), nv.get("x", nv.get("lon"))
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _get_min_edge_data(G: nx.Graph, u: Any, v: Any) -> float:
    """Return the minimum travel-time 'weight' between u and v in G."""
    data = G[u][v]
    if G.is_multigraph():
        return min(float(edge_attr.get("weight", 1.0)) for edge_attr in data.values())
    return float(data.get("weight", 1.0))


class AntColonyOptimizer:
    """Ant Colony Optimization solver for finding fastest routes on a road graph."""

    def __init__(
        self,
        G: nx.Graph,
        source: Any,
        destination: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the ACO optimizer for a weighted graph and OD pair.

        Args:
            G: NetworkX graph (DiGraph, MultiDiGraph, Graph, or MultiGraph)
                with numeric 'weight' attributes on edges representing travel time.
            source: Origin node ID.
            destination: Target node ID.
            params: Optional dictionary overriding ACO parameters
                (n_ants, n_iterations, alpha, beta, rho, Q, initial_pheromone,
                elitist_weight, seed).
        """
        self.G = G
        self.source = source
        self.destination = destination

        merged_params = _load_default_params()
        if params:
            merged_params.update({k: v for k, v in params.items() if v is not None})

        self.params = merged_params
        self.n_ants = int(merged_params["n_ants"])
        self.n_iterations = int(merged_params["n_iterations"])
        self.alpha = float(merged_params["alpha"])
        self.beta = float(merged_params["beta"])
        self.rho = float(merged_params["rho"])
        self.Q = float(merged_params["Q"])
        self.initial_pheromone = float(merged_params["initial_pheromone"])
        self.elitist_weight = float(merged_params.get("elitist_weight", 1.0))

        seed = merged_params.get("seed")
        self._rng = np.random.default_rng(seed)

        self.is_directed = G.is_directed()
        self.best_path: Optional[List[Any]] = None
        self.best_cost: float = float("inf")

        self._edge_to_idx: Dict[Tuple[Any, Any], int] = {}
        self._neighbors: Dict[Any, List[Any]] = {}
        self._neighbor_edge_idx: Dict[Any, np.ndarray] = {}
        self._neighbor_heuristics: Dict[Any, np.ndarray] = {}
        self._edge_weights: Dict[Tuple[Any, Any], float] = {}
        self._pheromone_arr: np.ndarray = np.empty(0, dtype=np.float64)
        self._allowed_nodes: Set[Any] = set()
        self._has_spatial_coords: bool = False

        self._build_graph_index()
        self._initialize_pheromones()

    def _canonical_edge(self, u: Any, v: Any) -> Tuple[Any, Any]:
        """Return edge key respecting directedness when directed."""
        if self.is_directed:
            return (u, v)
        if (u, v) in self._edge_to_idx:
            return (u, v)
        if (v, u) in self._edge_to_idx:
            return (v, u)
        return (u, v)

    def _compute_search_corridor(self) -> Set[Any]:
        """Compute reachable nodes (and elliptic corridor when coordinates exist)."""
        if self.source not in self.G or self.destination not in self.G:
            return set()

        # Nodes that can reach destination in the directed graph
        if self.is_directed:
            can_reach_dest = nx.ancestors(self.G, self.destination) | {self.destination}
        else:
            can_reach_dest = nx.node_connected_component(self.G, self.destination)

        if self.source not in can_reach_dest:
            return set()

        # If nodes have geographic coordinates (e.g., OSMnx Ranchi road network),
        # restrict search to a generous elliptic corridor around OD for scalability.
        src_attrs = self.G.nodes[self.source]
        dst_attrs = self.G.nodes[self.destination]
        if ("y" in src_attrs and "x" in src_attrs) and ("y" in dst_attrs and "x" in dst_attrs):
            self._has_spatial_coords = True
            direct_m = _node_haversine_m(self.G, self.source, self.destination)
            # Tighter elliptic corridor (1.45x direct + 1.2km buffer) for fast city-scale ACO,
            # automatically widening if disconnected.
            for mult, buf in ((1.45, 1200.0), (1.85, 2000.0), (2.5, 3000.0)):
                max_ellipse_m = max(mult * direct_m, direct_m + buf)
                corridor = {
                    n
                    for n in can_reach_dest
                    if (_node_haversine_m(self.G, self.source, n) + _node_haversine_m(self.G, n, self.destination))
                    <= max_ellipse_m
                }
                if self.source in corridor and self.destination in corridor:
                    sub = self.G.subgraph(corridor)
                    if nx.has_path(sub, self.source, self.destination):
                        return corridor

        return can_reach_dest

    def _build_graph_index(self) -> None:
        """Precompute compact adjacency, edge weights, and goal-aware heuristics."""
        self._allowed_nodes = self._compute_search_corridor()
        target_nodes = self._allowed_nodes if self._allowed_nodes else list(self.G.nodes())

        # Precompute topological free-flow remaining time estimate (in seconds)
        # using static road geometry (base_travel_time) so ants respect real street
        # connectivity instead of wandering into dead-ends toward straight-line bearings,
        # while dynamic weights (traffic, weather, signals) drive ACO optimization.
        rem_time_est: Dict[Any, float] = {}
        if self.destination in self.G:
            sub_graph = self.G.subgraph(target_nodes) if target_nodes else self.G
            try:
                rev_sub = sub_graph.reverse(copy=False) if self.is_directed else sub_graph
                # Use static base_travel_time if available, else length / 11.11 m/s
                def _static_cost(_u: Any, _v: Any, d: Dict[str, Any]) -> float:
                    if self.G.is_multigraph():
                        return min(
                            float(ed.get("base_travel_time", float(ed.get("length", 100.0)) / 11.11))
                            for ed in d.values()
                        )
                    return float(d.get("base_travel_time", float(d.get("length", 100.0)) / 11.11))

                rem_time_est = nx.single_source_dijkstra_path_length(
                    rev_sub, self.destination, weight=_static_cost
                )
            except Exception:
                if self._has_spatial_coords:
                    ref_speed_mps = 40.0 / 3.6
                    for n in target_nodes:
                        rem_time_est[n] = _node_haversine_m(self.G, n, self.destination) / ref_speed_mps

        edge_idx = 0
        for u in target_nodes:
            nbrs: List[Any] = []
            edge_indices: List[int] = []
            heuristics: List[float] = []
            u_rem = rem_time_est.get(u, 0.0)

            for v in self.G.neighbors(u):
                if u == v:
                    continue
                if self._allowed_nodes and v not in self._allowed_nodes:
                    continue
                weight = _get_min_edge_data(self.G, u, v)
                safe_weight = max(weight, _MIN_WEIGHT)
                self._edge_weights[(u, v)] = weight

                key = self._canonical_edge(u, v)
                if key not in self._edge_to_idx:
                    self._edge_to_idx[key] = edge_idx
                    edge_idx += 1

                if rem_time_est:
                    v_rem = rem_time_est.get(v, float("inf"))
                    if not np.isfinite(v_rem):
                        continue
                    # Marginal cost toward destination combining dynamic edge weight
                    # and static topological potential change.
                    marginal_cost = safe_weight + 1.2 * (v_rem - u_rem) + 0.05 * v_rem
                    effective_cost = max(0.5, marginal_cost)
                else:
                    effective_cost = safe_weight

                nbrs.append(v)
                edge_indices.append(self._edge_to_idx[key])
                heuristics.append((1.0 / effective_cost) ** self.beta)

            self._neighbors[u] = nbrs
            self._neighbor_edge_idx[u] = np.asarray(edge_indices, dtype=np.int64)
            self._neighbor_heuristics[u] = np.asarray(heuristics, dtype=np.float64)

    @property
    def pheromones(self) -> Dict[Tuple[Any, Any], float]:
        """Return a mapping of edge tuples to current pheromone levels."""
        return {
            edge: float(self._pheromone_arr[idx])
            for edge, idx in self._edge_to_idx.items()
        }

    def _initialize_pheromones(self) -> None:
        """Set initial pheromone level on all edges to initial_pheromone."""
        num_edges = len(self._edge_to_idx)
        self._pheromone_arr = np.full(
            num_edges, self.initial_pheromone, dtype=np.float64
        )

    def _select_next_node(self, current: Any, visited: Set[Any]) -> Optional[Any]:
        """Select the next unvisited neighbor probabilistically using ACO rules.

        P(next) = (pheromone[edge]^alpha * heuristic[edge]^beta) / sum_over_neighbors
        """
        nbrs = self._neighbors.get(current)
        if not nbrs:
            return None

        valid_positions = [i for i, n in enumerate(nbrs) if n not in visited]
        if not valid_positions:
            return None

        if len(valid_positions) == 1:
            return nbrs[valid_positions[0]]

        valid_edge_idx = self._neighbor_edge_idx[current][valid_positions]
        valid_heuristics = self._neighbor_heuristics[current][valid_positions]

        pheromones = self._pheromone_arr[valid_edge_idx]
        attractiveness = (pheromones ** self.alpha) * valid_heuristics

        total = float(np.sum(attractiveness))
        if total <= 0.0 or not np.isfinite(total):
            chosen_pos = int(self._rng.integers(0, len(valid_positions)))
            return nbrs[valid_positions[chosen_pos]]

        # Vectorized roulette-wheel selection via cumulative distribution
        cdf = np.cumsum(attractiveness / total)
        cdf[-1] = 1.0
        chosen_idx = int(np.searchsorted(cdf, self._rng.random(), side="right"))
        if chosen_idx >= len(valid_positions):
            chosen_idx = len(valid_positions) - 1
        return nbrs[valid_positions[chosen_idx]]

    def _construct_path(self, ant_id: int) -> Optional[List[Any]]:
        """Construct a single ant's cycle-free path from source to destination.

        Supports dead-end rollback when an ant enters a visited cul-de-sac so
        directed road graphs remain reliably traversable without cycles.
        """
        _ = ant_id
        if self.source not in self.G or self.destination not in self.G:
            return None

        current = self.source
        path: List[Any] = [current]
        visited_or_dead: Set[Any] = {current}
        max_steps = min(1200, max(300, len(self._neighbors)))
        steps = 0

        while current != self.destination and steps < max_steps:
            steps += 1
            next_node = self._select_next_node(current, visited_or_dead)
            if next_node is None:
                # Roll back from dead-end while keeping it marked dead for this ant
                if len(path) <= 1:
                    return None
                path.pop()
                current = path[-1]
                continue

            path.append(next_node)
            visited_or_dead.add(next_node)
            current = next_node

        return path if current == self.destination else None

    def _calculate_path_cost(self, path: Optional[List[Any]]) -> float:
        """Calculate total travel time (sum of edge weights) along a path."""
        if not path:
            return float("inf")
        if len(path) == 1:
            return 0.0

        total_cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            weight = self._edge_weights.get((u, v))
            if weight is None:
                if self.G.has_edge(u, v):
                    weight = _get_min_edge_data(self.G, u, v)
                else:
                    return float("inf")
            total_cost += weight
        return total_cost

    def _evaporate_pheromones(self) -> None:
        """Apply pheromone evaporation across all edges."""
        if self._pheromone_arr.size == 0:
            return
        self._pheromone_arr *= 1.0 - self.rho
        np.maximum(self._pheromone_arr, _MIN_PHEROMONE, out=self._pheromone_arr)

    def _deposit_on_path(self, path: List[Any], amount: float) -> None:
        """Add pheromone amount to every edge in the given path."""
        if len(path) < 2 or amount <= 0.0:
            return
        indices = [
            self._edge_to_idx[self._canonical_edge(path[i], path[i + 1])]
            for i in range(len(path) - 1)
        ]
        np.add.at(self._pheromone_arr, indices, amount)

    def _update_pheromones(
        self, paths: List[Optional[List[Any]]], costs: List[float]
    ) -> None:
        """Deposit pheromone for all valid paths plus elitist reinforcement."""
        valid_pairs = [
            (p, c)
            for p, c in zip(paths, costs)
            if p is not None and len(p) >= 2 and np.isfinite(c) and c > 0.0
        ]
        if not valid_pairs:
            return

        iter_best_path: Optional[List[Any]] = None
        iter_best_cost = float("inf")

        # Scale Q relative to initial pheromone and iteration best cost so pheromone
        # reinforcement remains meaningful across both small test graphs and large city routes.
        min_valid_cost = min(c for _, c in valid_pairs)
        cost_scale = max(min_valid_cost, 1.0)
        effective_q = max(self.Q, self.initial_pheromone * cost_scale * 0.5)

        for path, cost in valid_pairs:
            deposit = effective_q / max(cost, _MIN_WEIGHT)
            self._deposit_on_path(path, deposit)
            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_path = path

        # Elitist strategy: reinforce global best if available, else iteration best
        if (
            self.best_path is not None
            and len(self.best_path) >= 2
            and np.isfinite(self.best_cost)
            and self.best_cost <= iter_best_cost
        ):
            elite_path, elite_cost = self.best_path, self.best_cost
        else:
            elite_path, elite_cost = iter_best_path, iter_best_cost

        if elite_path is not None and len(elite_path) >= 2 and self.elitist_weight > 0.0:
            elite_deposit = self.elitist_weight * (effective_q / max(elite_cost, _MIN_WEIGHT))
            self._deposit_on_path(elite_path, elite_deposit)

    def optimize(self) -> Dict[str, Any]:
        """Execute the full Ant Colony Optimization algorithm.

        Returns:
            Dictionary containing:
                - best_path: List of node IDs for the best path found, or None.
                - best_cost: Total travel time in seconds (inf if no path exists).
                - convergence: List of best cost recorded after each iteration.
                - execution_time: Total wall-clock time in seconds.
                - iterations: Number of iterations executed.
                - ants_succeeded: Total number of ants that reached destination.
        """
        start_time = time.time()
        self.best_path = None
        self.best_cost = float("inf")
        self._initialize_pheromones()

        # Edge case: missing endpoints
        if self.source not in self.G or self.destination not in self.G:
            logger.warning(
                "Source (%s) or destination (%s) not found in graph.",
                self.source,
                self.destination,
            )
            return {
                "best_path": None,
                "best_cost": float("inf"),
                "convergence": [],
                "execution_time": time.time() - start_time,
                "iterations": 0,
                "ants_succeeded": 0,
            }

        # Edge case: trivial same-node route
        if self.source == self.destination:
            self.best_path = [self.source]
            self.best_cost = 0.0
            return {
                "best_path": list(self.best_path),
                "best_cost": 0.0,
                "convergence": [0.0] * self.n_iterations,
                "execution_time": time.time() - start_time,
                "iterations": self.n_iterations,
                "ants_succeeded": self.n_ants * self.n_iterations,
            }

        # Edge case: disconnected graph / no valid directed path
        if not nx.has_path(self.G, self.source, self.destination):
            logger.warning(
                "No path exists between source (%s) and destination (%s).",
                self.source,
                self.destination,
            )
            return {
                "best_path": None,
                "best_cost": float("inf"),
                "convergence": [],
                "execution_time": time.time() - start_time,
                "iterations": 0,
                "ants_succeeded": 0,
            }

        convergence: List[float] = []
        ants_succeeded = 0

        for iteration in range(self.n_iterations):
            iter_paths: List[Optional[List[Any]]] = []
            iter_costs: List[float] = []
            iter_valid_count = 0

            for ant_id in range(self.n_ants):
                path = self._construct_path(ant_id)
                if path is not None:
                    cost = self._calculate_path_cost(path)
                    iter_paths.append(path)
                    iter_costs.append(cost)
                    iter_valid_count += 1
                    ants_succeeded += 1

                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_path = list(path)
                else:
                    iter_paths.append(None)
                    iter_costs.append(float("inf"))

            self._evaporate_pheromones()
            self._update_pheromones(iter_paths, iter_costs)
            convergence.append(self.best_cost)

            if (iteration + 1) % 10 == 0:
                logger.info(
                    "ACO Iteration %d/%d | Best Cost: %.4fs | Valid Ants: %d/%d",
                    iteration + 1,
                    self.n_iterations,
                    self.best_cost,
                    iter_valid_count,
                    self.n_ants,
                )

        execution_time = time.time() - start_time
        return {
            "best_path": list(self.best_path) if self.best_path is not None else None,
            "best_cost": self.best_cost,
            "convergence": convergence,
            "execution_time": execution_time,
            "iterations": self.n_iterations,
            "ants_succeeded": ants_succeeded,
        }


def dijkstra_shortest_path(
    G: nx.Graph, source: Any, destination: Any
) -> Dict[str, Any]:
    """Compute shortest path using NetworkX Dijkstra for baseline comparison.

    Args:
        G: NetworkX graph with 'weight' attributes on edges.
        source: Origin node ID.
        destination: Target node ID.

    Returns:
        Dictionary matching AntColonyOptimizer.optimize() output format.
    """
    start_time = time.time()

    if source not in G or destination not in G:
        logger.warning(
            "Source (%s) or destination (%s) not found in graph.",
            source,
            destination,
        )
        return {
            "best_path": None,
            "best_cost": float("inf"),
            "convergence": [],
            "execution_time": time.time() - start_time,
            "iterations": 0,
            "ants_succeeded": 0,
        }

    try:
        path = nx.dijkstra_path(G, source, destination, weight="weight")
        cost = float(nx.dijkstra_path_length(G, source, destination, weight="weight"))
        execution_time = time.time() - start_time
        return {
            "best_path": list(path),
            "best_cost": cost,
            "convergence": [cost],
            "execution_time": execution_time,
            "iterations": 1,
            "ants_succeeded": 1,
        }
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        logger.warning(
            "Dijkstra found no path between source (%s) and destination (%s).",
            source,
            destination,
        )
        return {
            "best_path": None,
            "best_cost": float("inf"),
            "convergence": [],
            "execution_time": time.time() - start_time,
            "iterations": 0,
            "ants_succeeded": 0,
        }


def _calculate_path_distance_km(G: nx.Graph, path: Optional[List[Any]]) -> float:
    """Compute total physical road distance in kilometers along a node path."""
    if not path or len(path) < 2:
        return 0.0

    total_meters = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            continue
        data = G[u][v]
        if G.is_multigraph():
            best_edge = min(
                data.values(), key=lambda attr: float(attr.get("weight", 1.0))
            )
            length_m = float(best_edge.get("length", best_edge.get("length_m", 0.0)))
        else:
            length_m = float(data.get("length", data.get("length_m", 0.0)))
        total_meters += length_m
    return total_meters / 1000.0


def _format_pipeline_result(G: nx.Graph, result: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich standard optimizer output with route/distance_km/time_min fields."""
    path = result.get("best_path")
    cost = float(result.get("best_cost", float("inf")))
    return {
        **result,
        "route": list(path) if path is not None else [],
        "distance_km": _calculate_path_distance_km(G, path),
        "time_min": (cost / 60.0) if np.isfinite(cost) else 0.0,
        "convergence_history": list(result.get("convergence", [])),
    }


def run_aco(
    G: nx.Graph,
    source: Any,
    destination: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper running AntColonyOptimizer for pipeline callers."""
    optimizer = AntColonyOptimizer(G, source, destination, params=params)
    return _format_pipeline_result(G, optimizer.optimize())


def run_dijkstra(G: nx.Graph, source: Any, destination: Any) -> Dict[str, Any]:
    """Convenience wrapper running dijkstra_shortest_path for pipeline callers."""
    return _format_pipeline_result(G, dijkstra_shortest_path(G, source, destination))

