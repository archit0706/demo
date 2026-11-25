"""Configuration module for the Ranchi Ambulance Route Optimization System.

Defines geographic bounds, verified hospital locations, Ant Colony Optimization
(ACO) parameters, road speed limits, traffic and weather impact factors, API
credentials, and filesystem paths for models and cached road networks in
Ranchi, Jharkhand, India.
"""

import os
from typing import Any, Dict, List, Optional

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

RANCHI_BOUNDS: Dict[str, float] = {
    "north": 23.46,
    "south": 23.30,
    "east": 85.45,
    "west": 85.24,
}
"""Lat/lon bounding box covering Ranchi city and major peri-urban hospitals."""

HOSPITALS: List[Dict[str, Any]] = [
    {
        "name": "Rajendra Institute of Medical Sciences (RIMS)",
        "lat": 23.3915,
        "lon": 85.3485,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "Sadar Hospital",
        "lat": 23.3691,
        "lon": 85.3268,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "Central Institute of Psychiatry (CIP)",
        "lat": 23.4346,
        "lon": 85.3258,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "RINPAS",
        "lat": 23.4281,
        "lon": 85.3090,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "CCL Central Hospital",
        "lat": 23.4085,
        "lon": 85.3206,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "Paras HEC Hospital",
        "lat": 23.3136,
        "lon": 85.2772,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "ESIC Model Hospital",
        "lat": 23.3520,
        "lon": 85.3810,
        "type": "Government",
        "emergency": True,
    },
    {
        "name": "Ranchi Cancer Hospital (RCHRC)",
        "lat": 23.4376,
        "lon": 85.3028,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Bhagwan Mahavir Medica Superspecialty Hospital",
        "lat": 23.3960,
        "lon": 85.3780,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Medanta Hospital",
        "lat": 23.4507,
        "lon": 85.4372,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Orchid Medical Centre",
        "lat": 23.3782,
        "lon": 85.3344,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Raj Hospital",
        "lat": 23.3588,
        "lon": 85.3248,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Santevita Hospital",
        "lat": 23.3709,
        "lon": 85.3263,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Pulse Superspeciality Hospital",
        "lat": 23.3907,
        "lon": 85.3450,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Shree Jagannath Hospital (SJHRC)",
        "lat": 23.3820,
        "lon": 85.3236,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Guru Nanak Hospital",
        "lat": 23.3495,
        "lon": 85.3295,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Devkamal Hospital",
        "lat": 23.3732,
        "lon": 85.2721,
        "type": "Private",
        "emergency": True,
    },
    {
        "name": "Arogyam Hospital",
        "lat": 23.3650,
        "lon": 85.3080,
        "type": "Private",
        "emergency": True,
    },
]
"""Verified emergency hospitals in and around Ranchi, Jharkhand."""

ACO_PARAMS: Dict[str, float] = {
    "n_ants": 30,
    "n_iterations": 100,
    "alpha": 1.0,  # Pheromone importance
    "beta": 3.0,  # Heuristic importance
    "rho": 0.1,  # Evaporation rate
    "Q": 100.0,  # Pheromone deposit factor
    "initial_pheromone": 0.1,
}
"""Ant Colony Optimization hyperparameters for route search."""

SPEED_LIMITS: Dict[str, int] = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 30,
    "residential": 20,
    "unclassified": 20,
    "service": 15,
    "living_street": 10,
}
"""Default road speeds (km/h) mapped from OpenStreetMap highway types."""

TRAFFIC_SIGNAL_DELAY: int = 30
"""Average delay in seconds incurred at each traffic signal."""

WEATHER_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "weather_model.pkl")
"""Path to the serialized weather impact prediction model."""

GOOGLE_MAPS_API_KEY: Optional[str] = os.environ.get("GOOGLE_MAPS_API_KEY", None)
"""Google Maps API key loaded from the environment (None if unset)."""

DEFAULT_WEATHER: Dict[str, float] = {
    "rainfall_mm": 0,
    "lst_c": 25,
    "ndvi": 0.45,
}
"""Fallback weather parameters (rainfall in mm, LST in Celsius, NDVI)."""

WEATHER_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "clear": 1.0,
    "light_rain": 1.15,
    "moderate_rain": 1.3,
    "heavy_rain": 1.5,
    "extreme": 1.8,
}
"""Travel-time impact multipliers indexed by weather severity label."""

AMBULANCE_SPEED_MULTIPLIER: float = 1.3
"""Speed multiplier reflecting emergency priority relative to normal traffic."""

OSM_NETWORK_FILE: str = os.path.join(BASE_DIR, "data", "ranchi_network.graphml")
"""Path to the locally cached OpenStreetMap GraphML road network for Ranchi."""

RANCHI_WEATHER_FILE: str = os.path.join(BASE_DIR, "data", "ranchi_weather.csv")
"""Path to the extracted Ranchi district climate dataset."""

HOSPITALS_FILE: str = os.path.join(BASE_DIR, "data", "ranchi_hospitals.csv")
"""Path to the verified Ranchi hospitals dataset."""

OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
"""Directory for generated interactive maps, charts, and evaluation tables."""
