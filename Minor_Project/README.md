# Ambulance Route Optimization System — Ranchi, Jharkhand, India

A dynamic ambulance route optimization and evaluation system for **Ranchi, Jharkhand, India** using **Ant Colony Optimization (ACO)**, **Machine Learning weather severity prediction**, **OpenStreetMap drivable road networks (OSMnx)**, and **Google Maps Routes API** traffic-aware comparison (with a full **offline mode**).

---

## Project Structure

```
Minor_Project/
├── config.py                  # Bounds, verified Ranchi hospitals, ACO params, speed limits
├── weather_model.py           # Extracts Ranchi climate data & trains RandomForest weather classifier
├── road_network.py            # Loads/caches Ranchi OSMnx road network & hospital selection
├── edge_weight.py             # Dynamic travel-time edge weight calculation (seconds)
├── aco.py                     # Directed Ant Colony Optimization + Dijkstra baseline
├── google_maps.py             # Google Maps Routes API v2 client + offline haversine fallback
├── visualization.py           # Folium interactive map, convergence curve, comparison charts & table
├── app.py                     # Interactive Streamlit web frontend & dashboard
├── main.py                    # End-to-end CLI pipeline
├── requirements.txt           # Python dependencies
├── .env.example               # Template for GOOGLE_MAPS_API_KEY
├── satellite_climate_district_FIXED.csv  # Source Jharkhand satellite climate dataset
├── data/
│   ├── ranchi_hospitals.csv   # Verified dataset of 18 emergency hospitals in Ranchi
│   ├── ranchi_weather.csv     # Extracted Ranchi district climate subset (168 monthly records)
│   └── ranchi_network.graphml # Cached OSMnx drivable road network for Ranchi
├── models/
│   └── weather_model.pkl      # Trained RandomForestClassifier weather severity model
└── output/
    ├── route_map.html         # Interactive map (ACO green vs Google Maps blue dashed)
    ├── convergence.png        # ACO convergence curve (best travel time per iteration)
    ├── comparison.png         # Side-by-side travel time & distance bar chart
    └── evaluation_report.txt  # Formatted evaluation summary table
```

---

## 1. Datasets

1. **Ranchi Hospitals (`data/ranchi_hospitals.csv` & `config.HOSPITALS`)**:
   - 18 major government and private emergency hospitals in Ranchi (including RIMS, Sadar Hospital, CIP, RINPAS, CCL Central, Paras HEC, ESIC Namkum, RCHRC, Medica, Medanta, Orchid, Raj Hospital, Santevita, Pulse, SJHRC, Guru Nanak, Devkamal, and Arogyam) with WGS84 latitude/longitude coordinates cross-referenced from OpenStreetMap, Wikipedia, and hospital mapping directories.
2. **Road Network (`data/ranchi_network.graphml`)**:
   - Drivable directed road network (`network_type="drive"`) within Ranchi's bounding box (`23.30°N–23.46°N, 85.24°E–85.45°E`) downloaded via OSMnx, preserving one-way restrictions, edge lengths, road types, speed limits, junctions, and traffic signals.
3. **Weather Dataset (`data/ranchi_weather.csv`)**:
   - Extracted from `satellite_climate_district_FIXED.csv` (`District == "Ranchi"`, 168 monthly records from 2010–2023 with `LST_C`, `NDVI`, `Rainfall_mm`, `Month`, `Year`).

---

## 2. Dynamic Edge Weight Mathematical Formula

Each directed road segment $e = (u, v)$ is assigned an estimated ambulance travel time $W(e)$ in **seconds**:

$$W(e) = \frac{T_{\text{base}}(e) \times M_{\text{traffic}}(e) \times M_{\text{weather}} + D_{\text{signal}}(e) + D_{\text{road}}(e)}{M_{\text{ambulance}}}$$

Where:
- **Base Travel Time ($T_{\text{base}}$)**:
  $$T_{\text{base}}(e) = \frac{\text{length}_m(e)}{\text{speed\_limit}_{km/h}(e) / 3.6}$$
  Speed limits come from OSM `maxspeed` when present, or Indian urban road-type defaults in `config.SPEED_LIMITS` (`motorway: 80`, `trunk: 60`, `primary: 50`, `secondary: 40`, `tertiary: 30`, `residential: 20`, `service: 15`, `living_street: 10` km/h).
- **Traffic Multiplier ($M_{\text{traffic}}$)**:
  - *Online Mode*: Ratio $\text{duration} / \text{staticDuration}$ from the Google Maps Routes API (`TRAFFIC_AWARE`) when available.
  - *Offline Mode*: Time-of-day congestion estimate (`1.5–1.8` peak, `1.2–1.4` moderate, `1.0–1.1` off-peak, ±5% variation).
- **Weather Multiplier ($M_{\text{weather}}$)**:
  Predicted by the trained Random Forest weather severity classifier:
  `clear: 1.0`, `light_rain: 1.15`, `moderate_rain: 1.3`, `heavy_rain: 1.5`, `extreme: 1.8`.
- **Traffic Signal Delay ($D_{\text{signal}}$)**:
  `30 seconds` per traffic signal node (`highway == "traffic_signals"`).
- **Road-Specific Delay ($D_{\text{road}}$)**:
  `+10s` for roundabouts/rotaries, `+5s` for narrow `service`/`living_street` segments, `+3s` for bridges/tunnels.
- **Ambulance Priority Factor ($M_{\text{ambulance}} = 1.3$)**:
  Models emergency siren/right-of-way speed advantage over regular passenger cars.

---

## 3. Weather Prediction Model Scope & Assumptions

- The `RandomForestClassifier` in `weather_model.py` is trained **exclusively** on Ranchi climate records (`Month`, `Year`, `LST_C`, `NDVI`) to classify rainfall-derived `weather_severity` (`clear`, `light_rain`, `moderate_rain`, `heavy_rain`, `extreme`).
- **Important**: This model predicts **weather severity only** — it does **not** claim to predict road traffic congestion or travel time directly from climate data, as the dataset contains no traffic or travel-time labels.
- **Offline Assumptions**: When `GOOGLE_MAPS_API_KEY` is absent or `--offline` is used, traffic congestion uses a time-of-day heuristic and the baseline route comparison uses a haversine `1.4x` winding-factor estimate at `25 km/h` (explicitly labeled as `ESTIMATE ONLY`, never presented as live Google Maps data).
- **ETA vs Ambulance Time**: Google Maps Routes API ETA reflects normal passenger-vehicle driving conditions; ACO travel time incorporates weather severity, signal/road delays, and the `1.3x` ambulance priority multiplier.

---

## 4. Setup & Execution

### Install Dependencies
```bash
pip install -r requirements.txt
```

### (Optional) Configure Google Maps Routes API Key
Copy `.env.example` to `.env` or set the environment variable:
```powershell
$env:GOOGLE_MAPS_API_KEY="your_actual_google_routes_api_key"
```
Ensure the **Routes API** is enabled in your Google Cloud project. If no key is provided, the system automatically runs in **offline mode**.

### Train / Evaluate the Weather Model
```bash
python weather_model.py
```

### Launch Interactive Web Frontend (Recommended)
```bash
streamlit run app.py
```
Opens the interactive Ranchi dashboard in your browser with preset landmarks or custom coordinates, hospital selection, weather prediction, live Google Maps Routes API toggle (or offline mode), interactive Folium route map, convergence curve, and evaluation comparison table/charts.

### Run Route Optimization (CLI)
Default run (auto-selects nearest emergency hospital to patient, uses offline mode if no API key):
```bash
python main.py
```

Custom locations, hospital, month/year, and offline flag:
```bash
python main.py --ambulance-lat 23.3500 --ambulance-lon 85.3100 --patient-lat 23.3600 --patient-lon 85.3200 --hospital "RIMS" --month 7 --year 2024 --offline
```

CLI options:
- `--ambulance-lat`, `--ambulance-lon`: Starting coordinates of the ambulance
- `--patient-lat`, `--patient-lon`: Patient pickup coordinates
- `--hospital`: Hospital name (partial match) or 0-based index (default: nearest emergency hospital)
- `--month` (1–12), `--year`: Period for weather severity prediction
- `--offline`: Force offline mode without calling Google Maps Routes API
- `--train-weather`: Retrain `weather_model.pkl` before routing
- `--no-map`: Skip HTML map generation

