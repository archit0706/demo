"""Weather severity prediction module for the Ambulance Route Optimization System.

This module loads historical satellite climate records for Ranchi district from
``satellite_climate_district_FIXED.csv``, derives a rainfall-based weather
severity target, trains a ``RandomForestClassifier`` on climate features
(``Month``, ``Year``, ``LST_C``, ``NDVI``), and exposes utilities for loading the
persisted model, predicting weather severity, and mapping severity classes to
speed impact multipliers.

Important scope note:
    - The ``weather_severity`` target is derived strictly from ``Rainfall_mm``
      thresholds in the satellite climate dataset, NOT from any external
      traffic, road-speed, or travel-time dataset.
    - This model predicts weather severity from climate features only; it does
      NOT predict traffic conditions or ambulance travel time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RANCHI_DATASET_PATH = BASE_DIR / "data" / "ranchi_weather.csv"
DEFAULT_DATASET_PATH = BASE_DIR / "satellite_climate_district_FIXED.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "weather_model.pkl"

TARGET_DISTRICT = "Ranchi"
FEATURE_COLUMNS = ["Month", "Year", "LST_C", "NDVI"]
TARGET_COLUMN = "weather_severity"
REQUIRED_CSV_COLUMNS = ["District", "Year", "Month", "LST_C", "NDVI", "Rainfall_mm"]

SEVERITY_LABELS = (
    "clear",
    "light_rain",
    "moderate_rain",
    "heavy_rain",
    "extreme",
)
# Threshold boundaries in mm corresponding to SEVERITY_LABELS with [low, high) intervals:
# clear (<10), light_rain [10, 50), moderate_rain [50, 200),
# heavy_rain [200, 400), extreme (>=400).
RAINFALL_BINS = [-np.inf, 10.0, 50.0, 200.0, 400.0, np.inf]

WEATHER_MULTIPLIERS: dict[str, float] = {
    "clear": 1.0,
    "light_rain": 1.15,
    "moderate_rain": 1.3,
    "heavy_rain": 1.5,
    "extreme": 1.8,
}


def load_ranchi_climate_data(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    """Load the climate dataset filtered to Ranchi with derived weather severity.

    The ``weather_severity`` column is classified exclusively from monthly
    ``Rainfall_mm`` values in the satellite climate dataset (not from external
    traffic or travel-time data).

    Args:
        dataset_path: Path to ``satellite_climate_district_FIXED.csv``.

    Returns:
        A DataFrame containing Ranchi records with climate features and the
        derived ``weather_severity`` target column.

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
        ValueError: If no records are found for Ranchi.
    """
    path = Path(dataset_path)
    if not path.exists():
        if RANCHI_DATASET_PATH.exists():
            path = RANCHI_DATASET_PATH
        else:
            raise FileNotFoundError(f"Climate dataset not found at: {path}")

    df = pd.read_csv(path, usecols=REQUIRED_CSV_COLUMNS)
    ranchi_df = df[df["District"] == TARGET_DISTRICT].dropna(
        subset=FEATURE_COLUMNS + ["Rainfall_mm"]
    ).copy()

    if ranchi_df.empty:
        raise ValueError(f"No climate records found for District == '{TARGET_DISTRICT}'.")

    # Persist extracted Ranchi subset for reproducibility if not already created
    if not RANCHI_DATASET_PATH.exists() and path.resolve() != RANCHI_DATASET_PATH.resolve():
        try:
            RANCHI_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            full_df = pd.read_csv(path)
            full_df[full_df["District"] == TARGET_DISTRICT].to_csv(RANCHI_DATASET_PATH, index=False)
        except Exception:
            pass

    ranchi_df[TARGET_COLUMN] = pd.cut(
        ranchi_df["Rainfall_mm"],
        bins=RAINFALL_BINS,
        labels=list(SEVERITY_LABELS),
        right=False,
    ).astype(str)

    return ranchi_df.reset_index(drop=True)


@lru_cache(maxsize=4)
def _get_monthly_averages(dataset_path_str: str) -> dict[int, tuple[float, float]]:
    """Return cached Ranchi historical monthly averages as {month: (LST_C, NDVI)}."""
    ranchi_df = load_ranchi_climate_data(dataset_path_str)
    grouped = ranchi_df.groupby("Month")[["LST_C", "NDVI"]].mean()
    return {
        int(month): (float(row["LST_C"]), float(row["NDVI"]))
        for month, row in grouped.iterrows()
    }


def train_weather_model(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a RandomForestClassifier on Ranchi climate data and save it to disk.

    Predicts ``weather_severity`` (derived from ``Rainfall_mm``) using
    ``Month``, ``Year``, ``LST_C``, and ``NDVI`` with a stratified 80/20
    train/test split. Prints accuracy and classification report to stdout.

    Args:
        dataset_path: Path to the satellite climate CSV file.
        model_path: Destination path for the joblib-serialized model.
        test_size: Fraction of samples reserved for testing (default 0.2).
        random_state: Random seed for reproducible splitting and training.

    Returns:
        Dictionary containing ``accuracy``, ``classification_report`` (dict),
        ``train_samples``, ``test_samples``, and ``model_path``.
    """
    ranchi_df = load_ranchi_climate_data(dataset_path)
    X = ranchi_df[FEATURE_COLUMNS]
    y = ranchi_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(SEVERITY_LABELS),
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(SEVERITY_LABELS),
        zero_division=0,
        output_dict=True,
    )

    print("Ranchi Weather Severity Model Evaluation (RandomForestClassifier)")
    print("Note: Severity labels are derived from Rainfall_mm, not traffic/travel time.")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_text)

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    _get_monthly_averages.cache_clear()

    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": str(destination),
    }


def load_weather_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> RandomForestClassifier:
    """Load the persisted weather severity RandomForestClassifier from disk.

    Args:
        model_path: Path to the saved ``weather_model.pkl`` file.

    Returns:
        Loaded scikit-learn ``RandomForestClassifier`` instance.

    Raises:
        FileNotFoundError: If the saved model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weather model not found at: {path}. Run train_weather_model() first."
        )
    return joblib.load(path)


def predict_weather(
    month: int,
    year: int,
    lst_c: float | None = None,
    ndvi: float | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> str:
    """Predict weather severity for Ranchi given time and optional climate features.

    When ``lst_c`` or ``ndvi`` is omitted (``None``), the historical monthly
    mean for that month across Ranchi records is used.

    Args:
        month: Calendar month (1-12).
        year: Calendar year (e.g., 2024).
        lst_c: Optional Land Surface Temperature in degrees Celsius.
        ndvi: Optional Normalized Difference Vegetation Index.
        model_path: Path to the saved weather model pickle.
        dataset_path: Path to the satellite climate CSV used for monthly averages.

    Returns:
        Predicted weather severity label (one of ``clear``, ``light_rain``,
        ``moderate_rain``, ``heavy_rain``, ``extreme``).

    Raises:
        ValueError: If ``month`` is outside 1..12 or missing historical averages.
    """
    if isinstance(month, bool) or not isinstance(month, (int, np.integer)) or not (1 <= int(month) <= 12):
        raise ValueError(f"month must be an integer between 1 and 12, got {month!r}.")
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)):
        raise ValueError(f"year must be an integer, got {year!r}.")

    month_int = int(month)
    year_int = int(year)

    if lst_c is None or ndvi is None:
        monthly_avgs = _get_monthly_averages(str(Path(dataset_path).resolve()))
        if month_int not in monthly_avgs:
            raise ValueError(f"No historical Ranchi averages found for month {month_int}.")
        avg_lst_c, avg_ndvi = monthly_avgs[month_int]
        if lst_c is None:
            lst_c = avg_lst_c
        if ndvi is None:
            ndvi = avg_ndvi

    model = load_weather_model(model_path)
    features = pd.DataFrame(
        [[month_int, year_int, float(lst_c), float(ndvi)]],
        columns=FEATURE_COLUMNS,
    )
    return str(model.predict(features)[0])


def load_or_train_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> RandomForestClassifier:
    """Load the persisted weather model if available, otherwise train and save it."""
    path = Path(model_path)
    if path.exists():
        return load_weather_model(path)
    train_weather_model(dataset_path=dataset_path, model_path=path)
    return load_weather_model(path)


def predict_weather_details(
    month: int,
    year: int,
    lst_c: float | None = None,
    ndvi: float | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Predict weather severity and return structured details including multiplier."""
    month_int = int(month)
    year_int = int(year)
    resolved_ds = RANCHI_DATASET_PATH if RANCHI_DATASET_PATH.exists() else Path(dataset_path)
    monthly_avgs = _get_monthly_averages(str(resolved_ds.resolve()))
    avg_lst_c, avg_ndvi = monthly_avgs.get(month_int, (25.0, 0.45))

    used_lst = float(lst_c) if lst_c is not None else avg_lst_c
    used_ndvi = float(ndvi) if ndvi is not None else avg_ndvi

    severity = predict_weather(
        month=month_int,
        year=year_int,
        lst_c=used_lst,
        ndvi=used_ndvi,
        model_path=model_path,
        dataset_path=resolved_ds,
    )
    multiplier = get_weather_multiplier(severity)

    return {
        "severity": severity,
        "impact_multiplier": multiplier,
        "month": month_int,
        "year": year_int,
        "lst_c": round(used_lst, 2),
        "ndvi": round(used_ndvi, 3),
        "used_historical_averages": lst_c is None or ndvi is None,
    }


def get_weather_multiplier(severity: str) -> float:
    """Return the speed impact multiplier for a weather severity classification.

    Args:
        severity: Weather severity label (``clear``, ``light_rain``,
            ``moderate_rain``, ``heavy_rain``, or ``extreme``).

    Returns:
        Speed impact multiplier corresponding to ``severity``.

    Raises:
        ValueError: If ``severity`` is not a recognized weather severity label.
    """
    if severity not in WEATHER_MULTIPLIERS:
        valid = ", ".join(SEVERITY_LABELS)
        raise ValueError(f"Unknown weather severity {severity!r}. Expected one of: {valid}.")
    return WEATHER_MULTIPLIERS[severity]


if __name__ == "__main__":
    metrics = train_weather_model()
    print(f"Saved trained model to: {metrics['model_path']}")

    sample_cases = [
        (1, 2024, None, None),
        (7, 2024, None, None),
        (8, 2024, 24.5, 0.62),
    ]
    print("\nSample Predictions:")
    for sample_month, sample_year, sample_lst, sample_ndvi in sample_cases:
        predicted_severity = predict_weather(
            sample_month,
            sample_year,
            lst_c=sample_lst,
            ndvi=sample_ndvi,
        )
        multiplier = get_weather_multiplier(predicted_severity)
        print(
            f"  Month={sample_month}, Year={sample_year}, "
            f"LST_C={sample_lst}, NDVI={sample_ndvi} -> "
            f"severity={predicted_severity}, multiplier={multiplier}"
        )
