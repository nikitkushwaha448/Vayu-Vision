import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Health.csv"


def _aqi_band(aqi_series: np.ndarray) -> np.ndarray:
    """Map AQI to standard band buckets used as engineered feature."""
    aqi = aqi_series.astype(float)
    return np.select(
        [
            aqi <= 50,
            aqi <= 100,
            aqi <= 150,
            aqi <= 200,
            aqi <= 300,
        ],
        [0, 1, 2, 3, 4],
        default=5,
    )


def _feature_builder(x: np.ndarray) -> np.ndarray:
    """Build robust features from a single AQI input column."""
    aqi = np.asarray(x).astype(float).reshape(-1, 1)
    bands = _aqi_band(aqi.flatten()).reshape(-1, 1)
    return np.hstack([aqi, bands])


def train_target(df: pd.DataFrame, target_col: str, out_name: str) -> dict:
    x = df[["AQI"]].values
    y = df[target_col].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_builder", FunctionTransformer(_feature_builder, validate=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    out_path = BASE_DIR / out_name
    joblib.dump(pipeline, out_path)

    return {
        "target": target_col,
        "model_file": out_name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    if "AQI" not in df.columns:
        raise ValueError("Health.csv must contain AQI column")

    # Keep only required columns and valid AQI rows.
    required = ["AQI", "Health_Consequences", "General_Population", "Vulnerable_Population"]
    df = df[required].copy()
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)

    results = []
    results.append(train_target(df, "Health_Consequences", "model_health_v2.pkl"))
    results.append(train_target(df, "General_Population", "model_general_v2.pkl"))
    results.append(train_target(df, "Vulnerable_Population", "model_vulnerable_v2.pkl"))

    report = {
        "dataset_rows": int(len(df)),
        "features": ["AQI", "AQI_BAND"],
        "models": results,
    }

    report_path = BASE_DIR / "health_models_report_v2.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Health v2 models trained successfully")
    for r in results:
        print(f"{r['target']}: acc={r['accuracy']}, f1={r['f1_weighted']}, file={r['model_file']}")


if __name__ == "__main__":
    main()
