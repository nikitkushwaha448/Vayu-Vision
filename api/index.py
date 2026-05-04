"""Vayu-Vision API implemented with FastAPI.

Exports `app` (FastAPI instance) for ASGI/hosting platforms and Vercel.
If FastAPI is not installed, falls back to a minimal WSGI app for safety.
Additionally exports an optional `handler` (Mangum) when available for serverless
adapters that expect a Lambda-style handler (useful on some Vercel runtimes).
"""

import os
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

try:
    from fastapi import FastAPI
    fastapi_available = True
except Exception:
    FastAPI = None
    fastapi_available = False

# Optional serverless adapter (Mangum) — only used if installed.
handler = None
try:
    from mangum import Mangum  # type: ignore
    mangum_available = True
except Exception:
    Mangum = None
    mangum_available = False

if fastapi_available:
    app = FastAPI(title="Vayu-Vision")

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "vayu-vision"}

    @app.get("/")
    async def root():
        return {"message": "Vayu-Vision API is running", "hint": "Use /health for health checks"}

    # If Mangum is available, export a Lambda-compatible handler too.
    if mangum_available:
        handler = Mangum(app)

    # --- Prediction endpoint -------------------------------------------------
    from pydantic import BaseModel, Field, model_validator
    import joblib
    import numpy as np
    from typing import Dict, Optional

    class PredictRequest(BaseModel):
        """Request for `/predict`.

        - `city`: city key (e.g., "Delhi")
        - `pollutant_values`: mapping with keys `pm25, pm10, o3, no2, so2, co` and numeric values.
        """
        city: str = Field(..., description="City key used to select the AQI model.", examples=["Delhi"])
        pollutant_values: Dict[str, Optional[float]] = Field(
            ...,
            description="Pollutant concentration map with keys pm25, pm10, o3, no2, so2, co.",
            examples=[
                {
                    "pm25": 100.0,
                    "pm10": 80.0,
                    "o3": 20.0,
                    "no2": 30.0,
                    "so2": 5.0,
                    "co": 0.4,
                }
            ],
        )

        @model_validator(mode="before")
        @classmethod
        def check_pollutant_keys(cls, values):
            required = {"pm25", "pm10", "o3", "no2", "so2", "co"}
            pv = values.get("pollutant_values")
            if not isinstance(pv, dict):
                raise ValueError("pollutant_values must be an object with pollutant keys")
            missing = required - set(pv.keys())
            if missing:
                raise ValueError(f"Missing pollutant keys: {', '.join(sorted(missing))}")
            for k in required:
                v = pv.get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    raise ValueError(f"Pollutant {k} must be numeric")
                if fv < 0:
                    raise ValueError(f"Pollutant {k} must be non-negative")
                pv[k] = fv
            values["pollutant_values"] = pv
            return values

    class PredictResponse(BaseModel):
        city: str = Field(..., description="City key that was predicted.")
        predicted_aqi: float = Field(..., description="Predicted AQI value.")
        source: str = Field(..., description="Whether the value came from a model or fallback estimator.")


    # Minimal sklearn pickle compatibility helper (borrowed from app.py)
    def patch_sklearn_pickle_compat(obj):
        if obj is None:
            return obj

        estimators = getattr(obj, 'estimators_', None)
        if estimators is not None:
            for estimator in estimators:
                patch_sklearn_pickle_compat(estimator)

        if not hasattr(obj, 'monotonic_cst'):
            try:
                setattr(obj, 'monotonic_cst', None)
            except Exception:
                pass

        if hasattr(obj, 'statistics_') and not hasattr(obj, '_fill_dtype'):
            try:
                fill_dtype = obj.statistics_.dtype if getattr(obj, 'statistics_', None) is not None else np.float64
                setattr(obj, '_fill_dtype', fill_dtype)
            except Exception:
                pass

        return obj

    # Lazy model cache
    _model_cache: Dict[str, Dict] = {}

    def load_model_for_city(city_key: str):
        # Map a few common city keys to filenames used in app.py
        name_map = {
            "Delhi": "Delhi_random_forest_model.pkl",
            "Ghaziabad": "Ghaziabad_random_forest_model.pkl",
            "Hyderabad": "Hyderabad_random_forest_model.pkl",
            "Mumbai": "Mumbai_random_forest_model.pkl",
            "Ahmedabad": "Ahmedabad_random_forest_model.pkl",
        }
        model_file = name_map.get(city_key, None)
        if model_file is None:
            return None
        try:
            model = joblib.load(model_file)
            imputer = joblib.load('imputer.pkl')
            patch_sklearn_pickle_compat(model)
            patch_sklearn_pickle_compat(imputer)
            return {"model": model, "imputer": imputer}
        except Exception:
            return None


    @app.post(
        "/predict",
        response_model=PredictResponse,
        summary="Predict AQI from pollutant values",
        description="Predict AQI using a city-specific Random Forest model when available, otherwise use a conservative PM2.5 fallback.",
    )
    async def predict(req: PredictRequest):
        # Expect pollutant keys: pm25, pm10, o3, no2, so2, co
        keys = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        vals = [float(req.pollutant_values.get(k, 0.0) or 0.0) for k in keys]

        cache_entry = _model_cache.get(req.city)
        if cache_entry is None:
            cache_entry = load_model_for_city(req.city)
            if cache_entry:
                _model_cache[req.city] = cache_entry

        if not cache_entry:
            # Fallback: compute AQI from PM2.5 only (basic estimator)
            pm25 = vals[0]
            # Very simple linear fallback (not a production AQI); keep conservative values
            aqi = max(0.0, pm25 * 2.0)
            return {"city": req.city, "predicted_aqi": round(float(aqi), 2), "source": "fallback"}

        model = cache_entry['model']
        imputer = cache_entry['imputer']
        import pandas as pd

        input_df = pd.DataFrame([dict(zip(keys, vals))])
        input_imputed = imputer.transform(input_df)
        pred = float(model.predict(input_imputed)[0])
        return {"city": req.city, "predicted_aqi": round(pred, 2), "source": "model"}

else:
    # Minimal WSGI fallback so imports won't completely break if FastAPI isn't installed.
    import json

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path == "/health":
            payload = {"status": "ok", "service": "vayu-vision"}
        else:
            payload = {"message": "Vayu-Vision API running (WSGI fallback)", "hint": "Use /health"}

        body = json.dumps(payload).encode("utf-8")
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
        start_response("200 OK", headers)
        return [body]

# Helpful debug info when running locally; not required for deployment.
if os.getenv("VERCEL"):
    # On Vercel, it's harmless to have both `app` (ASGI) and `handler` available.
    pass
