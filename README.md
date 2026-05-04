# Air-Pulse

Air-Pulse is a Streamlit-based air quality intelligence project focused on Indian cities. It combines city-level pollutant data, AQI prediction models, and health-risk guidance to help users understand current conditions and make safer day-to-day decisions.

![Air-Pulse Banner](bg3.jpg)

![Home Experience](home_bg_trend.jpg)

## Visual Preview

| Home and Trends | Prediction Theme | City Dashboard Style |
| --- | --- | --- |
| ![Home Card](home_trend_card.jpg) | ![Background](bg.jpg) | ![Delhi Theme](delhi.jpg) |

## Highlights

- Real-time + model-assisted AQI view for supported cities
- AQI status interpretation from Good to Hazardous
- Health prediction and action recommendations
- Personal protection planner with exposure, mask, and purifier inputs
- Commute safety planner (General, School, Office profiles)
- Historical analysis with trends, top AQI months, and pollutant comparisons
- Downloadable analysis outputs (CSV) in the app

## Application Pages

The app sidebar includes:

- Home
- AQI Prediction
- Health Prediction
- Analysis

## City Experience Gallery

The app uses city-specific visuals to create a more immersive experience while switching across locations.

| City | Preview |
| --- | --- |
| Ahmedabad | ![Ahmedabad](ahmedabad.jpg) |
| Chennai | ![Chennai](chennai.jpg) |
| Gurgaon | ![Gurgaon](gurgaon.jpg) |
| Hyderabad | ![Hyderabad](hyderabad.jpg) |
| Mumbai | ![Mumbai](mumbai.jpg) |
| Nagaland | ![Nagaland](nagaland.jpg) |
| Punjab | ![Punjab](punjab.jpg) |
| Ghaziabad | ![Ghaziabad](ghaziabad.jpg) |
| Lucknow | ![Lucknow](lucknow.jpg) |
| Noida | ![Noida](noida.jpg) |

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn (Random Forest and health models)
- Matplotlib and Seaborn
- Joblib and Pickle
- Requests

## Project Structure

Key files and directories:

- app.py: Main Streamlit application
- analysis.py: Data processing and trend helpers used by the app
- train_all_models.py: Trains AQI models for available city datasets
- train_health_models_v2.py: Trains health-related model set
- test_models.py: Validates model artifacts
- predict_health.py: Health prediction utility script
- MODELS_SUMMARY.md: AQI model performance summary
- requirements.txt: Python dependencies
- *.csv: City-level historical pollutant and AQI datasets
- *_random_forest_model.pkl and model_*.pkl: Trained model artifacts

## Setup

### 1) Prerequisites

- Python 3.11 recommended
- Windows PowerShell (or any terminal)

Python 3.14 can require compiling native packages on Windows (for example pyarrow), which may fail without build tools. Python 3.11 is the most reliable choice for this project.

### 2) Create and activate virtual environment

```powershell
py -3.11 -m venv .venv311
& .\.venv311\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### 4) Run the app

Use VS Code: open the Run & Debug view, select "Run Air-Pulse (Streamlit)", then press F5.

Or command-line:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in terminal (typically http://localhost:8501).

## Model and Data Notes

- AQI models use pollutant inputs: pm25, pm10, o3, no2, so2, co
- Primary AQI output: AQI value
- Multiple city-specific Random Forest models are included
- Health models are available in both legacy and v2 artifacts

## Training and Validation

Train or retrain models:

```powershell
python train_all_models.py
python train_health_models_v2.py
```

Run validations:

```powershell
python test_models.py
```

## Troubleshooting

### 1) Streamlit not found

Ensure your virtual environment is activated, then reinstall dependencies:

```powershell
& .\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Model loading issues across scikit-learn versions

Some older model artifacts may require compatibility handling when loaded in newer scikit-learn versions. The app includes fallback logic for safer runtime behavior.

### 3) Health model v2 loading fallback

If the v2 health pickle cannot resolve a transformer reference at runtime, the app automatically falls back to legacy health model artifacts to keep predictions available.

## Future Improvements

- Add CI checks for model artifact compatibility
- Add automated dataset freshness checks
- Add deployment profile for Streamlit Community Cloud or containerized hosting
- Add live screenshot GIF walkthrough for each app page

## License

Add your preferred license file (for example MIT) before public reuse.

## Vercel Deployment (API)

This repository exposes a small API under `api/index.py` implemented with FastAPI. The `vercel.json` file maps all incoming routes to that module so Vercel's Python runtime will invoke it.

Notes:
- `api/index.py` exports both an ASGI `app` (FastAPI) and an optional `handler` (Mangum) when the `mangum` package is installed — this makes the module adaptable to several serverless execution environments.
- To test the API locally before deploying to Vercel, use the included PowerShell helper `vercel_deploy_test.ps1` which starts `uvicorn` and exercises `/health` and `/predict`.

Local test commands:

```powershell
& .\.venv311\Scripts\Activate.ps1
.\vercel_deploy_test.ps1
```

If you plan to deploy to Vercel, ensure `requirements.txt` includes all runtime dependencies (already updated) and then follow Vercel's Python deployment docs or use the Vercel CLI to deploy from this project root.

## GitHub Actions and Vercel Secrets

The CI workflow in `.github/workflows/ci.yml` runs tests on every pull request and push to `main`, then deploys to Vercel after a successful `main` branch build.

Before the deploy job can run, add these repository secrets in GitHub:

- `VERCEL_TOKEN` - your Vercel personal token
- `VERCEL_ORG_ID` - the Vercel team or account organization ID
- `VERCEL_PROJECT_ID` - the Vercel project ID for this repo

You can set them from the GitHub UI under Settings > Secrets and variables > Actions, or with the GitHub CLI:

```powershell
gh secret set VERCEL_TOKEN
gh secret set VERCEL_ORG_ID
gh secret set VERCEL_PROJECT_ID
```

If you want to deploy to a different project or branch, update the workflow file accordingly before merging to `main`.
