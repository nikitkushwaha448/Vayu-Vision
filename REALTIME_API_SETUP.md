# Air-Pulse Real-Time API Setup Guide

This guide walks you through setting up the real-time air quality API integration (OpenAQ + WAQI fallback).

## ✅ What's Already Done

- ✅ OpenAQ client (`realtime_api.py`) - free, no token needed
- ✅ WAQI fallback integration - uses your token
- ✅ Streamlit app integration - auto-detects and uses best available data source
- ✅ Project city detection - automatically finds CSV files you've added
- ✅ `.env` file with your WAQI token
- ✅ Config loader - reads `.env` at startup

## 🚀 Quick Start (3 Steps)

### Step 1: Verify `.env` File
The `.env` file should be located at: `e:\Air-Pulse2\Air-Pulse\.env`

Contents should look like:
```
# Air-Pulse Configuration
# WAQI (AQICN) API Token for air quality fallback
WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc
```

✅ Already done - your token is in place.

### Step 2: Activate Virtual Environment
Open PowerShell and run:
```powershell
cd e:\Air-Pulse2\Air-Pulse
.\.venv\Scripts\Activate.ps1
```

### Step 3: Run the Application

**Option A - Run Streamlit App (Recommended):**
```powershell
streamlit run app.py
```
Then:
- Open browser to `http://localhost:8501`
- Go to "AQI Prediction" tab
- Select a city from the dropdown
- Click "Predict AQI"
- App will automatically:
  1. Try OpenAQ direct lookup
  2. Try geocoding + nearest OpenAQ stations
  3. Fall back to WAQI using your token

**Option B - Run Test Script:**
```powershell
python scripts/test_api.py
```
Or with a specific city:
```powershell
python scripts/test_api.py "Delhi"
```

**Option C - Run Direct Checker:**
```powershell
python AQI.py
```

---

## 📊 Data Flow

```
Selected City (app.py)
    ↓
Try 1: OpenAQ direct city lookup
    ↓ (if no data)
Try 2: Geocode city → Find nearest OpenAQ stations
    ↓ (if no data)
Try 3: WAQI fallback (uses your token from .env)
    ↓
Display pollutant values + estimated AQI
```

## 🔍 What Gets Tested

The system fetches these pollutants:
- **PM2.5** (Fine particulate matter)
- **PM10** (Coarse particulate matter)
- **O3** (Ozone)
- **NO2** (Nitrogen dioxide)
- **SO2** (Sulfur dioxide)
- **CO** (Carbon monoxide)

If PM2.5 is available, AQI is estimated using US EPA breakpoints.

## 🗂️ Project City Detection

The app automatically detects cities you've added by looking for CSV files like:
- `ahmedabad-air-quality.csv` → City: "ahmedabad air quality"
- `delhi-air-quality.csv` → City: "delhi air quality"
- `mumbai-air-quality.csv` → City: "mumbai air quality"

These become available in the city dropdown in the Streamlit app.

## 📁 File Structure

```
Air-Pulse/
├── .env                          ← Your WAQI token (created)
├── .env.example                  ← Template reference
├── config.py                     ← Loads .env and provides tokens
├── realtime_api.py              ← OpenAQ + WAQI client + AQI calculator
├── app.py                        ← Streamlit app (updated with real-time lookup)
├── AQI.py                        ← Direct checker (updated with real-time lookup)
├── requirements.txt              ← Dependencies (already has requests)
└── scripts/
    ├── test_api.py              ← New comprehensive test script
    ├── test_openaq.py           ← Original test script
    └── net_check.py             ← Network connectivity check
```

## 🔧 Troubleshooting

### Issue: "No data available from any source"
**Solution:** 
- Check internet connection
- Verify `.env` file exists and has correct token
- Try a different city name
- Check city is supported by OpenAQ or WAQI

### Issue: WAQI token not working
**Solution:**
1. Verify token in `.env`: `e:\Air-Pulse2\Air-Pulse\.env`
2. Restart Python/Streamlit after editing `.env`
3. Get a new token from: https://aqicn.org/data-platform/token/

### Issue: "ModuleNotFoundError: No module named 'realtime_api'"
**Solution:**
- Ensure you're running from project directory: `cd e:\Air-Pulse2\Air-Pulse`
- Verify `realtime_api.py` exists in that directory

### Issue: Streamlit not found
**Solution:**
- Install dependencies: `pip install -r requirements.txt`
- Or activate venv: `.\.venv\Scripts\Activate.ps1`

## 📚 Advanced Usage

### Use WAQI Token Explicitly
```python
from realtime_api import fetch_from_waqi
data = fetch_from_waqi("Delhi", "your_token_here")
```

### Estimate AQI from PM2.5
```python
from realtime_api import compute_aqi_from_pm25
aqi = compute_aqi_from_pm25(35.5)  # PM2.5 = 35.5 µg/m³
print(aqi)  # Output: ~100 (Moderate)
```

### Geocode and Find Nearest Stations
```python
from realtime_api import fetch_nearest_by_city
data = fetch_nearest_by_city("New York", radius_m=100000)
```

## 🎯 Next Steps

1. ✅ **Test the setup:**
   - Run `streamlit run app.py`
   - Select a city
   - Click "Predict AQI"

2. 🔄 **Monitor results:**
   - Check which data source is used (OpenAQ direct, nearest, or WAQI)
   - Verify pollutant values make sense

3. 📊 **Integrate health predictions:**
   - Go to "Health Prediction" tab
   - Use the AQI value to get health insights

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Review error messages in terminal/Streamlit UI
3. Verify `.env` file and token
4. Test with `python scripts/test_api.py`

---

**Version:** 1.0  
**Last Updated:** April 29, 2026  
**Status:** ✅ Ready to use
