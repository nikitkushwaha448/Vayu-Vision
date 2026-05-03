# Sequential Setup Guide - WAQI Real-Time API
## Complete Step-by-Step Instructions

---

## 📌 STEP 1: Verify Virtual Environment

**What this does:** Ensures Python is set up correctly in an isolated environment.

### Windows PowerShell:
```powershell
cd e:\Air-Pulse2\Air-Pulse
```

**Expected output:**
```
E:\Air-Pulse2\Air-Pulse>
```

**Check if .venv exists:**
```powershell
ls .venv
```

**If .venv doesn't exist, create it:**
```powershell
python -m venv .venv
```

**✅ Completion Check:**
- `.venv` folder should exist in project directory

---

## 📌 STEP 2: Activate Virtual Environment

**What this does:** Activates the isolated Python environment so your project uses its own Python.

### PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

**Expected output:**
```
(.venv) E:\Air-Pulse2\Air-Pulse>
```

**Note:** Notice the `(.venv)` prefix - this means venv is active.

**✅ Completion Check:**
- Prompt should show `(.venv)` prefix

---

## 📌 STEP 3: Install Dependencies

**What this does:** Installs all required Python packages (requests, streamlit, pandas, etc.).

### Run:
```powershell
pip install -r requirements.txt
```

**Expected output:**
```
Collecting streamlit
Collecting pandas
...
Successfully installed streamlit pandas numpy scikit-learn ...
```

**If pip is old, upgrade it first:**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**✅ Completion Check:**
- All packages should install without errors
- Look for "Successfully installed" message at the end

---

## 📌 STEP 4: Verify Configuration Files

**What this does:** Ensures your WAQI token is configured and accessible.

### Check if `.env` file exists:
```powershell
Get-Content .env
```

**Expected output:**
```
# Air-Pulse Configuration
# WAQI (AQICN) API Token for air quality fallback
WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc
```

**If `.env` doesn't exist or is empty:**

1. **Create it:**
   ```powershell
   @"
   # Air-Pulse Configuration
   # WAQI (AQICN) API Token for air quality fallback
   WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc
   "@ | Out-File .env -Encoding UTF8
   ```

2. **Verify it was created:**
   ```powershell
   Get-Content .env
   ```

**✅ Completion Check:**
- `.env` file exists in project root
- Contains `WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc`

---

## 📌 STEP 5: Verify Core Modules

**What this does:** Ensures all necessary Python files are present.

### Check for required files:
```powershell
ls realtime_api.py, config.py, app.py, AQI.py
```

**Expected output:**
```
    Directory: E:\Air-Pulse2\Air-Pulse

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---          4/29/2026   10:00 AM       12345 realtime_api.py
-a---          4/29/2026   10:00 AM        5678 config.py
-a---          4/29/2026   10:00 AM       98765 app.py
-a---          4/29/2026   10:00 AM        4321 AQI.py
```

**If any files are missing:**
- Contact support or check project structure
- Files should be in `e:\Air-Pulse2\Air-Pulse\`

**✅ Completion Check:**
- All 4 files should be present
- Timestamps should be recent

---

## 📌 STEP 6: Run Verification Test

**What this does:** Tests that the real-time API integration works correctly.

### Run test script:
```powershell
python scripts/test_api.py
```

**Expected output (Option A - With OpenAQ data):**
```
Testing lookups for: ahmedabad air quality

1. Trying direct OpenAQ city lookup...
SUCCESS - OpenAQ direct city lookup returned data

Pollutant snapshot:
  pm25: 45.3
  pm10: 67.8
  o3: 25.4
  ...

Estimated AQI (from PM2.5): 115.2
```

**Expected output (Option B - With WAQI fallback):**
```
Testing lookups for: delhi air quality

1. Trying direct OpenAQ city lookup...
No data - trying nearest OpenAQ stations...
No data - trying WAQI fallback...
SUCCESS - WAQI fallback returned data

Pollutant snapshot:
  pm25: 89.5
  pm10: 142.3
  ...

Estimated AQI (from PM2.5): 180.5
```

**Expected output (Option C - No data):**
```
Testing lookups for: Unknown City

1. Trying direct OpenAQ city lookup...
No data - trying nearest OpenAQ stations...
No data - trying WAQI fallback...
No data available from any source
```

**✅ Completion Check:**
- Script runs without errors
- Shows at least one data source working (or shows "no data" gracefully)
- Completes without Python errors

---

## 📌 STEP 7: Run the Application

**What this does:** Launches the full Streamlit web app with real-time air quality features.

### Run Streamlit:
```powershell
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Hint: Press Q to quit
```

### Browser opens automatically:
- If not, manually open: `http://localhost:8501`

**Expected app behavior:**
1. Home page loads with "AQI Command Home"
2. Three tabs visible: "Home", "AQI Prediction", "Health Prediction", "Analysis"
3. System Snapshot shows city count

**✅ Completion Check:**
- Streamlit starts without errors
- Browser opens to localhost:8501
- App loads successfully

---

## 📌 STEP 8: Test Real-Time Lookup in App

**What this does:** Verifies the app can fetch real-time air quality data.

### In Streamlit app:
1. Click on **"AQI Prediction"** tab
2. Select a city from dropdown (e.g., "Ahmedabad", "Mumbai", etc.)
3. Click **"Predict AQI"** button

**Expected results:**
- Pollutant values appear (PM2.5, PM10, O3, NO2, SO2, CO)
- AQI value is shown
- Air quality status displayed (e.g., "Good", "Moderate", "Unhealthy")
- Data source shown (OpenAQ Direct / OpenAQ Nearest / WAQI Live)

**If you see an error:**
- Check internet connection
- Verify `.env` file has WAQI token
- Try a different city
- Check terminal for error messages

**✅ Completion Check:**
- At least one city returns pollutant data
- AQI is calculated and displayed
- Health recommendations appear in "Health Prediction" tab

---

## 📌 STEP 9: Verify Health Predictions

**What this does:** Tests the health impact calculations based on AQI.

### In Streamlit app:
1. Stay on **"AQI Prediction"** tab
2. Select a city and get AQI (if not already done)
3. Click on **"Health Prediction"** tab

**Expected results:**
- Shows current AQI and city
- Displays health predictions for:
  - Overall population
  - General population
  - Vulnerable population
- Recommended actions appear
- Advanced planner options available

**✅ Completion Check:**
- Health Prediction tab loads
- Shows health metrics for current AQI
- Interactive planner functions work

---

## 📌 STEP 10: Verify Analysis Tab

**What this does:** Tests historical data analysis from project CSV files.

### In Streamlit app:
1. Click on **"Analysis"** tab

**Expected results:**
- Historical AQI trends displayed
- Monthly analysis graphs show
- Pollutant correlation matrix appears
- Can filter by year and select specific pollutants

**✅ Completion Check:**
- Analysis tab loads without errors
- Historical data from CSV files displays
- Charts render correctly

---

## ✅ All Steps Complete!

Once you've verified all 10 steps, your Air-Pulse real-time API integration is fully functional.

### Summary of What's Working:
- ✅ Real-time OpenAQ data (primary source)
- ✅ WAQI fallback (when OpenAQ unavailable)
- ✅ Project city detection from CSV files
- ✅ AQI calculation from pollutants
- ✅ Health recommendations
- ✅ Historical analysis
- ✅ Personal protection planning

---

## 🆘 Quick Troubleshooting

| Step | Problem | Solution |
|------|---------|----------|
| 2 | `Activate.ps1` won't run | Try: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| 3 | pip install fails | Run: `python -m pip install --upgrade pip` first |
| 4 | Can't find .env | Create it manually with WAQI token |
| 6 | Test script errors | Check PYTHONPATH, run from project folder |
| 7 | Streamlit not found | Run step 3 again to install dependencies |
| 8 | No data from app | Check internet connection, verify WAQI token |
| 8 | "ModuleNotFoundError" | Ensure venv is activated (see `.venv` prefix) |

---

## 📞 Need Help?

1. Check relevant documentation:
   - `QUICK_START.md` - Quick reference
   - `REALTIME_API_SETUP.md` - Detailed setup guide
   - `INTEGRATION_SUMMARY.md` - Technical overview

2. Run test script for diagnostics:
   ```powershell
   python scripts/test_api.py
   ```

3. Check terminal output for specific error messages

---

**Setup Status: ✅ COMPLETE**  
**Date: April 29, 2026**  
**All systems ready to use!**
