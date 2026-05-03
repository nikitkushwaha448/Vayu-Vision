# 🎯 MASTER SETUP GUIDE - All Steps Sequential Wise

## 📍 START HERE

This document provides the complete, sequential path to get WAQI real-time API working.

---

## 🗺️ OVERVIEW - What Gets Set Up

```
Your Computer
    ↓
Activate Python Virtual Environment (.venv)
    ↓
Install Dependencies (streamlit, requests, etc.)
    ↓
Load Configuration (.env with WAQI token)
    ↓
Test Real-Time API Client (OpenAQ + WAQI)
    ↓
Launch Streamlit Web App
    ↓
Select City → Fetch Real-Time Air Quality Data
    ↓
Display Pollutants, AQI, Health Recommendations
```

---

## 📚 DOCUMENTATION FILES (Read in Order)

| Document | Purpose | Time |
|----------|---------|------|
| **This file** | Overview & quick reference | 2 min |
| **SEQUENTIAL_SETUP.md** | Detailed step-by-step guide | 30 min |
| **COMPLETE_CHECKLIST.md** | Verification checklist | 10 min |
| **QUICK_START.md** | Quick reference for running | 5 min |
| **REALTIME_API_SETUP.md** | Full technical setup | 15 min |
| **INTEGRATION_SUMMARY.md** | Technical overview | 10 min |

---

## ⚡ QUICK PATH (If you just want to run it)

### For experienced users:
```powershell
# 1. Navigate
cd e:\Air-Pulse2\Air-Pulse

# 2. Activate venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py

# 5. Select city → Click "Predict AQI"
```

**Time required:** 5-10 minutes

---

## 📋 FULL SETUP - Do These 10 Steps

### Phase 1: Environment (3 steps)

**STEP 1:** Open PowerShell and navigate
```powershell
cd e:\Air-Pulse2\Air-Pulse
```

**STEP 2:** Activate virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*Expected: Prompt shows `(.venv)` prefix*

**STEP 3:** Install dependencies
```powershell
pip install -r requirements.txt
```
*Expected: "Successfully installed" at end*

---

### Phase 2: Configuration (2 steps)

**STEP 4:** Verify .env file
```powershell
Get-Content .env
```
*Expected: Shows WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc*

**STEP 5:** If .env is missing, create it:
```powershell
@"
# Air-Pulse Configuration
WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc
"@ | Out-File .env -Encoding UTF8
```

---

### Phase 3: Testing (2 steps)

**STEP 6:** Run test script (no city specified - uses project CSV)
```powershell
python scripts/test_api.py
```
*Expected: Shows pollutant values OR "No data"*

**STEP 7:** Test with specific city
```powershell
python scripts/test_api.py "Mumbai"
```
*Expected: Returns air quality data if available*

---

### Phase 4: Application (2 steps)

**STEP 8:** Launch Streamlit
```powershell
streamlit run app.py
```
*Expected: Browser opens to http://localhost:8501*

**STEP 9:** In Streamlit browser:
- Click "AQI Prediction" tab
- Select a city from dropdown
- Click "Predict AQI" button
- View real-time pollutant data

**STEP 10:** Verify all features:
- [ ] AQI Prediction tab shows data
- [ ] Health Prediction tab calculates risks
- [ ] Analysis tab shows historical trends
- [ ] Download buttons work

---

## 🎮 LAUNCHER OPTIONS

### Option 1: Batch File (Easiest)
```
1. Double-click: launcher.bat
2. Select option 1
```
**Pros:** No terminal knowledge needed  
**Cons:** Slower

### Option 2: PowerShell Script
```powershell
.\launcher.ps1
# Select option 1
```
**Pros:** Clear menu, colorized output  
**Cons:** Requires PowerShell knowledge

### Option 3: Direct Command (Fastest)
```powershell
streamlit run app.py
```
**Pros:** Direct and fast  
**Cons:** Manual setup needed first

### Option 4: Automatic Setup Script
```
Double-click: setup.bat
```
**Pros:** Runs all setup automatically  
**Cons:** May need configuration after

---

## 🔄 TYPICAL FIRST-TIME FLOW

```
1. Open PowerShell
   ↓
2. cd e:\Air-Pulse2\Air-Pulse
   ↓
3. .\.venv\Scripts\Activate.ps1
   ↓
4. pip install -r requirements.txt
   ↓
5. streamlit run app.py
   ↓
6. Browser opens → Select city → Click "Predict AQI"
   ↓
7. View real-time air quality data
```

**Total time:** 15-20 minutes first time, 1-2 minutes subsequent times

---

## 📊 WHAT EACH PHASE DOES

### Phase 1: Environment Setup
- Ensures Python is isolated to this project
- Prevents conflicts with other Python projects
- Creates `.venv` folder with its own Python

### Phase 2: Dependencies Installation
- Installs streamlit (web framework)
- Installs pandas (data processing)
- Installs requests (HTTP library for APIs)
- Installs scikit-learn, joblib (ML/model loading)

### Phase 3: Configuration Loading
- Reads `.env` file with WAQI token
- Makes token available to all modules
- Enables WAQI fallback when OpenAQ unavailable

### Phase 4: Testing
- Verifies OpenAQ client works
- Tests WAQI fallback
- Confirms AQI calculation
- Validates end-to-end data flow

### Phase 5: Application Launch
- Starts Streamlit web server
- Opens browser to dashboard
- Makes app interactive

---

## 🔑 KEY POINTS TO REMEMBER

### `.env` File
- Location: `e:\Air-Pulse2\Air-Pulse\.env`
- Contains: WAQI token (keep private!)
- Auto-loaded by config.py
- **Token provided:** e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc

### Virtual Environment
- Must be activated before running Python code
- Prefix `(.venv)` in prompt = activated
- Ensures project uses correct Python packages
- Create once, activate each time you work

### Dependencies
- Defined in `requirements.txt`
- Install once with: `pip install -r requirements.txt`
- Add new packages by: `pip install package_name`

### Real-Time Data Sources
1. **OpenAQ** - Free, no token (primary)
2. **WAQI** - Free with token (fallback)
3. **Project CSVs** - Historical data (always available)

---

## ✅ FINAL VERIFICATION

When everything is set up correctly, you should see:

### In PowerShell Terminal:
```
(.venv) E:\Air-Pulse2\Air-Pulse> streamlit run app.py

  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### In Browser:
- Home page with city statistics
- City dropdown menu
- "Predict AQI" button
- Health Prediction and Analysis tabs

### When you click "Predict AQI":
- Real-time pollutant values (PM2.5, PM10, etc.)
- Estimated AQI
- Air quality status
- Data source attribution

---

## 🆘 COMMON ISSUES & FIXES

| Issue | Solution |
|-------|----------|
| `.venv` not found | Run: `python -m venv .venv` |
| `Activate.ps1` won't run | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError` | Make sure venv is activated (see `.venv` in prompt) |
| Streamlit not found | Run: `pip install -r requirements.txt` |
| No data returned | Check internet, verify `.env` has token |
| Port 8501 in use | Run: `streamlit run app.py --server.port 8502` |

---

## 📞 DETAILED HELP

For more detailed help, consult:
- **Step-by-step guide:** SEQUENTIAL_SETUP.md
- **Troubleshooting:** REALTIME_API_SETUP.md
- **Quick reference:** QUICK_START.md
- **Checklist:** COMPLETE_CHECKLIST.md

---

## 🎯 YOUR IMMEDIATE NEXT STEPS

1. **Choose your method** (launcher.bat or PowerShell commands)
2. **Open PowerShell** in the project folder
3. **Run the first 3 steps** (navigate, activate, install)
4. **Run test** (python scripts/test_api.py)
5. **Launch app** (streamlit run app.py)
6. **Select city and get air quality data!**

---

## ✨ Success! 

Once you see:
- ✅ Streamlit running in browser
- ✅ City dropdown populated
- ✅ Pollutant data displayed after clicking "Predict AQI"
- ✅ Health recommendations shown

**You've successfully integrated real-time air quality APIs!** 🎉

---

## 📅 Timeline

| Phase | Time | Action |
|-------|------|--------|
| Setup | 5 min | Activate venv + install |
| Test | 2 min | Run test script |
| Launch | 1 min | Start streamlit |
| Use | ∞ | Get air quality data |

**Total first-time setup: ~10-15 minutes**

---

**Status:** ✅ Complete & Ready  
**Date:** April 29, 2026  
**All systems operational**

**→ Start with SEQUENTIAL_SETUP.md for detailed instructions**
