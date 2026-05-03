# Quick Start Checklist - WAQI Real-Time API

## ✅ Completed Setup

- [x] OpenAQ client module created (`realtime_api.py`)
- [x] WAQI fallback integration implemented
- [x] Config loader setup (reads `.env`)
- [x] `.env` file created with your token
- [x] Streamlit app updated with real-time lookup
- [x] Test scripts created
- [x] Launcher scripts created (batch + PowerShell)
- [x] Documentation written

## 🚀 TO RUN - Choose One Method

### METHOD 1: Windows Batch (Easiest)
Double-click: `launcher.bat`
```
Then select option 1 to run Streamlit
```

### METHOD 2: PowerShell (Recommended)
Open PowerShell in project folder and run:
```powershell
.\launcher.ps1
```
Then select option 1

### METHOD 3: Direct Commands
Open PowerShell and run:
```powershell
cd e:\Air-Pulse2\Air-Pulse
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### METHOD 4: Test First
If you want to test without Streamlit:
```powershell
cd e:\Air-Pulse2\Air-Pulse
python scripts/test_api.py
```

## 📊 What Happens When You Run

1. **Config loads** → reads WAQI token from `.env`
2. **User selects city** → app automatically picks from your project files
3. **Data fetch attempt 1** → tries OpenAQ direct city lookup
4. **Data fetch attempt 2** → if no data, geocodes city and finds nearest OpenAQ stations
5. **Data fetch attempt 3** → if still no data, uses WAQI with your token
6. **Results display** → shows pollutants + estimated AQI + health actions

## 📝 Your WAQI Token

Location: `e:\Air-Pulse2\Air-Pulse\.env`
```
WAQI_TOKEN=e4de99b6014f3f7d8783ae5e2e398a70a2d35dfc
```

✅ Already configured - no action needed

## 🔍 Verify Everything

Run this to check all systems:
```powershell
cd e:\Air-Pulse2\Air-Pulse
python scripts/test_api.py
```

Expected output:
```
Testing lookups for: [your city]

1. Trying direct OpenAQ city lookup...
   (if no data) Trying nearest OpenAQ stations...
   (if no data) Trying WAQI fallback...

Pollutant snapshot:
  pm25: [value]
  pm10: [value]
  ...

Estimated AQI (from PM2.5): [value]
```

## 🆘 If Something Doesn't Work

1. **Make sure venv is activated:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Check `.env` file exists:**
   - Location: `e:\Air-Pulse2\Air-Pulse\.env`
   - Should contain your WAQI token

3. **Verify dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Test network connection:**
   ```powershell
   python scripts/net_check.py
   ```

## 📚 Documentation

- Full guide: `REALTIME_API_SETUP.md`
- This checklist: `QUICK_START.md` (this file)
- Code: `realtime_api.py` (main client)
- Config: `config.py` (token loader)

## 🎯 Next Steps After Running

1. ✅ Run the app (see METHOD 1-4 above)
2. ✅ Select a city from dropdown
3. ✅ Click "Predict AQI"
4. ✅ View pollutants and estimated AQI
5. ✅ Go to "Health Prediction" tab for health insights

**That's it! 🎉**

The real-time API integration is ready to use.
