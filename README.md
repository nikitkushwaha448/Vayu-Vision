# Air-Pulse — project requirements and run instructions

**Python version**: Use Python 3.11 (recommended). Python 3.14 may require building native wheels (pyarrow) and can fail on Windows.

**Required packages** (listed in `requirements.txt`):
- `streamlit` (web UI)
- `pandas`, `numpy` (data handling)
- `scikit-learn` (models)
- `matplotlib`, `seaborn` (plots)
- `joblib` (model saving/loading)
- `requests` (API calls)

Quick setup (PowerShell):
```powershell
# 1) Install Python 3.11 from python.org (if you don't have it)
# 2) From the project folder:
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# 3) Run the app (inside the activated venv):
python -m streamlit run app.py
```

Notes:
- If you prefer global/user installs, use `pip install --user -r requirements.txt` and run `python -m streamlit run app.py`.
- If installing on Windows and you must stay on Python 3.14, you'll likely need CMake + Visual Studio Build Tools to compile `pyarrow` (slow and error-prone); using Python 3.11 avoids that.

If you want, I can create the `.venv`, install dependencies and run the app for you now — tell me to proceed and whether you want automatic Python 3.11 install or you already have it installed.
