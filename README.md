# driver-monitoring-team2026280

## Quick Start (Windows PowerShell)

> Requires **Python 3.12**.

### First-time setup
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the webcam demo
```powershell
.venv\Scripts\activate
python scripts\demo_webcam.py
```

> **Tip:** If activation fails with "running scripts is disabled", run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
