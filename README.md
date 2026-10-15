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

## Raspberry Pi with an OV9281 camera

Tested target: Raspberry Pi OS Bookworm (64-bit).

### 1. Enable the camera
Add this to `/boot/firmware/config.txt` and reboot:
```
camera_auto_detect=0
dtoverlay=ov9281
```
(On a Pi 5 with the camera on the CAM0 port, use `dtoverlay=ov9281,cam0`.)

Check that it is detected: `rpicam-hello --list-cameras` should show `ov9281`.

### 2. Set up Python
Picamera2 comes from apt, not pip, so the venv needs access to system packages:
```bash
sudo apt install -y python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> If `import picamera2` fails with a numpy error after this, run `pip install "numpy<2"`.

### 3. Run the demo
```bash
python scripts/demo_webcam.py --picamera
```
Frames come in as grayscale at 640x400 by default. Use `--width`, `--height` and `--fps` to change that.
