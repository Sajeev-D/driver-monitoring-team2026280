# Eye Tracker + Rolling PERCLOS (ECE496 demo)

Integrates your original MediaPipe eye-feature tracker with `perclos.py`.
The webcam runs continuously; the PERCLOS engine recomputes using at most the
**last 60 seconds of timestamped samples**. A side panel shows the metrics and a
CSV records them once per second. No video or raw frames are automatically saved.

## Setup on Windows and macOS

Use **Python 3.12** and the same `requirements.txt` on both platforms.
Each teammate must create their own `.venv`; do not copy a virtual environment
between computers or operating systems.

The pinned [MediaPipe 0.10.32 release](https://pypi.org/project/mediapipe/0.10.32/#files)
has wheels for **Windows x64** and **Apple Silicon macOS 11+ (ARM64)**.
This setup does not cover Intel Macs or native Windows ARM64 Python.

### Windows (PowerShell)

Install **64-bit Python 3.12** from python.org, then open PowerShell in the
project directory (the directory containing `requirements.txt`):

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

These commands use the environment's Python directly, so activation and changes
to PowerShell's execution policy are unnecessary. Use that same Python for all
commands below. For example, after preparing your calibration photos:

```powershell
.\.venv\Scripts\python.exe scripts/demo_webcam_perclos.py --open_image calibration/open.jpg --closed_image calibration/closed.jpg
```

On later runs, reuse the saved calibration:

```powershell
.\.venv\Scripts\python.exe scripts/demo_webcam_perclos.py
```

Optional activation is `.\.venv\Scripts\Activate.ps1`; if your system blocks
it, keep using the direct Python commands above. In VS Code, choose
`.venv\Scripts\python.exe` via **Python: Select Interpreter**.

### macOS (Apple Silicon)

**1. Open Terminal at the root of this project** (the directory containing
`requirements.txt`). From the parent ECE496 workspace:

```bash
cd baseline_metrics/driver-monitoring-team2026280
```

Recommended: **native ARM64 Python 3.12**, not an Intel
Python under Rosetta. Confirm the terminal/Python architecture:

```bash
python3.12 --version
python3.12 -c "import platform; print(platform.machine())"
```

Expected architecture: `arm64`. If you do not have Python 3.12, install it via
Homebrew (`brew install python@3.12`), or download its ARM64 installer from
python.org. On Apple Silicon, Homebrew normally lives under `/opt/homebrew`.

**2. Create and activate a clean virtual environment:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The pinned MediaPipe 0.10.32 release includes a macOS ARM64 wheel; it uses the
`mp.tasks.vision.FaceLandmarker` API used by the code. The included
`models/face_landmarker.task` comes from the files you provided.

### Calibration and running (both platforms)

**3. Prepare TWO pictures of the SAME person:**

- `calibration/open.jpg`: clear, forward-facing image with eyes **fully open**.
- `calibration/closed.jpg`: image with eyes **fully or
  nearly closed**, rather than merely looking tired.

Keep camera angle, lighting, eyewear, and distance reasonably similar to the
webcam demo. Calibration is an EAR reference, **not a trained drowsiness
classifier**. Using a partially open-eyed image as `EAR_closed` makes the P80
threshold misleading. Create a `calibration` folder and add your own `open.jpg` and `closed.jpg`.
Photos and profiles are ignored by Git, so every fresh clone needs its own
calibration files. Run the command below to generate your profile.

Calibration requires two finite EAR values per state, with open EAR at least
`0.02` above closed EAR for each eye. For example, `0.30` open and `0.29` closed
fails this check. These numeric checks cannot confirm that the eyelids in the
closed photo are actually closed; inspect the photos yourself.

**4. Run your webcam and calibrate from the images:**

On macOS with the environment activated (Windows users use the direct
`.\.venv\Scripts\python.exe` command above):

```bash
python scripts/demo_webcam_perclos.py \
  --open_image calibration/open.jpg \
  --closed_image calibration/closed.jpg
```

The program detects eye landmarks in each still image using MediaPipe IMAGE
mode, saves the resulting **per-eye** open/closed EAR baselines to
`calibration/eye_calibration.json`, and launches your webcam. Future runs can
reuse the saved calibration with:

```bash
python scripts/demo_webcam_perclos.py
```

On macOS, for subsequent terminal sessions, activate the environment before running:

```bash
source .venv/bin/activate
python scripts/demo_webcam_perclos.py
```

You can also specify a saved profile using `--calibration some-file.json`.
Replacing the photos alone does not update a saved profile: rerun the command
with both image arguments to regenerate it.

**5. If the built-in camera does not open:**

- On Windows, enable camera access and **Let desktop apps access your camera**
  under **Settings → Privacy & security → Camera** (Windows 11), or
  **Settings → Privacy → Camera** (Windows 10). Close other camera apps.
  Replace `python` in the examples below with `.\.venv\Scripts\python.exe`
  when using PowerShell without activation.

- On macOS, give **Terminal** or **VS Code** camera permission under **System Settings →
  Privacy & Security → Camera** and restart the app.
- Try `python scripts/demo_webcam_perclos.py --camera 1` for an external webcam.
- Use `--video path/to/clip.mp4` to test without opening a camera (requires a
  calibration profile from images).

On macOS with VS Code, select the `.venv/bin/python` interpreter via **Python: Select
Interpreter**. If Python reports `x86_64` on your M1, switch to native ARM64
Python before recreating `.venv`.

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

### 3. Take calibration photos with the OV9281
Take the reference photos with the same camera and IR lighting you will run
with, so the EAR thresholds match what the demo sees:
```bash
mkdir -p calibration
rpicam-still -t 3000 -o calibration/open.jpg     # eyes fully open
rpicam-still -t 3000 -o calibration/closed.jpg   # eyelids fully closed
```

### 4. Run the demo
```bash
python scripts/demo_webcam_perclos.py --picamera --open_image calibration/open.jpg --closed_image calibration/closed.jpg
python scripts/demo_webcam_perclos.py --picamera   # later runs reuse the calibration
```
The original demo takes the same flag: `python scripts/demo_webcam.py --picamera`.
Frames come in as grayscale at 640x400 by default. Use `--width`, `--height` and `--fps` to change that.

## How PERCLOS is calculated

For each eye, the open and closed reference EARs define a personalized P80
threshold:

```text
P80 EAR threshold = open EAR - 0.80 × (open EAR - closed EAR)
Right PERCLOS = right-eye closed time / right-eye valid observed time
Left PERCLOS  = left-eye closed time / left-eye valid observed time
Mean PERCLOS (%) = 100 × (Right PERCLOS + Left PERCLOS) / 2
```

An eye is classified as closed when its estimated closure is at least 80%.
Closed time is accumulated over valid timestamp intervals, using the state at
the start of each interval. Missing endpoints and unusually long gaps are
excluded. Each eye has its own valid-time denominator; if either eye has no
valid observed time, the mean is unavailable.

For example, right-eye PERCLOS of 20% and left-eye PERCLOS of 10% gives a
mean of **15%**. One continuously closed eye and one continuously open eye
gives **50%**. The dashboard and CSV show this mean.

Blink counts and prolonged closures use a separate both-eye calculation:
both eyes must cross their own P80 thresholds together. The graph shows each
eye's personalized P80 threshold.

## What appears in the demo

| Metric | Meaning |
| --- | --- |
| PERCLOS P80 (%) | Arithmetic mean of right-eye and left-eye P80: each eye uses its own closed time / valid observed time and open/closed EAR references |
| Both-eye blink count | Number of completed 0.15–0.40 s P80 closure events in the rolling window |
| Average blink duration | Mean duration of completed both-eye closure events, in ms; `0 ms` means no completed blinks were measured |
| Longer closures (>400 ms) | Number of both-eye P80 closure events strictly longer than 0.40 s, including ongoing events |
| Valid data (%) | Fraction of the covered window with valid observations of both eyes |
| Face missing (%) | Proportion of captured frames without a detected face (separate from valid-time coverage) |
| Quality status | `WARMING_UP` before first minute; then `OK` if >=80% valid data, else `LOW_VALID_DATA` |
| EAR R/L | Current eye aspect ratio for each eye |
| P80 EAR R/L | Personalized eyelid closure thresholds for each eye |
| Head P/Y/R | Approximate MediaPipe head pose; not validated as distraction inference |
| FPS | Actual approximate processing frame rate |

`WARMING_UP` still shows **provisional** metrics calculated on the available
portion of the minute. At 60 seconds, the window becomes a true rolling minute
and older observations are dropped automatically. A blink spanning the exact
window boundary may only be partially visible. A closure already underway at
the start of the window can count as a blink if reopening is observed and its
visible duration is within the blink limits; its full duration may be longer.
Ongoing closures and closures ended by missing data or a gap are excluded from
completed blink counts. Quality status and valid/closed time describe the
both-eye calculation, rather than the individual-eye PERCLOS denominators. This is a prototype detector,
not a medically or operationally validated driver-alertness estimate.

The original PERCLOS implementation has been extended to accept both
`[N,6,2]` single-eye input and `[N,2,6,2]` two-eye input. It uses elapsed time,
not frame counts, for PERCLOS. The demo calculates PERCLOS independently for
each eye and averages the two fractions. Blink and prolonged-closure event counts still use both-eye mode,
which requires **both eyes** to cross their respective P80 thresholds.
CSV `perclos_raw` and `perclos_percent` contain the eye average; valid/closed
time and event fields describe the both-eye calculation. A missing face
is neither open nor closed. MediaPipe's per-frame tracking confidence is **not
exposed** by this extractor; the engine uses the `face_found` flag and finite
landmarks, rather than pretending an actual confidence measurement exists.

**Interpretation cautions:** The P80 metric *includes* ordinary blink closures
where they exceed P80; this matches the supplied initial `perclos.py`. Blink
duration here means duration of the P80 segment, not the complete motion of a
blink. The blink and prolonged-closure thresholds are provisional and require
validation against annotated videos before being treated as research results.
A closed-eye reference image is not ground truth for KSS or drowsiness severity.

## Files and outputs

```text
models/face_landmarker.task       MediaPipe model (included)
scripts/demo_webcam.py            Original eye-feature webcam demo (no calibration)
scripts/demo_webcam_perclos.py    PERCLOS demo + dashboard + CSV export
src/eye_features/extractor.py     Original landmark extractor
src/eye_features/buffer.py        Original rolling 60-second buffer
src/eye_features/landmarks.py     EAR and MediaPipe eye landmark indices
src/eye_features/perclos.py       Extended original PERCLOS engine
src/eye_features/calibration.py   Static-image EAR calibration, load/save JSON
src/eye_features/rolling_metrics.py  Buffer -> engine adapter
calibration/open.jpg             Your fully open-eye reference photo
calibration/closed.jpg           Your closed-eye reference photo
calibration/eye_calibration.json  Saved per-eye references; loaded automatically
recordings/metrics_*.csv         One row per second, created automatically
recordings/buffer_*.npz          Numeric landmarks, only when you press s
```

Press **q/Esc** to quit. Press **s** to save the numerical 60-second landmark
buffer to `.npz`. You can disable automatic CSV writing with `--no-log`, or
choose another file with `--log path/to/metrics.csv`.

**CSV fields:** elapsed time, observed window time, readiness/quality, PERCLOS
raw fraction and percent, blink count, average blink ms, prolonged closures,
valid data fraction, missing-face fraction, valid/closed time, current face detection status, both live
EARs, head pose, FPS, both open/closed reference EARs, and P80 thresholds.

## Tests

```bash
python -m pytest -q
```

`tests/test_perclos_integration.py` covers P80, both-eye handling, blinks,
prolonged closures, missing frames, low coverage, incomplete events, and
single-eye backwards compatibility. `tests/test_rolling_metrics.py` covers
calibration profiles, rolling-window readiness, tracking loss, and averaging
independent right/left PERCLOS values. `tests/test_eye_features.py` is preserved
from the original project. Testing the Mac webcam itself requires running the
demo locally with camera access; CI numerical tests do not substitute for
on-device testing or dataset validation.

## Original eye-feature demo

The original Git version is preserved in `scripts/demo_webcam.py`. It shows
landmarks, EAR, head pose, and the rolling buffer without PERCLOS calibration:

```bash
python scripts/demo_webcam.py
```

On Windows without activation, use
`.\.venv\Scripts\python.exe scripts/demo_webcam.py`.

## Sharing with teammates

Commit the source, tests, `requirements.txt`, README, and the included
`models/face_landmarker.task`. Each teammate installs dependencies locally and
creates their own calibration photos/profile. `.venv`, calibration images and
profiles, and generated recordings are ignored by Git.

The numerical tests can be run on Windows without activation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Camera permissions and webcam behavior must still be checked on each machine.

## Duration categories

Completed both-eye P80 segments from **150 through 400 ms**, inclusive, count
as candidate normal blinks. Segments **strictly over 400 ms** count as
longer-than-usual closures, including ongoing events once they exceed the
boundary. Segments below 150 ms remain in event records and contribute to
PERCLOS, but are not counted in either category.

The 150–400 ms range is discussed in
[Nyström et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11133197/),
citing Stern et al. (1984), for blink duration. Our implementation measures
only the P80 closure segment, not the full closing/opening movement. Applying
this range to P80 segments is a configurable prototype choice requiring
validation; longer closures are not confirmed microsleeps.

For compatibility, the CSV field `prolonged_closure_count_60s` and internal
`prolonged_closure_count` name are retained, but now count segments >400 ms.
Older CSVs generated with the previous defaults used a >=1 s threshold.
The saved `demo_webcam_previous.py` preserves the old layout but imports the
current engine and therefore uses the current duration thresholds.
