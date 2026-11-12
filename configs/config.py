from pathlib import Path

FEATURE_COLUMNS: list[str] = [
    "eye_openness",
    "perclos",
    "blink_rate_per_min",
    "avg_blink_duration_ms",
    "blink_duration_variability_ms",
    "yawning_rate_per_min",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
]

LABEL_COLUMN: str = "drowsiness_label"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "model.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)