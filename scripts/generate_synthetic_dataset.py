"""Generate a reproducible, educational driver-monitoring CSV dataset."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "driver_monitoring_synthetic.csv"
)

FIELDNAMES = [
    "participant_id",
    "session_order",
    "start_time",
    "sleep_duration",
    "wake_time",
    "elapsed_time_awake",
    "baseline_kss",
    "kss_rating",
    "eye_openness",
    "perclos",
    "blink_rate_per_min",
    "avg_blink_duration_ms",
    "blink_duration_variability_ms",
    "yawning_rate_per_min",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "observer_rating",
    "camera_frame_timestamp",
    "dropped_frame_indicator",
    "video_quality_score",
    "drowsiness_label",
]


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def rounded(value: float, places: int = 4) -> float:
    return round(value, places)


def generate_rows(
    participants: int,
    sessions: int,
    minutes_per_session: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    first_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for participant_index in range(1, participants + 1):
        participant_id = f"P{participant_index:03d}"
        baseline_kss = rng.randint(2, 5)
        participant_sensitivity = rng.uniform(-0.07, 0.07)

        for session_order in range(1, sessions + 1):
            session_day = (participant_index - 1) * sessions + session_order - 1
            start_time = first_date + timedelta(
                days=session_day,
                hours=rng.randint(8, 15),
                minutes=rng.choice([0, 15, 30, 45]),
            )
            initial_awake_minutes = rng.randint(30, 240)
            wake_time = start_time - timedelta(minutes=initial_awake_minutes)
            sleep_duration = rounded(rng.uniform(4.5, 9.0), 2)
            session_effect = rng.uniform(-0.04, 0.04)

            for minute in range(minutes_per_session):
                elapsed_awake_minutes = initial_awake_minutes + minute
                sleep_effect = max(0.0, 7.5 - sleep_duration) * 0.035
                fatigue = clamp(
                    0.12
                    + 0.0055 * minute
                    + sleep_effect
                    + participant_sensitivity
                    + session_effect
                    + rng.gauss(0, 0.035),
                    0.02,
                    0.98,
                )
                drowsiness_label = int(fatigue >= 0.62)

                camera_time = start_time + timedelta(minutes=minute)
                video_quality = clamp(rng.gauss(88, 9), 20, 100)
                if rng.random() < 0.04:
                    video_quality = clamp(rng.gauss(42, 12), 10, 65)

                dropped_frame = int(
                    rng.random() < (0.015 + max(0, 55 - video_quality) * 0.001)
                )
                measurement_noise = 1.0 + (100 - video_quality) / 70

                eye_openness = clamp(
                    0.94 - 0.48 * fatigue + rng.gauss(0, 0.035 * measurement_noise),
                    0.05,
                    1.0,
                )
                perclos = clamp(
                    0.025
                    + 0.39 * fatigue
                    + rng.gauss(0, 0.025 * measurement_noise),
                    0.0,
                    0.8,
                )
                blink_rate = clamp(
                    24 - 8 * fatigue + rng.gauss(0, 2.5 * measurement_noise),
                    2,
                    40,
                )
                blink_duration = clamp(
                    165
                    + 265 * fatigue
                    + rng.gauss(0, 32 * measurement_noise),
                    80,
                    800,
                )
                blink_variability = clamp(
                    280
                    + 900 * fatigue
                    + rng.gauss(0, 130 * measurement_noise),
                    20,
                    3000,
                )
                yawning_rate = clamp(
                    0.015
                    + 1.4 * fatigue**2
                    + rng.gauss(0, 0.16 * measurement_noise),
                    0,
                    4,
                )
                head_pitch = clamp(
                    1.0 + 9.0 * fatigue + rng.gauss(0, 3.0 * measurement_noise),
                    -20,
                    40,
                )
                head_yaw = clamp(rng.gauss(0, 5.0 * measurement_noise), -35, 35)
                head_roll = clamp(rng.gauss(0, 3.0 * measurement_noise), -25, 25)

                kss_rating = int(
                    clamp(round(1 + 8 * fatigue + rng.gauss(0, 1.15)), 1, 9)
                )
                observer_rating = int(
                    clamp(round(1 + 4 * fatigue + rng.gauss(0, 0.55)), 1, 5)
                )

                # Poor video can make individual sensor-derived measurements unavailable.
                if video_quality < 30 and rng.random() < 0.25:
                    eye_value: float | str = ""
                else:
                    eye_value = rounded(eye_openness)
                if dropped_frame and rng.random() < 0.2:
                    perclos_value: float | str = ""
                else:
                    perclos_value = rounded(perclos)

                rows.append(
                    {
                        "participant_id": participant_id,
                        "session_order": session_order,
                        "start_time": start_time.isoformat(),
                        "sleep_duration": sleep_duration,
                        "wake_time": wake_time.isoformat(),
                        "elapsed_time_awake": rounded(elapsed_awake_minutes / 60, 2),
                        "baseline_kss": baseline_kss,
                        "kss_rating": kss_rating,
                        "eye_openness": eye_value,
                        "perclos": perclos_value,
                        "blink_rate_per_min": rounded(blink_rate, 2),
                        "avg_blink_duration_ms": rounded(blink_duration, 2),
                        "blink_duration_variability_ms": rounded(blink_variability, 2),
                        "yawning_rate_per_min": rounded(yawning_rate, 3),
                        "head_pitch_deg": rounded(head_pitch, 2),
                        "head_yaw_deg": rounded(head_yaw, 2),
                        "head_roll_deg": rounded(head_roll, 2),
                        "observer_rating": observer_rating,
                        "camera_frame_timestamp": camera_time.isoformat(),
                        "dropped_frame_indicator": dropped_frame,
                        "video_quality_score": rounded(video_quality, 1),
                        "drowsiness_label": drowsiness_label,
                    }
                )

    return rows


def split_by_participant(
    rows: list[dict[str, object]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    participant_ids = sorted({str(row["participant_id"]) for row in rows})
    if len(participant_ids) < 2:
        raise ValueError("At least two participants are required for a train/test split.")

    rng = random.Random(seed)
    rng.shuffle(participant_ids)
    train_count = min(
        len(participant_ids) - 1,
        max(1, int(len(participant_ids) * train_fraction)),
    )
    train_participants = set(participant_ids[:train_count])
    train_rows = [row for row in rows if row["participant_id"] in train_participants]
    test_rows = [row for row in rows if row["participant_id"] not in train_participants]
    return train_rows, test_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CSV for driver-monitoring model experiments."
    )
    parser.add_argument("--participants", type=int, default=30)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--minutes-per-session", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name in ("participants", "sessions", "minutes_per_session"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.participants < 2:
        parser.error("--participants must be at least 2 to create train/test splits")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be greater than 0 and less than 1")
    return args


def main() -> None:
    args = parse_args()
    rows = generate_rows(
        participants=args.participants,
        sessions=args.sessions,
        minutes_per_session=args.minutes_per_session,
        seed=args.seed,
    )
    train_rows, test_rows = split_by_participant(
        rows,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    write_csv(args.output, rows)
    train_path = args.output.with_name(f"{args.output.stem}_train{args.output.suffix}")
    test_path = args.output.with_name(f"{args.output.stem}_test{args.output.suffix}")
    write_csv(train_path, train_rows)
    write_csv(test_path, test_rows)
    print(f"Wrote {len(rows)} synthetic rows to {args.output}")
    print(f"Wrote {len(train_rows)} training rows to {train_path}")
    print(f"Wrote {len(test_rows)} test rows to {test_path}")


if __name__ == "__main__":
    main()
