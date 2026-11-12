"""Adapter connecting the eye-feature time buffer to the PERCLOS engine."""

from .perclos import calculate_perclos, PERCLOSConfig


def compute_rolling_metrics(buffer, calibration, config=None):
    """Return current window metrics or None until >=2 samples exist.

    Missing-face samples are never classified as either open or closed.
    PERCLOS is the arithmetic mean of independently calculated eye values.
    Blink and prolonged-closure events still require both eyes closed together.
    """
    if len(buffer) < 2:
        return None
    arrays = buffer.as_arrays()
    result = calculate_perclos(
        landmarks=arrays["landmarks"],
        timestamps=arrays["timestamps_ms"].astype(float) / 1000.0,
        ear_open=calibration.ear_open,
        ear_closed=calibration.ear_closed,
        face_lost=~arrays["face_found"],
        config=config or PERCLOSConfig(),
    )
    eyes = {}
    for index, name in enumerate(("right", "left")):
        eyes[name] = calculate_perclos(
            landmarks=arrays["landmarks"][:, index],
            timestamps=arrays["timestamps_ms"].astype(float) / 1000.0,
            ear_open=calibration.ear_open[index],
            ear_closed=calibration.ear_closed[index],
            face_lost=~arrays["face_found"],
            config=config or PERCLOSConfig(),
        )
    result["eyes"] = eyes
    result["perclos_raw"] = sum(eye["perclos_raw"] for eye in eyes.values()) / 2
    result["perclos_percent"] = result["perclos_raw"] * 100
    result["window_span_s"] = buffer.span_s()
    # The engine's OK means data quality; a complete minute is a separate issue.
    result["window_ready"] = buffer.span_s() >= 59.5
    result["display_quality"] = (
        "WARMING_UP" if not result["window_ready"] else result["quality_flag"]
    )
    return result
