from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class PERCLOSConfig:
    # Ignore frames below this tracking confidence
    min_tracking_confidence: float = 0.5

    # P80 definition: eye is considered closed when estimated
    # eyelid closure is >= 80%
    closure_threshold: float = 0.80

    # Candidate blink duration range
    # These should eventually be validated/tuned on your dataset
    blink_min_s: float = 0.15
    blink_max_s: float = 0.40

    # Strict lower boundary for longer-than-usual P80 closure segments
    prolonged_closure_min_s: float = 0.40

    # Minimum amount of valid data required for a reliable result
    min_valid_fraction: float = 0.80

    # If frames are separated by much more than the expected
    # frame period, treat it as a break in tracking
    max_gap_factor: float = 2.5


# ============================================================
# EAR Calculation
# ============================================================

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Input:
        eye_landmarks: numpy array of shape [6, 2]

    Landmark ordering:
        p1, p4 = eye corners
        p2, p3 = upper eyelid
        p5, p6 = lower eyelid

    Returns:
        EAR value (float)
    """

    eye = np.asarray(eye_landmarks, dtype=float)

    if eye.shape != (6, 2):
        raise ValueError(
            f"Expected eye landmarks with shape [6, 2], got {eye.shape}"
        )

    # Invalid / missing landmark
    if not np.all(np.isfinite(eye)):
        return np.nan

    p1, p2, p3, p4, p5, p6 = eye

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    # Prevent division by zero
    if horizontal <= 1e-12:
        return np.nan

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return float(ear)


def calculate_ear_series(landmark_buffer):
    """
    Calculate EAR for every frame.

    Input:
        landmark_buffer: [N, 6, 2]

    Output:
        ear: [N]
    """

    return np.array(
        [calculate_ear(frame) for frame in landmark_buffer],
        dtype=float
    )


# ============================================================
# EAR -> Estimated Eyelid Closure
# ============================================================

def ear_to_closure_fraction(ear, ear_open, ear_closed):
    """
    Normalize EAR into approximate eyelid closure.

        0.0 = fully open
        1.0 = fully closed

    Example:
        EAR_open   = 0.30
        EAR_closed = 0.10

        EAR = 0.30 -> closure ~= 0%
        EAR = 0.20 -> closure ~= 50%
        EAR = 0.14 -> closure ~= 80%
        EAR = 0.10 -> closure ~= 100%
    """

    if ear_open <= ear_closed:
        raise ValueError(
            "ear_open must be greater than ear_closed."
        )

    ear = np.asarray(ear, dtype=float)

    closure = (
        (ear_open - ear)
        / (ear_open - ear_closed)
    )

    # Keep result between 0 and 1
    return np.clip(closure, 0.0, 1.0)


# ============================================================
# Closure Event Detection
# ============================================================

def detect_closure_events(
    closed_mask,
    valid_mask,
    timestamps,
    nominal_frame_period,
    max_gap_s
):
    """
    Groups consecutive closed-eye frames into individual events.

    Invalid frames or large timestamp gaps break an event.
    """

    events = []

    start_index = None
    previous_index = None

    def add_event(start, end, completed):
        duration = (
            timestamps[end]
            - timestamps[start]
            + nominal_frame_period
        )

        events.append({
            "start_index": int(start),
            "end_index": int(end),

            "start_time_s": float(timestamps[start]),
            "end_time_s": float(timestamps[end]),

            "duration_s": float(duration),
            "completed": bool(completed)
        })

    for i in range(len(closed_mask)):

        closed_and_valid = (
            valid_mask[i]
            and closed_mask[i]
        )

        # Check if there is an abnormally large timestamp gap
        gap_detected = (
            previous_index is not None
            and
            timestamps[i] - timestamps[previous_index]
            > max_gap_s
        )

        # Break current event if tracking gap occurs
        if gap_detected and start_index is not None:

            add_event(
                start_index,
                previous_index,
                False  # an observation gap cannot confirm a completed blink
            )

            start_index = None

        # Eye is closed
        if closed_and_valid:

            if start_index is None:
                start_index = i

            previous_index = i

        # Eye opened OR frame became invalid
        else:

            if (
                start_index is not None
                and previous_index is not None
            ):

                add_event(
                    start_index,
                    previous_index,
                    bool(valid_mask[i])  # observed eye reopening confirms end
                )

            start_index = None
            previous_index = None

    # Handle event continuing through final frame
    if (
        start_index is not None
        and previous_index is not None
    ):

        add_event(
            start_index,
            previous_index,
            False  # ongoing event at window boundary
        )

    return events


# ============================================================
# Main PERCLOS Engine
# ============================================================

def calculate_perclos(
    landmarks,
    timestamps,
    ear_open,
    ear_closed,
    tracking_confidence=None,
    face_lost=None,
    fps=None,
    config=None
):
    """
    Main PERCLOS calculation engine.

    ----------------------------------------------------------
    INPUT
    ----------------------------------------------------------

    landmarks:
        [N, 6, 2] for one eye, or [N, 2, 6, 2] for both eyes.
        Both-eye mode requires both eyes to cross P80 for a closed frame.

    timestamps:
        [N]
        Timestamp corresponding to each frame.

    ear_open:
        Driver's calibrated open-eye EAR (scalar or two per-eye values).

    ear_closed:
        Driver's calibrated fully-closed EAR (scalar or two per-eye values).

    tracking_confidence:
        [N]
        MediaPipe tracking confidence.
        Optional.

    face_lost:
        [N]
        True if the face/eye was not reliably detected.
        Optional.

    fps:
        Camera frame rate, e.g. 30 or 60.

    ----------------------------------------------------------
    OUTPUT
    ----------------------------------------------------------

    Dictionary containing:

        perclos_raw
        blink_count
        average_blink_duration_ms
        prolonged_closure_count
        valid_data_fraction
        quality_flag

    plus intermediate data useful for debugging.
    """

    if config is None:
        config = PERCLOSConfig()

    landmarks = np.asarray(
        landmarks,
        dtype=float
    )

    timestamps = np.asarray(
        timestamps,
        dtype=float
    )

    # --------------------------------------------------------
    # Validate input dimensions
    # --------------------------------------------------------

    if landmarks.ndim == 3 and landmarks.shape[1:] == (6, 2):
        eye_count = 1
    elif landmarks.ndim == 4 and landmarks.shape[1:] == (2, 6, 2):
        eye_count = 2
    else:
        raise ValueError("landmarks must have shape [N, 6, 2] or [N, 2, 6, 2]")

    ear_open_values = np.asarray(ear_open, dtype=float).reshape(-1)
    ear_closed_values = np.asarray(ear_closed, dtype=float).reshape(-1)
    if ear_open_values.size == 1:
        ear_open_values = np.repeat(ear_open_values, eye_count)
    if ear_closed_values.size == 1:
        ear_closed_values = np.repeat(ear_closed_values, eye_count)
    if (ear_open_values.size != eye_count or ear_closed_values.size != eye_count
            or not np.all(np.isfinite(ear_open_values))
            or not np.all(np.isfinite(ear_closed_values))
            or np.any(ear_open_values <= ear_closed_values)):
        raise ValueError("Each eye requires finite ear_open > ear_closed")

    N = landmarks.shape[0]

    if timestamps.shape != (N,):
        raise ValueError(
            f"timestamps must have shape [{N}]"
        )

    if N < 2:
        raise ValueError(
            "At least two frames are required."
        )

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "timestamps must be strictly increasing."
        )

    # --------------------------------------------------------
    # Handle optional Apurva status flags
    # --------------------------------------------------------

    if tracking_confidence is None:

        tracking_confidence = np.ones(
            N,
            dtype=float
        )

    else:

        tracking_confidence = np.asarray(
            tracking_confidence,
            dtype=float
        )

    if face_lost is None:

        face_lost = np.zeros(
            N,
            dtype=bool
        )

    else:

        face_lost = np.asarray(
            face_lost,
            dtype=bool
        )

    if tracking_confidence.shape != (N,):
        raise ValueError(
            "tracking_confidence must have shape [N]"
        )

    if face_lost.shape != (N,):
        raise ValueError(
            "face_lost must have shape [N]"
        )

    # --------------------------------------------------------
    # STEP 1:
    # Calculate EAR for every frame
    # --------------------------------------------------------

    per_eye_ear = (
        calculate_ear_series(landmarks)[:, None]
        if eye_count == 1 else
        np.column_stack([calculate_ear_series(landmarks[:, eye])
                         for eye in range(eye_count)])
    )
    # Keep scalar series for existing debug consumers and retain individual EARs.
    ear = (per_eye_ear[:, 0] if eye_count == 1 else
           np.mean(per_eye_ear, axis=1))

    # --------------------------------------------------------
    # STEP 2:
    # Determine which frames are actually trustworthy
    # --------------------------------------------------------

    valid_mask = (
        np.all(np.isfinite(per_eye_ear), axis=1)
        &
        np.isfinite(tracking_confidence)
        &
        (
            tracking_confidence
            >= config.min_tracking_confidence
        )
        &
        (~face_lost)
    )

    # --------------------------------------------------------
    # STEP 3:
    # Convert EAR -> closure percentage
    # --------------------------------------------------------

    per_eye_closure_fraction = np.column_stack([
        ear_to_closure_fraction(per_eye_ear[:, eye],
                                ear_open_values[eye], ear_closed_values[eye])
        for eye in range(eye_count)
    ])
    # Conservative both-eye definition: one open eye is not both eyes closed.
    closure_fraction = np.min(per_eye_closure_fraction, axis=1)

    # --------------------------------------------------------
    # STEP 4:
    # P80 classification
    #
    # True = >= 80% closure
    # --------------------------------------------------------

    closed_mask = (
        valid_mask
        &
        (
            closure_fraction
            >= config.closure_threshold
        )
    )

    # Personalized EAR corresponding to P80
    p80_ear_threshold = (ear_open_values - config.closure_threshold *
                         (ear_open_values - ear_closed_values))

    # --------------------------------------------------------
    # Determine expected frame spacing
    # --------------------------------------------------------

    if fps is not None:

        if fps <= 0:
            raise ValueError(
                "fps must be greater than zero."
            )

        nominal_frame_period = 1.0 / fps

    else:

        nominal_frame_period = float(
            np.median(
                np.diff(timestamps)
            )
        )

    max_gap_s = (
        config.max_gap_factor
        *
        nominal_frame_period
    )

    # --------------------------------------------------------
    # STEP 5:
    # Calculate PERCLOS USING TIME
    #
    # Important:
    # We don't simply do:
    #
    # closed_frames / total_frames
    #
    # because dropped frames could distort the result.
    # --------------------------------------------------------

    frame_intervals = np.diff(
        timestamps
    )

    # Only use an interval if both surrounding frames
    # contain valid eye tracking.
    valid_intervals = (
        valid_mask[:-1]
        &
        valid_mask[1:]
        &
        (
            frame_intervals
            <= max_gap_s
        )
    )

    closed_intervals = (
        valid_intervals
        &
        closed_mask[:-1]
    )

    valid_time_s = float(
        np.sum(
            frame_intervals[
                valid_intervals
            ]
        )
    )

    closed_time_s = float(
        np.sum(
            frame_intervals[
                closed_intervals
            ]
        )
    )

    if valid_time_s > 0:

        perclos_raw = (
            closed_time_s
            /
            valid_time_s
        )

    else:

        perclos_raw = np.nan

    # --------------------------------------------------------
    # STEP 6:
    # Find individual closure events
    # --------------------------------------------------------

    closure_events = (
        detect_closure_events(
            closed_mask,
            valid_mask,
            timestamps,
            nominal_frame_period,
            max_gap_s
        )
    )

    # --------------------------------------------------------
    # STEP 7:
    # Classify events
    # --------------------------------------------------------

    blink_events = [

        event

        for event in closure_events

        if (
            event["completed"]
            and config.blink_min_s
            <= round(event["duration_s"], 9)
            <= config.blink_max_s
        )
    ]

    prolonged_closure_events = [

        event

        for event in closure_events

        if (
            round(event["duration_s"], 9)
            > config.prolonged_closure_min_s
        )
    ]

    # --------------------------------------------------------
    # Blink statistics
    # --------------------------------------------------------

    blink_count = len(
        blink_events
    )

    if blink_count > 0:

        average_blink_duration_ms = (
            np.mean([
                event["duration_s"]
                for event in blink_events
            ])
            * 1000
        )

    else:

        average_blink_duration_ms = 0.0

    # --------------------------------------------------------
    # STEP 8:
    # Determine how much usable data we actually had
    # --------------------------------------------------------

    total_window_time = (
        timestamps[-1]
        -
        timestamps[0]
    )

    if total_window_time > 0:

        valid_data_fraction = (
            valid_time_s
            /
            total_window_time
        )

    else:

        valid_data_fraction = 0.0

    valid_data_fraction = float(
        np.clip(
            valid_data_fraction,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Quality flag
    # --------------------------------------------------------

    if (
        valid_data_fraction
        >= config.min_valid_fraction
    ):

        quality_flag = "OK"

    else:

        quality_flag = "LOW_VALID_DATA"

    # --------------------------------------------------------
    # Return output
    # --------------------------------------------------------

    return {

        # ----- Main outputs -----

        "perclos_raw":
            float(perclos_raw),

        "perclos_percent":
            float(perclos_raw * 100),

        "blink_count":
            blink_count,

        "average_blink_duration_ms":
            float(average_blink_duration_ms),

        "prolonged_closure_count":
            len(prolonged_closure_events),

        "valid_data_fraction":
            valid_data_fraction,

        "quality_flag":
            quality_flag,

        # ----- Useful metadata -----

        "valid_time_s":
            valid_time_s,

        "closed_time_s":
            closed_time_s,

        "ear_open": (float(ear_open_values[0]) if eye_count == 1 else
                     ear_open_values.tolist()),

        "ear_closed": (float(ear_closed_values[0]) if eye_count == 1 else
                       ear_closed_values.tolist()),

        "p80_ear_threshold": (float(p80_ear_threshold[0]) if eye_count == 1 else
                              p80_ear_threshold.tolist()),

        # ----- Debug / downstream data -----

        "ear":
            ear,

        "per_eye_ear": per_eye_ear,
        "per_eye_closure_fraction": per_eye_closure_fraction,

        "closure_fraction":
            closure_fraction,

        "valid_mask":
            valid_mask,

        "closed_mask":
            closed_mask,

        "closure_events":
            closure_events,

        "blink_events":
            blink_events,

        "prolonged_closure_events":
            prolonged_closure_events,

        "config":
            asdict(config)
    }