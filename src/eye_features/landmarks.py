"""Eye landmark indices and the eye aspect ratio (EAR).

Indices are from MediaPipe's 478-point face mesh. "Right" and "left" mean
the driver's own right and left eye (so the right eye is on the left side
of an unmirrored camera image).

Order for each eye is p1..p6 as used in the EAR formula:
    p1, p4 = eye corners
    p2, p3 = upper lid
    p6, p5 = lower lid (p2 sits above p6, p3 sits above p5)
"""

import numpy as np

RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Index 0 = right eye, index 1 = left eye, everywhere in this package.
EYES = (RIGHT_EYE, LEFT_EYE)


def eye_aspect_ratio(eye):
    """EAR for one eye.

    eye: array of shape [6, 2] in pixels, ordered p1..p6.
    Returns a float, or NaN if the eye is missing (NaN input) or has zero width.
    """
    eye = np.asarray(eye, dtype=np.float64)

    if np.isnan(eye).any():
        return float("nan")

    p1, p2, p3, p4, p5, p6 = eye
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return float("nan")

    return float(vertical / (2.0 * horizontal))
