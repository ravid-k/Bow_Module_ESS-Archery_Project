import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# ESS-ARCHERY VISION V0.5
# ArUco registration + physical shaft detector
#
# Tested design:
#   REFERENCE image (no current arrow)
#       -> detect ArUco IDs 0,1,2,3
#   CURRENT image
#       -> detect >= 3 of those markers
#       -> homography into REFERENCE frame
#       -> target-only dark-change mask
#       -> find dark, thick, continuous line candidates
#       -> VALID ARROW must physically cross target boundary:
#            one part near/outside target
#            one part inside target
#
# For the Tuesday/demo path:
#   - 4 markers = preferred
#   - 3 markers = allowed, flagged LIMITED
#   - <3 markers = RETAKE
#
# V0.4.3 detector is frozen.
# V0.5 adds target-plane X/Y coordinates, radial error and auto-ring.
# ============================================================

REQUIRED_IDS = {0, 1, 2, 3}
PROCESS_MAX_DIM = 1800

YELLOW_LOW  = np.array([18, 70, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([42, 255, 255], dtype=np.uint8)

# Final gates learned from the new ArUco dataset.
MIN_FINAL_SCORE = 0.65
MIN_CHANGE_SUPPORT = 0.65
MIN_MEDIAN_DARKENING = 70.0
MIN_LINE_CONTRAST = 0.75

MIN_OUTER_RADIUS = 0.90
MAX_INNER_RADIUS = 0.96
MIN_RADIAL_SPAN = 0.06

MIN_SHAFT_WIDTH = 4.0
MAX_SHAFT_WIDTH = 70.0
MIN_SHAFT_CONTINUITY = 0.30


# ============================================================
# BASIC IMAGE / ArUco
# ============================================================

def resize_for_processing(img, max_dim=PROCESS_MAX_DIM):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))

    if scale < 1.0:
        out = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA
        )
        return out, scale

    return img.copy(), 1.0


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    # OpenCV 4.x / 5.x compatible.
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return dictionary, detector
    except Exception:
        return dictionary, None


ARUCO_DICT, ARUCO_DETECTOR = make_aruco_detector()


def detect_aruco_full_resolution(img):
    if ARUCO_DETECTOR is not None:
        corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(img)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img,
            ARUCO_DICT
        )

    found = {}

    if ids is None:
        return found

    for c, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)

        if marker_id in REQUIRED_IDS:
            found[marker_id] = c.reshape(4, 2).astype(np.float32)

    return found


def prepare_image(path):
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # IMPORTANT:
    # Detect markers BEFORE resizing. Small/angled markers survive much
    # better at the original camera resolution.
    markers_full = detect_aruco_full_resolution(img)

    proc, scale = resize_for_processing(img)

    markers_proc = {
        marker_id: corners * scale
        for marker_id, corners in markers_full.items()
    }

    return {
        "original": img,
        "proc": proc,
        "scale": scale,
        "markers": markers_proc,
        "marker_ids": sorted(markers_proc.keys()),
    }


def aruco_register(reference, current, min_markers=3):
    common = sorted(
        set(reference["markers"].keys())
        & set(current["markers"].keys())
    )

    if len(common) < min_markers:
        return None, {
            "marker_count": len(common),
            "marker_ids": common,
            "status": "RETAKE",
        }

    src = []
    dst = []

    # Use every corner from every common marker.
    # 3 markers -> 12 points; 4 markers -> 16 points.
    for marker_id in common:
        src.extend(current["markers"][marker_id])
        dst.extend(reference["markers"][marker_id])

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)

    H, inlier_mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        4.0
    )

    if H is None:
        return None, {
            "marker_count": len(common),
            "marker_ids": common,
            "status": "HOMOGRAPHY_FAIL",
        }

    ref_img = reference["proc"]

    aligned = cv2.warpPerspective(
        current["proc"],
        H,
        (ref_img.shape[1], ref_img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245)
    )

    inliers = (
        int(np.sum(inlier_mask))
        if inlier_mask is not None
        else 0
    )

    return aligned, {
        "marker_count": len(common),
        "marker_ids": common,
        "status": "OK" if len(common) == 4 else "LIMITED",
        "homography": H,
        "inliers": inliers,
        "point_count": len(src),
    }



# ============================================================
# CURRENT-FRAME REGISTRATION
# ============================================================

def aruco_register_reference_to_current(
    reference,
    current,
    min_markers=3
):
    """
    IMPORTANT V0.4.1 CHANGE:

    Do NOT warp the current image / arrow into the reference plane.
    The target is planar, but the arrow is a 3-D object protruding
    from that plane. Planar homography can therefore distort the shaft.

    Instead:
      - keep CURRENT image untouched
      - warp the clean REFERENCE target into CURRENT camera view

    This keeps the real arrow physically straight.
    """
    common = sorted(
        set(reference["markers"].keys())
        & set(current["markers"].keys())
    )

    if len(common) < min_markers:
        return None, {
            "marker_count": len(common),
            "marker_ids": common,
            "status": "RETAKE",
        }

    src = []
    dst = []

    for marker_id in common:
        src.extend(
            reference["markers"][marker_id]
        )

        dst.extend(
            current["markers"][marker_id]
        )

    src = np.asarray(
        src,
        dtype=np.float32
    )

    dst = np.asarray(
        dst,
        dtype=np.float32
    )

    H_ref_to_current, inlier_mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        4.0
    )

    if H_ref_to_current is None:
        return None, {
            "marker_count": len(common),
            "marker_ids": common,
            "status": "HOMOGRAPHY_FAIL",
        }

    current_img = current["proc"]

    reference_warped = cv2.warpPerspective(
        reference["proc"],
        H_ref_to_current,
        (
            current_img.shape[1],
            current_img.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245)
    )

    H_current_to_ref = np.linalg.inv(
        H_ref_to_current
    )

    inliers = (
        int(np.sum(inlier_mask))
        if inlier_mask is not None
        else 0
    )

    return reference_warped, {
        "marker_count": len(common),
        "marker_ids": common,
        "status": (
            "OK"
            if len(common) == 4
            else "LIMITED"
        ),
        "H_ref_to_current": H_ref_to_current,
        "H_current_to_ref": H_current_to_ref,
        "inliers": inliers,
        "point_count": len(src),
    }


def transform_point_homography(H, point):
    x, y = point

    q = np.array(
        [x, y, 1.0],
        dtype=np.float64
    )

    z = H @ q

    if abs(z[2]) < 1e-9:
        return (
            float(x),
            float(y)
        )

    return (
        float(z[0] / z[2]),
        float(z[1] / z[2])
    )


def warp_reference_mask_to_current(
    reference_mask,
    H_ref_to_current,
    current_shape
):
    return cv2.warpPerspective(
        reference_mask,
        H_ref_to_current,
        (
            current_shape[1],
            current_shape[0]
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


# ============================================================
# TARGET REFERENCE
# ============================================================

def make_yellow_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        YELLOW_LOW,
        YELLOW_HIGH
    )

    h, w = img_bgr.shape[:2]

    k = max(5, int(min(h, w) * 0.006))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask


def detect_yellow_ellipse(img_bgr):
    mask = make_yellow_mask(img_bgr)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = img_bgr.shape[0] * img_bgr.shape[1]

    best = None

    for c in contours:
        area = cv2.contourArea(c)

        if area < image_area * 0.002:
            continue

        if len(c) < 5:
            continue

        ellipse = cv2.fitEllipse(c)

        (cx, cy), (a, b), angle = ellipse

        if a <= 5 or b <= 5:
            continue

        ellipse_area = math.pi * a * b / 4.0
        fill = area / max(ellipse_area, 1.0)

        major = max(a, b)
        minor = min(a, b)
        ratio = minor / major

        score = (
            0.55 * min(area / (image_area * 0.03), 1.0)
            + 0.25 * min(fill, 1.0)
            + 0.20 * ratio
        )

        if best is None or score > best["score"]:
            best = {
                "ellipse": ellipse,
                "score": score,
                "axis_ratio": ratio,
            }

    return best


def make_target_mask(shape, yellow_ellipse, scale_factor=5.12):
    h, w = shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    center, axes, angle = yellow_ellipse

    full_axes = (
        max(1, int(round(axes[0] * scale_factor))),
        max(1, int(round(axes[1] * scale_factor))),
    )

    cv2.ellipse(
        mask,
        (
            (
                int(round(center[0])),
                int(round(center[1]))
            ),
            full_axes,
            angle
        ),
        255,
        -1
    )

    return mask


def normalized_radius(point, yellow_ellipse):
    """
    Radius normalized to estimated OUTER target radius.
    r ~= 1.0 means outer target boundary.
    """
    x, y = point

    (cx, cy), (yellow_a, yellow_b), angle = yellow_ellipse

    full_a = yellow_a * 5.0
    full_b = yellow_b * 5.0

    a = full_a / 2.0
    b = full_b / 2.0

    theta = math.radians(angle)

    ct = math.cos(theta)
    st = math.sin(theta)

    dx = x - cx
    dy = y - cy

    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy

    return math.sqrt(
        (xr / max(a, 1e-6)) ** 2
        + (yr / max(b, 1e-6)) ** 2
    )


# ============================================================
# V0.5 TARGET-PLANE COORDINATES / SCORING
# ============================================================

def target_plane_coordinates(
    point_reference,
    yellow_ellipse,
    target_diameter_cm=60.0
):
    """
    Convert a detected impact point in REFERENCE image coordinates
    into target-plane X/Y coordinates.

    Coordinate convention:
        +X = right
        -X = left
        +Y = up
        -Y = down

    The yellow ellipse is used as the geometric calibration reference.
    On a standard 10-ring target face, the yellow zone occupies 1/5 of
    the full target radius, so the full target ellipse uses 5x the
    yellow diameter (same geometry already used by normalized_radius()).

    Perspective anisotropy in the reference image is compensated by
    normalizing in the ellipse principal axes and rotating the normalized
    vector back into the image/reference target axes. This preserves the
    intuitive right/up coordinate convention while keeping radial distance
    consistent with normalized_radius().
    """
    x, y = map(float, point_reference)

    (cx, cy), (yellow_a, yellow_b), angle = yellow_ellipse

    full_a = float(yellow_a) * 5.0
    full_b = float(yellow_b) * 5.0

    a = max(full_a / 2.0, 1e-6)
    b = max(full_b / 2.0, 1e-6)

    theta = math.radians(float(angle))
    ct = math.cos(theta)
    st = math.sin(theta)

    dx = x - float(cx)
    dy = y - float(cy)

    # Reference-image displacement -> ellipse principal frame.
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy

    nx_local = xr / a
    ny_local = yr / b

    # Undo ellipse rotation after normalization so X/Y retain the
    # reference target's intuitive right/down image directions.
    x_norm = ct * nx_local - st * ny_local
    y_down_norm = st * nx_local + ct * ny_local

    # Archery target convention used by Drona: +Y is UP.
    y_norm = -y_down_norm

    radial_norm = math.hypot(
        x_norm,
        y_norm
    )

    target_radius_cm = float(target_diameter_cm) / 2.0

    x_cm = x_norm * target_radius_cm
    y_cm = y_norm * target_radius_cm
    radial_cm = radial_norm * target_radius_cm

    # Standard 10 equal radial scoring zones.
    # Exact line boundaries are assigned to the higher-value ring.
    if radial_norm > 1.0:
        auto_ring = "MISS"
    else:
        zone = max(
            1,
            int(
                math.ceil(
                    max(radial_norm, 1e-9) * 10.0
                    - 1e-6
                )
            )
        )
        auto_ring = max(
            1,
            min(
                10,
                11 - zone
            )
        )

    return {
        "x_norm": float(x_norm),
        "y_norm": float(y_norm),
        "radial_norm": float(radial_norm),
        "x_cm": float(x_cm),
        "y_cm": float(y_cm),
        "radial_cm": float(radial_cm),
        "auto_ring": auto_ring,
        "target_diameter_cm": float(target_diameter_cm),
        "target_radius_cm": float(target_radius_cm),
    }


def coordinate_confidence(
    marker_count,
    shaft_score,
    impact_mode,
    radial_norm
):
    """
    Simple UI-oriented confidence label.

    This is NOT a statistical probability. It is a conservative status
    intended to tell the dashboard whether the automatic result should be
    trusted immediately or visually checked / manually overridden.
    """
    if radial_norm is None:
        return "LOW"

    if marker_count < 3:
        return "LOW"

    if impact_mode == "FAR_BLACK_GAP":
        return "MEDIUM"

    if marker_count == 4 and shaft_score >= 0.80:
        return "HIGH"

    if shaft_score >= 0.70:
        return "MEDIUM"

    return "LOW"


def save_shot_result_json(
    output_path,
    image_name,
    impact_current,
    impact_reference,
    coordinates,
    marker_count,
    shaft_score,
    impact_mode,
    confidence
):
    payload = {
        "schema": "ESS_ARCHERY_SHOT_RESULT_V0.5",
        "image": image_name,
        "status": "ARROW",
        "impact_current_px": {
            "x": int(impact_current[0]),
            "y": int(impact_current[1]),
        },
        "impact_reference_px": {
            "x": round(float(impact_reference[0]), 3),
            "y": round(float(impact_reference[1]), 3),
        },
        "auto_result": {
            "x_cm": round(coordinates["x_cm"], 2),
            "y_cm": round(coordinates["y_cm"], 2),
            "radial_cm": round(coordinates["radial_cm"], 2),
            "radial_norm": round(coordinates["radial_norm"], 5),
            "auto_ring": coordinates["auto_ring"],
        },
        "target": {
            "diameter_cm": coordinates["target_diameter_cm"],
            "radius_cm": coordinates["target_radius_cm"],
        },
        "vision": {
            "aruco_markers": int(marker_count),
            "shaft_score": round(float(shaft_score), 4),
            "impact_mode": impact_mode,
            "confidence": confidence,
        },
        "manual_override": {
            "supported": True,
            "recommended_when": [
                "impact marker is visibly misplaced",
                "shot lies close to a scoring-ring boundary",
                "vision confidence is LOW",
            ],
        },
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    return payload


# ============================================================
# DIFFERENCE MAP
# ============================================================

def match_gray_brightness(before_gray, after_gray, mask):
    m = mask > 0

    b = before_gray[m].astype(np.float32)
    a = after_gray[m].astype(np.float32)

    if len(b) < 100 or len(a) < 100:
        return after_gray.astype(np.float32)

    b_mean = float(np.mean(b))
    a_mean = float(np.mean(a))

    b_std = float(np.std(b))
    a_std = float(np.std(a))

    gain = b_std / max(a_std, 1e-6)
    gain = float(np.clip(gain, 0.75, 1.30))

    matched = (
        (after_gray.astype(np.float32) - a_mean)
        * gain
        + b_mean
    )

    return np.clip(
        matched,
        0,
        255
    )


def make_change_map(reference_img, aligned_img, target_mask):
    before_gray = cv2.cvtColor(
        reference_img,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        aligned_img,
        cv2.COLOR_BGR2GRAY
    )

    before_gray = cv2.GaussianBlur(
        before_gray,
        (3, 3),
        0
    )

    after_gray = cv2.GaussianBlur(
        after_gray,
        (3, 3),
        0
    )

    after_matched = match_gray_brightness(
        before_gray,
        after_gray,
        target_mask
    )

    darkening = (
        before_gray.astype(np.float32)
        - after_matched
    )

    vals = darkening[
        target_mask > 0
    ]

    med = float(np.median(vals))

    mad = float(
        np.median(
            np.abs(vals - med)
        )
    )

    robust_sigma = 1.4826 * mad

    threshold = max(
        13.0,
        med + 3.2 * robust_sigma
    )

    change = np.zeros(
        darkening.shape,
        dtype=np.uint8
    )

    change[
        darkening >= threshold
    ] = 255

    change = cv2.bitwise_and(
        change,
        target_mask
    )

    change = cv2.morphologyEx(
        change,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        ),
        iterations=1
    )

    change = cv2.morphologyEx(
        change,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )

    return change, darkening, threshold


# ============================================================
# SHAFT APPEARANCE
# ============================================================

def line_profile_metrics(gray, line):
    """
    Measure whether a Hough line lies on a thick, dark band.

    Hough commonly finds one EDGE of a shaft.
    This scans perpendicular offsets and recenters onto the dark shaft.
    """
    x1, y1, x2, y2 = map(float, line)

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)

    if length < 5:
        return None

    ux = dx / length
    uy = dy / length

    nx = -uy
    ny = ux

    h, w = gray.shape[:2]
    min_dim = min(h, w)

    n_samples = int(
        np.clip(
            length / 8.0,
            50,
            160
        )
    )

    ts = np.linspace(
        0.08,
        0.92,
        n_samples
    )

    bx = x1 + ts * dx
    by = y1 + ts * dy

    max_offset = int(
        np.clip(
            min_dim * 0.025,
            18,
            42
        )
    )

    step = 2

    offsets = np.arange(
        -max_offset,
        max_offset + 1,
        step,
        dtype=np.float32
    )

    X = bx[:, None] + nx * offsets[None, :]
    Y = by[:, None] + ny * offsets[None, :]

    Xi = np.clip(
        np.rint(X).astype(np.int32),
        0,
        w - 1
    )

    Yi = np.clip(
        np.rint(Y).astype(np.int32),
        0,
        h - 1
    )

    vals = gray[
        Yi,
        Xi
    ].astype(np.float32)

    edge_count = max(
        2,
        vals.shape[1] // 5
    )

    bg = np.median(
        np.concatenate(
            [
                vals[:, :edge_count],
                vals[:, -edge_count:]
            ],
            axis=1
        ),
        axis=1
    )

    delta = bg[:, None] - vals

    profile = np.median(
        delta,
        axis=0
    )

    best_idx = int(
        np.argmax(profile)
    )

    peak = float(
        profile[best_idx]
    )

    if peak <= 0:
        width = 0.0
    else:
        threshold = max(
            10.0,
            0.42 * peak
        )

        left = best_idx

        while (
            left > 0
            and profile[left - 1] >= threshold
        ):
            left -= 1

        right = best_idx

        while (
            right < len(profile) - 1
            and profile[right + 1] >= threshold
        ):
            right += 1

        width = float(
            (right - left + 1)
            * step
        )

    best_offset = float(
        offsets[best_idx]
    )

    core_delta = delta[
        :,
        best_idx
    ]

    continuity = float(
        np.mean(
            core_delta
            >= max(
                12.0,
                0.35 * peak
            )
        )
    )

    core_gray = float(
        np.median(
            vals[:, best_idx]
        )
    )

    contrast_score = float(
        np.clip(
            peak / 85.0,
            0.0,
            1.0
        )
    )

    width_score = float(
        np.clip(
            (width - 2.0) / 16.0,
            0.0,
            1.0
        )
    )

    continuity_score = float(
        np.clip(
            (continuity - 0.25) / 0.65,
            0.0,
            1.0
        )
    )

    darkness_score = float(
        np.clip(
            (190.0 - core_gray) / 140.0,
            0.0,
            1.0
        )
    )

    sx = nx * best_offset
    sy = ny * best_offset

    centered = (
        int(round(x1 + sx)),
        int(round(y1 + sy)),
        int(round(x2 + sx)),
        int(round(y2 + sy)),
    )

    return {
        "centered_line": centered,
        "band_contrast": contrast_score,
        "band_width_px": width,
        "width_score": width_score,
        "continuity": continuity,
        "continuity_score": continuity_score,
        "darkness_score": darkness_score,
    }


def sample_line_band(arr, line, n=120, half_width=0.0):
    x1, y1, x2, y2 = map(float, line)

    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    if half_width <= 0:
        xi = np.clip(
            np.rint(xs).astype(int),
            0,
            arr.shape[1] - 1
        )

        yi = np.clip(
            np.rint(ys).astype(int),
            0,
            arr.shape[0] - 1
        )

        return arr[yi, xi]

    dx = x2 - x1
    dy = y2 - y1

    L = max(
        math.hypot(dx, dy),
        1.0
    )

    nx = -dy / L
    ny = dx / L

    values = []

    for off in np.linspace(
        -half_width,
        half_width,
        5
    ):
        xi = np.clip(
            np.rint(
                xs + nx * off
            ).astype(int),
            0,
            arr.shape[1] - 1
        )

        yi = np.clip(
            np.rint(
                ys + ny * off
            ).astype(int),
            0,
            arr.shape[0] - 1
        )

        values.append(
            arr[yi, xi]
        )

    return np.max(
        np.stack(values),
        axis=0
    )


# ============================================================
# HYBRID PHYSICAL ARROW DETECTOR
# ============================================================

def detect_arrow(
    aligned_img,
    change_mask,
    darkening,
    yellow_ellipse
):
    gray = cv2.cvtColor(
        aligned_img,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        35,
        115
    )

    search_mask = make_target_mask(
        aligned_img.shape,
        yellow_ellipse,
        scale_factor=5.25
    )

    edges = cv2.bitwise_and(
        edges,
        search_mask
    )

    h, w = gray.shape[:2]
    min_dim = min(h, w)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(
            35,
            int(min_dim * 0.023)
        ),
        minLineLength=max(
            70,
            int(min_dim * 0.07)
        ),
        maxLineGap=max(
            30,
            int(min_dim * 0.04)
        )
    )

    if lines is None:
        return None, [], edges

    lines = np.asarray(
        lines
    ).reshape(-1, 4)

    # Change support gets a small dilation because the aligned arrow
    # may sit a few pixels away from the binary change-mask center.
    change_dilated = cv2.dilate(
        change_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11)
        ),
        iterations=1
    )

    candidates = []

    for raw in lines:
        metrics = line_profile_metrics(
            gray,
            raw
        )

        if metrics is None:
            continue

        line = metrics[
            "centered_line"
        ]

        x1, y1, x2, y2 = line

        length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        r1 = normalized_radius(
            (x1, y1),
            yellow_ellipse
        )

        r2 = normalized_radius(
            (x2, y2),
            yellow_ellipse
        )

        inner = min(r1, r2)
        outer = max(r1, r2)
        radial_span = outer - inner

        # ----------------------------------------------------
        # CRITICAL V0.4 PHYSICAL RULE
        #
        # A real embedded arrow must travel from the outside /
        # outer target region toward a point inside the target.
        #
        # This rejects:
        #   - internal printed diameter lines
        #   - target ring fragments
        #   - poles/edges completely outside target
        # ----------------------------------------------------
        if outer < MIN_OUTER_RADIUS:
            continue

        if inner > MAX_INNER_RADIUS:
            continue

        if radial_span < MIN_RADIAL_SPAN:
            continue

        width = metrics[
            "band_width_px"
        ]

        if not (
            MIN_SHAFT_WIDTH
            <= width
            <= MAX_SHAFT_WIDTH
        ):
            continue

        if (
            metrics["band_contrast"]
            < 0.08
        ):
            continue

        if (
            metrics["continuity"]
            < MIN_SHAFT_CONTINUITY
        ):
            continue

        n = max(
            80,
            int(length / 4)
        )

        change_values = sample_line_band(
            change_dilated,
            line,
            n=n,
            half_width=max(
                2.0,
                min(
                    width * 0.40,
                    10.0
                )
            )
        )

        dark_values = sample_line_band(
            darkening,
            line,
            n=n,
            half_width=max(
                2.0,
                min(
                    width * 0.30,
                    8.0
                )
            )
        )

        change_fraction = float(
            np.mean(
                change_values > 0
            )
        )

        dark_fraction = float(
            np.mean(
                dark_values > 13.0
            )
        )

        positive_dark = dark_values[
            dark_values > 0
        ]

        median_darkening = (
            float(
                np.median(
                    positive_dark
                )
            )
            if len(positive_dark)
            else 0.0
        )

        length_score = float(
            np.clip(
                length
                / (0.35 * min_dim),
                0.0,
                1.0
            )
        )

        width_score = (
            metrics["width_score"]
            * float(
                np.clip(
                    (70.0 - width)
                    / 30.0,
                    0.0,
                    1.0
                )
            )
        )

        crossing_score = (
            0.40
            * float(
                np.clip(
                    (outer - 0.88)
                    / 0.18,
                    0.0,
                    1.0
                )
            )
            +
            0.40
            * float(
                np.clip(
                    (0.96 - inner)
                    / 0.45,
                    0.0,
                    1.0
                )
            )
            +
            0.20
            * float(
                np.clip(
                    radial_span
                    / 0.45,
                    0.0,
                    1.0
                )
            )
        )

        score = (
            0.16 * length_score
            + 0.16 * metrics["band_contrast"]
            + 0.12 * width_score
            + 0.12 * metrics["continuity_score"]
            + 0.07 * metrics["darkness_score"]
            + 0.16 * change_fraction
            + 0.08 * dark_fraction
            + 0.05
              * float(
                  np.clip(
                      median_darkening / 70.0,
                      0.0,
                      1.0
                  )
              )
            + 0.08 * crossing_score
        )

        candidates.append({
            "line": line,
            "score": score,
            "length": length,
            "width": width,
            "contrast": metrics["band_contrast"],
            "continuity": metrics["continuity"],
            "change_fraction": change_fraction,
            "dark_fraction": dark_fraction,
            "median_darkening": median_darkening,
            "r1": r1,
            "r2": r2,
            "inner": inner,
            "outer": outer,
            "radial_span": radial_span,
            "crossing_score": crossing_score,
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    if not candidates:
        return None, [], edges

    best = candidates[0]

    # Final no-arrow rejection.
    accepted = (
        best["score"] >= MIN_FINAL_SCORE
        and best["change_fraction"] >= MIN_CHANGE_SUPPORT
        and best["median_darkening"] >= MIN_MEDIAN_DARKENING
        and best["contrast"] >= MIN_LINE_CONTRAST
    )

    if not accepted:
        return None, candidates, edges

    return best, candidates, edges



# ============================================================
# CURRENT-FRAME SHAFT DETECTOR
# ============================================================

def detect_arrow_current_frame(
    current_img,
    change_mask,
    darkening,
    target_mask_current,
    radius_function
):
    """
    Same physical V0.4 shaft logic, but executed in the ORIGINAL
    CURRENT camera geometry.

    This prevents the 3-D shaft from being bent / shifted by
    planar target rectification.
    """
    gray = cv2.cvtColor(
        current_img,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        35,
        115
    )

    search_mask = cv2.dilate(
        target_mask_current,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15)
        ),
        iterations=1
    )

    edges = cv2.bitwise_and(
        edges,
        search_mask
    )

    h, w = gray.shape[:2]
    min_dim = min(h, w)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(
            35,
            int(min_dim * 0.023)
        ),
        minLineLength=max(
            70,
            int(min_dim * 0.07)
        ),
        maxLineGap=max(
            30,
            int(min_dim * 0.04)
        )
    )

    if lines is None:
        return None, [], edges

    lines = np.asarray(
        lines
    ).reshape(-1, 4)

    change_dilated = cv2.dilate(
        change_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11)
        ),
        iterations=1
    )

    candidates = []

    for raw in lines:
        metrics = line_profile_metrics(
            gray,
            raw
        )

        if metrics is None:
            continue

        line = metrics[
            "centered_line"
        ]

        x1, y1, x2, y2 = line

        length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        r1 = radius_function(
            (x1, y1)
        )

        r2 = radius_function(
            (x2, y2)
        )

        inner = min(r1, r2)
        outer = max(r1, r2)
        radial_span = outer - inner

        # Real embedded arrow must connect outer target region
        # toward the target interior.
        if outer < MIN_OUTER_RADIUS:
            continue

        if inner > MAX_INNER_RADIUS:
            continue

        if radial_span < MIN_RADIAL_SPAN:
            continue

        width = metrics[
            "band_width_px"
        ]

        if not (
            MIN_SHAFT_WIDTH
            <= width
            <= MAX_SHAFT_WIDTH
        ):
            continue

        if (
            metrics["band_contrast"]
            < 0.08
        ):
            continue

        if (
            metrics["continuity"]
            < MIN_SHAFT_CONTINUITY
        ):
            continue

        n = max(
            80,
            int(length / 4)
        )

        change_values = sample_line_band(
            change_dilated,
            line,
            n=n,
            half_width=max(
                2.0,
                min(
                    width * 0.40,
                    10.0
                )
            )
        )

        dark_values = sample_line_band(
            darkening,
            line,
            n=n,
            half_width=max(
                2.0,
                min(
                    width * 0.30,
                    8.0
                )
            )
        )

        change_fraction = float(
            np.mean(
                change_values > 0
            )
        )

        dark_fraction = float(
            np.mean(
                dark_values > 13.0
            )
        )

        positive_dark = dark_values[
            dark_values > 0
        ]

        median_darkening = (
            float(
                np.median(
                    positive_dark
                )
            )
            if len(positive_dark)
            else 0.0
        )

        length_score = float(
            np.clip(
                length
                / (0.35 * min_dim),
                0.0,
                1.0
            )
        )

        width_score = (
            metrics["width_score"]
            * float(
                np.clip(
                    (70.0 - width)
                    / 30.0,
                    0.0,
                    1.0
                )
            )
        )

        crossing_score = (
            0.40
            * float(
                np.clip(
                    (outer - 0.88)
                    / 0.18,
                    0.0,
                    1.0
                )
            )
            +
            0.40
            * float(
                np.clip(
                    (0.96 - inner)
                    / 0.45,
                    0.0,
                    1.0
                )
            )
            +
            0.20
            * float(
                np.clip(
                    radial_span
                    / 0.45,
                    0.0,
                    1.0
                )
            )
        )

        score = (
            0.16 * length_score
            + 0.16 * metrics["band_contrast"]
            + 0.12 * width_score
            + 0.12 * metrics["continuity_score"]
            + 0.07 * metrics["darkness_score"]
            + 0.16 * change_fraction
            + 0.08 * dark_fraction
            + 0.05
              * float(
                  np.clip(
                      median_darkening / 70.0,
                      0.0,
                      1.0
                  )
              )
            + 0.08 * crossing_score
        )

        candidates.append({
            "line": line,
            "score": score,
            "length": length,
            "width": width,
            "contrast": metrics[
                "band_contrast"
            ],
            "continuity": metrics[
                "continuity"
            ],
            "change_fraction": change_fraction,
            "dark_fraction": dark_fraction,
            "median_darkening": median_darkening,
            "r1": r1,
            "r2": r2,
            "inner": inner,
            "outer": outer,
            "radial_span": radial_span,
            "crossing_score": crossing_score,
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    if not candidates:
        return None, [], edges

    best = candidates[0]

    accepted = (
        best["score"] >= MIN_FINAL_SCORE
        and best["change_fraction"] >= MIN_CHANGE_SUPPORT
        and best["median_darkening"] >= MIN_MEDIAN_DARKENING
        and best["contrast"] >= MIN_LINE_CONTRAST
    )

    if not accepted:
        return None, candidates, edges

    return best, candidates, edges


# ============================================================
# IMPACT REFINEMENT
# ============================================================

def sample_perpendicular_medians(
    gray,
    point,
    normal,
    core_half,
    side_offset
):
    h, w = gray.shape[:2]

    px, py = point
    nx, ny = normal

    core = []

    for off in np.linspace(
        -core_half,
        core_half,
        5
    ):
        x = int(
            round(
                px + nx * off
            )
        )

        y = int(
            round(
                py + ny * off
            )
        )

        if (
            0 <= x < w
            and 0 <= y < h
        ):
            core.append(
                float(
                    gray[y, x]
                )
            )

    side = []

    for base in (
        -side_offset,
        side_offset
    ):
        for extra in (
            -2.0,
            0.0,
            2.0
        ):
            x = int(
                round(
                    px
                    + nx
                    * (base + extra)
                )
            )

            y = int(
                round(
                    py
                    + ny
                    * (base + extra)
                )
            )

            if (
                0 <= x < w
                and 0 <= y < h
            ):
                side.append(
                    float(
                        gray[y, x]
                    )
                )

    if not core or not side:
        return None, None

    return (
        float(np.median(core)),
        float(np.median(side))
    )


def _sample_band_median(
    arr,
    point,
    normal,
    half_width,
    samples=7
):
    h, w = arr.shape[:2]

    values = []

    for off in np.linspace(
        -half_width,
        half_width,
        samples
    ):
        x = int(round(
            point[0]
            + normal[0]
            * off
        ))

        y = int(round(
            point[1]
            + normal[1]
            * off
        ))

        if (
            0 <= x < w
            and 0 <= y < h
        ):
            values.append(
                float(
                    arr[y, x]
                )
            )

    if not values:
        return 0.0

    return float(
        np.median(
            values
        )
    )


def _remove_short_active_runs(
    active,
    min_run=5
):
    active = active.astype(
        bool
    ).copy()

    i = 0

    while i < len(active):
        if not active[i]:
            i += 1
            continue

        j = i

        while (
            j < len(active)
            and active[j]
        ):
            j += 1

        if (
            j - i
            < min_run
        ):
            active[
                i:j
            ] = False

        i = j

    return active


def refine_impact(
    best,
    reference_warped_to_current,
    current_img,
    yellow_ellipse_reference,
    change_mask,
    target_mask_current,
    radius_function
):
    """
    V0.4.1 CURRENT-FRAME + RING-AWARE IMPACT TRACKER

    Why current frame?
      ArUco homography is valid for the flat target plane.
      The arrow shaft protrudes in 3-D, so rectifying the whole CURRENT
      image can geometrically distort the shaft.

    Therefore:
      - CURRENT shot stays untouched
      - REFERENCE target is warped INTO current camera view
      - shaft remains physically straight
      - before/after subtraction is still aligned on the target plane

    Impact logic:
      - trace from rear / target-entry side
      - do NOT assume impact is nearest target center
      - allow evidence loss through black scoring annulus
      - require sustained evidence to resume after a gap
    """
    x1, y1, x2, y2 = map(
        float,
        best["line"]
    )

    r1 = radius_function(
        (x1, y1)
    )

    r2 = radius_function(
        (x2, y2)
    )

    if r1 >= r2:
        rear_anchor = np.array(
            [x1, y1],
            dtype=np.float32
        )

        forward_anchor = np.array(
            [x2, y2],
            dtype=np.float32
        )

    else:
        rear_anchor = np.array(
            [x2, y2],
            dtype=np.float32
        )

        forward_anchor = np.array(
            [x1, y1],
            dtype=np.float32
        )

    vector = (
        forward_anchor
        - rear_anchor
    )

    base_len = float(
        np.linalg.norm(
            vector
        )
    )

    if base_len < 5:
        fallback = (
            int(round(
                forward_anchor[0]
            )),
            int(round(
                forward_anchor[1]
            ))
        )

        return (
            fallback,
            radius_function(
                fallback
            ),
            None
        )

    u = (
        vector
        / base_len
    )

    normal = np.array(
        [-u[1], u[0]],
        dtype=np.float32
    )

    # Effective current-view target radius from projected mask area.
    target_area = max(
        1,
        cv2.countNonZero(
            target_mask_current
        )
    )

    target_radius_px = math.sqrt(
        target_area
        / math.pi
    )

    step_px = 2.0

    s_values = np.arange(
        -0.35 * target_radius_px,
        2.55 * target_radius_px,
        step_px,
        dtype=np.float32
    )

    points = (
        rear_anchor[None, :]
        + s_values[:, None]
        * u[None, :]
    )

    radii = np.array([
        radius_function(
            (
                float(p[0]),
                float(p[1])
            )
        )
        for p in points
    ], dtype=np.float32)

    h, w = current_img.shape[:2]

    valid = (
        (points[:, 0] >= 3)
        & (points[:, 0] < w - 3)
        & (points[:, 1] >= 3)
        & (points[:, 1] < h - 3)
    )

    inside_indices = np.where(
        valid
        & (radii <= 1.02)
    )[0]

    if len(
        inside_indices
    ) == 0:
        fallback = (
            int(round(
                forward_anchor[0]
            )),
            int(round(
                forward_anchor[1]
            ))
        )

        return (
            fallback,
            radius_function(
                fallback
            ),
            None
        )

    entry_idx = int(
        inside_indices[0]
    )

    search_end = (
        len(s_values)
        - 1
    )

    reached_inner_target = False

    for i in range(
        entry_idx,
        len(s_values)
    ):
        if radii[i] < 0.68:
            reached_inner_target = True

        if (
            reached_inner_target
            and radii[i] > 1.06
        ):
            search_end = i
            break

    before_gray = cv2.cvtColor(
        reference_warped_to_current,
        cv2.COLOR_BGR2GRAY
    )

    after_gray = cv2.cvtColor(
        current_img,
        cv2.COLOR_BGR2GRAY
    )

    before_gray = cv2.GaussianBlur(
        before_gray,
        (3, 3),
        0
    )

    after_gray = cv2.GaussianBlur(
        after_gray,
        (3, 3),
        0
    )

    abs_difference = cv2.absdiff(
        before_gray,
        after_gray
    ).astype(
        np.float32
    )

    gx = cv2.Sobel(
        after_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        after_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        gx * gx
        + gy * gy
    )

    shaft_width = float(
        best.get(
            "width",
            20.0
        )
    )

    core_half = max(
        2.0,
        min(
            shaft_width
            * 0.30,
            8.0
        )
    )

    side_offset = max(
        8.0,
        min(
            shaft_width
            * 0.90,
            26.0
        )
    )

    combined = []

    for p in points:
        px = float(p[0])
        py = float(p[1])

        if not (
            3 <= px < w - 3
            and 3 <= py < h - 3
        ):
            combined.append(
                0.0
            )

            continue

        core_abs = _sample_band_median(
            abs_difference,
            p,
            normal,
            core_half,
            samples=7
        )

        left_abs = _sample_band_median(
            abs_difference,
            p
            + normal
            * side_offset,
            normal,
            max(
                2.0,
                core_half
                * 0.60
            ),
            samples=5
        )

        right_abs = _sample_band_median(
            abs_difference,
            p
            - normal
            * side_offset,
            normal,
            max(
                2.0,
                core_half
                * 0.60
            ),
            samples=5
        )

        side_abs = float(
            np.median([
                left_abs,
                right_abs
            ])
        )

        localized_change = max(
            0.0,
            core_abs
            - 0.65
            * side_abs
        )

        mask_values = []

        for off in np.linspace(
            -core_half,
            core_half,
            7
        ):
            x = int(round(
                px
                + normal[0]
                * off
            ))

            y = int(round(
                py
                + normal[1]
                * off
            ))

            if (
                0 <= x < w
                and 0 <= y < h
            ):
                mask_values.append(
                    float(
                        change_mask[
                            y,
                            x
                        ]
                    )
                    / 255.0
                )

        mask_support = (
            float(
                np.mean(
                    mask_values
                )
            )
            if mask_values
            else 0.0
        )

        edge_offset = max(
            3.0,
            shaft_width
            * 0.35
        )

        edge_left = _sample_band_median(
            gradient,
            p
            + normal
            * edge_offset,
            normal,
            2.0,
            samples=3
        )

        edge_right = _sample_band_median(
            gradient,
            p
            - normal
            * edge_offset,
            normal,
            2.0,
            samples=3
        )

        edge_value = float(
            np.mean([
                edge_left,
                edge_right
            ])
        )

        local_score = float(
            np.clip(
                localized_change
                / 55.0,
                0.0,
                1.0
            )
        )

        edge_score = float(
            np.clip(
                edge_value
                / 180.0,
                0.0,
                1.0
            )
        )

        combined.append(
            0.45
            * local_score
            +
            0.35
            * mask_support
            +
            0.20
            * edge_score
        )

    combined = np.asarray(
        combined,
        dtype=np.float32
    )

    if len(
        combined
    ) >= 5:
        evidence = np.convolve(
            combined,
            np.ones(
                5,
                dtype=np.float32
            )
            / 5.0,
            mode="same"
        )

    else:
        evidence = combined.copy()

    known_region = (
        (s_values >= 0.10 * base_len)
        & (s_values <= 0.80 * base_len)
    )

    known_values = evidence[
        known_region
    ]

    evidence_level = (
        float(
            np.percentile(
                known_values,
                60
            )
        )
        if len(
            known_values
        )
        else 0.45
    )

    evidence_threshold = max(
        0.18,
        min(
            0.38,
            0.42
            * evidence_level
        )
    )

    active = (
        evidence
        >= evidence_threshold
    )

    active = _remove_short_active_runs(
        active,
        min_run=5
    )

    start_search_samples = max(
        8,
        int(
            0.14
            * target_radius_px
            / step_px
        )
    )

    start_end = min(
        search_end,
        entry_idx
        + start_search_samples
    )

    starts = np.where(
        active[
            entry_idx:
            start_end + 1
        ]
    )[0]

    if len(
        starts
    ) == 0:
        fallback = (
            int(round(
                forward_anchor[0]
            )),
            int(round(
                forward_anchor[1]
            ))
        )

        debug = {
            "s_values": s_values,
            "radii": radii,
            "evidence": evidence,
            "threshold": evidence_threshold,
            "entry_idx": entry_idx,
            "impact_idx": None,
            "status": "NO_ENTRY_EVIDENCE",
        }

        return (
            fallback,
            radius_function(
                fallback
            ),
            debug
        )

    current_idx = (
        entry_idx
        + int(
            starts[0]
        )
    )

    impact_idx = None

    normal_gap_px = max(
        30.0,
        0.12
        * target_radius_px
    )

    black_gap_px = max(
        75.0,
        0.36
        * target_radius_px
    )

    min_recovery_run = 6

    while (
        current_idx
        < search_end
    ):
        if active[
            current_idx
        ]:
            current_idx += 1
            continue

        gap_start = current_idx

        long_look_end = min(
            search_end,
            gap_start
            + int(
                black_gap_px
                / step_px
            )
            + 2
        )

        look_radii = radii[
            gap_start:
            long_look_end + 1
        ]

        overlaps_black = bool(
            np.any(
                (
                    look_radii
                    >= 0.56
                )
                &
                (
                    look_radii
                    <= 0.84
                )
            )
        )

        allowed_gap_px = (
            black_gap_px
            if overlaps_black
            else normal_gap_px
        )

        look_end = min(
            search_end,
            gap_start
            + int(
                allowed_gap_px
                / step_px
            )
        )

        recovery_idx = None
        j = gap_start + 1

        while (
            j <= look_end
        ):
            if not active[j]:
                j += 1
                continue

            run_end = j

            while (
                run_end <= look_end
                and active[
                    run_end
                ]
            ):
                run_end += 1

            if (
                run_end - j
                >= min_recovery_run
            ):
                recovery_idx = j
                break

            j = run_end + 1

        if recovery_idx is not None:
            current_idx = recovery_idx
            continue

        impact_idx = max(
            entry_idx,
            gap_start - 1
        )

        break

    if impact_idx is None:
        active_indices = np.where(
            active[
                entry_idx:
                search_end + 1
            ]
        )[0]

        impact_idx = (
            entry_idx
            + int(
                active_indices[-1]
            )
            if len(
                active_indices
            )
            else entry_idx
        )

    impact_point = points[
        impact_idx
    ]

    impact = (
        int(round(
            float(
                impact_point[0]
            )
        )),
        int(round(
            float(
                impact_point[1]
            )
        )),
    )

    impact_radius = float(
        radius_function(
            impact
        )
    )

    debug = {
        "s_values": s_values,
        "radii": radii,
        "evidence": evidence,
        "threshold": evidence_threshold,
        "entry_idx": entry_idx,
        "impact_idx": impact_idx,
        "active": active,
        "status": "OK",
    }

    return (
        impact,
        impact_radius,
        debug
    )


def draw_impact_profile(
    debug,
    width=1100,
    height=420
):
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    if not debug:
        return canvas

    s_values = debug[
        "s_values"
    ]

    radii = debug[
        "radii"
    ]

    evidence = debug[
        "evidence"
    ]

    threshold = float(
        debug[
            "threshold"
        ]
    )

    entry_idx = int(
        debug[
            "entry_idx"
        ]
    )

    impact_idx = debug.get(
        "impact_idx"
    )

    left = 70
    right = width - 30
    top = 45
    bottom = height - 70

    s0 = float(
        s_values[
            entry_idx
        ]
    )

    relevant = (
        s_values >= s0
    )

    s_plot = s_values[
        relevant
    ]

    e_plot = evidence[
        relevant
    ]

    r_plot = radii[
        relevant
    ]

    max_s = max(
        float(
            np.max(
                s_plot - s0
            )
        ),
        1.0
    )

    max_e = max(
        float(
            np.percentile(
                e_plot,
                98
            )
        )
        if len(e_plot)
        else 1.0,
        threshold * 1.5,
        0.5
    )

    def x_from_s(s):
        return int(round(
            left
            + (
                right - left
            )
            * (
                float(s) - s0
            )
            / max_s
        ))

    inside_black = (
        (r_plot >= 0.56)
        & (r_plot <= 0.84)
    )

    run_start = None

    for i, flag in enumerate(
        inside_black
    ):
        if flag and run_start is None:
            run_start = i

        if (
            run_start is not None
            and (
                not flag
                or i
                == len(
                    inside_black
                ) - 1
            )
        ):
            run_end = (
                i
                if flag
                else i - 1
            )

            cv2.rectangle(
                canvas,
                (
                    x_from_s(
                        s_plot[
                            run_start
                        ]
                    ),
                    top
                ),
                (
                    x_from_s(
                        s_plot[
                            run_end
                        ]
                    ),
                    bottom
                ),
                (35, 35, 35),
                -1
            )

            run_start = None

    cv2.line(
        canvas,
        (left, bottom),
        (right, bottom),
        (170, 170, 170),
        1
    )

    cv2.line(
        canvas,
        (left, top),
        (left, bottom),
        (170, 170, 170),
        1
    )

    profile_points = []

    for s, ev in zip(
        s_plot,
        e_plot
    ):
        x = x_from_s(
            s
        )

        y = int(round(
            bottom
            - (
                bottom - top
            )
            * min(
                float(ev),
                max_e
            )
            / max_e
        ))

        profile_points.append(
            (x, y)
        )

    if len(
        profile_points
    ) >= 2:
        cv2.polylines(
            canvas,
            [
                np.asarray(
                    profile_points,
                    dtype=np.int32
                )
            ],
            False,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

    threshold_y = int(round(
        bottom
        - (
            bottom - top
        )
        * min(
            threshold,
            max_e
        )
        / max_e
    ))

    cv2.line(
        canvas,
        (left, threshold_y),
        (right, threshold_y),
        (0, 255, 255),
        2
    )

    if impact_idx is not None:
        impact_x = x_from_s(
            s_values[
                int(
                    impact_idx
                )
            ]
        )

        cv2.line(
            canvas,
            (impact_x, top),
            (impact_x, bottom),
            (0, 255, 0),
            3
        )

    cv2.putText(
        canvas,
        "Current-frame ring-aware shaft evidence",
        (left, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "white=evidence | yellow=threshold | green=impact | gray=black scoring annulus",
        (left, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    return canvas



# ============================================================
# V0.4.2 BLACK-RING DUAL-EDGE REFINEMENT
# ============================================================

def _remove_short_true_runs_bool(active, min_run=5):
    """
    Remove short isolated True runs from a 1-D boolean signal.
    """
    active = active.astype(bool).copy()

    i = 0
    n = len(active)

    while i < n:
        if not active[i]:
            i += 1
            continue

        j = i

        while j < n and active[j]:
            j += 1

        if (j - i) < min_run:
            active[i:j] = False

        i = j

    return active


def refine_black_ring_with_dual_edges(
    best,
    current_img,
    target_mask_current,
    radius_function,
    normal_impact,
    normal_radius
):
    """
    V0.4.2 SPECIAL CASE:
    refine impact only when the normal tracker terminates inside
    the black scoring annulus.

    Why this works:
      - the center of a black shaft can disappear against black paper;
      - the two cylindrical shaft edges often remain visible;
      - the shaft detector already gives us the correct axis and width.

    We therefore trace BOTH expected shaft edges using the image
    gradient perpendicular to the shaft direction.

    The tracker:
      1) starts on the already-confirmed shaft,
      2) follows dual-edge evidence forward,
      3) allows larger gaps while geometrically inside the black annulus,
      4) stops at the final sustained edge termination.

    Returns:
      (impact_point, impact_radius, debug)
    or
      (normal_impact, normal_radius, debug)
      if the special refinement is not reliable.
    """

    # Only use this specialized logic for the black annulus.
    if normal_radius is None or not (0.56 <= normal_radius <= 0.84):
        return (
            normal_impact,
            normal_radius,
            {
                "used": False,
                "reason": "NORMAL_IMPACT_NOT_IN_BLACK_ANNULUS",
            }
        )

    x1, y1, x2, y2 = map(float, best["line"])

    r1 = radius_function((x1, y1))
    r2 = radius_function((x2, y2))

    # Larger-radius endpoint is treated as rear / entry side.
    if r1 >= r2:
        rear_anchor = np.array([x1, y1], dtype=np.float32)
        forward_anchor = np.array([x2, y2], dtype=np.float32)
    else:
        rear_anchor = np.array([x2, y2], dtype=np.float32)
        forward_anchor = np.array([x1, y1], dtype=np.float32)

    vector = forward_anchor - rear_anchor
    base_len = float(np.linalg.norm(vector))

    if base_len < 5.0:
        return (
            normal_impact,
            normal_radius,
            {
                "used": False,
                "reason": "SHAFT_SEGMENT_TOO_SHORT",
            }
        )

    u = vector / base_len
    normal = np.array([-u[1], u[0]], dtype=np.float32)

    target_area = max(
        1,
        cv2.countNonZero(target_mask_current)
    )

    target_radius_px = math.sqrt(
        target_area / math.pi
    )

    step_px = 2.0

    s_values = np.arange(
        -0.20 * target_radius_px,
        2.60 * target_radius_px,
        step_px,
        dtype=np.float32
    )

    points = (
        rear_anchor[None, :]
        + s_values[:, None] * u[None, :]
    )

    radii = np.array(
        [
            radius_function(
                (float(p[0]), float(p[1]))
            )
            for p in points
        ],
        dtype=np.float32
    )

    h, w = current_img.shape[:2]

    valid = (
        (points[:, 0] >= 3)
        & (points[:, 0] < w - 3)
        & (points[:, 1] >= 3)
        & (points[:, 1] < h - 3)
    )

    inside_indices = np.where(
        valid & (radii <= 1.03)
    )[0]

    if len(inside_indices) == 0:
        return (
            normal_impact,
            normal_radius,
            {
                "used": False,
                "reason": "TARGET_ENTRY_NOT_FOUND",
            }
        )

    entry_idx = int(inside_indices[0])

    # Search through the target and stop after the far-side exit
    # if the arrow projection crosses the target center.
    search_end = len(s_values) - 1
    reached_inner = False

    for i in range(entry_idx, len(s_values)):
        if radii[i] < 0.55:
            reached_inner = True

        if reached_inner and radii[i] > 1.08:
            search_end = i
            break

    gray = cv2.cvtColor(
        current_img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    ).astype(np.float32)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Only gradients perpendicular to the shaft axis matter here.
    grad_normal = np.abs(
        gx * normal[0]
        + gy * normal[1]
    )

    shaft_width = float(
        best.get("width", 24.0)
    )

    half_width = max(
        4.0,
        0.50 * shaft_width
    )

    pair_score = []

    for p in points:
        side_peaks = []

        for sign in (-1.0, 1.0):
            offsets = np.linspace(
                sign * half_width * 0.55,
                sign * half_width * 1.45,
                9
            )

            values = []

            for off in offsets:
                x = int(round(
                    p[0] + normal[0] * off
                ))

                y = int(round(
                    p[1] + normal[1] * off
                ))

                if 0 <= x < w and 0 <= y < h:
                    values.append(
                        float(
                            grad_normal[y, x]
                        )
                    )
                else:
                    values.append(0.0)

            side_peaks.append(
                max(values)
                if values
                else 0.0
            )

        # BOTH sides must carry edge evidence.
        pair_score.append(
            min(
                side_peaks[0],
                side_peaks[1]
            )
        )

    pair_score = np.asarray(
        pair_score,
        dtype=np.float32
    )

    if len(pair_score) >= 7:
        edge_evidence = np.convolve(
            pair_score,
            np.ones(
                7,
                dtype=np.float32
            ) / 7.0,
            mode="same"
        )
    else:
        edge_evidence = pair_score.copy()

    # Conservative threshold tuned only for this black-ring fallback.
    #
    # We deliberately keep this separate from the normal impact tracker
    # so the already-working colored-background behavior is untouched.
    edge_threshold = 18.0

    edge_active = (
        edge_evidence
        >= edge_threshold
    )

    edge_active = _remove_short_true_runs_bool(
        edge_active,
        min_run=5
    )

    # Start from the already-confirmed shaft area.
    idx_s0 = int(
        np.argmin(
            np.abs(s_values)
        )
    )

    start_search_end = min(
        search_end,
        idx_s0
        + int(
            (
                base_len
                + 0.15 * target_radius_px
            )
            / step_px
        )
    )

    start_candidates = np.where(
        edge_active[
            idx_s0:
            start_search_end + 1
        ]
    )[0]

    if len(start_candidates) == 0:
        return (
            normal_impact,
            normal_radius,
            {
                "used": False,
                "reason": "NO_DUAL_EDGE_START",
                "s_values": s_values,
                "radii": radii,
                "edge_evidence": edge_evidence,
                "edge_threshold": edge_threshold,
                "edge_active": edge_active,
            }
        )

    current_idx = (
        idx_s0
        + int(start_candidates[0])
    )

    last_active_idx = current_idx

    # Normal gaps must be short.
    normal_gap_px = max(
        26.0,
        0.06 * target_radius_px
    )

    # Black-on-black can temporarily hide both edges.
    # A substantially larger gap is allowed ONLY when that gap lies
    # in the known black scoring annulus.
    black_gap_px = max(
        90.0,
        0.26 * target_radius_px
    )

    min_recovery_run = 5

    i = current_idx

    while i <= search_end:

        if edge_active[i]:
            j = i

            while (
                j <= search_end
                and edge_active[j]
            ):
                j += 1

            if (j - i) >= min_recovery_run:
                last_active_idx = j - 1

            i = j
            continue

        gap_start = i
        j = i

        while (
            j <= search_end
            and not edge_active[j]
        ):
            j += 1

        gap_px = (
            j - gap_start
        ) * step_px

        gap_radii = radii[
            gap_start:
            min(
                j + 1,
                len(radii)
            )
        ]

        overlaps_black = bool(
            np.any(
                (gap_radii >= 0.56)
                & (gap_radii <= 0.84)
            )
        )

        allowed_gap_px = (
            black_gap_px
            if overlaps_black
            else normal_gap_px
        )

        if (
            j <= search_end
            and gap_px <= allowed_gap_px
        ):
            run_end = j

            while (
                run_end <= search_end
                and edge_active[run_end]
            ):
                run_end += 1

            if (
                run_end - j
                >= min_recovery_run
            ):
                i = j
                continue

        # No reliable recovery after this point.
        break

    edge_impact_idx = int(
        last_active_idx
    )

    edge_impact_point = points[
        edge_impact_idx
    ]

    edge_impact = (
        int(round(
            float(
                edge_impact_point[0]
            )
        )),
        int(round(
            float(
                edge_impact_point[1]
            )
        ))
    )

    edge_radius = float(
        radius_function(
            edge_impact
        )
    )

    # Convert the original normal impact into the same shaft coordinate
    # so we know whether the dual-edge tracker actually extended toward
    # the physical shaft termination instead of jumping backwards.
    normal_vec = (
        np.asarray(
            normal_impact,
            dtype=np.float32
        )
        - rear_anchor
    )

    normal_s = float(
        np.dot(
            normal_vec,
            u
        )
    )

    edge_s = float(
        s_values[
            edge_impact_idx
        ]
    )

    # Sanity gates:
    # - remain on/near target
    # - do not jump backwards substantially
    # - refinement should be supported by a real tracked edge run
    accepted = (
        edge_radius <= 1.02
        and (
            edge_s >= (
                normal_s
                - 0.18 * target_radius_px
            )
            or (
                0.60 <= edge_radius <= 0.78
                and normal_radius is not None
                and 0.78 <= normal_radius <= 0.86
            )
        )
    )

    debug = {
        "used": bool(accepted),
        "reason": (
            "BLACK_RING_DUAL_EDGE"
            if accepted
            else "DUAL_EDGE_REJECTED"
        ),
        "s_values": s_values,
        "radii": radii,
        "edge_evidence": edge_evidence,
        "edge_threshold": edge_threshold,
        "edge_active": edge_active,
        "edge_impact_idx": edge_impact_idx,
        "normal_s": normal_s,
        "edge_s": edge_s,
        "normal_impact": normal_impact,
        "edge_impact": edge_impact,
        "normal_radius": normal_radius,
        "edge_radius": edge_radius,
    }

    if accepted:
        return (
            edge_impact,
            edge_radius,
            debug
        )

    return (
        normal_impact,
        normal_radius,
        debug
    )


def draw_black_edge_profile(
    debug,
    width=1100,
    height=420
):
    """
    V0.4.2 debug image:
      white = dual-edge evidence
      yellow = edge threshold
      green = selected edge impact
      gray = black annulus geometry
    """
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    if (
        not debug
        or "edge_evidence" not in debug
    ):
        cv2.putText(
            canvas,
            "Black-ring edge refinement not used",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return canvas

    s_values = debug["s_values"]
    radii = debug["radii"]
    evidence = debug["edge_evidence"]
    threshold = float(
        debug["edge_threshold"]
    )

    impact_idx = debug.get(
        "edge_impact_idx"
    )

    left = 70
    right = width - 30
    top = 45
    bottom = height - 70

    s0 = float(
        np.min(s_values)
    )

    s1 = float(
        np.max(s_values)
    )

    s_span = max(
        s1 - s0,
        1.0
    )

    max_evidence = max(
        float(
            np.percentile(
                evidence,
                98
            )
        )
        if len(evidence)
        else 1.0,
        threshold * 1.6,
        30.0
    )

    def x_from_s(s):
        return int(round(
            left
            + (
                right - left
            )
            * (
                float(s) - s0
            )
            / s_span
        ))

    # Black annulus shading.
    inside_black = (
        (radii >= 0.56)
        & (radii <= 0.84)
    )

    run_start = None

    for i, flag in enumerate(
        inside_black
    ):
        if flag and run_start is None:
            run_start = i

        is_last = (
            i == len(
                inside_black
            ) - 1
        )

        if (
            run_start is not None
            and (
                not flag
                or is_last
            )
        ):
            run_end = (
                i
                if flag and is_last
                else i - 1
            )

            cv2.rectangle(
                canvas,
                (
                    x_from_s(
                        s_values[
                            run_start
                        ]
                    ),
                    top
                ),
                (
                    x_from_s(
                        s_values[
                            run_end
                        ]
                    ),
                    bottom
                ),
                (35, 35, 35),
                -1
            )

            run_start = None

    profile_points = []

    for s, value in zip(
        s_values,
        evidence
    ):
        x = x_from_s(s)

        y = int(round(
            bottom
            - (
                bottom - top
            )
            * min(
                float(value),
                max_evidence
            )
            / max_evidence
        ))

        profile_points.append(
            (x, y)
        )

    if len(profile_points) >= 2:
        cv2.polylines(
            canvas,
            [
                np.asarray(
                    profile_points,
                    dtype=np.int32
                )
            ],
            False,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

    threshold_y = int(round(
        bottom
        - (
            bottom - top
        )
        * min(
            threshold,
            max_evidence
        )
        / max_evidence
    ))

    cv2.line(
        canvas,
        (left, threshold_y),
        (right, threshold_y),
        (0, 255, 255),
        2
    )

    if impact_idx is not None:
        impact_x = x_from_s(
            s_values[
                int(
                    impact_idx
                )
            ]
        )

        cv2.line(
            canvas,
            (impact_x, top),
            (impact_x, bottom),
            (0, 255, 0),
            3
        )

    cv2.putText(
        canvas,
        "V0.4.2 dual-edge black-ring evidence",
        (left, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "white=paired shaft edges | yellow=threshold | green=impact | gray=black annulus",
        (left, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    return canvas



# ============================================================
# V0.4.3 FAR-SIDE BLACK TERMINATION FALLBACK
# ============================================================

def refine_far_side_black_gap(
    best,
    impact_debug,
    normal_impact,
    normal_radius,
    radius_function
):
    """
    Very narrow fallback for the remaining V0.4.2 corner case:

      - shaft projection crosses near the target center,
      - normal tracker overshoots to the OUTER edge of the far-side
        black annulus,
      - dual-edge tracker cannot form a trustworthy pair.

    In this geometry a true black-ring impact often appears as:
        strong shaft evidence
        -> sustained disappearance inside far-side black
        -> false recovery caused by printed/ring structure.

    We therefore find the FIRST substantial evidence gap on the
    far-side black annulus after the center crossing.

    IMPORTANT:
    This is intentionally conservative and is NOT used for:
      - normal colored-ring hits,
      - near-side black hits,
      - white/outer-ring hits,
      - shots that never cross the central region.
    """

    debug = {
        "used": False,
        "reason": "NOT_APPLICABLE",
    }

    if impact_debug is None:
        debug["reason"] = "NO_NORMAL_PROFILE"
        return normal_impact, normal_radius, debug

    # This fallback is only for the exact overshoot pattern we observed.
    if normal_radius is None or not (0.78 <= normal_radius <= 0.86):
        debug["reason"] = "NORMAL_RADIUS_NOT_OUTER_BLACK"
        return normal_impact, normal_radius, debug

    # Shaft detector must show that the projection reaches well toward
    # the target center; otherwise this is not the cross-center case.
    if float(best.get("inner", 1.0)) > 0.50:
        debug["reason"] = "SHAFT_DOES_NOT_CROSS_INNER_TARGET"
        return normal_impact, normal_radius, debug

    s_values = np.asarray(
        impact_debug["s_values"],
        dtype=np.float32
    )

    radii = np.asarray(
        impact_debug["radii"],
        dtype=np.float32
    )

    active = np.asarray(
        impact_debug["active"],
        dtype=bool
    )

    entry_idx = int(
        impact_debug["entry_idx"]
    )

    if len(radii) < 10:
        debug["reason"] = "PROFILE_TOO_SHORT"
        return normal_impact, normal_radius, debug

    # Find the center-crossing / minimum-radius point.
    search_r = radii[entry_idx:]

    min_rel = int(
        np.argmin(search_r)
    )

    center_idx = entry_idx + min_rel

    # Verify that after this point we are moving outward again.
    if center_idx + 6 >= len(radii):
        debug["reason"] = "NO_FAR_SIDE"
        return normal_impact, normal_radius, debug

    # If there is sustained real evidence in the far-side WHITE ring,
    # the arrow may genuinely continue beyond black. Do not interfere.
    white_far = (
        (np.arange(len(radii)) > center_idx)
        & (radii >= 0.86)
        & (radii <= 1.03)
        & active
    )

    white_indices = np.where(
        white_far
    )[0]

    if len(white_indices) >= 8:
        # Count whether at least one sustained active run exists.
        run_start = None
        sustained_white = False

        for idx in white_indices:
            if run_start is None:
                run_start = idx
                prev = idx
                continue

            if idx == prev + 1:
                prev = idx
                continue

            if prev - run_start + 1 >= 8:
                sustained_white = True
                break

            run_start = idx
            prev = idx

        if (
            run_start is not None
            and prev - run_start + 1 >= 8
        ):
            sustained_white = True

        if sustained_white:
            debug["reason"] = "SHAFT_CONTINUES_INTO_WHITE"
            return normal_impact, normal_radius, debug

    # Search only the far-side black annulus.
    i = center_idx + 1

    candidate_gap = None

    while i < len(active):
        # We care only once we reach black on the far side.
        if radii[i] < 0.58:
            i += 1
            continue

        if radii[i] > 0.82:
            break

        if active[i]:
            i += 1
            continue

        j = i

        while (
            j < len(active)
            and not active[j]
            and radii[j] <= 0.84
        ):
            j += 1

        gap_len = j - i

        if gap_len > 0:
            r_start = float(radii[i])
            r_end = float(radii[j - 1])
            radial_span = abs(r_end - r_start)

            # Long enough to be a physical disappearance, not a one-pixel
            # texture fluctuation. 2 px/sample in the current tracker.
            if (
                gap_len >= 12
                and radial_span >= 0.025
                and max(r_start, r_end) >= 0.62
            ):
                candidate_gap = (
                    i,
                    j - 1,
                    r_start,
                    r_end,
                    gap_len,
                )
                break

        i = max(j, i + 1)

    if candidate_gap is None:
        debug["reason"] = "NO_FAR_BLACK_GAP"
        return normal_impact, normal_radius, debug

    gap_start, gap_end, r_start, r_end, gap_len = candidate_gap

    # For these black-on-black impacts, the shaft can become invisible
    # for part of its final embedded section. Empirically the FAR END of
    # the first substantial disappearance is a much better estimate than
    # the later printed-ring recovery that caused V0.4.2 to overshoot.
    impact_idx = int(gap_end)

    x1, y1, x2, y2 = map(
        float,
        best["line"]
    )

    r1 = radius_function(
        (x1, y1)
    )

    r2 = radius_function(
        (x2, y2)
    )

    if r1 >= r2:
        rear_anchor = np.array(
            [x1, y1],
            dtype=np.float32
        )
        forward_anchor = np.array(
            [x2, y2],
            dtype=np.float32
        )
    else:
        rear_anchor = np.array(
            [x2, y2],
            dtype=np.float32
        )
        forward_anchor = np.array(
            [x1, y1],
            dtype=np.float32
        )

    vector = forward_anchor - rear_anchor
    length = float(
        np.linalg.norm(vector)
    )

    if length < 5:
        debug["reason"] = "SHAFT_TOO_SHORT"
        return normal_impact, normal_radius, debug

    u = vector / length

    p = (
        rear_anchor
        + float(s_values[impact_idx]) * u
    )

    impact = (
        int(round(float(p[0]))),
        int(round(float(p[1]))),
    )

    impact_radius = float(
        radius_function(
            impact
        )
    )

    # Final safety gate: candidate must remain in black annulus.
    if not (0.60 <= impact_radius <= 0.78):
        debug["reason"] = "CANDIDATE_NOT_BLACK"
        debug["candidate_radius"] = impact_radius
        return normal_impact, normal_radius, debug

    debug.update({
        "used": True,
        "reason": "FAR_SIDE_BLACK_GAP",
        "gap_start_idx": gap_start,
        "gap_end_idx": gap_end,
        "gap_len": gap_len,
        "gap_r_start": r_start,
        "gap_r_end": r_end,
        "impact_idx": impact_idx,
        "impact": impact,
        "impact_radius": impact_radius,
    })

    return impact, impact_radius, debug


# ============================================================
# VISUALIZATION
# ============================================================

def draw_marker_overlay(img, marker_dict):
    out = img.copy()

    for marker_id, corners in marker_dict.items():
        poly = np.rint(
            corners
        ).astype(np.int32)

        cv2.polylines(
            out,
            [poly],
            True,
            (0, 255, 0),
            4
        )

        center = np.mean(
            corners,
            axis=0
        )

        cv2.putText(
            out,
            f"ID {marker_id}",
            (
                int(center[0]) - 25,
                int(center[1]) - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return out


def draw_target_reference(img, yellow_ellipse):
    center, axes, angle = yellow_ellipse

    cv2.ellipse(
        img,
        (
            center,
            axes,
            angle
        ),
        (0, 255, 255),
        3
    )

    cv2.ellipse(
        img,
        (
            center,
            (
                axes[0] * 5.0,
                axes[1] * 5.0
            ),
            angle
        ),
        (0, 220, 0),
        3
    )

    cv2.drawMarker(
        img,
        (
            int(round(center[0])),
            int(round(center[1]))
        ),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        32,
        3
    )


def create_result_image(
    aligned,
    yellow_ellipse,
    marker_info,
    best,
    candidates,
    impact=None,
    impact_radius=None,
    impact_mode="NORMAL"
):
    out = aligned.copy()

    draw_target_reference(
        out,
        yellow_ellipse
    )

    # Light alternate candidates.
    for c in candidates[:4]:
        x1, y1, x2, y2 = c["line"]

        cv2.line(
            out,
            (x1, y1),
            (x2, y2),
            (155, 155, 155),
            2
        )

    cv2.rectangle(
        out,
        (20, 20),
        (1120, 300),
        (10, 10, 10),
        -1
    )

    count = marker_info[
        "marker_count"
    ]

    ids = marker_info[
        "marker_ids"
    ]

    marker_text = (
        f"ARUCO {count}/4  IDs {ids}"
    )

    marker_color = (
        (70, 230, 120)
        if count == 4
        else (0, 190, 255)
    )

    cv2.putText(
        out,
        "ESS-ARCHERY VISION V0.5",
        (40, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        out,
        marker_text,
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        marker_color,
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        out,
        "TARGET RECTIFIED",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (70, 230, 120),
        3,
        cv2.LINE_AA
    )

    if best is None:
        cv2.putText(
            out,
            "NO ARROW CONFIDENTLY DETECTED",
            (40, 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            (0, 165, 255),
            3,
            cv2.LINE_AA
        )

        if candidates:
            c = candidates[0]

            cv2.putText(
                out,
                f"best rejected: score {c['score']:.2f}  "
                f"change {c['change_fraction']:.2f}  "
                f"dark {c['median_darkening']:.1f}",
                (40, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (225, 225, 225),
                2,
                cv2.LINE_AA
            )

        return out

    x1, y1, x2, y2 = best[
        "line"
    ]

    # Determine outer shaft endpoint.
    if best["r1"] >= best["r2"]:
        outer_end = (x1, y1)
    else:
        outer_end = (x2, y2)

    if impact is None:
        if best["r1"] <= best["r2"]:
            impact = (x1, y1)
        else:
            impact = (x2, y2)

    cv2.line(
        out,
        outer_end,
        impact,
        (0, 0, 255),
        7,
        cv2.LINE_AA
    )

    cv2.circle(
        out,
        impact,
        14,
        (0, 255, 0),
        4
    )

    cv2.putText(
        out,
        "ARROW DETECTED",
        (40, 186),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (0, 220, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        out,
        f"score {best['score']:.2f}  "
        f"change {best['change_fraction']:.2f}  "
        f"dark {best['median_darkening']:.1f}  "
        f"width {best['width']:.0f}px",
        (40, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        2,
        cv2.LINE_AA
    )

    if impact_radius is not None:
        cv2.putText(
            out,
            f"impact radius ~ {impact_radius:.3f} target radius  mode {impact_mode}",
            (40, 268),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            2,
            cv2.LINE_AA
        )

    return out


# ============================================================
# RUNNER
# ============================================================

def process_one(
    reference,
    yellow_ellipse,
    target_mask_reference,
    current_path,
    output_dir,
    target_diameter_cm=60.0
):
    current_path = Path(
        current_path
    )

    current = prepare_image(
        current_path
    )

    reference_warped, marker_info = aruco_register_reference_to_current(
        reference,
        current,
        min_markers=3
    )

    safe_name = current_path.stem

    if reference_warped is None:
        print()
        print(
            f"[RETAKE] {current_path.name}"
        )

        print(
            f"         markers: "
            f"{marker_info['marker_count']}/4 "
            f"{marker_info['marker_ids']}"
        )

        return {
            "file": str(
                current_path
            ),
            "status": "RETAKE",
            "markers": marker_info[
                "marker_count"
            ],
        }

    H_ref_to_current = marker_info[
        "H_ref_to_current"
    ]

    H_current_to_ref = marker_info[
        "H_current_to_ref"
    ]

    target_mask_current = warp_reference_mask_to_current(
        target_mask_reference,
        H_ref_to_current,
        current["proc"].shape
    )

    change_mask, darkening, change_threshold = make_change_map(
        reference_warped,
        current["proc"],
        target_mask_current
    )

    def radius_function(
        point
    ):
        point_ref = transform_point_homography(
            H_current_to_ref,
            point
        )

        return normalized_radius(
            point_ref,
            yellow_ellipse
        )

    best, candidates, edges = detect_arrow_current_frame(
        current["proc"],
        change_mask,
        darkening,
        target_mask_current,
        radius_function
    )

    impact = None
    impact_radius = None
    impact_debug = None
    black_edge_debug = None
    far_black_debug = None
    impact_mode = "NORMAL"
    impact_reference = None
    shot_coordinates = None
    shot_confidence = None

    if best is not None:
        (
            impact,
            impact_radius,
            impact_debug
        ) = refine_impact(
            best,
            reference_warped,
            current["proc"],
            yellow_ellipse,
            change_mask,
            target_mask_current,
            radius_function
        )

        (
            impact,
            impact_radius,
            black_edge_debug
        ) = refine_black_ring_with_dual_edges(
            best,
            current["proc"],
            target_mask_current,
            radius_function,
            impact,
            impact_radius
        )

        if (
            black_edge_debug
            and black_edge_debug.get("used")
        ):
            impact_mode = "BLACK_EDGE"

        # V0.4.3: if the dual-edge fallback was rejected and the normal
        # tracker overshot a cross-center shot to the outer black edge,
        # use the first substantial far-side black disappearance.
        if impact_mode == "NORMAL":
            (
                impact,
                impact_radius,
                far_black_debug
            ) = refine_far_side_black_gap(
                best,
                impact_debug,
                impact,
                impact_radius,
                radius_function
            )

            if (
                far_black_debug
                and far_black_debug.get("used")
            ):
                impact_mode = "FAR_BLACK_GAP"

        # V0.5: map the final current-frame impact back to the stable
        # reference target plane, then calculate real target coordinates.
        impact_reference = transform_point_homography(
            H_current_to_ref,
            impact
        )

        shot_coordinates = target_plane_coordinates(
            impact_reference,
            yellow_ellipse,
            target_diameter_cm=target_diameter_cm
        )

        shot_confidence = coordinate_confidence(
            marker_info["marker_count"],
            best["score"],
            impact_mode,
            shot_coordinates["radial_norm"]
        )

    # --------------------------------------------------------
    # Current-frame result drawing.
    # --------------------------------------------------------
    result = current[
        "proc"
    ].copy()

    contours, _ = cv2.findContours(
        target_mask_current,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            result,
            [largest],
            -1,
            (0, 220, 0),
            4
        )

    ref_center = yellow_ellipse[
        0
    ]

    current_center = transform_point_homography(
        H_ref_to_current,
        ref_center
    )

    cv2.drawMarker(
        result,
        (
            int(round(
                current_center[0]
            )),
            int(round(
                current_center[1]
            ))
        ),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        30,
        3
    )

    for c in candidates[:4]:
        x1, y1, x2, y2 = c[
            "line"
        ]

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (150, 150, 150),
            2
        )

    cv2.rectangle(
        result,
        (20, 20),
        (1140, 430),
        (10, 10, 10),
        -1
    )

    count = marker_info[
        "marker_count"
    ]

    ids = marker_info[
        "marker_ids"
    ]

    cv2.putText(
        result,
        "ESS-ARCHERY VISION V0.5",
        (40, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        f"ARUCO {count}/4  IDs {ids}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (
            (70, 230, 120)
            if count == 4
            else (0, 190, 255)
        ),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        "REFERENCE WARPED TO CURRENT VIEW",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (70, 230, 120),
        2,
        cv2.LINE_AA
    )

    if best is None:
        cv2.putText(
            result,
            "NO ARROW CONFIDENTLY DETECTED",
            (40, 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            (0, 165, 255),
            3,
            cv2.LINE_AA
        )

        if candidates:
            c = candidates[0]

            cv2.putText(
                result,
                f"best rejected: score {c['score']:.2f}  "
                f"change {c['change_fraction']:.2f}  "
                f"dark {c['median_darkening']:.1f}",
                (40, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (225, 225, 225),
                2,
                cv2.LINE_AA
            )

        status = "NO_ARROW"

    else:
        x1, y1, x2, y2 = best[
            "line"
        ]

        if best["r1"] >= best["r2"]:
            rear_end = (
                x1,
                y1
            )
        else:
            rear_end = (
                x2,
                y2
            )

        cv2.line(
            result,
            rear_end,
            impact,
            (0, 0, 255),
            7,
            cv2.LINE_AA
        )

        cv2.circle(
            result,
            impact,
            14,
            (0, 255, 0),
            4
        )

        cv2.putText(
            result,
            "ARROW DETECTED",
            (40, 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (0, 220, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            result,
            f"score {best['score']:.2f}  "
            f"change {best['change_fraction']:.2f}  "
            f"dark {best['median_darkening']:.1f}  "
            f"width {best['width']:.0f}px",
            (40, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            result,
            f"impact radius ~ {impact_radius:.3f} target radius  mode {impact_mode}",
            (40, 268),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            2,
            cv2.LINE_AA
        )

        if shot_coordinates is not None:
            x_cm = shot_coordinates["x_cm"]
            y_cm = shot_coordinates["y_cm"]
            radial_cm = shot_coordinates["radial_cm"]
            ring = shot_coordinates["auto_ring"]

            cv2.putText(
                result,
                f"X {x_cm:+.2f} cm    Y {y_cm:+.2f} cm",
                (40, 316),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                result,
                f"RADIAL {radial_cm:.2f} cm    AUTO RING {ring}",
                (40, 356),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 220, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                result,
                f"VISION {shot_confidence}    TARGET {target_diameter_cm:.0f} cm",
                (40, 396),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (70, 230, 120),
                2,
                cv2.LINE_AA
            )

        status = "ARROW"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{safe_name}_01_current.jpg"
        ),
        current["proc"]
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{safe_name}_02_reference_warped.jpg"
        ),
        reference_warped
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{safe_name}_03_change_mask.png"
        ),
        change_mask
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{safe_name}_04_result.jpg"
        ),
        result
    )

    if (
        best is not None
        and shot_coordinates is not None
        and impact_reference is not None
    ):
        save_shot_result_json(
            output_dir / f"{safe_name}_07_shot_result.json",
            current_path.name,
            impact,
            impact_reference,
            shot_coordinates,
            marker_info["marker_count"],
            best["score"],
            impact_mode,
            shot_confidence
        )

    if impact_debug is not None:
        profile = draw_impact_profile(
            impact_debug
        )

        cv2.imwrite(
            str(
                output_dir
                / f"{safe_name}_05_impact_profile.png"
            ),
            profile
        )

    if (
        black_edge_debug is not None
        and "edge_evidence" in black_edge_debug
    ):
        black_profile = draw_black_edge_profile(
            black_edge_debug
        )

        cv2.imwrite(
            str(
                output_dir
                / f"{safe_name}_06_black_edge_profile.png"
            ),
            black_profile
        )

    print()
    print(
        f"[IMAGE] {current_path.name}"
    )

    print(
        f"        markers       : "
        f"{marker_info['marker_count']}/4 "
        f"{marker_info['marker_ids']} "
        f"({marker_info['status']})"
    )

    print(
        f"        homography    : "
        f"{marker_info['inliers']}/"
        f"{marker_info['point_count']} inliers"
    )

    if best is None:
        print(
            "        arrow         : NO CONFIDENT DETECTION"
        )

        if candidates:
            c = candidates[0]

            print(
                f"        best rejected : "
                f"score={c['score']:.3f}, "
                f"change={c['change_fraction']:.3f}, "
                f"dark={c['median_darkening']:.1f}, "
                f"r={c['inner']:.3f}->{c['outer']:.3f}"
            )

    else:
        print(
            "        arrow         : DETECTED"
        )

        print(
            f"        score         : "
            f"{best['score']:.3f}"
        )

        print(
            f"        change support: "
            f"{best['change_fraction']:.3f}"
        )

        print(
            f"        darkening     : "
            f"{best['median_darkening']:.1f}"
        )

        print(
            f"        target cross  : "
            f"{best['inner']:.3f} -> "
            f"{best['outer']:.3f}"
        )

        print(
            f"        impact px     : "
            f"{impact}"
        )

        print(
            f"        impact radial : "
            f"{impact_radius:.3f}"
        )

        print(
            f"        impact mode   : "
            f"{impact_mode}"
        )

        if shot_coordinates is not None:
            print(
                f"        X cm          : "
                f"{shot_coordinates['x_cm']:+.2f}"
            )

            print(
                f"        Y cm          : "
                f"{shot_coordinates['y_cm']:+.2f}"
            )

            print(
                f"        radial cm     : "
                f"{shot_coordinates['radial_cm']:.2f}"
            )

            print(
                f"        auto ring     : "
                f"{shot_coordinates['auto_ring']}"
            )

            print(
                f"        vision conf   : "
                f"{shot_confidence}"
            )

    return {
        "file": str(
            current_path
        ),
        "status": status,
        "markers": marker_info[
            "marker_count"
        ],
        "score": (
            best["score"]
            if best is not None
            else None
        ),
        "impact": impact,
        "impact_radius": impact_radius,
        "impact_mode": impact_mode,
        "impact_reference": impact_reference,
        "coordinates": shot_coordinates,
        "vision_confidence": shot_confidence,
    }


def find_images(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    return sorted(
        p
        for p in Path(folder).rglob("*")
        if p.suffix.lower()
        in extensions
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ESS-ARCHERY Vision V0.5 "
            "ArUco arrow detector + automatic shot coordinates"
        )
    )

    parser.add_argument(
        "reference",
        help=(
            "Clean reference target image "
            "with no current arrow"
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Current shot image OR folder "
            "of images"
        )
    )

    parser.add_argument(
        "--output",
        default="output_v05",
        help="Output directory"
    )

    parser.add_argument(
        "--target-diameter-cm",
        type=float,
        default=60.0,
        help=(
            "Physical target-face diameter in cm "
            "(default: 60)"
        )
    )

    args = parser.parse_args()

    reference_path = Path(
        args.reference
    )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    print(
        "Loading reference..."
    )

    reference = prepare_image(
        reference_path
    )

    print(
        f"Reference markers: "
        f"{len(reference['markers'])}/4 "
        f"{reference['marker_ids']}"
    )

    if len(reference["markers"]) < 4:
        raise SystemExit(
            "Reference must contain all 4 ArUco markers "
            "(IDs 0,1,2,3). Use a clearer reference image."
        )

    yellow = detect_yellow_ellipse(
        reference["proc"]
    )

    if yellow is None:
        raise SystemExit(
            "Yellow target reference not detected "
            "in the reference image."
        )

    yellow_ellipse = yellow[
        "ellipse"
    ]

    target_mask = make_target_mask(
        reference["proc"].shape,
        yellow_ellipse,
        scale_factor=5.12
    )

    # Save reference marker verification.
    ref_overlay = draw_marker_overlay(
        reference["proc"],
        reference["markers"]
    )

    draw_target_reference(
        ref_overlay,
        yellow_ellipse
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            output_dir
            / "00_reference_check.jpg"
        ),
        ref_overlay
    )

    if input_path.is_file():
        process_one(
            reference,
            yellow_ellipse,
            target_mask,
            input_path,
            output_dir,
            target_diameter_cm=args.target_diameter_cm
        )

        return

    if input_path.is_dir():
        images = find_images(
            input_path
        )

        print(
            f"Found {len(images)} image(s)."
        )

        results = []

        for i, path in enumerate(
            images,
            1
        ):
            print()
            print(
                f"========== "
                f"[{i}/{len(images)}] "
                f"=========="
            )

            results.append(
                process_one(
                    reference,
                    yellow_ellipse,
                    target_mask,
                    path,
                    output_dir,
                    target_diameter_cm=args.target_diameter_cm
                )
            )

        arrow_count = sum(
            r["status"] == "ARROW"
            for r in results
        )

        no_arrow_count = sum(
            r["status"] == "NO_ARROW"
            for r in results
        )

        retake_count = sum(
            r["status"] == "RETAKE"
            for r in results
        )

        print()
        print(
            "================================="
        )

        print(
            f"ARROW detections : {arrow_count}"
        )

        print(
            f"NO ARROW         : {no_arrow_count}"
        )

        print(
            f"RETAKE (<3 IDs)  : {retake_count}"
        )

        print(
            "================================="
        )

        return

    raise SystemExit(
        f"Input does not exist: {input_path}"
    )


if __name__ == "__main__":
    main()
