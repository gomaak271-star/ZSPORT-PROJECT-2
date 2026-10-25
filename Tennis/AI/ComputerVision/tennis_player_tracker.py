"""ZSport tennis player tracking prototype (v2, fully automatic).

Outputs (same names and same columns as v1):
- player_coordinates.csv
- player_coordinates.json
- trajectory_heatmap.png (top-down court heatmap)
- tracking_debug.avi

What changed vs v1:
- Court calibration uses several frames (median of the Hough estimates), not frame 1 only.
- Positions behind the baseline / outside the doubles lines are KEPT (v1 dropped them).
- The target is chosen by where the feet land on the court (near half), not only by pixel size.
- If the target is lost longer than MAX_LOST_SECONDS (e.g. a changeover), it is re-acquired
  automatically instead of staying lost for the rest of the video.
- Foot position is smoothed (EMA) before speed / trajectory, so speed is much less noisy.
- Debug video no longer darkens the frame when drawing the trajectory.

Important: this is a robust prototype, not a guarantee of 100% identity accuracy.
"""

import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. USER SETTINGS
# ============================================================
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "/home/alihany/Desktop/ZSport/Tennis/AI/Data/tenis_backview/video2.mp4"

# Used only when the court mapping is unavailable (pixel-based fallback selection).
AUTO_INIT_MIN_Y_RATIO = 0.40

TRACKER_CONFIG = "bytetrack.yaml"
PERSON_CLASS_ID = 0  # COCO person class
DETECTION_CONFIDENCE = 0.25   # minimum confidence for a detection to be used by this script
TRACKER_LOW_CONFIDENCE = 0.10  # passed to the tracker so ByteTrack can recover weak detections
IMAGE_SIZE = 960

# Re-identification guardrails (heuristics; evaluate on your real video).
MAX_LOST_SECONDS = 3.0
MAX_REID_HIST_DISTANCE = 0.62       # Bhattacharyya distance: 0 is most similar
MAX_DIRECT_HIST_DISTANCE = 0.78
MIN_REID_SCORE = 0.42
MIN_REID_SCORE_MARGIN = 0.08
MIN_SIZE_RATIO = 0.35
MAX_SIZE_RATIO = 2.80
HISTOGRAM_UPDATE_ALPHA = 0.02
HISTOGRAM_UPDATE_MAX_DISTANCE = 0.45

# After this many seconds without the target, pick the near-court player again.
REACQUIRE_AFTER_SECONDS = MAX_LOST_SECONDS

# Smoothing of the foot point used for speed and trajectory (1.0 = no smoothing).
FOOT_SMOOTHING_ALPHA = 0.45

# Court geometry is always estimated automatically; no manual calibration window.
CALIBRATION_SAMPLE_FRAMES = 6        # frames sampled for the court estimate
CALIBRATION_WINDOW_SECONDS = 5.0     # sampled from the first N seconds
COURT_WIDTH_METERS = 10.97   # Doubles court width
COURT_LENGTH_METERS = 23.77  # Standard tennis court length
COURT_PLAYABLE_WIDTH_PX = 720
COURT_PLAYABLE_HEIGHT_PX = int(round(
    COURT_PLAYABLE_WIDTH_PX * COURT_LENGTH_METERS / COURT_WIDTH_METERS
))
COURT_MARGIN_PX = 260  # ~4 m of free space around the court (players stand 1-3 m behind the baseline)
REFERENCE_WIDTH_PX = 1280.0  # pixel thresholds below are tuned for this width and scaled by width/1280
SIDELINE_OUTER_BAND = 0.03   # keep only the outermost sideline cluster (doubles line, not singles)
COURT_CANVAS_WIDTH = COURT_PLAYABLE_WIDTH_PX + 2 * COURT_MARGIN_PX
COURT_CANVAS_HEIGHT = COURT_PLAYABLE_HEIGHT_PX + 2 * COURT_MARGIN_PX
COURT_HEATMAP_RADIUS = 22
TRAJECTORY_THICKNESS = 2
OUTPUT_ROOT = "COORDS_OUT"

FALLBACK_METHOD = "automatic rear-view fallback (approximate; check court overlay)"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def extract_torso_histogram(frame_bgr, box):
    """Return an L1-normalized HSV histogram for the central torso area."""
    image_h, image_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    if box_w < 8 or box_h < 16:
        return None

    crop_x1 = max(0, min(image_w, int(x1 + box_w * 0.20)))
    crop_x2 = max(0, min(image_w, int(x2 - box_w * 0.20)))
    crop_y1 = max(0, min(image_h, int(y1 + box_h * 0.12)))
    crop_y2 = max(0, min(image_h, int(y1 + box_h * 0.62)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0:
        return None
    hist = hist / total
    return hist.astype(np.float32)


def histogram_distance(hist_a, hist_b):
    """Return Bhattacharyya distance, or None if either histogram is missing."""
    if hist_a is None or hist_b is None:
        return None
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def foot_point(box):
    """Approximate ground-contact point: center of the lower box edge."""
    x1, y1, x2, y2 = map(float, box)
    return (int(round((x1 + x2) / 2.0)), int(round(y2)))


def box_area(box):
    x1, y1, x2, y2 = map(float, box)
    return max(1.0, x2 - x1) * max(1.0, y2 - y1)


def extract_detections(result, frame):
    """Convert Ultralytics tracked person boxes to a consistent Python list."""
    detections = []
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return detections

    xyxy = boxes_obj.xyxy.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else np.ones(len(xyxy))
    ids_tensor = boxes_obj.id
    ids = ids_tensor.cpu().numpy() if ids_tensor is not None else None

    # A track ID is required for stable selection and recovery in this script.
    if ids is None:
        return detections

    for box, conf, track_id in zip(xyxy, confs, ids):
        if float(conf) < DETECTION_CONFIDENCE:
            continue
        hist = extract_torso_histogram(frame, box)
        detections.append({
            "box": np.asarray(box, dtype=np.float32),
            "id": int(track_id),
            "conf": float(conf),
            "hist": hist,
            "foot": foot_point(box),
            "area": box_area(box),
            "size": (max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1]))),
        })
    return detections


def select_target_detection(detections, frame_shape, homography=None, calibration_ok=False):
    """Pick the target automatically.

    With a trustworthy court mapping: the most prominent person whose feet land in the
    NEAR half of the court (including the area behind the baseline). This avoids picking
    the far-court player, a ball kid or an umpire standing outside the playing area.
    Otherwise: the old pixel heuristic (lower / larger person).
    """
    height, width = frame_shape[:2]
    if not detections:
        return None

    if homography is not None and calibration_ok:
        net_y = COURT_MARGIN_PX + (COURT_PLAYABLE_HEIGHT_PX - 1) / 2.0
        candidates = []
        for det in detections:
            projected = project_to_court_float(det["foot"], homography)
            if projected is None:
                continue
            cx, cy = projected
            in_near_half = net_y <= cy <= COURT_CANVAS_HEIGHT - 1
            in_width = 0 <= cx <= COURT_CANVAS_WIDTH - 1
            if in_near_half and in_width:
                candidates.append(det)
        if candidates:
            return max(candidates, key=lambda d: d["area"] * (0.5 + d["conf"]))

    lower = [d for d in detections if d["box"][3] >= height * AUTO_INIT_MIN_Y_RATIO]
    if not lower:
        return None
    return max(lower, key=lambda d: float(d["box"][3]) * 1.5 + d["area"] * 0.001)


def predicted_foot(last_foot, velocity_per_frame, frames_ahead):
    """Short-horizon linear prediction; limits extrapolation to avoid drift."""
    if last_foot is None:
        return None
    horizon = min(max(0, int(frames_ahead)), 5)
    vx, vy = velocity_per_frame if velocity_per_frame is not None else (0.0, 0.0)
    return np.array([last_foot[0] + vx * horizon, last_foot[1] + vy * horizon], dtype=np.float32)


def add_heat_point(heatmap, point, radius):
    """Accumulate a circular footprint locally so repeated visits add up."""
    x, y = map(int, point)
    height, width = heatmap.shape
    x1, x2 = max(0, x - radius), min(width, x + radius + 1)
    y1, y2 = max(0, y - radius), min(height, y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return

    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    patch = heatmap[y1:y2, x1:x2]
    patch[circle_mask] += 1.0


def choose_reid_candidate(
    detections,
    target_hist,
    last_known_foot,
    last_known_box_size,
    motion_velocity,
    frames_lost,
    frame_width,
    px_scale=1.0,
):
    """Conservative Re-ID: returns a candidate only if the best match is strong and unambiguous."""
    if (
        not detections
        or target_hist is None
        or last_known_foot is None
        or last_known_box_size is None
    ):
        return None

    max_distance = min(
        frame_width * 0.35,
        max(frame_width * 0.07, 80.0 * px_scale * max(1, frames_lost)),
    )
    expected = predicted_foot(last_known_foot, motion_velocity, frames_lost)
    if expected is None:
        return None

    scored = []
    last_w, last_h = last_known_box_size
    for det in detections:
        candidate_hist_distance = histogram_distance(target_hist, det["hist"])
        if candidate_hist_distance is None or candidate_hist_distance > MAX_REID_HIST_DISTANCE:
            continue

        candidate_foot = np.asarray(det["foot"], dtype=np.float32)
        distance = float(np.linalg.norm(candidate_foot - expected))
        if distance > max_distance:
            continue

        candidate_w, candidate_h = det["size"]
        size_ratio = (candidate_w * candidate_h) / max(1.0, last_w * last_h)
        if not (MIN_SIZE_RATIO <= size_ratio <= MAX_SIZE_RATIO):
            continue

        appearance_score = max(0.0, 1.0 - candidate_hist_distance)
        distance_score = max(0.0, 1.0 - distance / max(1.0, max_distance))
        size_score = math.exp(-abs(math.log(max(size_ratio, 1e-6))))
        confidence_score = max(0.0, min(1.0, det["conf"]))
        score = (
            0.55 * appearance_score
            + 0.30 * distance_score
            + 0.10 * size_score
            + 0.05 * confidence_score
        )
        scored.append((score, det))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_detection = scored[0]
    if best_score < MIN_REID_SCORE:
        return None

    # If the first and second candidates are too similar, do not guess.
    if len(scored) > 1 and (best_score - scored[1][0]) < MIN_REID_SCORE_MARGIN:
        return None

    return best_detection


def update_histogram(template_hist, new_hist, alpha):
    if template_hist is None:
        return new_hist.copy() if new_hist is not None else None
    if new_hist is None:
        return template_hist
    updated = cv2.addWeighted(template_hist, 1.0 - alpha, new_hist, alpha, 0.0)
    total = float(updated.sum())
    if total > 0:
        updated /= total
    return updated.astype(np.float32)


# ============================================================
# 2B. TENNIS COURT CALIBRATION AND TOP-DOWN HEATMAP
# ============================================================
def _line_x_at_y(line, y):
    """Evaluate a line represented as x = a*y + b."""
    a, b = line
    return float(a * y + b)


def _fit_line_from_segments(segments):
    """Fit x=a*y+b from Hough line segment endpoints, weighted by segment length."""
    points_y = []
    points_x = []
    weights = []
    for x1, y1, x2, y2, length in segments:
        points_y.extend([float(y1), float(y2)])
        points_x.extend([float(x1), float(x2)])
        weights.extend([math.sqrt(max(1.0, length))] * 2)
    if len(points_y) < 2:
        return None
    try:
        a, b = np.polyfit(
            np.asarray(points_y),
            np.asarray(points_x),
            1,
            w=np.asarray(weights),
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite([a, b]).all():
        return None
    return float(a), float(b)


def _fallback_corners(image_w, image_h):
    return np.asarray(
        [
            [image_w * 0.07, image_h * 0.91],
            [image_w * 0.93, image_h * 0.91],
            [image_w * 0.65, image_h * 0.36],
            [image_w * 0.35, image_h * 0.36],
        ],
        dtype=np.float32,
    )


def estimate_court_quadrilateral(frame):
    """Estimate court corners automatically with Hough lines.

    Works best for a fixed rear-baseline camera where the court forms a trapezoid.
    If the court lines are not detectable, a camera-view prior is used so the
    pipeline can still run unattended (flagged as a fallback in the method string).

    Returns:
        corners: float32 array ordered near-left, near-right, far-right, far-left
        method: human-readable description of the estimate
    """
    image_h, image_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)

    min_line_len = max(35, int(image_w * 0.10))
    max_gap = max(12, int(image_w * 0.025))
    threshold = max(35, int(min(image_w, image_h) * 0.07))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_gap,
    )

    horizontal_segments = []
    left_segments = []
    right_segments = []
    if raw is not None:
        # OpenCV may return (N, 1, 4) or (N, 4); normalize to (N, 4).
        raw_segments = np.asarray(raw).reshape(-1, 4)
        for item in raw_segments:
            x1, y1, x2, y2 = [int(v) for v in item]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_len:
                continue

            # Baselines / net / service lines are approximately horizontal.
            if abs(dy) <= max(5, int(image_h * 0.018)) and abs(dx) >= image_w * 0.12:
                ym = (y1 + y2) / 2.0
                if image_h * 0.12 <= ym <= image_h * 0.99:
                    horizontal_segments.append((min(x1, x2), ym, max(x1, x2), length))
                continue

            # Candidate sidelines need meaningful horizontal and vertical extent.
            if abs(dy) < image_h * 0.12 or abs(dx) < image_w * 0.025:
                continue

            top_y = image_h * 0.36
            bottom_y = image_h * 0.90
            a = dx / float(dy)
            x_top = x1 + (top_y - y1) * a
            x_bottom = x1 + (bottom_y - y1) * a

            if not (-0.10 * image_w <= x_top <= 1.10 * image_w and
                    -0.10 * image_w <= x_bottom <= 1.10 * image_w):
                continue

            # Rear-view perspective: left sideline leans right toward the far end;
            # right sideline leans left.
            if x_bottom < image_w * 0.50 and x_top > x_bottom + image_w * 0.04:
                if x_bottom < image_w * 0.48 and x_top < image_w * 0.78:
                    left_segments.append((x1, y1, x2, y2, length))
            elif x_bottom > image_w * 0.50 and x_top < x_bottom - image_w * 0.04:
                if x_bottom > image_w * 0.52 and x_top > image_w * 0.22:
                    right_segments.append((x1, y1, x2, y2, length))

    # Singles and doubles sidelines are close together: keep only the OUTERMOST cluster
    # so the fit follows the doubles line instead of a line between the two.
    def _x_at_bottom(seg):
        sx1, sy1, sx2, sy2, _ = seg
        if sy2 == sy1:
            return float(sx1)
        return sx1 + (image_h * 0.90 - sy1) * (sx2 - sx1) / float(sy2 - sy1)

    band = image_w * SIDELINE_OUTER_BAND
    if left_segments:
        outer = min(_x_at_bottom(s) for s in left_segments)
        left_segments = [s for s in left_segments if _x_at_bottom(s) <= outer + band]
    if right_segments:
        outer = max(_x_at_bottom(s) for s in right_segments)
        right_segments = [s for s in right_segments if _x_at_bottom(s) >= outer - band]

    left_line = _fit_line_from_segments(left_segments)
    right_line = _fit_line_from_segments(right_segments)

    far_y = image_h * 0.36
    near_y = image_h * 0.90
    corners = None

    if left_line is not None and right_line is not None:
        baseline_rows = []
        for x1, y, x2, length in horizontal_segments:
            left_x = _line_x_at_y(left_line, y)
            right_x = _line_x_at_y(right_line, y)
            predicted_width = right_x - left_x
            if predicted_width <= image_w * 0.12:
                continue
            overlap = max(0.0, min(float(x2), right_x) - max(float(x1), left_x))
            coverage = overlap / predicted_width
            if coverage >= 0.55:
                baseline_rows.append((float(y), float(coverage), float(length)))

        # BOTH baselines must really be detected; otherwise the calibration is rejected
        # (v2 silently kept default rows, which could squash the court length).
        near_found = False
        far_found = False
        if baseline_rows:
            near_candidates = [r for r in baseline_rows if r[0] > image_h * 0.60]
            if near_candidates:
                near_y = max(near_candidates, key=lambda r: r[0])[0]
                near_found = True
                far_candidates = [
                    r for r in baseline_rows
                    if image_h * 0.15 < r[0] < near_y - image_h * 0.18
                ]
                if far_candidates:
                    far_y = min(far_candidates, key=lambda r: r[0])[0]
                    far_found = True
        if not (near_found and far_found):
            left_line = None  # forces the fallback branch below

        if left_line is None:
            return _fallback_corners(image_w, image_h), FALLBACK_METHOD
        x_near_left = _line_x_at_y(left_line, near_y)
        x_far_left = _line_x_at_y(left_line, far_y)
        x_near_right = _line_x_at_y(right_line, near_y)
        x_far_right = _line_x_at_y(right_line, far_y)
        valid_sides = (
            x_near_left < x_near_right
            and x_far_left < x_far_right
            and x_near_left < image_w * 0.48
            and x_near_right > image_w * 0.52
            and (x_near_right - x_near_left) > image_w * 0.30
            and (x_far_right - x_far_left) > image_w * 0.08
            and (x_far_right - x_far_left) < (x_near_right - x_near_left) * 1.15
        )
        valid_rows = (
            image_h * 0.10 < far_y < image_h * 0.72
            and image_h * 0.62 < near_y < image_h * 0.995
            and near_y - far_y > image_h * 0.20
        )
        if valid_sides and valid_rows:
            corners = np.asarray(
                [
                    [x_near_left, near_y],
                    [x_near_right, near_y],
                    [x_far_right, far_y],
                    [x_far_left, far_y],
                ],
                dtype=np.float32,
            )

    method = "automatic Hough-line estimate"
    if corners is None:
        corners = _fallback_corners(image_w, image_h)
        method = FALLBACK_METHOD

    contour = corners.reshape((-1, 1, 2)).astype(np.float32)
    valid = (
        np.isfinite(corners).all()
        and cv2.isContourConvex(contour)
        and abs(cv2.contourArea(contour)) >= image_w * image_h * 0.01
    )
    if not valid:
        corners = _fallback_corners(image_w, image_h)
        method = FALLBACK_METHOD

    corners[:, 0] = np.clip(corners[:, 0], 0, image_w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, image_h - 1)
    return corners, method


def estimate_court_from_frames(frames):
    """Run the court estimate on several frames and take the median of the good ones.

    Returns (corners, method, calibration_ok). calibration_ok is False when only the
    fixed-prior fallback was available (then positions in meters are only approximate).
    """
    good = []
    first_corners, first_method = None, FALLBACK_METHOD
    for frame in frames:
        corners, method = estimate_court_quadrilateral(frame)
        if first_corners is None:
            first_corners, first_method = corners, method
        if method != FALLBACK_METHOD:
            good.append(corners)

    if len(good) >= 2:
        stacked = np.stack(good, axis=0)
        median = np.median(stacked, axis=0).astype(np.float32)
        contour = median.reshape((-1, 1, 2)).astype(np.float32)
        if cv2.isContourConvex(contour):
            return median, f"automatic Hough-line estimate (median of {len(good)} frames)", True
        # Median not convex: keep a good single-frame estimate instead of discarding them all.
        return good[0], "automatic Hough-line estimate (1 frame; median rejected)", True
    if len(good) == 1:
        return good[0], "automatic Hough-line estimate (1 frame)", True
    return first_corners, first_method, False


def build_court_homography(frames):
    """Create the top-down court transform automatically without mouse input."""
    source_points, method, calibration_ok = estimate_court_from_frames(frames)
    m = COURT_MARGIN_PX
    w = COURT_PLAYABLE_WIDTH_PX
    h = COURT_PLAYABLE_HEIGHT_PX
    destination_points = np.asarray(
        [
            [m, m + h - 1],          # near-left
            [m + w - 1, m + h - 1],  # near-right
            [m + w - 1, m],          # far-right
            [m, m],                  # far-left
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_points, destination_points)
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        image_h, image_w = frames[0].shape[:2]
        homography = cv2.getPerspectiveTransform(
            _fallback_corners(image_w, image_h), destination_points
        )
        method = FALLBACK_METHOD
        calibration_ok = False
    return homography, method, calibration_ok


def project_to_court_float(point, homography):
    """Project a video pixel to top-down court-canvas coordinates (float, unclipped)."""
    src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(src, homography)[0, 0]
    if not np.isfinite(projected).all():
        return None
    return float(projected[0]), float(projected[1])


def video_point_to_court(point, homography):
    """Project a video pixel into court-canvas pixels.

    Unlike v1, the whole canvas (court + margin) is accepted, so a player standing
    behind the baseline or outside the doubles lines is still recorded.
    """
    projected = project_to_court_float(point, homography)
    if projected is None:
        return None
    x, y = projected
    if not (0 <= x <= COURT_CANVAS_WIDTH - 1 and 0 <= y <= COURT_CANVAS_HEIGHT - 1):
        return None
    return int(round(x)), int(round(y))


def court_px_to_meters(x_px, y_px):
    """Convert canvas pixels to meters measured from the far-left corner of the doubles court.

    Values can be negative or exceed the court size when the player is outside the lines.
    """
    x_m = (x_px - COURT_MARGIN_PX) / max(1, COURT_PLAYABLE_WIDTH_PX - 1) * COURT_WIDTH_METERS
    y_m = (y_px - COURT_MARGIN_PX) / max(1, COURT_PLAYABLE_HEIGHT_PX - 1) * COURT_LENGTH_METERS
    return round(x_m, 3), round(y_m, 3)


def draw_court_lines(image):
    """Draw a regulation-proportion top-down doubles court, including singles and service lines."""
    m = COURT_MARGIN_PX
    w = COURT_PLAYABLE_WIDTH_PX
    h = COURT_PLAYABLE_HEIGHT_PX
    x_left, x_right = m, m + w - 1
    y_top, y_bottom = m, m + h - 1
    white = (238, 245, 238)
    thickness = 3

    cv2.rectangle(image, (x_left, y_top), (x_right, y_bottom), white, thickness, cv2.LINE_AA)

    singles_width_ratio = 8.23 / COURT_WIDTH_METERS
    singles_half_px = int(round(w * singles_width_ratio / 2.0))
    center_x = (x_left + x_right) // 2
    singles_left = center_x - singles_half_px
    singles_right = center_x + singles_half_px
    cv2.line(image, (singles_left, y_top), (singles_left, y_bottom), white, thickness, cv2.LINE_AA)
    cv2.line(image, (singles_right, y_top), (singles_right, y_bottom), white, thickness, cv2.LINE_AA)

    net_y = (y_top + y_bottom) // 2
    cv2.line(image, (x_left, net_y), (x_right, net_y), (220, 225, 220), 5, cv2.LINE_AA)
    cv2.line(image, (x_left, net_y), (x_right, net_y), white, 2, cv2.LINE_AA)

    service_offset = int(round(h * 6.40 / COURT_LENGTH_METERS))
    service_top = net_y - service_offset
    service_bottom = net_y + service_offset
    cv2.line(image, (singles_left, service_top), (singles_right, service_top), white, thickness, cv2.LINE_AA)
    cv2.line(image, (singles_left, service_bottom), (singles_right, service_bottom), white, thickness, cv2.LINE_AA)
    cv2.line(image, (center_x, service_top), (center_x, service_bottom), white, thickness, cv2.LINE_AA)

    mark_len = max(5, int(round(w * 0.10 / COURT_WIDTH_METERS)))
    cv2.line(image, (center_x, y_top), (center_x, y_top + mark_len), white, thickness, cv2.LINE_AA)
    cv2.line(image, (center_x, y_bottom - mark_len), (center_x, y_bottom), white, thickness, cv2.LINE_AA)
    return image


def save_court_heatmap(heatmap, output_path):
    """Save the accumulated positional heatmap over a clean top-down tennis court.

    The heat layer now covers the whole canvas (court + margin), so time spent behind
    the baseline is visible.
    """
    background = np.full(
        (COURT_CANVAS_HEIGHT, COURT_CANVAS_WIDTH, 3),
        (48, 112, 58), dtype=np.uint8,
    )
    draw_court_lines(background)

    blurred = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=18, sigmaY=18)
    peak = float(blurred.max()) if blurred.size else 0.0
    if peak > 0:
        normalized = np.clip(blurred / peak * 255.0, 0, 255).astype(np.uint8)
        heat_colors = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        mask = blurred > max(0.05, peak * 0.015)
        blended = cv2.addWeighted(background, 0.38, heat_colors, 0.62, 0)
        background[mask] = blended[mask]
        # Redraw court markings on top so they remain visible through the heat layer.
        draw_court_lines(background)

    cv2.putText(background, "ZSport | Player Movement Heatmap", (COURT_MARGIN_PX, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return bool(cv2.imwrite(str(output_path), background))


def verify_video_file(path):
    """Check that the output can be reopened and at least one frame read."""
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    check = cv2.VideoCapture(path)
    try:
        opened = check.isOpened()
        ok, _ = check.read() if opened else (False, None)
        return bool(opened and ok)
    finally:
        check.release()


def sample_calibration_frames(cap, fps, reported_total_frames):
    """Read a few frames from the start of the video for the court estimate, then rewind."""
    frames = []
    if reported_total_frames and reported_total_frames > 1:
        last = min(reported_total_frames - 1, int(fps * CALIBRATION_WINDOW_SECONDS))
        indices = sorted(set(int(i) for i in np.linspace(0, last, CALIBRATION_SAMPLE_FRAMES)))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


# ============================================================
# 3. MAIN PROCESS
# ============================================================
def main():
    video_file = Path(VIDEO_PATH).expanduser()
    if not video_file.is_file():
        raise FileNotFoundError(f"Video not found: {video_file}")

    output_dir = Path(OUTPUT_ROOT) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}" / video_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "player_coordinates.csv"
    json_path = output_dir / "player_coordinates.json"
    heatmap_path = output_dir / "trajectory_heatmap.png"
    debug_video_path = output_dir / "tracking_debug.avi"

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: width={width}, height={height}, fps={fps}")

    writer = None
    heatmap = np.zeros((COURT_CANVAS_HEIGHT, COURT_CANVAS_WIDTH), dtype=np.float32)
    court_homography = None
    calibration_ok = False
    trajectory_layer = np.zeros((height, width, 3), dtype=np.uint8)
    records = []

    # The CSV is written row by row while processing, so a crash in the middle of a long
    # video does not lose everything. JSON and heatmap are saved in the finally block.
    csv_columns = [
        "frame_index", "time_s", "player_id", "tracking_status",
        "pixel_x", "pixel_y", "pixel_position", "speed_pixels_per_sec",
        "confidence", "bounding_box", "court_x_px", "court_y_px",
        "court_x_meters_estimated", "court_y_meters_estimated", "court_mapping_method",
        "foot_clipped",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer_csv = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer_csv.writeheader()
    heatmap_saved = False
    json_saved = False

    px_scale = width / REFERENCE_WIDTH_PX  # keeps pixel thresholds resolution-independent

    frame_index = 0
    target_player_id = None
    target_hist = None
    last_known_foot = None
    last_known_box_size = None
    last_seen_frame_index = None
    motion_velocity = np.array([0.0, 0.0], dtype=np.float32)
    smooth_foot = None       # EMA-smoothed foot point (float array)
    prev_smooth_foot = None  # previous smoothed point, for speed and trajectory
    prev_time_s = None

    tracked_frames_count = 0
    recovered_frames_count = 0
    lost_frames_count = 0
    reid_events_count = 0
    processing_completed = False
    court_calibration_method = "not initialized"

    reacquire_after_frames = max(1, int(round(fps * REACQUIRE_AFTER_SECONDS)))
    max_lost_frames = max(1, int(round(fps * MAX_LOST_SECONDS)))

    try:
        # Court mapping from several early frames (median), then rewind.
        calibration_frames = sample_calibration_frames(cap, fps, reported_total_frames)

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("The video contains no readable frames.")
        if not calibration_frames:
            calibration_frames = [first_frame]

        court_homography, court_calibration_method, calibration_ok = build_court_homography(
            calibration_frames
        )
        print(f"Court mapping: {court_calibration_method}")
        if not calibration_ok:
            print("WARNING: court lines were not detected; meters are only a rough estimate "
                  "and the target is chosen by pixel position.")

        frame_index = 1
        writer = cv2.VideoWriter(
            str(debug_video_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not initialize the AVI output writer.")

        frame = first_frame
        while True:
            current_time_s = (frame_index - 1) / fps
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                classes=[PERSON_CLASS_ID],
                # Low threshold so ByteTrack can use weak detections in its 2nd association
                # stage; detections below DETECTION_CONFIDENCE are still dropped in
                # extract_detections().
                conf=TRACKER_LOW_CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )
            result = results[0] if results else None
            detections = extract_detections(result, frame) if result is not None else []

            best_detection = None
            tracking_status = "Lost"
            court_point = None
            court_x_m = None
            court_y_m = None
            speed_px_s = None
            foot_clipped = None

            if target_player_id is None:
                # Fully automatic initialization; retries on later frames if needed.
                initial = select_target_detection(
                    detections, frame.shape, court_homography, calibration_ok
                )
                if initial is not None:
                    best_detection = initial
                    target_player_id = initial["id"]
                    target_hist = initial["hist"].copy() if initial["hist"] is not None else None
                    tracking_status = "Initialized"
                else:
                    tracking_status = "SearchingTarget"

            else:
                # 1) Prefer the same tracker ID, but reject obvious appearance/distance jumps.
                direct = next((d for d in detections if d["id"] == target_player_id), None)
                frames_since_last = (
                    max(1, frame_index - last_seen_frame_index)
                    if last_seen_frame_index is not None
                    else 1
                )

                if direct is not None:
                    candidate_hist_dist = histogram_distance(target_hist, direct["hist"])
                    direct_ok = True
                    if last_known_foot is not None:
                        expected = predicted_foot(last_known_foot, motion_velocity, frames_since_last)
                        direct_distance = float(np.linalg.norm(
                            np.asarray(direct["foot"], dtype=np.float32) - expected
                        ))
                        direct_max_distance = min(
                            width * 0.35,
                            max(width * 0.07, 80.0 * px_scale * frames_since_last),
                        )
                        if direct_distance > direct_max_distance:
                            direct_ok = False
                    if candidate_hist_dist is not None and candidate_hist_dist > MAX_DIRECT_HIST_DISTANCE:
                        direct_ok = False

                    if direct_ok:
                        best_detection = direct
                        tracking_status = "Tracking"

                # 2) ID missing or inconsistent: conservative Re-ID for short gaps.
                if best_detection is None and last_seen_frame_index is not None:
                    frames_lost = frame_index - last_seen_frame_index
                    if 1 <= frames_lost <= max_lost_frames:
                        recovered = choose_reid_candidate(
                            detections=detections,
                            target_hist=target_hist,
                            last_known_foot=last_known_foot,
                            last_known_box_size=last_known_box_size,
                            motion_velocity=motion_velocity,
                            frames_lost=frames_lost,
                            frame_width=width,
                            px_scale=px_scale,
                        )
                        if recovered is not None:
                            best_detection = recovered
                            target_player_id = recovered["id"]
                            tracking_status = "Recovered"
                            reid_events_count += 1

                    # 3) Lost for a long time (e.g. changeover, leaving the frame):
                    # pick the near-court player again instead of staying lost forever.
                    elif frames_lost >= reacquire_after_frames:
                        reacquired = select_target_detection(
                            detections, frame.shape, court_homography, calibration_ok
                        )
                        if reacquired is not None:
                            best_detection = reacquired
                            target_player_id = reacquired["id"]
                            if reacquired["hist"] is not None:
                                target_hist = reacquired["hist"].copy()
                            tracking_status = "Recovered"
                            reid_events_count += 1

            if best_detection is not None:
                if tracking_status == "Recovered":
                    # Do not compute speed or draw a line across a possible identity change.
                    smooth_foot = None
                    prev_smooth_foot = None
                    prev_time_s = None
                    # The old person's velocity must not predict the new person's position.
                    motion_velocity = np.array([0.0, 0.0], dtype=np.float32)

                tracked_frames_count += 1
                if tracking_status == "Recovered":
                    recovered_frames_count += 1

                box = best_detection["box"]
                x1, y1, x2, y2 = map(int, box)
                this_foot = (int(best_detection["foot"][0]), int(best_detection["foot"][1]))
                this_foot_arr = np.asarray(this_foot, dtype=np.float32)

                # EMA smoothing of the foot point (speed and trajectory use this).
                if smooth_foot is None:
                    smooth_foot = this_foot_arr.copy()
                else:
                    smooth_foot = (
                        FOOT_SMOOTHING_ALPHA * this_foot_arr
                        + (1.0 - FOOT_SMOOTHING_ALPHA) * smooth_foot
                    )

                if prev_smooth_foot is not None and prev_time_s is not None:
                    delta_time = current_time_s - prev_time_s
                    if delta_time > 0:
                        step = smooth_foot - prev_smooth_foot
                        speed_px_s = round(float(np.linalg.norm(step)) / delta_time, 2)
                        measured_velocity = step / max(1.0, delta_time * fps)
                        max_velocity = max(8.0 * px_scale, width * 0.15)
                        speed_vec = float(np.linalg.norm(measured_velocity))
                        if speed_vec > max_velocity:
                            measured_velocity = measured_velocity * (max_velocity / speed_vec)
                        motion_velocity = 0.70 * motion_velocity + 0.30 * measured_velocity

                    cv2.line(
                        trajectory_layer,
                        (int(round(prev_smooth_foot[0])), int(round(prev_smooth_foot[1]))),
                        (int(round(smooth_foot[0])), int(round(smooth_foot[1]))),
                        (0, 165, 255),
                        TRAJECTORY_THICKNESS,
                    )

                prev_smooth_foot = smooth_foot.copy()
                prev_time_s = current_time_s

                last_known_foot = this_foot
                last_known_box_size = best_detection["size"]
                last_seen_frame_index = frame_index

                # Update the appearance template only after a confident, consistent direct track.
                hist_dist_now = histogram_distance(target_hist, best_detection["hist"])
                if (
                    tracking_status in ("Tracking", "Initialized")
                    and best_detection["conf"] >= 0.45
                    and best_detection["hist"] is not None
                    and (hist_dist_now is None or hist_dist_now <= HISTOGRAM_UPDATE_MAX_DISTANCE)
                ):
                    target_hist = update_histogram(
                        target_hist, best_detection["hist"], HISTOGRAM_UPDATE_ALPHA
                    )

                # Court position uses the SMOOTHED foot so meters and speed are consistent.
                foot_clipped = bool(y2 >= height - 2)
                court_point = video_point_to_court(
                    (float(smooth_foot[0]), float(smooth_foot[1])), court_homography
                )
                if court_point is not None:
                    add_heat_point(heatmap, court_point, COURT_HEATMAP_RADIUS)
                    court_x_m, court_y_m = court_px_to_meters(court_point[0], court_point[1])

                color = (0, 255, 0) if tracking_status in ("Tracking", "Initialized") else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                speed_label = f"{speed_px_s:.2f} px/s" if speed_px_s is not None else "N/A"
                label = f"Target ID: {target_player_id} | {tracking_status} | {speed_label}"
                cv2.putText(
                    frame, label, (x1, max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
                )
                cv2.circle(frame, this_foot, 5, (0, 0, 255), -1)
            else:
                lost_frames_count += 1
                smooth_foot = None
                prev_smooth_foot = None
                prev_time_s = None
                status_text = (
                    "Searching for a player automatically..."
                    if target_player_id is None
                    else f"Target ID: {target_player_id} | LOST / identity uncertain"
                )
                cv2.putText(
                    frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # Draw the trajectory on top without darkening or tinting the video.
            output_frame = frame.copy()
            trail_mask = trajectory_layer.any(axis=2)
            output_frame[trail_mask] = trajectory_layer[trail_mask]
            writer.write(output_frame)

            record = {
                "frame_index": frame_index,
                "time_s": round(current_time_s, 4),
                "player_id": best_detection["id"] if best_detection is not None else None,
                "tracking_status": tracking_status,
                "pixel_x": int(best_detection["foot"][0]) if best_detection is not None else None,
                "pixel_y": int(best_detection["foot"][1]) if best_detection is not None else None,
                "pixel_position": list(best_detection["foot"]) if best_detection is not None else None,
                "speed_pixels_per_sec": speed_px_s if best_detection is not None else None,
                "confidence": round(best_detection["conf"], 4) if best_detection is not None else None,
                "bounding_box": [int(v) for v in best_detection["box"]] if best_detection is not None else None,
                "court_x_px": int(court_point[0]) if court_point is not None else None,
                "court_y_px": int(court_point[1]) if court_point is not None else None,
                "court_x_meters_estimated": court_x_m,
                "court_y_meters_estimated": court_y_m,
                "court_mapping_method": court_calibration_method,
                # True when the box touches the bottom edge: the real feet may be below the image.
                "foot_clipped": foot_clipped,
            }
            records.append(record)
            row = record.copy()
            row["pixel_position"] = json.dumps(row["pixel_position"]) if row["pixel_position"] is not None else ""
            row["bounding_box"] = json.dumps(row["bounding_box"]) if row["bounding_box"] is not None else ""
            writer_csv.writerow(row)
            if frame_index % 100 == 0:
                csv_file.flush()

            ret, next_frame = cap.read()
            if not ret:
                break
            frame_index += 1
            frame = next_frame

        processing_completed = True

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        csv_file.close()

        # Save whatever was processed even if something failed mid-video.
        try:
            heatmap_saved = save_court_heatmap(heatmap, heatmap_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            json_saved = json_path.exists() and json_path.stat().st_size > 0
        except Exception as save_exc:  # keep the original error visible, if any
            print(f"WARNING: could not save heatmap/JSON: {save_exc}", file=sys.stderr)

    # ========================================================
    # 4. VERIFY OUTPUTS AND REPORT
    # ========================================================
    csv_saved = csv_path.exists() and csv_path.stat().st_size > 0
    video_saved = verify_video_file(str(debug_video_path))

    total_frames_processed = len(records)
    track_coverage = (tracked_frames_count / total_frames_processed * 100.0) if total_frames_processed else 0.0
    lost_coverage = (lost_frames_count / total_frames_processed * 100.0) if total_frames_processed else 0.0

    print("\n========== ZSport tracking report ==========")
    print(f"Video: {video_file}")
    print(f"Court mapping method: {court_calibration_method}")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Frames with selected target: {tracked_frames_count} ({track_coverage:.2f}%)")
    print(f"Frames marked Lost / uncertain: {lost_frames_count} ({lost_coverage:.2f}%)")
    print(f"Re-ID / re-acquire events: {reid_events_count}")
    print("Note: coverage measures detection continuity, not identity accuracy.")
    print("Speed values are pixels/second, NOT meters/second.")
    print("\nOutput checks:")
    print(f"Heatmap: {'OK' if heatmap_saved else 'FAILED'} - {heatmap_path}")
    print(f"JSON:    {'OK' if json_saved else 'FAILED'} - {json_path}")
    print(f"CSV:     {'OK' if csv_saved else 'FAILED'} - {csv_path}")
    print(f"Video:   {'OK' if video_saved else 'FAILED'} - {debug_video_path}")

    if processing_completed and heatmap_saved and json_saved and csv_saved and video_saved:
        print(f"\nCompleted. Output directory:\n{output_dir.resolve()}")
    else:
        print("\nWARNING: processing or one or more output checks did not complete successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as exc:
        import traceback
        print(f"\nERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
