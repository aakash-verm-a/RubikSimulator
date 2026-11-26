"""
Automatic Rubik's Cube Face Detection & Color Recognition Pipeline.

Replaces fixed-point grid sampling with automatic cube-face detection
anywhere in the webcam frame:
  1. Multi-pass square sticker contour detection (Adaptive Thresholding from
     Color_recog-v2.py + Canny Edge from cube-crop.py).
  2. Size- and proximity-based sticker clustering to reject background noise.
  3. Minimum-area rotated rectangle fit + perspective deskew (warpPerspective)
     so tilted cubes held in front of a webcam are straightened automatically.
  4. Median HSV ROI sampling + Color_recog-v2.py HSV classification with
     nearest-color fallback, plus live overlay polygon coordinates and an
     annotated crop preview.
"""

from __future__ import annotations
import base64
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# Exact HSV ranges from Color_recog-v2.py
RUBIKS_COLOR_RANGES = {
    "R": [
        (np.array([0, 100, 100]), np.array([8, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
    ],
    "O": [(np.array([9, 100, 100]), np.array([22, 255, 255]))],
    "Y": [(np.array([23, 100, 100]), np.array([35, 255, 255]))],
    "G": [(np.array([36, 100, 100]), np.array([85, 255, 255]))],
    "B": [(np.array([90, 100, 100]), np.array([130, 255, 255]))],
    "W": [(np.array([0, 0, 180]), np.array([180, 60, 255]))],
}

PROTOTYPE_HSV = {
    "R": (3.0, 200.0, 190.0),
    "O": (15.0, 210.0, 220.0),
    "Y": (28.0, 200.0, 230.0),
    "G": (62.0, 180.0, 180.0),
    "B": (110.0, 200.0, 180.0),
    "W": (0.0, 20.0, 230.0),
}

COLOR_BGR_DRAW = {
    "R": (60, 60, 240),
    "O": (30, 140, 255),
    "Y": (40, 230, 250),
    "G": (60, 210, 60),
    "B": (240, 110, 40),
    "W": (245, 245, 245),
}


def decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data.")
    return img


def encode_jpg_data_url(image: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def _extract_square_candidates(binary: np.ndarray, img_area: int) -> List[Dict[str, Any]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(180, int(img_area * 0.0008))
    max_area = int(img_area * 0.08)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        # Interior angles check (allow perspective tilt)
        angles = []
        for i in range(4):
            p0 = approx[i][0].astype(np.float32)
            p1 = approx[(i + 1) % 4][0].astype(np.float32)
            p2 = approx[(i + 2) % 4][0].astype(np.float32)
            v1 = p0 - p1
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
            angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
        if not all(62.0 < a < 118.0 for a in angles):
            continue

        rect = cv2.minAreaRect(approx)
        (cx, cy), (rw, rh), _ = rect
        if rw < 10 or rh < 10:
            continue
        aspect = max(rw, rh) / float(min(rw, rh) + 1e-5)
        if aspect > 1.45:
            continue
        # Fill ratio against minAreaRect
        if area / float(rw * rh + 1e-5) < 0.72:
            continue

        side = 0.5 * (rw + rh)
        candidates.append({
            "cx": float(cx),
            "cy": float(cy),
            "side": float(side),
            "area": float(area),
            "pts": approx.reshape(4, 2).astype(np.float32),
        })
    return candidates


def _deduplicate_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate detections of the same sticker from inner/outer edges."""
    if not cands:
        return []
    cands_sorted = sorted(cands, key=lambda c: c["area"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for c in cands_sorted:
        too_close = False
        for k in kept:
            dist = np.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"])
            if dist < 0.55 * min(c["side"], k["side"]):
                too_close = True
                break
        if not too_close:
            kept.append(c)
    return kept


def _find_best_sticker_cluster(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find the largest spatially coherent cluster of similarly-sized square
    stickers (a 3x3 cube face has up to 9 stickers of nearly identical size
    within ~3.6 * sticker_side of each other).
    """
    if len(cands) < 3:
        return []

    best_cluster: List[Dict[str, Any]] = []
    best_score = -1.0

    for seed in cands:
        s_side = seed["side"]
        # Collect neighbors with similar sticker size and within 3.6x side length
        cluster = [
            c for c in cands
            if (0.62 * s_side <= c["side"] <= 1.48 * s_side)
            and (np.hypot(c["cx"] - seed["cx"], c["cy"] - seed["cy"]) <= 3.6 * s_side)
        ]
        if len(cluster) < 3:
            continue

        # Refine around cluster centroid so we don't drift
        mean_x = float(np.mean([c["cx"] for c in cluster]))
        mean_y = float(np.mean([c["cy"] for c in cluster]))
        med_side = float(np.median([c["side"] for c in cluster]))
        cluster = [
            c for c in cluster
            if (0.68 * med_side <= c["side"] <= 1.40 * med_side)
            and (np.hypot(c["cx"] - mean_x, c["cy"] - mean_y) <= 2.35 * med_side)
        ]
        if len(cluster) < 3:
            continue

        # Compute bounding squareness of this cluster
        all_pts = np.vstack([c["pts"] for c in cluster])
        (_, _), (bw, bh), _ = cv2.minAreaRect(all_pts)
        if min(bw, bh) < 1:
            continue
        ratio = max(bw, bh) / float(min(bw, bh))
        if ratio > 1.55:
            continue
        # Expected face width is ~3.1 * sticker_side
        expected_w = 3.15 * med_side
        size_err = abs(0.5 * (bw + bh) - expected_w) / (expected_w + 1e-5)

        score = len(cluster) * 10.0 - ratio * 2.5 - size_err * 4.0
        if len(cluster) <= 9 and score > best_score:
            best_score = score
            best_cluster = cluster

    return best_cluster


def _detect_outer_square_fallback(binary: np.ndarray, img_area: int) -> Optional[np.ndarray]:
    """
    Fallback when individual sticker borders are faint (e.g., stickerless cube):
    find a prominent square contour matching a whole cube face.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(img_area * 0.04)
    max_area = int(img_area * 0.65)
    best_quad = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        (_, _), (rw, rh), _ = rect
        if min(rw, rh) < 50:
            continue
        aspect = max(rw, rh) / float(min(rw, rh) + 1e-5)
        if aspect < 1.3 and area / float(rw * rh + 1e-5) > 0.78:
            if area > best_area:
                best_area = area
                best_quad = order_quad_points(approx)
    return best_quad


def detect_cube_face_quad(image: np.ndarray) -> Tuple[Optional[np.ndarray], str, int]:
    """
    Automatically locate the Rubik's cube face anywhere in the BGR frame.
    Returns (ordered_4x2_quad, method_name, sticker_count).
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Pass 1: Adaptive thresholding (Color_recog-v2.py)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Pass 2: Canny edges + dilation (cube-crop.py)
    canny = cv2.Canny(blurred, 35, 110)
    dilated = cv2.dilate(canny, kernel, iterations=2)

    cands = _extract_square_candidates(closed, img_area) + _extract_square_candidates(dilated, img_area)
    cands = _deduplicate_candidates(cands)
    cluster = _find_best_sticker_cluster(cands)

    if len(cluster) >= 4:
        all_pts = np.vstack([c["pts"] for c in cluster])
        rect = cv2.minAreaRect(all_pts)
        (cx, cy), (rw, rh), angle = rect
        # Expand slightly if partial cluster (e.g. 4-8 stickers) and ensure square-ish face
        side = max(rw, rh)
        med_side = float(np.median([c["side"] for c in cluster]))
        # Ensure total face width is at least ~2.9 * sticker_side when a corner sticker is missed
        if len(cluster) < 9:
            side = max(side, 2.95 * med_side)
        box = cv2.boxPoints(((cx, cy), (side, side), angle))
        quad = order_quad_points(box)
        return quad, f"sticker_cluster ({len(cluster)} stickers)", len(cluster)

    # Fallback: outer square face contour
    outer = _detect_outer_square_fallback(dilated, img_area)
    if outer is not None:
        return outer, "outer_face_contour", 0

    return None, "not_found", 0


def classify_hsv(hsv_value: np.ndarray) -> Tuple[str, bool]:
    """
    Classifies an HSV median value using the exact Color_recog-v2.py rules first,
    then falls back to nearest-color HSV distance for varied webcam lighting.
    """
    h, s, v = float(hsv_value[0]), float(hsv_value[1]), float(hsv_value[2])

    if s < 65 and v > 135:
        return "W", True

    if v >= 50:
        for cname, ranges in RUBIKS_COLOR_RANGES.items():
            if cname == "W":
                continue
            for lower, upper in ranges:
                if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                    return cname, True

    if s < 75 and v > 105:
        return "W", False

    best_color = "W"
    best_dist = float("inf")
    for cname, (ph, ps, pv) in PROTOTYPE_HSV.items():
        if cname == "W":
            continue
        dh = min(abs(h - ph), 180.0 - abs(h - ph))
        ds = abs(s - ps) * 0.25
        dv = abs(v - pv) * 0.15
        dist = dh * 2.5 + ds + dv
        if dist < best_dist:
            best_dist = dist
            best_color = cname
    return best_color, False


def visual_grid_to_arduino_string(grid_3x3: List[str]) -> str:
    top = grid_3x3[0:3]
    mid = grid_3x3[3:6]
    bot = grid_3x3[6:9]
    return "".join(bot + mid + top)


def arduino_string_to_visual_grid(arduino_str: str) -> List[str]:
    bot = list(arduino_str[0:3])
    mid = list(arduino_str[3:6])
    top = list(arduino_str[6:9])
    return top + mid + bot


def detect_face_from_image(
    image: np.ndarray,
    include_preview: bool = True,
) -> Dict[str, Any]:
    """
    Automatically detect the Rubik's cube face anywhere in `image`, perspective-
    warp it to a flat 300x300 square, and sample the 9 sticker colors.
    """
    h_img, w_img = image.shape[:2]
    quad, method, sticker_count = detect_cube_face_quad(image)

    if quad is None:
        return {
            "detected": False,
            "method": method,
            "sticker_count": 0,
            "message": "Cube face not detected. Hold a cube face clearly in view with good contrast.",
        }

    # Perspective-warp the detected cube face to an upright 300x300 square
    out_size = 300
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (out_size, out_size))

    hsv_image = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    cell = out_size // 3
    half_roi = 14

    visual_colors: List[str] = []
    exact_flags: List[bool] = []
    hsv_samples: List[List[int]] = []
    rgb_samples: List[List[int]] = []

    preview_img = warped.copy() if include_preview else None

    for row in range(3):
        for col in range(3):
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            roi_hsv = hsv_image[cy - half_roi : cy + half_roi, cx - half_roi : cx + half_roi]
            roi_bgr = warped[cy - half_roi : cy + half_roi, cx - half_roi : cx + half_roi]

            med_hsv = np.median(roi_hsv, axis=(0, 1))
            med_bgr = np.median(roi_bgr, axis=(0, 1))
            cname, is_exact = classify_hsv(med_hsv)

            visual_colors.append(cname)
            exact_flags.append(is_exact)
            hsv_samples.append([int(round(med_hsv[0])), int(round(med_hsv[1])), int(round(med_hsv[2]))])
            rgb_samples.append([int(round(med_bgr[2])), int(round(med_bgr[1])), int(round(med_bgr[0]))])

            if preview_img is not None:
                cv2.rectangle(
                    preview_img,
                    (col * cell + 3, row * cell + 3),
                    ((col + 1) * cell - 3, (row + 1) * cell - 3),
                    (255, 255, 255),
                    1,
                )
                cv2.circle(preview_img, (cx, cy), 14, (15, 23, 42), -1)
                cv2.circle(preview_img, (cx, cy), 14, COLOR_BGR_DRAW.get(cname, (200, 200, 200)), 2)
                cv2.putText(
                    preview_img,
                    cname,
                    (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    quad_norm = [[round(float(x) / w_img, 4), round(float(y) / h_img, 4)] for x, y in quad]

    result: Dict[str, Any] = {
        "detected": True,
        "method": method,
        "sticker_count": sticker_count,
        "quad_norm": quad_norm,
        "visual_grid": visual_colors,
        "arduino_string": visual_grid_to_arduino_string(visual_colors),
        "exact_matches": exact_flags,
        "hsv_samples": hsv_samples,
        "rgb_samples": rgb_samples,
    }
    if preview_img is not None:
        result["annotated_preview"] = encode_jpg_data_url(preview_img, quality=82)
    return result
