from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _create_angles_dict(pt: tuple[float, list[int]], mt: tuple[float, list[int]], tl: tuple[float, list[int]]) -> dict:
    """Build API-compatible angle dictionary."""
    return {
        "pt": {
            "angle": float(pt[0]),
            "idxs": [int(pt[1][0]), int(pt[1][1])],
        },
        "mt": {
            "angle": float(mt[0]),
            "idxs": [int(mt[1][0]), int(mt[1][1])],
        },
        "tl": {
            "angle": float(tl[0]),
            "idxs": [int(tl[1][0]), int(tl[1][1])],
        },
    }


def _is_s_curve(midpoints: list[list[float]]) -> bool:
    """Return True if the midpoint trend indicates an S-curve."""
    num = len(midpoints)
    if num < 3:
        return False

    denom_y = midpoints[0][1] - midpoints[num - 1][1]
    denom_x = midpoints[0][0] - midpoints[num - 1][0]
    if denom_y == 0 or denom_x == 0:
        return False

    ll = np.zeros((num - 2, 1), dtype=float)
    for i in range(num - 2):
        ll[i] = (
            (midpoints[i][1] - midpoints[num - 1][1]) / denom_y
            - (midpoints[i][0] - midpoints[num - 1][0]) / denom_x
        )

    quadratic = np.dot(ll, ll.T)
    return float(np.sum(quadratic)) != float(np.sum(np.abs(quadratic)))


def _safe_angle_deg(dot_value: float, norm_a: float, norm_b: float) -> float:
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    value = np.clip(dot_value / denom, -1.0, 1.0)
    return float(np.arccos(value) * 180.0 / math.pi)


def cobb_angle_cal(landmark_xy: Iterable[float], image_shape: tuple[int, ...]):
    """
    Calculate Cobb angles and metadata.

    Args:
        landmark_xy: [x1, x2, ..., xn, y1, y2, ..., yn]
        image_shape: Image shape tuple where image_shape[0] is height.

    Returns:
        (cobb_angles_list, angles_with_pos, curve_type, midpoint_lines)
    """
    landmark_xy = [float(v) for v in landmark_xy]

    if len(landmark_xy) % 2 != 0:
        raise ValueError("landmark_xy length must be even.")

    ap_num = len(landmark_xy) // 2
    if ap_num < 8:
        raise ValueError("At least 8 points (2 vertebrae x 4 keypoints) are required.")

    vnum = ap_num // 4
    if vnum < 2:
        raise ValueError("At least 2 vertebrae are required for Cobb angle calculation.")

    first_half = landmark_xy[:ap_num]
    second_half = landmark_xy[ap_num:]

    cob_angles = np.zeros(3, dtype=float)
    angles_with_pos: dict = {}
    curve_type = None

    # Midpoints (2 points per vertebra pair)
    mid_p_v: list[list[float]] = []
    for i in range(len(landmark_xy) // 4):
        x = first_half[2 * i: 2 * i + 2]
        y = second_half[2 * i: 2 * i + 2]
        if len(x) < 2 or len(y) < 2:
            continue
        mid_p_v.append([(x[0] + x[1]) / 2.0, (y[0] + y[1]) / 2.0])

    # Midpoint lines used by frontend drawing and angle references
    mid_p: list[list[float]] = []
    for i in range(vnum):
        x = first_half[4 * i: 4 * i + 4]
        y = second_half[4 * i: 4 * i + 4]
        if len(x) < 4 or len(y) < 4:
            continue
        point1 = [(x[0] + x[2]) / 2.0, (y[0] + y[2]) / 2.0]
        point2 = [(x[3] + x[1]) / 2.0, (y[3] + y[1]) / 2.0]
        mid_p.extend([point1, point2])

    if len(mid_p) < 4:
        raise ValueError("Insufficient midpoint pairs to compute Cobb angles.")

    vec_m = []
    for i in range(len(mid_p) // 2):
        p1, p2 = mid_p[2 * i], mid_p[2 * i + 1]
        vec_m.append([p2[0] - p1[0], p2[1] - p1[1]])

    vec_m_np = np.asarray(vec_m, dtype=float)
    mod_v = np.linalg.norm(vec_m_np, axis=1)
    dot_v = np.dot(vec_m_np, vec_m_np.T)

    denom = np.outer(mod_v, mod_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_angles = np.divide(dot_v, denom, where=denom != 0)
        cos_angles[denom == 0] = 1.0
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    maxt = np.amax(angles, axis=0)
    pos1 = np.argmax(angles, axis=0)

    pt_rad = float(np.amax(maxt))
    pos2 = int(np.argmax(maxt))
    ref_idx = int(pos1[pos2])

    pt = pt_rad * 180.0 / math.pi
    cob_angles[0] = pt

    if not _is_s_curve(mid_p_v):
        mt = _safe_angle_deg(
            float(np.dot(vec_m_np[0], vec_m_np[pos2])),
            float(np.linalg.norm(vec_m_np[0])),
            float(np.linalg.norm(vec_m_np[pos2])),
        )
        tl = _safe_angle_deg(
            float(np.dot(vec_m_np[vnum - 1], vec_m_np[ref_idx])),
            float(np.linalg.norm(vec_m_np[vnum - 1])),
            float(np.linalg.norm(vec_m_np[ref_idx])),
        )

        cob_angles[1] = mt
        cob_angles[2] = tl

        # Spine Type C
        angles_with_pos = _create_angles_dict(
            mt=(float(pt), [pos2, ref_idx]),
            pt=(float(mt), [0, pos2]),
            tl=(float(tl), [ref_idx, vnum - 1]),
        )
        curve_type = "C"
    else:
        if (mid_p_v[pos2 * 2][1] + mid_p_v[ref_idx * 2][1]) < image_shape[0]:
            # Upper Cobb angle
            upper_candidates = vec_m_np[0:pos2]
            if len(upper_candidates) == 0:
                raise ValueError("Insufficient upper candidates for S-curve calculation.")

            mod_v_p = float(np.linalg.norm(vec_m_np[pos2]))
            mod_v1 = np.linalg.norm(upper_candidates, axis=1)
            dot_v1 = np.dot(vec_m_np[pos2], upper_candidates.T)

            angles1 = np.arccos(np.clip(dot_v1 / np.clip(mod_v_p * mod_v1, 1e-12, None), -1, 1))
            mt = float(np.amax(angles1) * 180.0 / math.pi)
            pos1_1 = int(np.argmax(angles1, axis=0))
            cob_angles[1] = mt

            # Lower Cobb angle
            lower_candidates = vec_m_np[ref_idx:vnum]
            mod_v_p2 = float(np.linalg.norm(vec_m_np[ref_idx]))
            mod_v2 = np.linalg.norm(lower_candidates, axis=1)
            dot_v2 = np.dot(vec_m_np[ref_idx], lower_candidates.T)

            angles2 = np.arccos(np.clip(dot_v2 / np.clip(mod_v_p2 * mod_v2, 1e-12, None), -1, 1))
            tl = float(np.amax(angles2) * 180.0 / math.pi)
            pos1_2 = int(np.argmax(angles2, axis=0)) + ref_idx - 1
            cob_angles[2] = tl

            # Spine Type S (up and down)
            angles_with_pos = _create_angles_dict(
                mt=(float(pt), [pos2, ref_idx]),
                pt=(float(mt), [pos1_1, pos2]),
                tl=(float(tl), [ref_idx, pos1_2]),
            )
            curve_type = "S"
        else:
            # Upper Cobb angle
            upper_candidates = vec_m_np[0:pos2]
            if len(upper_candidates) == 0:
                raise ValueError("Insufficient upper candidates for S-curve calculation.")

            mod_v_p = float(np.linalg.norm(vec_m_np[pos2]))
            mod_v1 = np.linalg.norm(upper_candidates, axis=1)
            dot_v1 = np.dot(vec_m_np[pos2], upper_candidates.T)

            angles1 = np.arccos(np.clip(dot_v1 / np.clip(mod_v_p * mod_v1, 1e-12, None), -1, 1))
            mt = float(np.amax(angles1) * 180.0 / math.pi)
            pos1_1 = int(np.argmax(angles1, axis=0))
            cob_angles[1] = mt

            # Upper-upper Cobb angle
            upper_upper_candidates = vec_m_np[0:pos1_1 + 1]
            mod_v_p2 = float(np.linalg.norm(vec_m_np[pos1_1]))
            mod_v2 = np.linalg.norm(upper_upper_candidates, axis=1)
            dot_v2 = np.dot(vec_m_np[pos1_1], upper_upper_candidates.T)

            angles2 = np.arccos(np.clip(dot_v2 / np.clip(mod_v_p2 * mod_v2, 1e-12, None), -1, 1))
            tl = float(np.amax(angles2) * 180.0 / math.pi)
            pos1_2 = int(np.argmax(angles2, axis=0))
            cob_angles[2] = tl

            # Spine Type S (up and upper-up)
            angles_with_pos = _create_angles_dict(
                tl=(float(pt), [pos2, ref_idx]),
                mt=(float(mt), [pos1_1, pos2]),
                pt=(float(tl), [pos1_2, pos1_1]),
            )
            curve_type = "S"

    midpoint_lines = []
    for i in range(len(mid_p) // 2):
        midpoint_lines.append([
            [int(mid_p[i * 2][0]), int(mid_p[i * 2][1])],
            [int(mid_p[i * 2 + 1][0]), int(mid_p[i * 2 + 1][1])],
        ])

    cobb_angles_list = [float(c) for c in cob_angles]
    for key in angles_with_pos.keys():
        angles_with_pos[key]["angle"] = float(angles_with_pos[key]["angle"])
        angles_with_pos[key]["idxs"] = [int(i) for i in angles_with_pos[key]["idxs"]]

    return cobb_angles_list, angles_with_pos, curve_type, midpoint_lines


def keypoints_to_landmark_xy(keypoints: Iterable[Iterable[Iterable[float]]]) -> list[float]:
    """Convert keypoints [[[x,y],...], ...] to [x..., y...]."""
    x_points: list[float] = []
    y_points: list[float] = []

    for kps in keypoints:
        for kp in kps:
            x_points.append(float(kp[0]))
            y_points.append(float(kp[1]))

    return x_points + y_points
