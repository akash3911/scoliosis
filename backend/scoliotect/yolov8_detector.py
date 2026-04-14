from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from scoliotect.cobb_angle_cal import cobb_angle_cal, keypoints_to_landmark_xy


_YOLO_MODEL = None
_KEYPOINTS_PER_VERTEBRA = 4


def _get_model_path() -> Path:
    # backend/scoliotect/yolov8_detector.py -> backend/models/best.pt
    return Path(__file__).resolve().parents[1] / "models" / "best.pt"


def get_yolov8_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    model_path = _get_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    _YOLO_MODEL = YOLO(str(model_path))
    return _YOLO_MODEL


def predict_image_to_api_format(image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> dict[str, Any]:
    model = get_yolov8_model()
    result = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        max_det=18,
        verbose=False,
        device="cpu",
    )[0]

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
    classes = result.boxes.cls.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
    keypoints_xy = (
        result.keypoints.xy.detach().cpu().numpy()
        if getattr(result, "keypoints", None) is not None and getattr(result.keypoints, "xy", None) is not None
        else np.empty((0, 0, 2))
    )

    # Top scores first, then sort by ymin to preserve top-to-bottom vertebra order.
    score_order = np.argsort(-scores)
    boxes_xyxy = boxes_xyxy[score_order][:18]
    scores = scores[score_order][:18]
    classes = classes[score_order][:18]
    if len(keypoints_xy) > 0:
        keypoints_xy = keypoints_xy[score_order][:18]

    if len(boxes_xyxy) > 0:
        y_order = np.argsort(boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[y_order]
        scores = scores[y_order]
        classes = classes[y_order]
        if len(keypoints_xy) > 0:
            keypoints_xy = keypoints_xy[y_order]

    detections = []
    keypoints = []
    landmarks = []

    names = model.names if hasattr(model, "names") else {}

    for idx in range(len(boxes_xyxy)):
        xmin, ymin, xmax, ymax = boxes_xyxy[idx].tolist()
        score = float(scores[idx])
        class_id = int(classes[idx])
        class_name = names.get(class_id, "vert") if isinstance(names, dict) else "vert"

        bbox_int = [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]

        detections.append(
            {
                "class": class_id,
                "confidence": score,
                "name": class_name,
                "xmin": bbox_int[0],
                "ymin": bbox_int[1],
                "xmax": bbox_int[2],
                "ymax": bbox_int[3],
            }
        )

        # Use model keypoints directly for v3 output/calculation.
        if idx < len(keypoints_xy):
            raw_kps = keypoints_xy[idx].tolist()
            kps = [[float(kp[0]), float(kp[1])] for kp in raw_kps[:_KEYPOINTS_PER_VERTEBRA]]
        else:
            kps = []

        if len(kps) > 0:
            keypoints.append(kps)

        for kp in kps:
            landmarks.append(kp[0])
            landmarks.append(kp[1])

    try:
        _, angles, curve_type, midpoint_lines = cobb_angle_cal(keypoints_to_landmark_xy(keypoints), image_bgr.shape)
    except Exception:
        angles = None
        curve_type = None
        midpoint_lines = None

    return {
        "detections": detections,
        "landmarks": landmarks,
        "angles": angles,
        "curve_type": curve_type,
        "midpoint_lines": midpoint_lines,
    }
