from __future__ import annotations

from typing import Any
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO


def _load_model_with_safe_globals(weights_path: str) -> YOLO:
    import torch
    from ultralytics.nn.tasks import DetectionModel

    serialization = getattr(torch, "serialization", None)
    if serialization is None:
        raise RuntimeError("torch.serialization is unavailable in this torch version.")

    allowed_globals = [DetectionModel]

    if hasattr(serialization, "safe_globals"):
        with serialization.safe_globals(allowed_globals):
            return YOLO(weights_path)

    if hasattr(serialization, "add_safe_globals"):
        serialization.add_safe_globals(allowed_globals)
        return YOLO(weights_path)

    raise RuntimeError(
        "This torch version does not support safe_globals/add_safe_globals required for allowlisting."
    )


def load_model(weights_path: str) -> YOLO:
    if not weights_path:
        raise ValueError("weights_path must be a non-empty string.")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found: {weights_path}")

    try:
        return YOLO(weights_path)
    except Exception as exc:
        error_text = str(exc)
        if (
            "Weights only load failed" in error_text
            or "weights_only" in error_text
            or "Unsupported global" in error_text
        ):
            try:
                return _load_model_with_safe_globals(weights_path)
            except Exception as safe_exc:
                raise RuntimeError(
                    "Failed to load YOLO model with default loader. "
                    f"Original error: {exc}. Safe-global retry error: {safe_exc}"
                ) from safe_exc

        raise RuntimeError(f"Failed to load YOLO model from '{weights_path}': {exc}") from exc


def _normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    if image_array.dtype == np.uint8:
        return image_array

    image_array = image_array.astype(np.float32)

    if image_array.size == 0:
        return np.zeros_like(image_array, dtype=np.uint8)

    min_val = float(np.min(image_array))
    max_val = float(np.max(image_array))

    if 0.0 <= min_val and max_val <= 1.0:
        image_array = image_array * 255.0
        return np.clip(image_array, 0, 255).astype(np.uint8)

    low = float(np.percentile(image_array, 1))
    high = float(np.percentile(image_array, 99))

    if high <= low:
        low = min_val
        high = max_val

    if high > low:
        image_array = np.clip(image_array, low, high)
        image_array = (image_array - low) / (high - low)
        image_array = image_array * 255.0
        return np.clip(image_array, 0, 255).astype(np.uint8)

    return np.zeros_like(image_array, dtype=np.uint8)


def prepare_display_image(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        raise TypeError("image must be a PIL image or a NumPy array.")

    if image_array.ndim == 2:
        image_array = _normalize_to_uint8(image_array)
        image_array = np.stack([image_array, image_array, image_array], axis=-1)

    elif image_array.ndim == 3:
        channels = image_array.shape[2]

        if channels == 1:
            image_array = image_array[:, :, 0]
            image_array = _normalize_to_uint8(image_array)
            image_array = np.stack([image_array, image_array, image_array], axis=-1)

        elif channels == 3:
            image_array = _normalize_to_uint8(image_array)

        elif channels == 4:
            image_array = image_array[:, :, :3]
            image_array = _normalize_to_uint8(image_array)

        else:
            raise ValueError("Unsupported channel count in image array.")
    else:
        raise ValueError(
            "Image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    return np.ascontiguousarray(image_array)


def _resolve_class_name(class_id: int, names: Any) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def run_inference(model: YOLO, image: Any, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
    if model is None or not hasattr(model, "predict"):
        raise ValueError("model must be a loaded YOLO model instance.")

    if not isinstance(conf_threshold, (int, float)) or not 0 <= float(conf_threshold) <= 1:
        raise ValueError("conf_threshold must be between 0 and 1.")

    prepared_image = prepare_display_image(image)
    results = model.predict(source=prepared_image, conf=float(conf_threshold), verbose=False)

    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None or cls is None or conf is None:
        return []

    xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    cls_np = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
    conf_np = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

    if xyxy_np.size == 0 or cls_np.size == 0 or conf_np.size == 0:
        return []

    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", None)

    detections: list[dict[str, Any]] = []

    for bbox, class_id_raw, confidence_raw in zip(xyxy_np, cls_np, conf_np):
        class_id = int(class_id_raw)
        confidence = float(confidence_raw)
        x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]

        detections.append(
            {
                "class_id": class_id,
                "class_name": _resolve_class_name(class_id, names),
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections