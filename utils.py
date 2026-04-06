from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DISPLAY_NAME_MAP = {
    "metal": "Metal Implant",
    "fracture": "Fracture Region",
    "anomaly": "Abnormal Bone Region",
    "soft tissue": "Soft Tissue Finding",
}


def get_display_class_name(class_name: str) -> str:
    key = str(class_name).strip().lower()
    return DISPLAY_NAME_MAP.get(key, class_name)


def save_uploaded_file(uploaded_file, save_dir):
    """Save a Streamlit uploaded file to disk and return the saved file path."""
    if uploaded_file is None:
        raise ValueError("uploaded_file cannot be None.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = Path(getattr(uploaded_file, "name", "uploaded_image")).name
    if not filename:
        filename = "uploaded_image"

    target_path = save_path / filename

    if target_path.exists():
        stem, suffix = target_path.stem, target_path.suffix
        counter = 1
        while True:
            candidate = save_path / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                target_path = candidate
                break
            counter += 1

    try:
        with open(target_path, "wb") as out_file:
            if hasattr(uploaded_file, "getbuffer"):
                out_file.write(uploaded_file.getbuffer())
            elif hasattr(uploaded_file, "read"):
                out_file.write(uploaded_file.read())
            else:
                raise TypeError("uploaded_file must support getbuffer() or read().")
    except Exception as exc:
        raise RuntimeError(f"Failed to save uploaded file to '{target_path}'.") from exc

    return str(target_path)


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize image safely using percentile-based contrast stretching.
    This is much better for darker / high-dynamic-range X-rays.
    """
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if 0.0 <= min_val and max_val <= 1.0:
        img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    low = float(np.percentile(img, 1))
    high = float(np.percentile(img, 99))

    if high <= low:
        low = min_val
        high = max_val

    if high > low:
        img = np.clip(img, low, high)
        img = (img - low) / (high - low)
        img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    return np.zeros_like(img, dtype=np.uint8)


def _to_rgb_uint8(image: Any) -> np.ndarray:
    """Convert PIL/NumPy input to a 3-channel uint8 RGB NumPy array."""
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError("image must be a PIL image or a NumPy array.")

    if img.ndim == 2:
        img = _normalize_to_uint8(img)
        img = np.stack([img, img, img], axis=-1)

    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
        img = _normalize_to_uint8(img)
        img = np.stack([img, img, img], axis=-1)

    elif img.ndim == 3 and img.shape[2] == 3:
        img = _normalize_to_uint8(img)

    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
        img = _normalize_to_uint8(img)

    else:
        raise ValueError("Unsupported image shape. Expected HxW, HxWx1, HxWx3, or HxWx4.")

    return img.copy()


def draw_detections(image, detections):
    """Draw bounding boxes and labels on an image and return an annotated RGB NumPy array."""
    annotated = _to_rgb_uint8(image)

    if not detections:
        return annotated

    height, width = annotated.shape[:2]

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        raw_class_name = str(det.get("class_name", "unknown"))
        class_name = get_display_class_name(raw_class_name)
        confidence = float(det.get("confidence", 0.0))
        label = f"{class_name} {confidence:.2f}"

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        box_color = (255, 70, 70) if confidence >= 0.5 else (255, 180, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        text_y = y1 - 10 if y1 > 25 else y1 + 20
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def generate_summary(detections):
    """Create a simple summary dictionary from detection results."""
    detections = detections or []
    total_detections = len(detections)

    detected_class_names = sorted(
        {get_display_class_name(str(d.get("class_name", "unknown"))) for d in detections}
    )
    highest_confidence = max((float(d.get("confidence", 0.0)) for d in detections), default=0.0)

    if total_detections == 0:
        status_message = "No strong fracture indication found"
    elif highest_confidence >= 0.7:
        status_message = "Possible fracture detected with high confidence"
    elif highest_confidence >= 0.4:
        status_message = "Possible fracture detected with moderate confidence"
    else:
        status_message = "Low-confidence abnormal finding detected"

    return {
        "total_detections": total_detections,
        "detected_class_names": detected_class_names,
        "highest_confidence": round(highest_confidence, 4),
        "status_message": status_message,
    }