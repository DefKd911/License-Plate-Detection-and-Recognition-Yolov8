from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

# EasyOCR for license plate text recognition
_easyocr_reader = None
_easyocr_import_error: str | None = None
try:
    import easyocr
except Exception as e:
    easyocr = None  # type: ignore
    _easyocr_import_error = str(e)


@dataclass(frozen=True)
class PlateDetection:
    bbox_xyxy: list[int]  # [x1, y1, x2, y2]
    confidence: float
    text: str


def _clean_text(text: str) -> str:
    """Clean OCR text output."""
    text = text.strip()
    text = text.replace("\n", "").replace("\r", "").replace("\t", " ")
    return " ".join(text.split())


def ocr_plate_text(plate_bgr: np.ndarray) -> str:
    """Extract text from license plate region using EasyOCR."""
    if plate_bgr.size == 0:
        return ""

    if easyocr is None:  # type: ignore
        if _easyocr_import_error:
            import warnings
            warnings.warn(f"OCR disabled: EasyOCR import failed: {_easyocr_import_error}", UserWarning)
        return ""

    # Lazy initialize EasyOCR reader (first call downloads models)
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception as e:
            import warnings
            error_msg = str(e)
            if "decompressing" in error_msg.lower() or "invalid block" in error_msg.lower():
                warnings.warn(
                    f"EasyOCR model corrupted. Clear cache: {os.path.expanduser('~/.EasyOCR/model')}",
                    UserWarning
                )
            else:
                warnings.warn(f"Failed to initialize EasyOCR: {e}", UserWarning)
            return ""

    try:
        results = _easyocr_reader.readtext(plate_bgr)
        if not results:
            return ""
        texts = [text for (_, text, conf) in results if conf >= 0.3]
        return _clean_text(" ".join(texts))
    except Exception as e:
        import warnings
        warnings.warn(f"EasyOCR processing failed: {e}", UserWarning)
        return ""


def _env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.getenv(name)
    return float(val) if val else default


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load YOLOv8 model (cached)."""
    model_path = Path(os.getenv("MODEL_PATH", "kbest.pt"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Set MODEL_PATH env var or place 'kbest.pt' in project root."
        )
    return YOLO(str(model_path))


def predict_image(
    image_bgr: np.ndarray,
    conf: float | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, list[PlateDetection]]:
    """Detect license plates and extract text. Returns annotated image and detections."""
    if conf is None:
        conf = _env_float("YOLO_CONF", 0.25)
    if device is None:
        device = os.getenv("YOLO_DEVICE", "cpu")

    model = get_model()
    results = model.predict(image_bgr, device=device, conf=conf, verbose=False)

    annotated = image_bgr.copy()
    detections: list[PlateDetection] = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])

            h, w = image_bgr.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            plate_crop = image_bgr[y1c:y2c, x1c:x2c]

            text = ocr_plate_text(plate_crop)

            cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            label = f"{text or 'UNKNOWN'} ({confidence * 100:.1f}%)"
            cv2.putText(annotated, label, (x1c, max(0, y1c - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            detections.append(PlateDetection(bbox_xyxy=[x1c, y1c, x2c, y2c], confidence=confidence, text=text))

    return annotated, detections


def detections_to_dict(detections: list[PlateDetection]) -> list[dict[str, Any]]:
    """Convert PlateDetection objects to dictionaries."""
    return [{"bbox_xyxy": d.bbox_xyxy, "confidence": d.confidence, "text": d.text} for d in detections]
