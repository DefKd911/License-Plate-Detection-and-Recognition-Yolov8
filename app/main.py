from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.inference import detections_to_dict, get_model, predict_image


app = FastAPI(
    title="License Plate Detection & Recognition API",
    version="1.0.0",
    description="YOLOv8-based license plate detection + EasyOCR.",
)


@app.on_event("startup")
def _warmup() -> None:
    # Fail fast if model weights are missing.
    get_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ocr")
def health_ocr() -> dict[str, Any]:
    """Check if OCR (EasyOCR) is working"""
    from app.inference import easyocr, _easyocr_import_error
    
    if easyocr is None:
        return {
            "ocr_available": False,
            "error": _easyocr_import_error or "EasyOCR not imported",
            "fix_hint": "Run: pip install easyocr"
        }
    
    try:
        # Try to initialize reader (this will download models on first run)
        import easyocr as ocr
        reader = ocr.Reader(['en'], gpu=False)
        return {
            "ocr_available": True,
            "ocr_engine": "EasyOCR",
            "status": "ok",
            "note": "First OCR call will be slower (~2-3s) as models load"
        }
    except Exception as e:
        return {
            "ocr_available": False,
            "error": str(e),
            "status": "easyocr_initialization_failed"
        }


def _read_image_as_bgr(upload: UploadFile, raw: bytes) -> np.ndarray:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold"),
) -> dict[str, Any]:
    raw = await file.read()
    image_bgr = _read_image_as_bgr(file, raw)

    _annotated, detections = predict_image(image_bgr, conf=conf)

    return {
        "count": len(detections),
        "detections": detections_to_dict(detections),
        "texts": [d.text for d in detections if d.text],
    }


@app.post("/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold"),
) -> Response:
    raw = await file.read()
    image_bgr = _read_image_as_bgr(file, raw)

    annotated, _detections = predict_image(image_bgr, conf=conf)

    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
