"""Leafwatch web and inference service using the public IP102 model."""
import io
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 12 * 1024 * 1024
MODEL_REPOSITORY = "underdogquality/yolo11s-pest-detection"
MODEL_FILENAME = "best.pt"


def group_pest(label):
    """Map IP102 species labels to the advice groups used by Leafwatch."""
    name = label.lower()
    if "aphid" in name or "green bug" in name:
        return "aphid"
    if "aleuro" in name or "trialeurodes" in name or "whitefly" in name:
        return "whitefly"
    caterpillar_terms = ("caterpillar", "army worm", "cutworm", "borer", "leaf roller", "budworm", "moth")
    if any(term in name for term in caterpillar_terms):
        return "caterpillar"
    return "other"


class Detector:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weights = hf_hub_download(repo_id=MODEL_REPOSITORY, filename=MODEL_FILENAME)
        self.model = YOLO(weights, task="detect")
        self.lock = threading.Lock()

    def detect(self, image):
        with self.lock:
            result = self.model.predict(image, conf=0.35, imgsz=640, max_det=300, verbose=False)[0]
            boxes = result.boxes.xyxyn.tolist()
            scores = result.boxes.conf.tolist()
            labels = result.boxes.cls.int().tolist()
        detections = []
        for (x1, y1, x2, y2), confidence, label in zip(boxes, scores, labels):
            if x2 > x1 and y2 > y1:
                species = str(result.names[label])
                detections.append({
                    "pest": group_pest(species),
                    "species": species,
                    "confidence": round(float(confidence), 4),
                    "box": [round(x1, 5), round(y1, 5), round(x2 - x1, 5), round(y2 - y1, 5)],
                })
        return detections


app = FastAPI(title="Leafwatch inference service")
allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or ["*"], allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Field"])
app.mount("/assets", StaticFiles(directory=ROOT / "dist"), name="assets")

detector = None
model_error = None


@app.on_event("startup")
def load_model():
    global detector, model_error
    try:
        detector = Detector()
        model_error = None
    except Exception as error:
        logging.exception("Could not start the IP102 detector")
        detector = None
        model_error = str(error)


@app.get("/api/status")
def model_status():
    return {"ready": detector is not None, "model": "YOLO11s trained on IP102", "modelSource": f"https://huggingface.co/{MODEL_REPOSITORY}", "datasetSource": "https://github.com/xpwu95/IP102", "error": model_error}


@app.post("/api/scan")
async def scan_photo(photo: UploadFile = File(...), x_field: str = Header(..., alias="X-Field")):
    if detector is None:
        raise HTTPException(status_code=503, detail="The detector is still starting. Please try again shortly.")
    field = unquote(x_field).strip()
    if not field or len(field) > 80:
        raise HTTPException(status_code=400, detail="Enter a field name of up to 80 characters.")
    if photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Use a JPEG, PNG or WebP photo.")
    data = await photo.read(MAX_UPLOAD + 1)
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Choose a photo under 12 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 50_000_000:
                raise ValueError("Image exceeds the pixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Could not read this photo.") from error
    try:
        detections = detector.detect(image)
    except Exception as error:
        logging.exception("Inference failed")
        raise HTTPException(status_code=500, detail="The detector could not analyze this photo.") from error
    return {"field": field, "capturedAt": datetime.now(timezone.utc).isoformat(), "detections": detections}


@app.get("/{path:path}")
def web_app(path: str):
    static_root = (ROOT / "dist").resolve()
    requested = (static_root / path).resolve()
    if path and static_root in requested.parents and requested.is_file():
        return FileResponse(requested)
    return FileResponse(static_root / "index.html")
