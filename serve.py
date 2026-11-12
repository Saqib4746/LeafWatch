"""Leafwatch web and inference service using the IP102 pest detection model."""
import gc
import hmac
import io
import logging
import os
import threading
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 12 * 1024 * 1024
MODEL_REPOSITORY = os.getenv("MODEL_REPOSITORY", "underdogquality/yolo11s-pest-detection")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best.pt")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

if not os.getenv("YOLO_CONFIG_DIR"):
    local_config = ROOT / ".model-config"
    os.environ["YOLO_CONFIG_DIR"] = str(local_config if local_config.exists() else Path("/tmp/Ultralytics"))


def group_pest(label):
    """Map IP102 species labels to the advice groups used by Leafwatch."""
    name = " ".join(label.lower().split())
    aphid_terms = ("aphid", "aphis", "green bug", "therioaphis", "toxoptera")
    if any(term in name for term in aphid_terms):
        return "aphid"
    if "aleuro" in name or "trialeurodes" in name or "whitefly" in name:
        return "whitefly"
    caterpillar_terms = (
        "caterpillar", "army worm", "cutworm", "borer",
        "leaf roller", "budworm", "moth", "prodenia litura",
    )
    if any(term in name for term in caterpillar_terms):
        return "caterpillar"
    return "other"


def resolve_weights():
    """Resolve model weights from explicit config, local files, or public fallback."""
    configured = os.getenv("MODEL_PATH", "").strip()
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Configured MODEL_PATH does not exist: {candidate}")
        return candidate, "local"

    model_url = os.getenv("MODEL_URL", "").strip()
    if model_url:
        suffix = Path(urlparse(model_url).path).suffix.lower()
        if suffix not in {".pt", ".pth"}:
            suffix = ".pt"
        cached = Path(os.getenv("MODEL_CACHE_DIR", "/tmp")) / f"leafwatch-model{suffix}"
        if not cached.is_file():
            cached.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(model_url, cached)
        return cached, "url"

    for candidate in (
        ROOT / "model-source" / MODEL_FILENAME,
        ROOT / "models" / MODEL_FILENAME,
        ROOT / MODEL_FILENAME,
        ROOT / "training-runs" / "mobilenet-ip102" / "best.pth",
    ):
        if candidate.is_file():
            return candidate, "local"

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=MODEL_REPOSITORY, filename=MODEL_FILENAME)), "huggingface"


def clamp_box(x1, y1, x2, y2):
    """Clamp and round normalized [x, y, w, h] coordinates so x+w <= 1 and y+h <= 1."""
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))
    rx = round(x1, 5)
    ry = round(y1, 5)
    rw = round(min(1.0 - rx, max(0.0, x2 - rx)), 5)
    rh = round(min(1.0 - ry, max(0.0, y2 - ry)), 5)
    return [rx, ry, rw, rh]


class Detector:
    def __init__(self):
        import torch

        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
        weights, source = resolve_weights()
        self.source = source
        if weights.suffix.lower() == ".pth":
            from mobile_detector import MobileDetector

            self._mobile = MobileDetector(weights)
            self.model_name = "MobileNetV3 SSDLite trained on IP102"
            self.lock = self._mobile.lock
        else:
            from ultralytics import YOLO

            self._mobile = None
            self.model = YOLO(str(weights), task="detect")
            self.model_name = "YOLO11s trained on IP102"
            self.lock = threading.Lock()

    def detect(self, image):
        if self._mobile is not None:
            detections = []
            for item in self._mobile.detect(image):
                x, y, w, h = item["box"]
                box = clamp_box(x, y, x + w, y + h)
                if box[2] > 0 and box[3] > 0:
                    detections.append({
                        "pest": item["pest"],
                        "species": item["pest"],
                        "confidence": round(float(item["confidence"]), 4),
                        "box": box,
                    })
            return detections

        with self.lock:
            result = self.model.predict(image, conf=0.35, imgsz=640, max_det=300, verbose=False)[0]
            boxes = result.boxes.xyxyn.tolist()
            scores = result.boxes.conf.tolist()
            labels = result.boxes.cls.int().tolist()

        detections = []
        for (x1, y1, x2, y2), confidence, label in zip(boxes, scores, labels):
            box = clamp_box(x1, y1, x2, y2)
            if box[2] > 0 and box[3] > 0:
                species = str(result.names[label])
                detections.append({
                    "pest": group_pest(species),
                    "species": species,
                    "confidence": round(float(confidence), 4),
                    "box": box,
                })
        return detections


detector = None
model_error = None


def load_model():
    global detector, model_error
    try:
        detector = Detector()
        model_error = None
    except Exception as error:
        logging.exception("Could not start the IP102 detector")
        detector = None
        model_error = str(error)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Leafwatch inference service", lifespan=lifespan)
allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Field", "Authorization"],
)


@app.get("/api/status")
def model_status():
    return {
        "ready": detector is not None,
        "model": detector.model_name if detector is not None else "YOLO11s trained on IP102",
        "modelSource": f"https://huggingface.co/{MODEL_REPOSITORY}",
        "datasetSource": "https://github.com/xpwu95/IP102",
        "error": model_error,
    }


@app.post("/api/scan")
async def scan_photo(
    photo: UploadFile = File(...),
    x_field: str = Header(..., alias="X-Field"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    expected_token = os.getenv("INFERENCE_API_TOKEN", "").strip()
    if expected_token:
        provided = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided, expected_token):
            raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    if detector is None:
        raise HTTPException(status_code=503, detail="The detector is still starting. Please try again shortly.")
    field = unquote(x_field).strip()
    if not field or len(field) > 80:
        raise HTTPException(status_code=400, detail="Enter a field name of up to 80 characters.")
    content_type = (photo.content_type or "").lower()
    suffix = Path(photo.filename or "").suffix.lower()
    if content_type not in ALLOWED_IMAGE_TYPES and not (
        content_type in {"", "application/octet-stream"} and suffix in IMAGE_SUFFIXES
    ):
        raise HTTPException(status_code=400, detail="Use a JPEG, PNG or WebP photo.")
    data = await photo.read(MAX_UPLOAD + 1)
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Choose a photo under 12 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 50_000_000:
                raise ValueError("Image exceeds the pixel limit.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > 1280:
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Could not read this photo.") from error
    try:
        detections = detector.detect(image)
        gc.collect()
    except Exception as error:
        logging.exception("Inference failed")
        raise HTTPException(status_code=500, detail="The detector could not analyze this photo.") from error
    return {
        "field": field,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
    }


@app.get("/{path:path}")
def web_app(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found.")
    static_root = (ROOT / "dist").resolve()
    requested = (static_root / path).resolve()
    if path and static_root in requested.parents and requested.is_file():
        return FileResponse(requested)
    return FileResponse(static_root / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
