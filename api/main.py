"""FastAPI serving surface with explicit upload and model safety boundaries."""

from contextlib import asynccontextmanager
import io

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from src.config import CFG
from src.inference import DISCLAIMER, load_checkpoint_model, run_inference


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


@asynccontextmanager
async def lifespan(application):
    """Load one model per process when a local checkpoint is available."""
    application.state.model = None
    application.state.model_error = None
    try:
        application.state.model = load_checkpoint_model()
    except Exception as exc:
        application.state.model_error = str(exc)
    yield
    application.state.model = None


app = FastAPI(
    title="ChestXplain API",
    description="Research-only explainable chest X-ray classification API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
    }


@app.get("/metadata")
def metadata():
    return {
        "labels": CFG.DISEASE_LABELS,
        "num_classes": CFG.NUM_CLASSES,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload must be a PNG or JPEG image.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Upload exceeds the 10 MB limit.")

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("unsupported image format")
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Upload must be a readable PNG or JPEG image.")

    if request.app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is unavailable; prediction service is not ready.",
        )

    try:
        result = run_inference(model=request.app.state.model, image=image)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return {"disclaimer": DISCLAIMER, **result}
