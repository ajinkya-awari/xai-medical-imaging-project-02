import base64
import io

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app
from src.config import CFG
from src.inference import preprocess_image, probabilities_payload


@pytest.fixture(autouse=True)
def isolate_tests_from_local_checkpoint(monkeypatch, tmp_path):
    """Keep contract tests independent of pre-existing weights and native libs."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-checkpoint.pth"))


def _png_bytes(size=(32, 24)):
    image = Image.new("RGB", size, color=(120, 80, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocess_image_returns_normalized_batched_tensor():
    tensor = preprocess_image(Image.open(io.BytesIO(_png_bytes())))

    assert tuple(tensor.shape) == (1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    assert str(tensor.dtype) == "torch.float32"
    assert bool(np.isfinite(tensor.numpy()).all())


def test_probabilities_payload_preserves_all_labels_and_top_prediction():
    probabilities = np.linspace(0.01, 0.99, CFG.NUM_CLASSES, dtype=np.float32)

    payload = probabilities_payload(probabilities)

    assert set(payload["probabilities"]) == set(CFG.DISEASE_LABELS)
    assert payload["top_prediction"]["label"] == CFG.DISEASE_LABELS[-1]
    assert payload["confidence"] == pytest.approx(0.99, abs=1e-5)


def test_health_and_metadata_are_available_without_model_weights():
    with TestClient(app) as client:
        health = client.get("/health")
        metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert metadata.status_code == 200
    assert metadata.json()["labels"] == CFG.DISEASE_LABELS
    assert metadata.json()["disclaimer"]


def test_predict_rejects_non_image_upload_before_model_access():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert "PNG or JPEG" in response.json()["detail"]


def test_predict_returns_structured_result_for_a_fake_model(monkeypatch):
    expected_overlay = base64.b64encode(b"overlay").decode("ascii")
    fake_result = {
        "probabilities": {label: 0.1 for label in CFG.DISEASE_LABELS},
        "top_prediction": {"label": CFG.DISEASE_LABELS[0], "probability": 0.1},
        "confidence": 0.1,
        "gradcam_png_base64": expected_overlay,
    }
    monkeypatch.setattr("api.main.run_inference", lambda **_: fake_result)

    with TestClient(app) as client:
        app.state.model = object()
        response = client.post(
            "/predict",
            files={"file": ("xray.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {**fake_result, "disclaimer": response.json()["disclaimer"]}
