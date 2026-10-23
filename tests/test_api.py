from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.inference import ModelNotReady


def png_bytes(size=(300, 300)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "purple").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(registry, monkeypatch):
    monkeypatch.setitem(main.state, "registry", registry)
    monkeypatch.setitem(main.state, "error", None)
    # lifespan сам ищет модели в model/, здесь реестр уже подставлен
    with TestClient(main.app) as test_client:
        monkeypatch.setitem(main.state, "registry", registry)
        yield test_client


def test_health_reports_loaded_models(client):
    assert client.get("/api/health").json() == {"ready": True, "models": 2}


def test_models_endpoint_lists_both(client):
    body = client.get("/api/models").json()
    assert body["default"] == "model"
    assert [m["id"] for m in body["models"]] == ["extra", "model"]
    main_model = next(m for m in body["models"] if m["id"] == "model")
    assert main_model == {
        "id": "model", "name": "Flowers102", "architecture": "dummy",
        "num_classes": 102, "accuracy": 0.5,
    }


def test_predict_uses_selected_model(client):
    files = {"image": ("f.png", png_bytes(), "image/png")}
    default = client.post("/api/predict", files=files).json()
    chosen = client.post("/api/predict", files=files, data={"model": "extra"}).json()

    assert default["model"] == "model"
    assert chosen["model"] == "extra"
    # У моделей разное число классов, поэтому и распределения разные.
    assert default["top"] != chosen["top"]


def test_unknown_model_rejected(client):
    response = client.post(
        "/api/predict",
        files={"image": ("f.png", png_bytes(), "image/png")},
        data={"model": "../../etc/passwd"},
    )
    assert response.status_code == 404


def test_predict_returns_sorted_top3_probabilities(client):
    body = client.post("/api/predict", files={"image": ("f.png", png_bytes(), "image/png")}).json()

    assert len(body["top"]) == 3
    probabilities = [item["probability"] for item in body["top"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert body["uncertain"] is (probabilities[0] < 0.4)
    assert body["latency_ms"] >= 0


def test_non_image_content_type_rejected(client):
    response = client.post("/api/predict", files={"image": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_corrupted_image_rejected(client):
    response = client.post("/api/predict", files={"image": ("a.png", b"\x89PNG broken", "image/png")})
    assert response.status_code == 400


def test_oversized_upload_rejected(client):
    oversized = b"\x89PNG" + b"0" * (10 * 1024 * 1024)
    response = client.post("/api/predict", files={"image": ("big.png", oversized, "image/png")})
    assert response.status_code == 413


def test_oversized_body_rejected_before_parsing(client):
    """Отказ по Content-Length, до того как форма окажется во временном файле."""
    response = client.post(
        "/api/predict",
        content=b"x",
        headers={"content-length": str(50 * 1024 * 1024), "content-type": "image/png"},
    )
    assert response.status_code == 413


def test_empty_upload_rejected(client):
    response = client.post("/api/predict", files={"image": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_missing_model_yields_503(monkeypatch):
    monkeypatch.setitem(main.state, "registry", None)
    monkeypatch.setitem(main.state, "error", "веса не найдены")
    # TestClient без контекста не запускает lifespan, модель остаётся пустой
    response = TestClient(main.app).post(
        "/api/predict", files={"image": ("f.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert "веса не найдены" in response.json()["detail"]


def test_mismatched_labels_are_detected(model_dir, tmp_path):
    import json
    from app.inference import Classifier

    broken = tmp_path / "labels.json"
    meta = json.loads((model_dir / "labels.json").read_text())
    meta["classes"] = meta["classes"][:10]
    broken.write_text(json.dumps(meta))

    with pytest.raises(ModelNotReady, match="разных прогонов"):
        Classifier(model_dir / "model.onnx", broken)


def test_static_route_does_not_escape_dist(client):
    """Обход каталога через «..» в URL: раньше так утекали исходники сервера."""
    for attack in (
        "/%2e%2e/%2e%2e/app/inference.py",
        "/..%2f..%2fapp%2finference.py",
        "/%2e%2e%2f%2e%2e%2fpyproject.toml",
    ):
        response = client.get(attack)
        assert response.status_code in (404, 400), attack
        assert "onnxruntime" not in response.text
        assert "[project]" not in response.text


def test_decompression_bomb_rejected(client):
    """Маленький файл, который разворачивается в гигабайты пикселей."""
    buffer = io.BytesIO()
    Image.new("L", (20000, 20000)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    assert len(raw) < 10 * 1024 * 1024  # лимит на вес он проходит

    response = client.post("/api/predict", files={"image": ("bomb.png", raw, "image/png")})
    assert response.status_code == 400
    assert "слишком большое" in response.json()["detail"]


def test_errors_do_not_leak_server_paths(monkeypatch):
    from app.inference import Classifier, ModelNotReady

    with pytest.raises(ModelNotReady) as info:
        Classifier(Path("/nonexistent/model.onnx"), Path("/nonexistent/labels.json"))
    assert "/nonexistent" not in str(info.value)

    monkeypatch.setitem(main.state, "registry", None)
    monkeypatch.setitem(main.state, "error", str(info.value))
    body = TestClient(main.app).get("/api/health").json()
    assert "/" not in body["error"].replace("model/", "").replace("notebooks/", "")


def test_registry_ignores_model_without_description(tmp_path, model_dir):
    """Одинокий .onnx без описания не должен ломать список моделей."""
    import shutil
    from app.inference import Registry

    for name in ("model.onnx", "labels.json"):
        shutil.copy(model_dir / name, tmp_path / name)
    shutil.copy(model_dir / "extra.onnx", tmp_path / "orphan.onnx")

    assert [i.id for i in Registry(tmp_path).infos] == ["model"]
