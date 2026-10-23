"""FastAPI-приложение: API предсказаний плюс раздача собранного фронтенда."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import (
    MODEL_DIR,
    UNCERTAIN_BELOW,
    ModelNotReady,
    Registry,
    UnknownModel,
)
from app.preprocess import BadImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

state: dict[str, Registry | str | None] = {"registry": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузим один раз на старте. Её отсутствие не должно ронять
    # сервер: интерфейс обязан открыться и внятно сказать, чего не хватает.
    try:
        registry = Registry()
        state["registry"] = registry
        names = ", ".join(f"{i.id} ({i.num_classes} классов)" for i in registry.infos)
        print(f"моделей найдено: {len(registry.infos)} — {names}")
    except ModelNotReady as exc:
        state["error"] = str(exc)
        print(f"{exc} (ожидаются в {MODEL_DIR})")
    yield


app = FastAPI(title="Flowers-102 classifier", lifespan=lifespan)


@app.middleware("http")
async def reject_oversized_bodies(request: Request, call_next):
    """Отсечь слишком большое тело до разбора multipart.

    Проверка внутри обработчика опаздывает: к этому моменту Starlette уже
    собрал форму и слил её во временный файл на диске. На публичном хостинге
    тот же лимит стоит продублировать на обратном прокси — Content-Length
    можно не прислать вовсе.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        return JSONResponse({"detail": "файл больше 10 МБ"}, status_code=413)
    return await call_next(request)


def get_registry() -> Registry:
    registry = state["registry"]
    if registry is None:
        raise HTTPException(status_code=503, detail=state["error"] or "модель не загружена")
    return registry


@app.get("/api/health")
async def health():
    registry = state["registry"]
    if registry is None:
        return {"ready": False, "error": state["error"]}
    return {"ready": True, "models": len(registry.infos)}


@app.get("/api/models")
async def models():
    registry = get_registry()
    return {
        "models": [info.as_dict() for info in registry.infos],
        "default": registry.default_id,
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), model: str | None = Form(None)):
    if image.content_type and not image.content_type.startswith("image/"):
        # Содержимое заголовка в ответ не возвращаем: это ввод клиента.
        raise HTTPException(status_code=415, detail="ожидалось изображение")

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="файл больше 10 МБ")
    if not raw:
        raise HTTPException(status_code=400, detail="пустой файл")

    registry = get_registry()
    try:
        classifier = registry.get(model)
    except UnknownModel as exc:
        raise HTTPException(status_code=404, detail=f"нет модели «{model}»") from exc

    try:
        top, latency_ms = classifier.predict(raw)
    except BadImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model": model or registry.default_id,
        "top": [{"label": p.label, "probability": p.probability} for p in top],
        "uncertain": top[0].probability < UNCERTAIN_BELOW,
        "latency_ms": round(latency_ms, 1),
    }


# Статика монтируется последней, чтобы не перехватывать /api/*.
# StaticFiles сам проверяет, что запрошенный путь не выходит за пределы
# каталога: самодельная отдача файлов здесь дала бы обход каталога
# через «..» в URL.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
