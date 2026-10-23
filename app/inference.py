"""Загрузка ONNX-моделей и предсказание. Без torch."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.preprocess import preprocess

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

TOP_K = 3
# Модель знает только свой список видов и для кота всё равно назовёт растение.
# Ниже этого порога честнее признаться, что уверенности нет.
# Порог невысокий намеренно: label smoothing при обучении срезает
# сверхуверенность, и верные ответы часто идут с вероятностью 0.3-0.5.
UNCERTAIN_BELOW = 0.25

NO_MODELS = (
    "ни одной модели не найдено: в папке model/ нужна пара файлов с общим "
    "именем, например flowers102.onnx и flowers102.labels.json. Запусти "
    "ноутбук из notebooks/ в Colab и положи файлы туда."
)


class ModelNotReady(RuntimeError):
    """Веса не найдены или не совпадают с описанием."""


class UnknownModel(KeyError):
    """Запрошена модель, которой нет в папке."""


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


@dataclass(frozen=True)
class ModelInfo:
    """Описание модели, прочитанное без загрузки самого графа."""

    id: str
    name: str
    architecture: str
    num_classes: int
    accuracy: float | None


class Classifier:
    def __init__(self, model_path: Path, labels_path: Path):
        if not model_path.exists() or not labels_path.exists():
            # Без абсолютных путей: этот текст показывается в браузере,
            # а раскрывать структуру каталогов сервера незачем.
            raise ModelNotReady(NO_MODELS)

        self.meta = json.loads(labels_path.read_text(encoding="utf-8"))
        self.classes: list[str] = self.meta["classes"]

        spec = self.meta.get("preprocessing", {})
        self.resize_size = int(spec.get("resize_shorter_side", 256))
        self.crop_size = int(spec.get("center_crop", 224))
        self.input_name = spec.get("input_name", "pixel_values")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

        graph_input = self.session.get_inputs()[0]
        if graph_input.name != self.input_name:
            self.input_name = graph_input.name

        actual_classes = self.session.get_outputs()[0].shape[-1]
        if isinstance(actual_classes, int) and actual_classes != len(self.classes):
            raise ModelNotReady(
                f"модель отдаёт {actual_classes} классов, а в описании их "
                f"{len(self.classes)} — файлы из разных прогонов"
            )

    def predict(self, raw: bytes) -> tuple[list[Prediction], float]:
        tensor = preprocess(raw, self.resize_size, self.crop_size)

        started = time.perf_counter()
        probabilities = self.session.run(None, {self.input_name: tensor})[0][0]
        latency_ms = (time.perf_counter() - started) * 1000

        top_indices = np.argsort(probabilities)[::-1][:TOP_K]
        top = [
            Prediction(self.classes[int(i)], float(probabilities[int(i)]))
            for i in top_indices
        ]
        return top, latency_ms


def _labels_for(model_path: Path) -> Path | None:
    """Описание модели: сначала <имя>.labels.json, потом общий labels.json.

    Ноутбуки называют файлы по датасету (flowers102.onnx и рядом
    flowers102.labels.json). Пара model.onnx + labels.json поддержана
    как раскладка на одну модель — так выглядели выгрузки прежних версий
    ноутбуков, и ломать их при обновлении незачем.
    """
    sidecar = model_path.with_suffix(".labels.json")
    if sidecar.exists():
        return sidecar
    shared = model_path.parent / "labels.json"
    if model_path.stem == "model" and shared.exists():
        return shared
    return None


class Registry:
    """Все модели из папки model/. Графы грузятся лениво, при первом запросе.

    Лениво — потому что каждый ONNX это около 100 МБ в памяти, а описание
    (имя, число классов, точность) читается из маленького JSON и нужно
    сразу, чтобы интерфейс показал список.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self._paths: dict[str, tuple[Path, Path]] = {}
        self._infos: list[ModelInfo] = []
        self._loaded: dict[str, Classifier] = {}
        # Запросы обслуживаются в нескольких потоках, а первая загрузка
        # графа занимает секунду: без блокировки два потока прочитали бы
        # один и тот же файл на 100 МБ одновременно.
        self._lock = threading.Lock()

        for model_path in sorted(model_dir.glob("*.onnx")):
            labels_path = _labels_for(model_path)
            if labels_path is None:
                continue
            try:
                meta = json.loads(labels_path.read_text(encoding="utf-8"))
                info = ModelInfo(
                    id=model_path.stem,
                    name=meta.get("dataset") or model_path.stem,
                    architecture=meta.get("architecture", "неизвестно"),
                    num_classes=len(meta["classes"]),
                    accuracy=meta.get("accuracy", {}).get("test_top1_finetuned"),
                )
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # битое описание — модель просто не показываем
            self._paths[info.id] = (model_path, labels_path)
            self._infos.append(info)

        if not self._infos:
            raise ModelNotReady(NO_MODELS)

    @property
    def infos(self) -> list[ModelInfo]:
        return list(self._infos)

    @property
    def default_id(self) -> str:
        # Раскладка на одну модель однозначна; при нескольких берём первую
        # по алфавиту — порядок задан сортировкой при обходе папки.
        return "model" if "model" in self._paths else self._infos[0].id

    def get(self, model_id: str | None = None) -> Classifier:
        model_id = model_id or self.default_id
        if model_id not in self._paths:
            raise UnknownModel(model_id)
        with self._lock:
            if model_id not in self._loaded:
                self._loaded[model_id] = Classifier(*self._paths[model_id])
            return self._loaded[model_id]
