"""Игрушечные ONNX-модели с тем же контрактом, что у настоящих.

Позволяют тестировать весь путь bytes -> onnxruntime -> JSON, не дожидаясь
обучения в Colab и не тащя torch в зависимости. Моделей две, чтобы
проверять выбор модели.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

CROP = 224
MAIN_CLASSES = 102
EXTRA_CLASSES = 10


def _write_model(path, num_classes):
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.normal(0, 1, (num_classes, 3)).astype(np.float32), "W")
    bias = numpy_helper.from_array(np.zeros(num_classes, dtype=np.float32), "B")

    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["pixel_values"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "dummy",
        [helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, ["batch", 3, CROP, CROP])],
        [helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, ["batch", num_classes])],
        [weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_labels(path, num_classes, dataset):
    path.write_text(json.dumps({
        "architecture": "dummy",
        "dataset": dataset,
        "num_classes": num_classes,
        "classes": [f"flower {i:03d}" for i in range(num_classes)],
        "accuracy": {"test_top1_finetuned": 0.5},
        "preprocessing": {
            "resize_shorter_side": 256,
            "center_crop": CROP,
            "input_name": "pixel_values",
        },
    }))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("model")

    # Основная модель: пара model.onnx + labels.json, как её кладёт ноутбук.
    _write_model(directory / "model.onnx", MAIN_CLASSES)
    _write_labels(directory / "labels.json", MAIN_CLASSES, "Flowers102")

    # Вторая модель: описание рядом, через <имя>.labels.json.
    _write_model(directory / "extra.onnx", EXTRA_CLASSES)
    _write_labels(directory / "extra.labels.json", EXTRA_CLASSES, "Второй набор")

    return directory


@pytest.fixture(scope="session")
def registry(model_dir):
    from app.inference import Registry

    return Registry(model_dir)


@pytest.fixture(scope="session")
def classifier(registry):
    return registry.get()
