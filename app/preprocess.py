"""Препроцессинг изображений без torchvision.

Повторяет transforms.Compose([Resize(256), CenterCrop(224), ToTensor()]),
которым обучалась модель. Нормализация ImageNet сюда не входит: она
встроена в ONNX-граф, поэтому на выходе значения остаются в [0, 1].
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Ограничение на размер распакованного изображения. Лимит на вес файла
# от «бомбы сжатия» не спасает: PNG 20000x20000 весит 0.4 МБ, а в памяти
# занимает больше гигабайта. 40 Мп с запасом перекрывает любую фотографию.
MAX_PIXELS = 40_000_000


class BadImageError(ValueError):
    """Файл не является изображением, повреждён или неприемлемо велик."""


def _too_large(size: str) -> BadImageError:
    return BadImageError(
        f"изображение слишком большое ({size}), "
        f"допустимо до {MAX_PIXELS // 1_000_000} мегапикселей"
    )


def load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        # Pillow сам отказывается открывать совсем уж гигантские файлы.
        raise _too_large("отказался открывать даже Pillow") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise BadImageError("не удалось прочитать изображение") from exc

    # Заголовок уже прочитан, пиксели ещё нет — проверяем размер до распаковки.
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise _too_large(f"{width}x{height}")

    try:
        image.load()
    except OSError as exc:
        raise BadImageError("не удалось прочитать изображение") from exc

    # Фотографии с телефона несут поворот в EXIF; без этого модель
    # получает картинку, лежащую на боку.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_shorter_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width <= height:
        new_width = size
        new_height = max(1, round(size * height / width))
    else:
        new_height = size
        new_width = max(1, round(size * width / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = round((width - size) / 2)
    top = round((height - size) / 2)
    return image.crop((left, top, left + size, top + size))


def preprocess(raw: bytes, resize_size: int, crop_size: int) -> np.ndarray:
    """bytes -> тензор NCHW float32 в диапазоне [0, 1]."""
    image = load_image(raw)
    image = center_crop(resize_shorter_side(image, resize_size), crop_size)

    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC
    return array.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
