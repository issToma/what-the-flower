from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.preprocess import BadImageError, load_image, preprocess

RESIZE, CROP = 256, 224


def encode(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (224, 224), (50, 1200)])
def test_output_shape_is_always_nchw_224(size):
    tensor = preprocess(encode(Image.new("RGB", size, "green")), RESIZE, CROP)
    assert tensor.shape == (1, 3, CROP, CROP)
    assert tensor.dtype == np.float32


def test_values_scaled_to_unit_range():
    tensor = preprocess(encode(Image.new("RGB", (300, 300), (255, 0, 0))), RESIZE, CROP)
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    # Канал R заполнен, G и B пусты — значит порядок каналов не перепутан.
    assert tensor[0, 0].mean() == pytest.approx(1.0, abs=0.02)
    assert tensor[0, 1].mean() == pytest.approx(0.0, abs=0.02)


def test_grayscale_and_rgba_are_converted():
    for mode in ("L", "RGBA", "P"):
        raw = encode(Image.new(mode, (300, 300), 128), fmt="PNG")
        assert preprocess(raw, RESIZE, CROP).shape == (1, 3, CROP, CROP)


def test_shorter_side_resize_preserves_aspect_ratio():
    image = load_image(encode(Image.new("RGB", (800, 400), "blue")))
    from app.preprocess import resize_shorter_side

    assert resize_shorter_side(image, 256).size == (512, 256)


def test_crop_takes_the_centre():
    # Левая половина чёрная, правая белая: центральный кроп квадратного
    # изображения должен захватить обе в равной пропорции.
    image = Image.new("RGB", (400, 400), "black")
    image.paste(Image.new("RGB", (200, 400), "white"), (200, 0))
    tensor = preprocess(encode(image, "PNG"), RESIZE, CROP)
    assert tensor.mean() == pytest.approx(0.5, abs=0.02)


def test_garbage_input_is_rejected():
    with pytest.raises(BadImageError):
        preprocess(b"this is not an image", RESIZE, CROP)
