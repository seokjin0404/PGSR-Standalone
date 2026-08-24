"""Image/array I/O for PGSR evaluation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image


def read_float(path: str | None, size: tuple[int, int], channels: int) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not decode evaluation target: {path}")
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] >= 3:
        image = image[..., :3][..., ::-1]
    if channels == 1:
        image = image[..., 0]
    elif image.shape[-1] == 1:
        image = np.repeat(image, channels, axis=-1)
    width, height = size
    if image.shape[:2] != (height, width):
        interpolation = cv2.INTER_NEAREST if channels == 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return image.astype(np.float32)


def read_mask(path: str | None, size: tuple[int, int]) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not decode evaluation mask: {path}")
    if image.shape != (size[1], size[0]):
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    return image > 127


def quantize_rgb(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return encoded, encoded.astype(np.float32) / 255.0


def save_rgb(path: Path, array_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="RGB").save(path)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))
