"""Deterministic, cross-view-comparable visualizations."""

from __future__ import annotations

import cv2
import numpy as np


def colorize_scalar(
    values: np.ndarray,
    minimum: float,
    maximum: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    normalized = np.clip((values - minimum) / max(maximum - minimum, 1.0e-8), 0.0, 1.0)
    gray = np.rint(np.nan_to_num(normalized) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[..., ::-1]
    color[~valid] = 0
    return color


def normal_rgb(normals: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit = normals / np.maximum(norm, 1.0e-8)
    encoded = np.rint(np.clip(unit * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    if valid is None:
        valid = np.isfinite(normals).all(axis=-1) & (norm[..., 0] > 1.0e-8)
    encoded[~valid] = 0
    return encoded


def scene_depth_range(records: list[dict]) -> tuple[float, float]:
    targets = []
    predictions = []
    for record in records:
        if record["gt_depth"] is not None:
            values = record["gt_depth"]
            mask = np.isfinite(values) & (values > 0.0) & (values < 1.0e4)
            targets.append(values[mask])
        values = record["pred_depth"]
        mask = np.isfinite(values) & (values > 0.0) & (record["alpha"] > 1.0e-3)
        predictions.append(values[mask])
    source = targets if any(values.size for values in targets) else predictions
    available = [values for values in source if values.size]
    if not available:
        return 0.0, 1.0
    packed = np.concatenate(available)
    low, high = np.percentile(packed, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)
