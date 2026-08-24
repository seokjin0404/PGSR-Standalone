"""Metric definitions shared by the standalone PGSR exporter."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from utils.loss_utils import ssim


CSV_FIELDS = [
    "scene",
    "view_id",
    "rgb_psnr",
    "rgb_ssim",
    "rgb_lpips",
    "sl_psnr",
    "sl_ssim",
    "depth_mae",
    "depth_rmse",
    "depth_median_ae",
    "depth_p90_ae",
    "depth_p95_ae",
    "depth_acc_1mm",
    "depth_acc_2mm",
    "depth_acc_5mm",
    "depth_abs_rel",
    "depth_delta_1.25",
    "depth_delta_1.25^2",
    "depth_delta_1.25^3",
    "normal_mean_angle",
    "normal_median_angle",
    "normal_p90_angle",
    "normal_acc_10",
    "normal_acc_20",
    "normal_acc_22.5",
    "normal_acc_30",
]

SUMMARY_FIELDS = [field for field in CSV_FIELDS if field not in {"scene", "view_id"}]


def _nan_values(names: Iterable[str]) -> dict[str, float]:
    return {name: float("nan") for name in names}


def rgb_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module,
) -> dict[str, float]:
    """Compute RGB metrics on the exact quantized images written to disk."""
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((prediction - target) ** 2)
    psnr = float(-10.0 * torch.log10(mse.clamp_min(1.0e-12)))
    with torch.no_grad():
        return {
            "rgb_psnr": psnr,
            "rgb_ssim": float(ssim(prediction[None], target[None])),
            "rgb_lpips": float(lpips_model(prediction[None], target[None]).mean()),
            "sl_psnr": float("nan"),
            "sl_ssim": float("nan"),
        }


def depth_valid_mask(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray,
    alpha: np.ndarray,
    foreground: np.ndarray | None,
    alpha_threshold: float,
) -> np.ndarray:
    valid = (
        np.isfinite(prediction_mm)
        & np.isfinite(target_mm)
        & (target_mm > 0.0)
        & (target_mm < 1.0e4)
        & (alpha > alpha_threshold)
    )
    if foreground is not None:
        valid &= foreground
    return valid


def depth_metrics(
    prediction_mm: np.ndarray,
    target_mm: np.ndarray | None,
    valid: np.ndarray | None,
) -> dict[str, float]:
    names = [
        "depth_mae",
        "depth_rmse",
        "depth_median_ae",
        "depth_p90_ae",
        "depth_p95_ae",
        "depth_acc_1mm",
        "depth_acc_2mm",
        "depth_acc_5mm",
        "depth_abs_rel",
        "depth_delta_1.25",
        "depth_delta_1.25^2",
        "depth_delta_1.25^3",
    ]
    if target_mm is None or valid is None or not np.any(valid):
        return _nan_values(names)
    pred = prediction_mm[valid].astype(np.float64)
    target = target_mm[valid].astype(np.float64)
    error = np.abs(pred - target)
    ratio = np.maximum(pred / target, target / np.maximum(pred, 1.0e-12))
    return {
        "depth_mae": float(np.mean(error)),
        "depth_rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "depth_median_ae": float(np.median(error)),
        "depth_p90_ae": float(np.percentile(error, 90)),
        "depth_p95_ae": float(np.percentile(error, 95)),
        "depth_acc_1mm": float(np.mean(error <= 1.0)),
        "depth_acc_2mm": float(np.mean(error <= 2.0)),
        "depth_acc_5mm": float(np.mean(error <= 5.0)),
        "depth_abs_rel": float(np.mean(error / target)),
        "depth_delta_1.25": float(np.mean(ratio < 1.25)),
        "depth_delta_1.25^2": float(np.mean(ratio < 1.25**2)),
        "depth_delta_1.25^3": float(np.mean(ratio < 1.25**3)),
    }


def erode_one_pixel(mask: np.ndarray) -> np.ndarray:
    """Require a valid four-neighbourhood to suppress boundary mixtures."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return eroded


def normal_errors(
    prediction_world: np.ndarray,
    target_world: np.ndarray | None,
    valid_geometry: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if target_world is None or valid_geometry is None:
        return None, None
    pred_norm = np.linalg.norm(prediction_world, axis=-1)
    target_norm = np.linalg.norm(target_world, axis=-1)
    valid = (
        valid_geometry
        & np.isfinite(prediction_world).all(axis=-1)
        & np.isfinite(target_world).all(axis=-1)
        & (target_norm > 1.0e-8)
    )
    if not np.any(valid):
        return np.full(valid.shape, np.nan, dtype=np.float32), valid
    pred_unit = prediction_world / np.maximum(pred_norm[..., None], 1.0e-8)
    target_unit = target_world / np.maximum(target_norm[..., None], 1.0e-8)
    # Deliberately signed: orientation errors are not hidden with abs(dot).
    dot = np.sum(pred_unit * target_unit, axis=-1)
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))).astype(np.float32)
    angle[~valid] = np.nan
    return angle, valid


def normal_metrics(
    prediction_world: np.ndarray,
    target_world: np.ndarray | None,
    valid_geometry: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray | None]:
    names = [
        "normal_mean_angle",
        "normal_median_angle",
        "normal_p90_angle",
        "normal_acc_10",
        "normal_acc_20",
        "normal_acc_22.5",
        "normal_acc_30",
    ]
    angle, valid = normal_errors(prediction_world, target_world, valid_geometry)
    if angle is None or valid is None or not np.any(valid):
        return _nan_values(names), angle
    values = angle[valid].astype(np.float64)
    return (
        {
            "normal_mean_angle": float(np.mean(values)),
            "normal_median_angle": float(np.median(values)),
            "normal_p90_angle": float(np.percentile(values, 90)),
            "normal_acc_10": float(np.mean(values <= 10.0)),
            "normal_acc_20": float(np.mean(values <= 20.0)),
            "normal_acc_22.5": float(np.mean(values <= 22.5)),
            "normal_acc_30": float(np.mean(values <= 30.0)),
        },
        angle,
    )


def aggregate(rows: list[dict]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "aggregation": "arithmetic mean over views, independently ignoring NaN",
        "num_views": len(rows),
    }
    for field in SUMMARY_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[field] = float(np.mean(finite)) if finite.size else float("nan")
    return summary
