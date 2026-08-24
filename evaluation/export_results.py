#!/usr/bin/env python3
"""Render, evaluate, export, and verify one trained PGSR scene."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from evaluation.io_utils import (
    quantize_rgb,
    read_float,
    read_mask,
    save_array,
    save_rgb,
)
from evaluation.metrics import (
    CSV_FIELDS,
    SUMMARY_FIELDS,
    aggregate,
    depth_metrics,
    depth_valid_mask,
    erode_one_pixel,
    normal_metrics,
    rgb_metrics,
)
from evaluation.visualization import (
    colorize_scalar,
    normal_rgb,
    scene_depth_range,
)
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene


ALPHA_THRESHOLD = 1.0e-3
DEPTH_ERROR_VIS_MAX_MM = 10.0
NORMAL_ERROR_VIS_MAX_DEG = 90.0
REQUIRED_VIEW_FILES = (
    "gt_rgb.png",
    "pred_rgb.png",
    "rgb_error.png",
    "gt_depth.npy",
    "pred_depth.npy",
    "gt_depth_vis.png",
    "pred_depth_vis.png",
    "depth_error.png",
    "gt_normal.npy",
    "pred_normal.npy",
    "gt_normal.png",
    "pred_normal.png",
    "normal_error.png",
)


def load_training_config(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing PGSR training config: {cfg_path}")
    config = eval(
        cfg_path.read_text(encoding="utf-8"),
        {"__builtins__": {}, "Namespace": Namespace},
    )
    if not isinstance(config, Namespace):
        raise TypeError(f"unexpected cfg_args content in {cfg_path}")
    config.model_path = str(model_path)
    return config


def world_normals(view, native_camera_normals: np.ndarray) -> np.ndarray:
    w2c = view.world_view_transform.detach().cpu().numpy().T
    camera_to_world = np.linalg.inv(w2c)
    return native_camera_normals @ camera_to_world[:3, :3].T


def save_checkpoint_alias(model_path: Path, iteration: int) -> None:
    source = model_path / f"chkpnt{iteration}.pth"
    if not source.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {source}")
    destination = model_path / "checkpoints" / "final.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    point_cloud = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if point_cloud.is_file():
        shutil.copy2(point_cloud, destination.parent / "final_point_cloud.ply")


def finite_or_nan(value):
    return value if isinstance(value, str) or math.isfinite(float(value)) else "NaN"


def write_metrics(model_path: Path, rows: list[dict], scene_name: str) -> None:
    metrics_dir = model_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "per_view_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: finite_or_nan(row[field]) for field in CSV_FIELDS})
    summary = {"scene": scene_name, **aggregate(rows)}
    (metrics_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


def verify_outputs(model_path: Path, expected_views: int) -> None:
    checkpoint_dir = model_path / "checkpoints"
    if not any(checkpoint_dir.glob("final.*")):
        raise RuntimeError("verification failed: checkpoints/final.* is missing")
    csv_path = model_path / "metrics" / "per_view_metrics.csv"
    summary_path = model_path / "metrics" / "summary_metrics.json"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError("verification failed: metric CSV schema differs")
        if sum(1 for _ in reader) != expected_views:
            raise RuntimeError("verification failed: metric row count differs")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in SUMMARY_FIELDS:
        if key not in summary:
            raise RuntimeError(f"verification failed: summary lacks {key}")
    for required in (model_path / "config.json", model_path / "run.log"):
        if not required.is_file():
            raise RuntimeError(f"verification failed: {required} is missing")
    render_root = model_path / "renders" / "val"
    view_dirs = sorted(path for path in render_root.iterdir() if path.is_dir())
    if len(view_dirs) != expected_views:
        raise RuntimeError("verification failed: visualization view count differs")
    for view_dir in view_dirs:
        missing = [name for name in REQUIRED_VIEW_FILES if not (view_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"verification failed: {view_dir} lacks {missing}")


def export(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    prepared_scene = args.prepared_scene.resolve()
    manifest = json.loads(
        (prepared_scene / "pgsr_input.json").read_text(encoding="utf-8")
    )
    if manifest["dataset_type"] != args.dataset_type:
        raise ValueError("prepared dataset type does not match exporter argument")
    depth_to_mm = manifest.get("pred_depth_to_mm")
    if depth_to_mm is None:
        raise RuntimeError(
            "metric-scale depth is unavailable; real data needs "
            "scale_estimation/scale_result.json with scale_mm_per_colmap"
        )
    depth_to_mm = float(depth_to_mm)

    config = load_training_config(model_path)
    config.source_path = str(prepared_scene)
    gaussians = GaussianModel(config.sh_degree)
    scene = Scene(config, gaussians, load_iteration=args.iteration, shuffle=False)
    pipeline = SimpleNamespace(
        convert_SHs_python=bool(getattr(config, "convert_SHs_python", False)),
        compute_cov3D_python=bool(getattr(config, "compute_cov3D_python", False)),
        debug=False,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if config.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    lpips_model = LPIPS("vgg", "0.1").cuda().eval()
    target_by_name = {
        record["image_name"]: record
        for record in manifest.get("evaluation_records", [])
    }

    records = []
    with torch.no_grad():
        for index, view in enumerate(tqdm(scene.getTestCameras(), desc="Exporting val")):
            output = render(view, gaussians, pipeline, background)
            prediction = output["render"].clamp(0.0, 1.0)
            target, _ = view.get_image()
            prediction_np = prediction.permute(1, 2, 0).cpu().numpy()
            target_np = target.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            pred_u8, pred_metric = quantize_rgb(prediction_np)
            gt_u8, gt_metric = quantize_rgb(target_np)
            height, width = pred_metric.shape[:2]

            pred_depth = (
                output["plane_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
                * depth_to_mm
            )
            alpha = output["rendered_alpha"].squeeze().detach().cpu().numpy().astype(np.float32)
            camera_normal = (
                output["rendered_normal"].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            )
            pred_normal = world_normals(view, camera_normal).astype(np.float32)
            if view.image_name not in target_by_name:
                raise RuntimeError(f"no evaluation manifest record for {view.image_name}")
            target_record = target_by_name.get(view.image_name, {})
            gt_depth = read_float(target_record.get("gt_depth"), (width, height), 1)
            gt_normal = read_float(target_record.get("gt_normal"), (width, height), 3)
            foreground = read_mask(target_record.get("mask"), (width, height))
            records.append(
                {
                    "index": index,
                    "image_name": view.image_name,
                    "pred_u8": pred_u8,
                    "gt_u8": gt_u8,
                    "pred_metric": pred_metric,
                    "gt_metric": gt_metric,
                    "pred_depth": pred_depth,
                    "gt_depth": gt_depth,
                    "pred_normal": pred_normal,
                    "gt_normal": gt_normal,
                    "alpha": alpha,
                    "foreground": foreground,
                }
            )

    if not records:
        raise RuntimeError("PGSR test split contains no evaluation views")
    depth_min, depth_max = scene_depth_range(records)
    rows = []
    render_root = model_path / "renders" / "val"
    for record in tqdm(records, desc="Writing metrics and visualizations"):
        view_dir = render_root / f"{record['index']:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(view_dir / "gt_rgb.png", record["gt_u8"])
        save_rgb(view_dir / "pred_rgb.png", record["pred_u8"])
        save_rgb(
            view_dir / "rgb_error.png",
            np.abs(record["pred_u8"].astype(np.int16) - record["gt_u8"].astype(np.int16)).astype(np.uint8),
        )

        empty_depth = np.full_like(record["pred_depth"], np.nan, dtype=np.float32)
        gt_depth = record["gt_depth"] if record["gt_depth"] is not None else empty_depth
        save_array(view_dir / "gt_depth.npy", gt_depth)
        save_array(view_dir / "pred_depth.npy", record["pred_depth"])
        geometry_valid = None
        if record["gt_depth"] is not None:
            geometry_valid = depth_valid_mask(
                record["pred_depth"],
                record["gt_depth"],
                record["alpha"],
                record["foreground"],
                ALPHA_THRESHOLD,
            )
        save_rgb(
            view_dir / "gt_depth_vis.png",
            colorize_scalar(gt_depth, depth_min, depth_max, geometry_valid),
        )
        pred_valid = np.isfinite(record["pred_depth"]) & (record["alpha"] > ALPHA_THRESHOLD)
        if record["foreground"] is not None:
            pred_valid &= record["foreground"]
        save_rgb(
            view_dir / "pred_depth_vis.png",
            colorize_scalar(record["pred_depth"], depth_min, depth_max, pred_valid),
        )
        depth_error = np.abs(record["pred_depth"] - gt_depth)
        save_rgb(
            view_dir / "depth_error.png",
            colorize_scalar(depth_error, 0.0, DEPTH_ERROR_VIS_MAX_MM, geometry_valid),
        )

        empty_normal = np.full_like(record["pred_normal"], np.nan, dtype=np.float32)
        gt_normal = record["gt_normal"] if record["gt_normal"] is not None else empty_normal
        save_array(view_dir / "gt_normal.npy", gt_normal)
        save_array(view_dir / "pred_normal.npy", record["pred_normal"])
        normal_geometry = erode_one_pixel(geometry_valid) if geometry_valid is not None else pred_valid
        normal_values, angle = normal_metrics(
            record["pred_normal"], record["gt_normal"], normal_geometry
        )
        normal_valid = np.isfinite(angle) if angle is not None else np.zeros(pred_valid.shape, dtype=bool)
        save_rgb(view_dir / "gt_normal.png", normal_rgb(gt_normal, normal_valid))
        save_rgb(view_dir / "pred_normal.png", normal_rgb(record["pred_normal"], pred_valid))
        angle_image = angle if angle is not None else np.full(pred_valid.shape, np.nan, dtype=np.float32)
        save_rgb(
            view_dir / "normal_error.png",
            colorize_scalar(angle_image, 0.0, NORMAL_ERROR_VIS_MAX_DEG, normal_valid),
        )

        pred_tensor = torch.from_numpy(record["pred_metric"]).permute(2, 0, 1).cuda()
        gt_tensor = torch.from_numpy(record["gt_metric"]).permute(2, 0, 1).cuda()
        row = {"scene": args.scene, "view_id": f"{record['index']:03d}"}
        row.update(rgb_metrics(pred_tensor, gt_tensor, lpips_model))
        row.update(depth_metrics(record["pred_depth"], record["gt_depth"], geometry_valid))
        row.update(normal_values)
        rows.append(row)

    write_metrics(model_path, rows, args.scene)
    save_checkpoint_alias(model_path, args.iteration)
    exported_config = {
        "scene": args.scene,
        "dataset_type": args.dataset_type,
        "iteration": args.iteration,
        "prepared_scene": str(prepared_scene),
        "source_scene": manifest["source"],
        "prediction_depth_scale_to_mm": depth_to_mm,
        "normal_frame": "world",
        "normal_orientation": "signed dot product; no absolute-value orientation folding",
        "validity": {
            "alpha_threshold": ALPHA_THRESHOLD,
            "depth": "finite GT in (0,10000) mm intersected with alpha and optional foreground mask",
            "normal": "depth validity eroded by one 4-neighbour pixel; finite nonzero GT normal",
        },
        "visualization": {
            "depth_range_mm": [depth_min, depth_max],
            "depth_error_range_mm": [0.0, DEPTH_ERROR_VIS_MAX_MM],
            "normal_error_range_degrees": [0.0, NORMAL_ERROR_VIS_MAX_DEG],
        },
        "metrics_aggregation": "arithmetic mean over views, independently ignoring NaN",
        "training_config": vars(config),
        "manifest": manifest,
    }
    (model_path / "config.json").write_text(
        json.dumps(exported_config, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    verify_outputs(model_path, len(records))
    print(f"Verified {args.scene}: {len(records)} views in {model_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prepared-scene", type=Path, required=True)
    parser.add_argument("--dataset-type", choices=("synthetic", "real_world"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    export(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
