#!/usr/bin/env python3
"""Make a prepared COLMAP real-world scene directly readable by PGSR.

Images and the selected COLMAP model are materialized with hard links when
possible (copies otherwise). Pixel values are never decoded or transformed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pattern", type=int, default=1)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.pattern != 1:
        raise ValueError("prepared real-world COLMAP models require --pattern 1")

    source = args.source.resolve()
    destination = args.destination.resolve()
    split_path = source / "split.json"
    summary_path = source / "colmap_workspace" / "run_summary.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"missing {split_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing {summary_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split.get("train_pose_ids", [])]
    eval_ids = [
        int(value) for value in split.get(f"{args.eval_split}_pose_ids", [])
    ]
    if not train_ids:
        raise RuntimeError(f"train_pose_ids is empty in {split_path}")
    if not eval_ids:
        raise RuntimeError(
            f"{args.eval_split}_pose_ids is empty in {split_path}; "
            "choose an available --eval-split"
        )

    model_id = int(summary["best_model_id"])
    colmap_source = source / "colmap_workspace" / "sparse" / str(model_id)
    required_model_files = ("cameras.bin", "images.bin", "points3D.bin")
    for filename in required_model_files:
        if not (colmap_source / filename).is_file():
            raise FileNotFoundError(f"missing COLMAP model file: {colmap_source / filename}")

    modes = set()
    for filename in required_model_files:
        modes.add(
            link_or_copy(
                colmap_source / filename, destination / "sparse" / filename
            )
        )

    def materialize_images(split_name: str, pose_ids: list[int]) -> list[str]:
        stems = []
        for pose_id in pose_ids:
            filename = f"r_{pose_id}_{args.pattern}.png"
            image_source = source / split_name / "rgb" / filename
            if not image_source.is_file():
                raise FileNotFoundError(f"missing real-world RGB image: {image_source}")
            modes.add(link_or_copy(image_source, destination / "images" / filename))
            stems.append(Path(filename).stem)
        return stems

    train_stems = materialize_images("train", train_ids)
    eval_stems = materialize_images(args.eval_split, eval_ids)
    pgsr_split = {"train": train_stems, "test": eval_stems}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "split.json").write_text(
        json.dumps(pgsr_split, indent=2) + "\n", encoding="utf-8"
    )

    scale_path = source / "scale_estimation" / "scale_result.json"
    scale = None
    if scale_path.is_file():
        scale = json.loads(scale_path.read_text(encoding="utf-8")).get(
            "scale_mm_per_colmap"
        )
    manifest = {
        "source": str(source),
        "dataset_type": "real_world",
        "pattern_index": args.pattern,
        "evaluation_split": args.eval_split,
        "colmap_model": str(colmap_source),
        "scale_mm_per_colmap": scale,
        "pred_depth_to_mm": scale,
        "coordinate_policy": "native COLMAP coordinates; PGSR camera/point geometry unchanged",
        "color_space": "linear_rgb",
        "color_policy": "linear RGB values preserved; no transfer-function conversion",
        "materialization": sorted(modes),
        "train_views": len(train_stems),
        "evaluation_views": len(eval_stems),
        "evaluation_records": [
            {
                "image_name": f"r_{pose_id}_{args.pattern}",
                "source_image": str(
                    (source / args.eval_split / "rgb" / f"r_{pose_id}_{args.pattern}.png").resolve()
                ),
                "frame_id": pose_id,
                "gt_depth": str(
                    (source / args.eval_split / "depth" / f"r_{pose_id}.exr").resolve()
                )
                if (source / args.eval_split / "depth" / f"r_{pose_id}.exr").is_file()
                else None,
                "gt_normal": str(
                    (source / args.eval_split / "normal_world" / f"r_{pose_id}.exr").resolve()
                )
                if (source / args.eval_split / "normal_world" / f"r_{pose_id}.exr").is_file()
                else None,
                "mask": str(
                    (source / args.eval_split / "mask" / f"r_{pose_id}.png").resolve()
                )
                if (source / args.eval_split / "mask" / f"r_{pose_id}.png").is_file()
                else None,
            }
            for pose_id in eval_ids
        ],
    }
    (destination / "pgsr_input.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared real-world {source.name}: {len(train_stems)} train / "
        f"{len(eval_stems)} {args.eval_split} views; color unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
