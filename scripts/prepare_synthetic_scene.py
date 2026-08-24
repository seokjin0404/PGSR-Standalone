#!/usr/bin/env python3
"""Make a Blender/NeRF synthetic scene directly readable by PGSR.

Standard Blender frames are preserved as-is. Extended structured-light datasets
that explicitly provide ``pattern_index`` can select one RGB pattern. Millimetre
camera translations are detected by their large magnitude and converted to
metres; ordinary Blender scenes retain their original coordinate scale.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median


def selected_frames(payload: dict, pattern: int) -> tuple[list[dict], bool]:
    source_frames = payload.get("frames", [])
    has_pattern_metadata = any("pattern_index" in frame for frame in source_frames)
    if has_pattern_metadata:
        frames = [
            frame
            for frame in source_frames
            if int(frame.get("pattern_index", -1)) == pattern
        ]
    else:
        frames = list(source_frames)
    if not frames:
        raise RuntimeError(f"no frames selected for pattern_index={pattern}")
    return frames, has_pattern_metadata


def infer_translation_scale(frames: list[dict], structured_light: bool) -> float:
    radii = []
    for frame in frames:
        matrix = frame["transform_matrix"]
        radii.append(sum(float(matrix[axis][3]) ** 2 for axis in range(3)) ** 0.5)
    # The supported structured-light synthetic captures use millimetres and have
    # camera radii in the hundreds. Standard Blender/NeRF scenes are near unit scale.
    return 0.001 if structured_light and median(radii) > 10.0 else 1.0


def convert(
    payload: dict,
    source: Path,
    pattern: int,
    scale: float,
) -> tuple[dict, bool]:
    source_frames, has_pattern_metadata = selected_frames(payload, pattern)
    frames = []
    for frame in source_frames:
        image = (source / frame["file_path"]).with_suffix(".png").resolve()
        if not image.is_file():
            raise FileNotFoundError(
                f"missing image referenced by transforms JSON: {image}"
            )
        matrix = [
            [float(value) for value in row] for row in frame["transform_matrix"]
        ]
        for axis in range(3):
            matrix[axis][3] *= scale
        frames.append(
            {"file_path": str(image.with_suffix("")), "transform_matrix": matrix}
        )
    converted = {key: value for key, value in payload.items() if key != "frames"}
    converted["frames"] = frames
    return converted, has_pattern_metadata


def evaluation_records(payload: dict, source: Path, pattern: int) -> list[dict]:
    """Record unmodified evaluation targets for the standalone exporter."""
    frames, _ = selected_frames(payload, pattern)
    records = []
    for frame in frames:
        relative = Path(frame["file_path"])
        image = (source / relative).with_suffix(".png").resolve()
        match = re.search(r"r_(\d+)(?:_\d+)?$", relative.stem)
        frame_id = int(match.group(1)) if match else None
        split_name = relative.parts[0] if len(relative.parts) > 1 else "val"
        target_stem = f"r_{frame_id}" if frame_id is not None else relative.stem

        def optional_target(folder: str, suffix: str) -> str | None:
            path = (source / split_name / folder / f"{target_stem}{suffix}").resolve()
            return str(path) if path.is_file() else None

        records.append(
            {
                "image_name": image.stem,
                "source_image": str(image),
                "frame_id": frame_id,
                "gt_depth": optional_target("depth", ".exr"),
                "gt_normal": optional_target("normal_world", ".exr"),
                "mask": optional_target("mask", ".png"),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pattern", type=int, default=1)
    parser.add_argument("--translation-scale", type=float)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    train_path = source / "transforms_train.json"
    eval_path = source / f"transforms_{args.eval_split}.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing {train_path}")
    if not eval_path.is_file() and args.eval_split == "val":
        eval_path = source / "transforms_test.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing evaluation transforms in {source}")

    train_payload = json.loads(train_path.read_text(encoding="utf-8"))
    candidate_frames, structured_light = selected_frames(train_payload, args.pattern)
    scale = (
        args.translation_scale
        if args.translation_scale is not None
        else infer_translation_scale(candidate_frames, structured_light)
    )
    train, _ = convert(train_payload, source, args.pattern, scale)
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    test, test_has_patterns = convert(
        eval_payload,
        source,
        args.pattern,
        scale,
    )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "transforms_train.json").write_text(
        json.dumps(train, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "transforms_test.json").write_text(
        json.dumps(test, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": str(source),
        "dataset_type": "synthetic",
        "structured_light_extensions": structured_light or test_has_patterns,
        "pattern_index": args.pattern if structured_light else None,
        "translation_scale": scale,
        "pred_depth_to_mm": 1.0 / scale,
        "evaluation_source": str(eval_path),
        "evaluation_records": evaluation_records(
            eval_payload, source, args.pattern
        ),
        "train_views": len(train["frames"]),
        "evaluation_views": len(test["frames"]),
    }
    (destination / "pgsr_input.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared synthetic {source.name}: {len(train['frames'])} train / "
        f"{len(test['frames'])} evaluation views; translation_scale={scale:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
