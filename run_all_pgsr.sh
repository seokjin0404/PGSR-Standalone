#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_all_pgsr.sh --data-root PATH [options]

Options:
  --data-root PATH       One supported scene or a directory containing scenes (required)
  --output-root PATH     Output root (default: output/batch)
  --gpu ID               One GPU to use (default: 0)
  --iterations N         Training iterations (default: 30000)
  --pattern N            RGB pattern index (default: 1)
  --eval-split val|test  Evaluation split (default: val)
  --eval-only            Skip training and evaluate existing checkpoints
  --skip-existing        Skip scenes with a final checkpoint
  --white-background     Composite synthetic RGBA images on white
  --dry-run              Only print detected scenes and their types
  -h, --help             Show this message

Synthetic scenes are detected by transforms_train.json. Prepared real-world
scenes are detected by split.json. All scenes run sequentially on one GPU.
PGSR/.venv/bin/python is used automatically.
EOF
}

DATA_ROOT=""
OUTPUT_ROOT="output/batch"
GPU_ID="${GPU_ID:-0}"
ITERATIONS=30000
PATTERN=1
EVAL_SPLIT="val"
EVAL_ONLY=0
SKIP_EXISTING=0
WHITE_BACKGROUND=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --eval-split) EVAL_SPLIT="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --white-background) WHITE_BACKGROUND=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "--data-root is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset root does not exist: $DATA_ROOT" >&2
    exit 1
fi
if [[ "$EVAL_SPLIT" != "val" && "$EVAL_SPLIT" != "test" ]]; then
    echo "--eval-split must be val or test" >&2
    exit 2
fi
if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--iterations must be a positive integer" >&2
    exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd)"
PYTHON_BIN="${PGSR_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${PGSR_PYTHON:-$(command -v python3 || true)}"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found. Run: bash setup_pgsr.sh" >&2
    exit 1
fi

declare -a SCENES=()
if [[ -f "$DATA_ROOT/transforms_train.json" || -f "$DATA_ROOT/split.json" ]]; then
    SCENES+=("$DATA_ROOT")
else
    while IFS= read -r -d '' metadata; do
        SCENES+=("$(dirname -- "$metadata")")
    done < <(
        find "$DATA_ROOT" -type f \
            \( -name transforms_train.json -o -name split.json \) \
            -print0 | sort -z
    )
fi
if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "No synthetic or prepared real-world scenes found under: $DATA_ROOT" >&2
    exit 1
fi

declare -A SEEN_NAMES=()
for scene_dir in "${SCENES[@]}"; do
    scene_name="$(basename -- "$scene_dir")"
    if [[ -n "${SEEN_NAMES[$scene_name]:-}" ]]; then
        echo "Duplicate scene name '$scene_name' below data root" >&2
        exit 1
    fi
    SEEN_NAMES[$scene_name]=1
done

echo "PGSR batch: ${#SCENES[@]} scene(s), GPU=$GPU_ID, iteration=$ITERATIONS"
for scene_dir in "${SCENES[@]}"; do
    if [[ -f "$scene_dir/transforms_train.json" ]]; then
        dataset_kind="synthetic"
    elif [[ -f "$scene_dir/split.json" ]]; then
        dataset_kind="real-world"
    else
        echo "Unrecognized scene: $scene_dir" >&2
        exit 1
    fi
    echo "  [$dataset_kind] $scene_dir"
done
if [[ "$DRY_RUN" -eq 1 ]]; then exit 0; fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

cd "$REPO_ROOT"
for scene_dir in "${SCENES[@]}"; do
    scene_name="$(basename -- "$scene_dir")"
    prepared="$OUTPUT_ROOT/_prepared/$scene_name"
    model_dir="$OUTPUT_ROOT/$scene_name"
    log_dir="$model_dir/logs"

    if [[ -f "$scene_dir/transforms_train.json" ]]; then
        dataset_kind="synthetic"
    else
        dataset_kind="real-world"
    fi
    if [[ "$EVAL_ONLY" -eq 0 && -f "$model_dir/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]]; then
        if [[ "$SKIP_EXISTING" -eq 1 ]]; then
            echo "[$dataset_kind][$scene_name] final checkpoint exists; skipping"
            continue
        fi
        echo "[$dataset_kind][$scene_name] output already exists: $model_dir" >&2
        echo "Use --eval-only, --skip-existing, or another --output-root." >&2
        exit 1
    fi

    mkdir -p "$log_dir"
    if [[ "$EVAL_ONLY" -eq 0 ]]; then
        if [[ "$dataset_kind" == "synthetic" ]]; then
            "$PYTHON_BIN" scripts/prepare_synthetic_scene.py \
                "$scene_dir" "$prepared" --pattern "$PATTERN" \
                --eval-split "$EVAL_SPLIT"
        else
            "$PYTHON_BIN" scripts/prepare_realworld_scene.py \
                "$scene_dir" "$prepared" --pattern "$PATTERN" \
                --eval-split "$EVAL_SPLIT"
        fi
        train_args=(
            "$PYTHON_BIN" train.py --source_path "$prepared" --model_path "$model_dir"
            --eval --iterations "$ITERATIONS"
            --test_iterations 7000 "$ITERATIONS"
            --save_iterations 7000 "$ITERATIONS"
            --checkpoint_iterations "$ITERATIONS"
        )
        if [[ "$dataset_kind" == "synthetic" && "$WHITE_BACKGROUND" -eq 1 ]]; then
            train_args+=(--white_background)
        fi
        echo "[$dataset_kind][$scene_name] training on GPU $GPU_ID"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "${train_args[@]}" 2>&1 | tee "$log_dir/training.log"
    elif [[ ! -f "$model_dir/point_cloud/iteration_$ITERATIONS/point_cloud.ply" ]]; then
        echo "[$dataset_kind][$scene_name] checkpoint missing: $model_dir" >&2
        exit 1
    fi

    echo "[$dataset_kind][$scene_name] rendering evaluation images"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" render.py \
        --model_path "$model_dir" --iteration "$ITERATIONS" --skip_train \
        2>&1 | tee "$log_dir/render.log"
    echo "[$dataset_kind][$scene_name] computing RGB metrics"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" metrics.py --model_paths "$model_dir" \
        2>&1 | tee "$log_dir/metrics.log"
done

echo "PGSR batch complete: $OUTPUT_ROOT"
