#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 {synthetic|real} SCENE SOURCE_DIR OUTPUT_DIR" >&2
    exit 2
fi

DATASET_KIND="$1"
SCENE="$2"
SOURCE_DIR="$3"
OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PGSR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
ITERATION=30000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python was not found. Run bash setup_pgsr.sh first." >&2
    exit 1
fi

PREPARED_DIR="${REPO_ROOT}/outputs/pgsr/_prepared/${DATASET_KIND}/${SCENE}"
mkdir -p "${PREPARED_DIR}" "${OUTPUT_DIR}"

if [[ "${DATASET_KIND}" == "synthetic" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_synthetic_scene.py" \
        "${SOURCE_DIR}" "${PREPARED_DIR}" \
        --pattern 1 \
        --eval-split val
    EXPORT_TYPE="synthetic"
elif [[ "${DATASET_KIND}" == "real" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_realworld_scene.py" \
        "${SOURCE_DIR}" "${PREPARED_DIR}" \
        --pattern 1 \
        --eval-split val
    EXPORT_TYPE="real_world"
else
    echo "Unknown dataset kind: ${DATASET_KIND}" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] Training ${DATASET_KIND}/${SCENE}"
"${PYTHON_BIN}" "${REPO_ROOT}/train.py" \
    --source_path "${PREPARED_DIR}" \
    --model_path "${OUTPUT_DIR}" \
    --eval \
    --iterations "${ITERATION}" \
    --test_iterations 7000 "${ITERATION}" \
    --save_iterations 7000 "${ITERATION}" \
    --checkpoint_iterations "${ITERATION}"

FINAL_CHECKPOINT="${OUTPUT_DIR}/chkpnt${ITERATION}.pth"
FINAL_POINT_CLOUD="${OUTPUT_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
[[ -f "${FINAL_CHECKPOINT}" ]] || { echo "Missing ${FINAL_CHECKPOINT}" >&2; exit 1; }
[[ -f "${FINAL_POINT_CLOUD}" ]] || { echo "Missing ${FINAL_POINT_CLOUD}" >&2; exit 1; }

echo "[$(date --iso-8601=seconds)] Rendering and evaluating ${DATASET_KIND}/${SCENE}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m evaluation.export_results \
    --model-path "${OUTPUT_DIR}" \
    --prepared-scene "${PREPARED_DIR}" \
    --dataset-type "${EXPORT_TYPE}" \
    --scene "${SCENE}" \
    --iteration "${ITERATION}"
echo "[$(date --iso-8601=seconds)] Completed and verified ${DATASET_KIND}/${SCENE}"
