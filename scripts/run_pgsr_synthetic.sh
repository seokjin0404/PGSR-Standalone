#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# COLLABORATOR: CHANGE ONLY THIS LINE.
DATA_ROOT="/CHANGE/ONLY/THIS/PATH"
# ============================================================

SCENES=("chair" "drum" "ficus" "lego" "mic" "ship")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/pgsr/synthetic"

if [[ "${DATA_ROOT}" == "/CHANGE/ONLY/THIS/PATH" ]]; then
    echo "Set DATA_ROOT at the top of $0 before running." >&2
    exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "DATA_ROOT is not a directory: ${DATA_ROOT}" >&2
    exit 2
fi

resolve_scene() {
    local scene="$1"
    local candidate
    for candidate in \
        "${DATA_ROOT}/${scene}" \
        "${DATA_ROOT}/synthetic/${scene}" \
        "${DATA_ROOT}/${scene}2" \
        "${DATA_ROOT}/synthetic/${scene}2"; do
        if [[ -f "${candidate}/transforms_train.json" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    local matches=()
    while IFS= read -r candidate; do
        [[ -f "${candidate}/transforms_train.json" ]] && matches+=("${candidate}")
    done < <(find "${DATA_ROOT}" -maxdepth 4 -type d \( -name "${scene}" -o -name "${scene}2" \) -print)
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "Expected exactly one synthetic dataset for ${scene}; found ${#matches[@]}." >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

mkdir -p "${OUTPUT_ROOT}"
for scene in "${SCENES[@]}"; do
    source_dir="$(resolve_scene "${scene}")"
    scene_output="${OUTPUT_ROOT}/${scene}"
    mkdir -p "${scene_output}"
    echo "=== ${scene}: ${source_dir} ==="
    bash "${SCRIPT_DIR}/run_pgsr_standard_scene.sh" \
        synthetic "${scene}" "${source_dir}" "${scene_output}" \
        2>&1 | tee "${scene_output}/run.log"
done

echo "All synthetic PGSR scenes completed successfully: ${OUTPUT_ROOT}"
