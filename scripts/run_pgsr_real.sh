#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# COLLABORATOR: CHANGE ONLY THIS LINE.
DATA_ROOT="/CHANGE/ONLY/THIS/PATH"
# ============================================================

SCENES=("fruits" "lego" "stair" "statues")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/pgsr/real"

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
        "${DATA_ROOT}/real/${scene}" \
        "${DATA_ROOT}/real_world/${scene}" \
        "${DATA_ROOT}/real-world/${scene}"; do
        if [[ -f "${candidate}/split.json" && -f "${candidate}/colmap_workspace/run_summary.json" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    local matches=()
    while IFS= read -r candidate; do
        if [[ -f "${candidate}/split.json" && -f "${candidate}/colmap_workspace/run_summary.json" ]]; then
            matches+=("${candidate}")
        fi
    done < <(find "${DATA_ROOT}" -maxdepth 4 -type d -name "${scene}" -print)
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "Expected exactly one real-world dataset for ${scene}; found ${#matches[@]}." >&2
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
        real "${scene}" "${source_dir}" "${scene_output}" \
        2>&1 | tee "${scene_output}/run.log"
done

echo "All real-world PGSR scenes completed successfully: ${OUTPUT_ROOT}"
