#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/path/to/PartNext_mesh}"
CATEGORIES="${CATEGORIES:-Hammer}"
ALIAS_CONFIG="${ALIAS_CONFIG:-${ROOT_DIR}/configs/hammer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/visualizations/support_query}"

read -r -a CATEGORY_ARGS <<< "${CATEGORIES}"

cd "${ROOT_DIR}"
python tools/visualize_partnext_support_query_points.py \
  --dataset-root "${DATASET_ROOT}" \
  --categories "${CATEGORY_ARGS[@]}" \
  --label-level config \
  --alias-config "${ALIAS_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
