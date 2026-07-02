#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/semantic_field/latest/best.pt}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/PartNext_mesh}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/visualizations/semantic_field}"
DEVICE="${DEVICE:-cuda}"

cd "${ROOT_DIR}"
python tools/visualize_utonia_universal_field.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --feature-methods semantic \
  "$@"
