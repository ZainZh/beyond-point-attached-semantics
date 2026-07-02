#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/path/to/PartNext_mesh}"
UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT:-auto}"
CATEGORIES="${CATEGORIES:-Hammer}"
ALIAS_CONFIG="${ALIAS_CONFIG:-${ROOT_DIR}/configs/hammer.json}"
RUN_NAME="${RUN_NAME:-semantic_field_hammer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/semantic_field}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DEVICE="${DEVICE:-cuda}"

read -r -a CATEGORY_ARGS <<< "${CATEGORIES}"

cd "${ROOT_DIR}"
python -m train.train_utonia_universal_field \
  --train-mode semantic \
  --dataset-root "${DATASET_ROOT}" \
  --categories "${CATEGORY_ARGS[@]}" \
  --label-level config \
  --alias-config "${ALIAS_CONFIG}" \
  --utonia-checkpoint "${UTONIA_CHECKPOINT}" \
  --device "${DEVICE}" \
  --batch-size 6 \
  --num-workers 8 \
  --epochs 4000 \
  --balanced-sampling-ratio 0 \
  --rotation-mode so3 \
  --jitter-std 0.005 \
  --triplane-resolution 64 \
  --sem-ce-weight 1.0 \
  --sem-contrastive-weight 0.2 \
  --sem-consistency-weight 0.1 \
  --geo-dense-correspondence-weight 0.0 \
  --geo-metric-weight 0.0 \
  --geo-consistency-weight 0.0 \
  --occ-weight 0.0 \
  --amp \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --wandb-project object-centric-semantic-field \
  --wandb-mode "${WANDB_MODE}" \
  "$@"
