#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
semantic_ckpt_A=${4:-none}
semantic_ckpt_B=${5:-none}
semantic_device=${6:-cuda:0}
object_placeholders=${7:-\{A\},\{B\}}
semantic_point_num=${8:-128}
target_num_points=${9:-1024}
output_suffix=${10:--objpc-semantic-pointwise-hybrid-semdebugref}
semantic_input_color_mode=${11:-debug_placeholder}
semantic_forward_mode=${12:-reference}

extra_args=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    extra_args+=(--semantic_model "{A}=${semantic_ckpt_A}")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    extra_args+=(--semantic_model "{B}=${semantic_ckpt_B}")
fi

python scripts/process_data_semantic_pointwise_hybrid.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --semantic_device "${semantic_device}" \
    --semantic_num_points "${semantic_point_num}" \
    --semantic_input_color_mode "${semantic_input_color_mode}" \
    --semantic_forward_mode "${semantic_forward_mode}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    "${extra_args[@]}"
