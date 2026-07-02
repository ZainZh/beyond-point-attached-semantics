import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from incremental_objpc_zarr import append_episode_to_buffer, open_or_reset_replay_buffer
from eef_action_utils import add_eef_preprocess_args, eef_arrays_for_episode, validate_eef_dataset_frame
from object_extraction_utils import summarize_modes
from object_pointcloud_utils import (
    default_placeholder_order,
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    merge_object_point_clouds,
    parse_placeholder_list,
    parse_target_extents,
    valid_xyz_centroid,
)
from pointwise_context_utils import build_context_point_cloud, resolve_context_placeholders
from pointwise_preprocess_meta import load_or_init_meta, reconcile_episode_stats, write_meta
from semantic_feature_utils import compute_semantic_pointwise_cloud, load_semantic_model


def parse_placeholder_model_args(values):
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected PLACEHOLDER=CHECKPOINT, got {item!r}")
        placeholder, checkpoint = item.split("=", 1)
        placeholder = placeholder.strip()
        checkpoint = checkpoint.strip()
        if not placeholder:
            raise ValueError(f"Missing placeholder in {item!r}")
        if not checkpoint:
            continue
        result[placeholder] = checkpoint
    return result


def placeholder_pointcloud_key(placeholder: str) -> str:
    return f"semantic_point_cloud_{placeholder.strip('{}')}"


def infer_placeholder_order(scene_info: dict, first_episode: dict) -> list[str]:
    placeholders = []
    seen = set()

    def append_placeholder(value):
        value = str(value)
        if value not in seen:
            seen.add(value)
            placeholders.append(value)

    for key in default_placeholder_order(scene_info, first_episode):
        append_placeholder(key)

    if isinstance(scene_info, dict):
        episode_keys = sorted(
            [key for key in scene_info.keys() if isinstance(key, str) and key.startswith("episode_")]
        )
        for episode_key in episode_keys:
            episode_info = scene_info.get(episode_key, {})
            if not isinstance(episode_info, dict):
                continue
            object_pc_info = episode_info.get("object_pointcloud", {})
            targets = object_pc_info.get("targets", {}) if isinstance(object_pc_info, dict) else {}
            if isinstance(targets, dict):
                for placeholder in targets.keys():
                    append_placeholder(placeholder)

    object_pointcloud = first_episode.get("object_pointcloud", {})
    if isinstance(object_pointcloud, dict):
        for placeholder in sorted(object_pointcloud.keys()):
            append_placeholder(placeholder)

    return placeholders


def build_parser():
    parser = argparse.ArgumentParser(
        description="Process RoboTwin episodes into DP3 zarr using context object point clouds and point-wise semantic embeddings."
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--object_placeholders", type=str, default="")
    parser.add_argument("--semantic_model", action="append", default=[], help="Repeated PLACEHOLDER=CHECKPOINT mapping.")
    parser.add_argument("--semantic_device", type=str, default="cuda:0")
    parser.add_argument("--semantic_num_points", type=int, default=128)
    parser.add_argument(
        "--semantic_input_color_mode",
        choices=["debug_placeholder", "stored_scaled", "stored"],
        default="debug_placeholder",
    )
    parser.add_argument("--semantic_forward_mode", choices=["reference", "dp3"], default="reference")
    parser.add_argument("--output_suffix", type=str, default="-objpc-semantic-pointwise")
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--cluster_eps", type=float, default=0.04)
    parser.add_argument("--min_cluster_points", type=int, default=24)
    parser.add_argument("--table_quantile", type=float, default=0.08)
    parser.add_argument("--table_margin", type=float, default=0.01)
    parser.add_argument("--save_placeholder_point_clouds", action="store_true")
    parser.add_argument("--keep_feature_placeholders_in_context", action="store_true")
    add_eef_preprocess_args(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    task_name = args.task_name
    task_config = args.task_config
    num = int(args.expert_data_num)
    load_dir = os.path.join("../../data", str(task_name), str(task_config))
    validate_eef_dataset_frame(
        action_mode=args.action_mode,
        eef_frame_mode=args.eef_frame_mode,
        load_dir=load_dir,
    )
    save_dir = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}.zarr"
    meta_path = f"./data/{task_name}-{task_config}-{num}{args.output_suffix}_meta.json"

    scene_info = load_scene_info(os.path.join(load_dir, "scene_info.json"))
    first_episode = load_hdf5(os.path.join(load_dir, "data/episode0.hdf5"))
    placeholders = parse_placeholder_list(args.object_placeholders)
    if len(placeholders) == 0:
        placeholders = infer_placeholder_order(scene_info, first_episode)
    if len(placeholders) == 0:
        raise RuntimeError("No object_pointcloud placeholders were found in the collected data.")

    model_paths = parse_placeholder_model_args(args.semantic_model)
    if len(model_paths) == 0:
        raise RuntimeError("Semantic pointwise preprocessing requires at least one --semantic_model PLACEHOLDER=CHECKPOINT.")

    device = torch.device(args.semantic_device if torch.cuda.is_available() else "cpu")
    model_by_placeholder = {}
    for placeholder, checkpoint in model_paths.items():
        model_by_placeholder[placeholder] = load_semantic_model(
            checkpoint=checkpoint,
            device=device,
        )
    semantic_feat_dims = {
        placeholder: int(artifacts["sem_embedding_dim"])
        for placeholder, artifacts in model_by_placeholder.items()
    }
    unique_semantic_dims = sorted(set(semantic_feat_dims.values()))
    if len(unique_semantic_dims) != 1:
        raise RuntimeError(
            f"Semantic pointwise preprocessing requires consistent semantic embedding dimensions, got {semantic_feat_dims!r}."
        )
    semantic_feat_dim = int(unique_semantic_dims[0])

    feature_placeholders = [placeholder for placeholder in placeholders if placeholder in model_by_placeholder]
    context_placeholders = resolve_context_placeholders(
        placeholders,
        feature_placeholders,
        keep_feature_placeholders_in_context=bool(args.keep_feature_placeholders_in_context),
    )
    if len(feature_placeholders) == 0:
        raise RuntimeError("None of the requested object placeholders has a semantic model configured.")

    replay_buffer, start_episode = open_or_reset_replay_buffer(save_dir)
    meta = load_or_init_meta(
        meta_path,
        task_name=task_name,
        task_config=task_config,
        expert_data_num=num,
    )
    meta.update(
        {
            "output_zarr": str(Path(save_dir).resolve()),
            "object_placeholders": placeholders,
            "context_placeholders": context_placeholders,
            "feature_placeholders": feature_placeholders,
            "semantic_models": model_paths,
            "semantic_num_points": int(args.semantic_num_points),
            "semantic_feat_dim": semantic_feat_dim,
            "semantic_feat_dims": semantic_feat_dims,
            "semantic_input_color_mode": str(args.semantic_input_color_mode),
            "semantic_forward_mode": str(args.semantic_forward_mode),
            "keep_feature_placeholders_in_context": bool(args.keep_feature_placeholders_in_context),
        }
    )
    episode_stats = reconcile_episode_stats(meta.get("episodes", []), start_episode=start_episode)

    print(f"Resuming semantic pointwise preprocessing from episode {start_episode + 1} / {num}")

    for current_ep in range(start_episode, num):
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        episode = load_hdf5(load_path)
        vector_all = episode["vector"]
        eef_arrays = eef_arrays_for_episode(args, episode)
        prev_centroids = {placeholder: None for placeholder in placeholders}
        local_modes = {placeholder: Counter() for placeholder in placeholders}
        asset_specs = {}
        has_exact = {}

        for placeholder in placeholders:
            _, asset_spec = parse_target_extents(scene_info, current_ep, placeholder)
            asset_specs[placeholder] = asset_spec
            has_exact[placeholder] = placeholder in episode["object_pointcloud"]

        episode_point_cloud_arrays = []
        episode_semantic_arrays = {placeholder: [] for placeholder in feature_placeholders}
        episode_placeholder_point_cloud_arrays = {placeholder: [] for placeholder in placeholders}
        episode_state_arrays = []
        episode_joint_action_arrays = []

        for frame_idx in range(vector_all.shape[0]):
            per_placeholder_point_clouds = {}
            for placeholder in placeholders:
                target_extents, _ = parse_target_extents(scene_info, current_ep, placeholder)
                object_pc, extract_meta = extract_placeholder_point_cloud(
                    episode,
                    frame_idx=frame_idx,
                    placeholder=placeholder,
                    target_num_points=int(args.target_num_points),
                    target_extents=target_extents,
                    prev_centroid=prev_centroids[placeholder],
                    cluster_eps=float(args.cluster_eps),
                    min_cluster_points=int(args.min_cluster_points),
                    table_quantile=float(args.table_quantile),
                    table_margin=float(args.table_margin),
                )
                per_placeholder_point_clouds[placeholder] = object_pc
                local_modes[placeholder][str(extract_meta.get("mode", "unknown"))] += 1
                centroid = valid_xyz_centroid(object_pc)
                if centroid is not None:
                    prev_centroids[placeholder] = centroid

            if frame_idx != vector_all.shape[0] - 1:
                context_point_cloud, _ = build_context_point_cloud(
                    per_placeholder_point_clouds,
                    placeholders=placeholders,
                    feature_placeholders=feature_placeholders,
                    target_num_points=int(args.target_num_points),
                    keep_feature_placeholders_in_context=bool(args.keep_feature_placeholders_in_context),
                )
                episode_point_cloud_arrays.append(context_point_cloud.astype(np.float32))
                episode_state_arrays.append(eef_arrays[0][frame_idx] if eef_arrays is not None else vector_all[frame_idx])

                for placeholder in placeholders:
                    object_pc = per_placeholder_point_clouds[placeholder]
                    if args.save_placeholder_point_clouds:
                        episode_placeholder_point_cloud_arrays[placeholder].append(object_pc.astype(np.float32))

                for placeholder in feature_placeholders:
                    episode_semantic_arrays[placeholder].append(
                        compute_semantic_pointwise_cloud(
                            artifacts=model_by_placeholder[placeholder],
                            object_point_cloud=per_placeholder_point_clouds[placeholder],
                            target_num_points=int(args.semantic_num_points),
                            placeholder=placeholder,
                            semantic_input_color_mode=str(args.semantic_input_color_mode),
                            semantic_forward_mode=str(args.semantic_forward_mode),
                        ).astype(np.float32)
                    )

            if frame_idx != 0:
                episode_joint_action_arrays.append(eef_arrays[1][frame_idx - 1] if eef_arrays is not None else vector_all[frame_idx])

        episode_data = {
            "point_cloud": np.asarray(episode_point_cloud_arrays, dtype=np.float32),
            "state": np.asarray(episode_state_arrays, dtype=np.float32),
            "action": np.asarray(episode_joint_action_arrays, dtype=np.float32),
        }
        for placeholder in feature_placeholders:
            point_cloud_key = placeholder_pointcloud_key(placeholder)
            episode_data[point_cloud_key] = np.asarray(episode_semantic_arrays[placeholder], dtype=np.float32)
        if args.save_placeholder_point_clouds:
            for placeholder in placeholders:
                point_cloud_key = f"object_point_cloud_{placeholder.strip('{}')}"
                episode_data[point_cloud_key] = np.asarray(
                    episode_placeholder_point_cloud_arrays[placeholder],
                    dtype=np.float32,
                )

        append_episode_to_buffer(replay_buffer, episode_data)

        episode_record = {
            "episode": current_ep,
            "placeholders": placeholders,
            "context_placeholders": context_placeholders,
            "feature_placeholders": feature_placeholders,
            "target_assets": asset_specs,
            "has_exact_object_pointcloud": has_exact,
            "has_semantic_model": {
                placeholder: placeholder in model_by_placeholder
                for placeholder in placeholders
            },
            "modes": {
                placeholder: summarize_modes(local_modes[placeholder])
                for placeholder in placeholders
            },
        }
        if len(episode_stats) == current_ep:
            episode_stats.append(episode_record)
        else:
            episode_stats[current_ep] = episode_record
        meta["episodes"] = episode_stats
        write_meta(meta_path, meta)

    print()
    write_meta(meta_path, meta)


if __name__ == "__main__":
    main()
