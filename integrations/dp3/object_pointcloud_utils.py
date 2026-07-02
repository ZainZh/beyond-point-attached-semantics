import os
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


def _read_optional(group: h5py.Group, path: str):
    if path not in group:
        return None
    return group[path][()]


def load_hdf5(dataset_path: str) -> dict:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        try:
            head_group = root["/observation/head_camera"]
        except KeyError:
            head_group = None
        data = {
            "vector": root["/joint_action/vector"][()],
            "control": _read_optional(root, "/joint_action/control"),
            "eef_pose_base": _read_optional(root, "/eef_action/base_pose6"),
            "pointcloud": root["/pointcloud"][()],
            "intrinsic_cv": _read_optional(head_group, "intrinsic_cv") if head_group is not None else None,
            "extrinsic_cv": _read_optional(head_group, "extrinsic_cv") if head_group is not None else None,
            "cam2world_gl": _read_optional(head_group, "cam2world_gl") if head_group is not None else None,
            "mesh_segmentation": _read_optional(head_group, "mesh_segmentation") if head_group is not None else None,
            "object_pointcloud": {},
        }
        if "object_pointcloud" in root:
            object_group = root["/object_pointcloud"]
            for key in object_group.keys():
                data["object_pointcloud"][str(key)] = object_group[key][()]
    return data


def frame_matrix(arr, idx: int):
    if arr is None:
        return None
    if arr.ndim == 3:
        return arr[idx]
    return arr


def frame_image(arr, idx: int):
    if arr is None:
        return None
    if arr.ndim == 4:
        return arr[idx]
    return arr


def parse_target_extents(scene_info: dict, episode_idx: int, placeholder: str):
    from object_extraction_utils import load_asset_extents, parse_asset_spec

    episode_info = scene_info.get(f"episode_{episode_idx}", {}) if isinstance(scene_info, dict) else {}
    info_dict = episode_info.get("info", {}) if isinstance(episode_info, dict) else {}
    asset_spec = info_dict.get(placeholder, None)
    model_name, model_id = parse_asset_spec(asset_spec)
    extents = load_asset_extents(model_name, model_id)
    return extents, asset_spec


def load_scene_info(scene_info_path: str) -> Dict:
    from object_extraction_utils import load_scene_info as _load_scene_info

    return _load_scene_info(scene_info_path)


def ensure_point_cloud_channels(point_cloud: np.ndarray, channels: int = 6) -> np.ndarray:
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected point cloud with shape (N, C), got {point_cloud.shape}")
    if point_cloud.shape[1] == channels:
        return point_cloud.astype(np.float32)
    if point_cloud.shape[1] > channels:
        return point_cloud[:, :channels].astype(np.float32)
    if point_cloud.shape[1] == 3 and channels == 6:
        padding = np.zeros((point_cloud.shape[0], 3), dtype=np.float32)
        return np.concatenate([point_cloud, padding], axis=1)
    raise ValueError(f"Cannot coerce point cloud shape {point_cloud.shape} to {channels} channels")


def strip_zero_points(point_cloud: np.ndarray) -> np.ndarray:
    point_cloud = ensure_point_cloud_channels(point_cloud, channels=6)
    xyz = point_cloud[:, :3]
    nonzero_mask = ~np.isclose(xyz, 0.0).all(axis=1)
    return point_cloud[nonzero_mask]


def _farthest_point_sample_indices(points_xyz: np.ndarray, target_num: int) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    num_points = len(points_xyz)
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)
    if num_points <= target_num:
        return np.arange(num_points, dtype=np.int64)

    selected = np.zeros((target_num,), dtype=np.int64)
    distances = np.full((num_points,), np.inf, dtype=np.float32)
    farthest = int(np.random.randint(0, num_points))
    for i in range(target_num):
        selected[i] = farthest
        centroid = points_xyz[farthest]
        dist = np.sum((points_xyz - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def resample_point_cloud(point_cloud: np.ndarray, target_num_points: int) -> np.ndarray:
    point_cloud = strip_zero_points(point_cloud)
    if target_num_points <= 0:
        return point_cloud.astype(np.float32)
    if len(point_cloud) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32)
    if len(point_cloud) < target_num_points:
        reps = target_num_points - len(point_cloud)
        extra_idx = np.random.choice(len(point_cloud), size=reps, replace=True)
        point_cloud = np.concatenate([point_cloud, point_cloud[extra_idx]], axis=0)
        return point_cloud.astype(np.float32)
    if len(point_cloud) == target_num_points:
        return point_cloud.astype(np.float32)
    selected = _farthest_point_sample_indices(point_cloud[:, :3], target_num_points)
    return point_cloud[selected].astype(np.float32)


def merge_object_point_clouds(point_clouds: Iterable[np.ndarray], target_num_points: int) -> np.ndarray:
    cleaned = []
    for point_cloud in point_clouds:
        cleaned_pc = strip_zero_points(point_cloud)
        if len(cleaned_pc) > 0:
            cleaned.append(cleaned_pc)
    if len(cleaned) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32)
    merged = cleaned[0] if len(cleaned) == 1 else np.concatenate(cleaned, axis=0)
    return resample_point_cloud(merged, target_num_points)


def valid_xyz_centroid(point_cloud: np.ndarray) -> Optional[np.ndarray]:
    point_cloud = ensure_point_cloud_channels(point_cloud, channels=6)
    xyz = point_cloud[:, :3]
    nonzero_mask = ~np.isclose(xyz, 0.0).all(axis=1)
    if not np.any(nonzero_mask):
        return None
    return xyz[nonzero_mask].mean(axis=0)


def parse_placeholder_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        values: List[str] = []
        for item in text:
            values.extend(parse_placeholder_list(item))
        return values
    return [item.strip() for item in str(text).split(",") if item.strip()]


def default_placeholder_order(scene_info: dict, episode: dict) -> List[str]:
    try:
        targets = scene_info["episode_0"]["object_pointcloud"]["targets"]
        if isinstance(targets, dict) and len(targets) > 0:
            return [str(key) for key in targets.keys()]
    except Exception:
        pass
    return sorted(str(key) for key in episode.get("object_pointcloud", {}).keys())


def extract_placeholder_point_cloud(
    episode: Dict[str, np.ndarray],
    *,
    frame_idx: int,
    placeholder: str,
    target_num_points: int,
    target_extents: Optional[np.ndarray] = None,
    prev_centroid: Optional[np.ndarray] = None,
    cluster_eps: float = 0.04,
    min_cluster_points: int = 24,
    table_quantile: float = 0.08,
    table_margin: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, object]]:
    exact_object_pc_all = episode["object_pointcloud"].get(placeholder)
    if exact_object_pc_all is not None:
        object_pc = ensure_point_cloud_channels(exact_object_pc_all[frame_idx], channels=6)
        object_pc = resample_point_cloud(object_pc, target_num_points)
        return object_pc, {
            "mode": "exact_collected",
            "point_count_in": int(object_pc.shape[0]),
            "point_count_selected": int(object_pc.shape[0]),
        }

    scene_pc = ensure_point_cloud_channels(episode["pointcloud"][frame_idx], channels=6)
    from object_extraction_utils import extract_object_point_cloud

    object_pc_xyz, extract_meta = extract_object_point_cloud(
        scene_point_cloud=scene_pc,
        intrinsic_cv=frame_matrix(episode["intrinsic_cv"], frame_idx),
        extrinsic_cv=frame_matrix(episode["extrinsic_cv"], frame_idx),
        cam2world_gl=frame_matrix(episode["cam2world_gl"], frame_idx),
        mesh_segmentation=frame_image(episode["mesh_segmentation"], frame_idx),
        target_extents=target_extents,
        prev_centroid=prev_centroid,
        target_num_points=int(target_num_points),
        cluster_eps=float(cluster_eps),
        min_cluster_points=int(min_cluster_points),
        table_quantile=float(table_quantile),
        table_margin=float(table_margin),
    )
    object_pc = ensure_point_cloud_channels(object_pc_xyz, channels=6)
    return object_pc, extract_meta


__all__ = [
    "default_placeholder_order",
    "ensure_point_cloud_channels",
    "extract_placeholder_point_cloud",
    "frame_image",
    "frame_matrix",
    "load_hdf5",
    "load_scene_info",
    "merge_object_point_clouds",
    "parse_placeholder_list",
    "parse_target_extents",
    "resample_point_cloud",
    "strip_zero_points",
    "valid_xyz_centroid",
]
