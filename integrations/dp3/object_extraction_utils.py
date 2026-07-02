from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_scene_info(scene_info_path: str) -> Dict:
    if not os.path.isfile(scene_info_path):
        return {}
    with open(scene_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_asset_spec(asset_spec: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not isinstance(asset_spec, str) or "/base" not in asset_spec:
        return None, None
    model_name, base_id = asset_spec.rsplit("/base", 1)
    try:
        return model_name, int(base_id)
    except ValueError:
        return None, None


def _concat_meshes(meshes: List[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    valid = [mesh for mesh in meshes if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0]
    if not valid:
        return None
    return valid[0] if len(valid) == 1 else trimesh.util.concatenate(valid)


def _load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    if not path.exists():
        return None
    try:
        mesh_or_scene = trimesh.load(path, force="scene")
    except Exception:
        return None
    if isinstance(mesh_or_scene, trimesh.Scene):
        return _concat_meshes(list(mesh_or_scene.geometry.values()))
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    return None


def load_asset_extents(model_name: Optional[str], model_id: Optional[int]) -> Optional[np.ndarray]:
    if model_name is None or model_id is None:
        return None
    model_dir = REPO_ROOT / "assets" / "objects" / model_name
    candidates = [
        model_dir / f"base{model_id}.glb",
        model_dir / f"textured{model_id}.obj",
        model_dir / "visual" / f"base{model_id}.glb",
        model_dir / "visual" / f"textured{model_id}.obj",
        model_dir / "collision" / f"base{model_id}.glb",
        model_dir / "collision" / f"textured{model_id}.obj",
    ]
    for candidate in candidates:
        mesh = _load_mesh(candidate)
        if mesh is None or mesh.bounds is None:
            continue
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        extents = bounds[1] - bounds[0]
        if np.all(extents > 1e-6):
            return extents.astype(np.float32)
    return None


def farthest_point_sample(points: np.ndarray, target_num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    target_num = int(target_num)
    if len(points) == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :3], target_num, axis=0)

    if len(points) < target_num:
        extra_idx = np.random.choice(len(points), size=target_num - len(points), replace=True)
        points = np.concatenate([points, points[extra_idx]], axis=0)
    elif len(points) > target_num:
        xyz = points[:, :3]
        selected = np.zeros((target_num,), dtype=np.int64)
        distances = np.full((len(points),), np.inf, dtype=np.float32)
        farthest = int(np.random.randint(0, len(points)))
        for i in range(target_num):
            selected[i] = farthest
            centroid = xyz[farthest]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = int(np.argmax(distances))
        points = points[selected]
    return points[:, :3].astype(np.float32)


def _project_points_with_matrix(
    points_world: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_hw
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    pts_cam = (extrinsic_cv @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((len(points_world), 2), -1, dtype=np.int64)
    if not np.any(valid):
        return uv, valid
    pts_cam_valid = pts_cam[valid]
    uv_float = (intrinsic_cv @ pts_cam_valid.T).T
    uv_float = uv_float[:, :2] / np.clip(uv_float[:, 2:3], 1e-6, None)
    u = np.round(uv_float[:, 0]).astype(np.int64)
    v = np.round(uv_float[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_idx = np.nonzero(valid)[0]
    valid[valid_idx] &= inside
    uv[valid_idx[inside], 0] = u[inside]
    uv[valid_idx[inside], 1] = v[inside]
    return uv, valid


def sample_mesh_colors_for_points(
    points_world: np.ndarray,
    intrinsic_cv: Optional[np.ndarray],
    extrinsic_cv: Optional[np.ndarray],
    cam2world_gl: Optional[np.ndarray],
    mesh_segmentation: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if mesh_segmentation is None or intrinsic_cv is None:
        return None
    image_hw = mesh_segmentation.shape[:2]
    matrices: List[np.ndarray] = []
    if extrinsic_cv is not None:
        matrices.append(np.asarray(extrinsic_cv, dtype=np.float32))
        try:
            matrices.append(np.linalg.inv(np.asarray(extrinsic_cv, dtype=np.float32)))
        except np.linalg.LinAlgError:
            pass
    if cam2world_gl is not None:
        try:
            matrices.append(np.linalg.inv(np.asarray(cam2world_gl, dtype=np.float32)))
        except np.linalg.LinAlgError:
            pass

    best_colors = None
    best_valid = -1
    for matrix in matrices:
        uv, valid = _project_points_with_matrix(
            points_world=points_world,
            intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32),
            extrinsic_cv=matrix,
            image_hw=image_hw,
        )
        valid_count = int(valid.sum())
        if valid_count <= best_valid:
            continue
        colors = np.zeros((len(points_world), 3), dtype=np.uint8)
        if valid_count > 0:
            colors[valid] = mesh_segmentation[uv[valid, 1], uv[valid, 0], :3]
        best_colors = colors
        best_valid = valid_count
    return best_colors


def _connected_components(points: np.ndarray, eps: float, min_points: int) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    sq_eps = float(eps) ** 2
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    neighbors = dist2 <= sq_eps
    visited = np.zeros((len(points),), dtype=bool)
    components: List[np.ndarray] = []
    for idx in range(len(points)):
        if visited[idx]:
            continue
        queue = [idx]
        visited[idx] = True
        comp = []
        while queue:
            current = queue.pop()
            comp.append(current)
            for next_idx in np.nonzero(neighbors[current])[0].tolist():
                if not visited[next_idx]:
                    visited[next_idx] = True
                    queue.append(next_idx)
        comp_arr = np.asarray(comp, dtype=np.int64)
        if len(comp_arr) >= min_points:
            components.append(comp_arr)
    return components


def _cluster_score(
    points: np.ndarray,
    *,
    target_extents: Optional[np.ndarray],
    prev_centroid: Optional[np.ndarray],
    seg_colors: Optional[np.ndarray],
) -> float:
    extents = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-4)
    score = 0.0
    ref_scale = float(np.max(extents))
    if target_extents is not None:
        obs_ext = np.sort(extents)
        tgt_ext = np.sort(np.maximum(np.asarray(target_extents, dtype=np.float32), 1e-4))
        score += float(np.mean(np.abs(np.log(obs_ext / tgt_ext))))
        ref_scale = max(ref_scale, float(np.max(tgt_ext)))
    else:
        score += 0.25 / max(float(len(points)), 1.0)

    if prev_centroid is not None:
        centroid = points.mean(axis=0)
        score += 0.5 * float(np.linalg.norm(centroid - prev_centroid)) / max(ref_scale, 1e-3)
    else:
        score -= 0.05 * float(points[:, 2].mean())

    if seg_colors is not None and len(seg_colors) == len(points):
        valid = np.any(seg_colors > 0, axis=1)
        if np.any(valid):
            _, counts = np.unique(seg_colors[valid], axis=0, return_counts=True)
            purity = float(counts.max() / counts.sum())
            score -= 0.1 * purity
    return score


def _extract_by_segmentation_colors(
    points_xyz: np.ndarray,
    seg_colors: Optional[np.ndarray],
    *,
    target_extents: Optional[np.ndarray],
    prev_centroid: Optional[np.ndarray],
    min_points: int,
) -> Optional[np.ndarray]:
    if seg_colors is None or len(seg_colors) != len(points_xyz):
        return None
    valid = np.any(seg_colors > 0, axis=1)
    if not np.any(valid):
        return None

    best_points = None
    best_score = np.inf
    unique_colors, counts = np.unique(seg_colors[valid], axis=0, return_counts=True)
    for color, count in zip(unique_colors, counts):
        if int(count) < min_points:
            continue
        color_mask = np.all(seg_colors == color, axis=1)
        cluster_points = points_xyz[color_mask]
        if len(cluster_points) < min_points:
            continue
        score = _cluster_score(
            cluster_points,
            target_extents=target_extents,
            prev_centroid=prev_centroid,
            seg_colors=seg_colors[color_mask],
        )
        if score < best_score:
            best_score = score
            best_points = cluster_points
    return best_points


def extract_object_point_cloud(
    scene_point_cloud: np.ndarray,
    *,
    intrinsic_cv: Optional[np.ndarray],
    extrinsic_cv: Optional[np.ndarray],
    cam2world_gl: Optional[np.ndarray],
    mesh_segmentation: Optional[np.ndarray],
    target_extents: Optional[np.ndarray],
    prev_centroid: Optional[np.ndarray],
    target_num_points: int = 1024,
    cluster_eps: float = 0.04,
    min_cluster_points: int = 24,
    table_quantile: float = 0.08,
    table_margin: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if scene_point_cloud.ndim != 2 or scene_point_cloud.shape[1] < 3:
        return np.zeros((target_num_points, 3), dtype=np.float32), {"mode": "invalid_pointcloud"}

    points_xyz = np.asarray(scene_point_cloud[:, :3], dtype=np.float32)
    points_xyz = points_xyz[~np.isclose(points_xyz, 0.0).all(axis=1)]
    if len(points_xyz) == 0:
        return np.zeros((target_num_points, 3), dtype=np.float32), {"mode": "empty_scene"}

    seg_colors = sample_mesh_colors_for_points(
        points_world=points_xyz,
        intrinsic_cv=intrinsic_cv,
        extrinsic_cv=extrinsic_cv,
        cam2world_gl=cam2world_gl,
        mesh_segmentation=mesh_segmentation,
    )

    table_z = float(np.quantile(points_xyz[:, 2], table_quantile))
    foreground_mask = points_xyz[:, 2] > (table_z + float(table_margin))
    points_fg = points_xyz[foreground_mask]
    seg_fg = seg_colors[foreground_mask] if seg_colors is not None else None
    if len(points_fg) < min_cluster_points:
        points_fg = points_xyz
        seg_fg = seg_colors

    components = _connected_components(points_fg, eps=cluster_eps, min_points=min_cluster_points)
    best_points = None
    best_mode = "full_scene_fallback"
    best_score = np.inf

    for comp in components:
        cluster_points = points_fg[comp]
        cluster_seg = seg_fg[comp] if seg_fg is not None else None
        score = _cluster_score(
            cluster_points,
            target_extents=target_extents,
            prev_centroid=prev_centroid,
            seg_colors=cluster_seg,
        )
        if score < best_score:
            best_score = score
            best_points = cluster_points
            best_mode = "cluster"

    if best_points is None:
        best_points = _extract_by_segmentation_colors(
            points_xyz=points_fg,
            seg_colors=seg_fg,
            target_extents=target_extents,
            prev_centroid=prev_centroid,
            min_points=max(8, min_cluster_points // 2),
        )
        if best_points is not None:
            best_mode = "segmentation_color"

    if best_points is None:
        best_points = points_fg if len(points_fg) > 0 else points_xyz

    sampled = farthest_point_sample(best_points[:, :3], target_num_points)
    metadata = {
        "mode": best_mode,
        "point_count_in": int(len(points_xyz)),
        "point_count_selected": int(len(sampled)),
        "table_z": table_z,
        "has_segmentation": bool(seg_colors is not None),
    }
    return sampled.astype(np.float32), metadata


def summarize_modes(mode_counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(mode_counter.items(), key=lambda kv: kv[0])}


__all__ = [
    "extract_object_point_cloud",
    "load_asset_extents",
    "load_scene_info",
    "parse_asset_spec",
    "summarize_modes",
]
