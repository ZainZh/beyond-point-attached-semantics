from __future__ import annotations

import argparse
import colorsys
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import trimesh

from my_datasets import PartNextUniversalFieldDataset, partnext_universal_field_collate_fn
from my_datasets.partnext_canonical_field import (
    UTONIA,
    _concatenate_meshes,
    _load_geometry_meshes,
    _sample_surface_points,
)
from my_datasets.partnext_occupancy import _sample_queries
from my_datasets.partnext_universal_field import _normalize_multiple_arrays
from models import UtoniaUniversalFieldNet
from myutils import move_to_device

IGNORE_LABEL = -1


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return path


FEATURE_METHODS = ("semantic", "utonia", "dinov2")
FEATURE_EMBEDDING_KEYS = {
    "semantic": "sem_embeddings",
    "utonia": "utonia_features",
    "dinov2": "dinov2_features",
}
FEATURE_OUTPUT_STEMS = {
    "semantic": "surface_sem_embedding",
    "utonia": "surface_utonia_feature",
    "dinov2": "surface_dinov2_feature",
}


def parse_feature_methods(methods: Sequence[str] | str | None = None) -> list[str]:
    if methods is None:
        return ["semantic"]
    values = [methods] if isinstance(methods, str) else list(methods)
    parsed: list[str] = []
    for value in values:
        for token in str(value).split(","):
            method = token.strip().lower()
            if not method:
                continue
            if method not in FEATURE_METHODS:
                raise ValueError(
                    f"feature method must be one of {list(FEATURE_METHODS)}, got: {method}"
                )
            if method not in parsed:
                parsed.append(method)
    return parsed or ["semantic"]


def _feature_embedding_key(method: str) -> str:
    return FEATURE_EMBEDDING_KEYS[method]


def _feature_output_stem(method: str) -> str:
    return FEATURE_OUTPUT_STEMS[method]


def _feature_local_file_key(method: str) -> str:
    return f"{_feature_output_stem(method)}_pca"


def _feature_joint_file_key(method: str) -> str:
    return f"{_feature_output_stem(method)}_joint_pca"


def _build_palette(num_classes: int) -> np.ndarray:
    colors: list[list[int]] = []
    for index in range(max(num_classes, 1)):
        # 色相 (Hue)：保持原来的黄金分割比例不变，这是生成不重复颜色的最佳方式
        hue = (index * 0.6180339887498949) % 1.0

        # 修改 1 - 大幅提高饱和度 (Saturation)：
        # 将区间从 [0.25~0.5] 提升到 [0.7~1.0]。颜色会从“发灰”变为“极其饱满鲜艳”。
        saturation = 0.7 + 0.3 * ((index % 3) / 2.0)

        # 修改 2 - 调整明度并增加相邻差异 (Value)：
        # 将区间设为 [0.85~1.0]，并在相邻索引间交替（%2），从而最大化相邻生成的颜色对比度。
        value = 0.85 + 0.15 * (index % 2)

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(channel * 255) for channel in rgb])
    return np.asarray(colors, dtype=np.uint8)

def _probability_to_rgb(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
    red = probability
    green = 0.25 + 0.75 * probability
    blue = 1.0 - probability
    rgb = np.stack([red, green, blue], axis=1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _pca_to_rgb(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings.astype(np.float32) - embeddings.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        repeated = np.repeat(centered[:, :1], 3, axis=1)
        return np.clip((repeated + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

    matrix = torch.from_numpy(centered)
    _, _, components = torch.pca_lowrank(matrix, q=min(3, matrix.shape[1]))
    projected = (matrix @ components[:, :3]).numpy()
    projected = projected - projected.min(axis=0, keepdims=True)
    denom = projected.max(axis=0, keepdims=True)
    denom[denom < 1e-8] = 1.0
    projected = projected / denom
    return np.clip(projected * 255.0, 0.0, 255.0).astype(np.uint8)


def _fit_shared_pca_projection(embedding_sets: list[np.ndarray]) -> dict[str, np.ndarray | int]:
    valid_sets = [
        np.asarray(embeddings, dtype=np.float32)
        for embeddings in embedding_sets
        if embeddings is not None and len(embeddings) > 0
    ]
    if not valid_sets:
        raise ValueError("Shared PCA requires at least one non-empty embedding set.")

    all_embeddings = np.concatenate(valid_sets, axis=0).astype(np.float32, copy=False)
    mean = all_embeddings.mean(axis=0, keepdims=True)
    centered = all_embeddings - mean

    projection_dim = min(3, centered.shape[1]) if centered.ndim == 2 else 1
    components = np.zeros((centered.shape[1], 3), dtype=np.float32)

    if centered.shape[0] >= 3 and centered.shape[1] >= 2:
        matrix = torch.from_numpy(centered)
        _, _, basis = torch.pca_lowrank(matrix, q=projection_dim)
        basis_np = basis[:, :projection_dim].numpy().astype(np.float32, copy=False)
        components[:, :projection_dim] = basis_np
    else:
        components[:projection_dim, :projection_dim] = np.eye(projection_dim, dtype=np.float32)

    projected = centered @ components
    projected_min = projected.min(axis=0, keepdims=True)
    projected_max = projected.max(axis=0, keepdims=True)
    span = projected_max - projected_min
    span[span < 1e-8] = 1.0

    return {
        "mean": mean.astype(np.float32, copy=False),
        "components": components.astype(np.float32, copy=False),
        "projected_min": projected_min.astype(np.float32, copy=False),
        "projected_span": span.astype(np.float32, copy=False),
        "embedding_dim": int(all_embeddings.shape[1]),
        "num_points": int(all_embeddings.shape[0]),
    }


def _apply_shared_pca_projection_to_rgb(
    embeddings: np.ndarray,
    projection: dict[str, np.ndarray | int],
) -> np.ndarray:
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    centered = embeddings_np - np.asarray(projection["mean"], dtype=np.float32)
    projected = centered @ np.asarray(projection["components"], dtype=np.float32)
    projected = projected - np.asarray(projection["projected_min"], dtype=np.float32)
    projected = projected / np.asarray(projection["projected_span"], dtype=np.float32)
    return np.clip(projected * 255.0, 0.0, 255.0).astype(np.uint8)


def _interpolate_support_features_to_query(
        *,
        support_points: torch.Tensor,
        support_features: torch.Tensor,
        query_points: torch.Tensor,
        knn: int = 3,
        chunk_size: int = 1024,
) -> torch.Tensor:
    if support_points.ndim != 2 or query_points.ndim != 2:
        raise ValueError("support_points and query_points must have shape (N, 3).")
    if support_features.ndim != 2:
        raise ValueError("support_features must have shape (N, C).")
    if support_points.shape[0] != support_features.shape[0]:
        raise ValueError(
            f"support point/feature mismatch: {support_points.shape[0]} vs {support_features.shape[0]}"
        )
    if support_points.shape[0] == 0:
        raise ValueError("Cannot interpolate from an empty support point set.")

    k = max(1, min(int(knn), int(support_points.shape[0])))
    chunk = max(1, int(chunk_size))
    results: list[torch.Tensor] = []
    for start in range(0, int(query_points.shape[0]), chunk):
        query_chunk = query_points[start:start + chunk]
        distances = torch.cdist(query_chunk, support_points)
        if k == 1:
            nearest = distances.argmin(dim=1)
            results.append(support_features[nearest])
            continue
        nearest_distances, nearest_indices = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
        )
        weights = torch.reciprocal(nearest_distances.clamp_min(1e-6))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        gathered = support_features[nearest_indices]
        results.append((gathered * weights.unsqueeze(-1)).sum(dim=1))
    return torch.cat(results, dim=0)


def _extract_raw_utonia_query_features(
        *,
        model: UtoniaUniversalFieldNet,
        batch: dict,
        sample: dict,
        device: torch.device,
        knn: int = 3,
        chunk_size: int = 1024,
) -> np.ndarray:
    support_points = batch["support_points_view1"][0].to(device=device, dtype=torch.float32)
    query_points = batch["query_surface_points_view1"][0].to(device=device, dtype=torch.float32)
    num_support_points = int(sample["support_points_view1"].shape[0])
    with torch.no_grad():
        support_features = model.feature_extractor(
            batch["utonia_input_view1"],
            num_support_points=num_support_points,
        )[0].to(device=device, dtype=torch.float32)
        query_features = _interpolate_support_features_to_query(
            support_points=support_points,
            support_features=support_features,
            query_points=query_points,
            knn=int(knn),
            chunk_size=int(chunk_size),
        )
    return query_features.detach().cpu().numpy().astype(np.float32, copy=False)


def _render_view_directions(num_views: int) -> list[np.ndarray]:
    base = np.asarray(
        [
            [1.0, -1.0, 0.65],
            [-1.0, -1.0, 0.65],
            [0.0, 1.0, 0.85],
            [1.0, 1.0, 0.65],
            [-1.0, 1.0, 0.65],
        ],
        dtype=np.float32,
    )
    directions: list[np.ndarray] = []
    for index in range(max(1, int(num_views))):
        direction = base[index % len(base)].copy()
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        directions.append(direction)
    return directions


def _orthographic_camera_frame(view_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(view_direction, dtype=np.float32).reshape(3)
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up_hint, forward)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(forward, right)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    return right.astype(np.float32), up.astype(np.float32), forward.astype(np.float32)


def _project_orthographic(
        points: np.ndarray,
        *,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        extent: float,
        image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    x = points_np @ right
    y = points_np @ up
    depth = points_np @ forward
    scale = max(float(extent), 1e-6)
    pixels = np.stack(
        [
            (x / scale * 0.45 + 0.5) * float(image_size - 1),
            (0.5 - y / scale * 0.45) * float(image_size - 1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    valid = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= float(image_size - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= float(image_size - 1))
    )
    return pixels, valid, depth.astype(np.float32, copy=False)


def _pseudo_point_colors(points: np.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points_np.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    minimum = points_np.min(axis=0, keepdims=True)
    span = points_np.max(axis=0, keepdims=True) - minimum
    span[span < 1e-6] = 1.0
    colors = 0.2 + 0.75 * ((points_np - minimum) / span)
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


def _render_pseudo_rgb_view(
        *,
        support_points: np.ndarray,
        query_points: np.ndarray,
        view_direction: np.ndarray,
        image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support_np = np.asarray(support_points, dtype=np.float32).reshape(-1, 3)
    query_np = np.asarray(query_points, dtype=np.float32).reshape(-1, 3)
    right, up, forward = _orthographic_camera_frame(view_direction)
    combined = np.concatenate([support_np, query_np], axis=0) if len(query_np) else support_np
    xy = np.stack([combined @ right, combined @ up], axis=1)
    extent = max(float(np.abs(xy).max()) * 1.08, 1e-3)

    support_pixels, support_valid, support_depth = _project_orthographic(
        support_np,
        right=right,
        up=up,
        forward=forward,
        extent=extent,
        image_size=image_size,
    )
    query_pixels, query_valid, _ = _project_orthographic(
        query_np,
        right=right,
        up=up,
        forward=forward,
        extent=extent,
        image_size=image_size,
    )

    rgb = np.full((image_size, image_size, 3), 238, dtype=np.uint8)
    depth_buffer = np.full((image_size, image_size), -np.inf, dtype=np.float32)
    colors = _pseudo_point_colors(support_np)
    radius = max(1, int(round(float(image_size) / 224.0)))
    for index in np.flatnonzero(support_valid):
        px = int(round(float(support_pixels[index, 0])))
        py = int(round(float(support_pixels[index, 1])))
        depth = float(support_depth[index])
        shade = 0.65 + 0.35 * max(0.0, float(np.dot(support_np[index], forward)))
        color = np.clip(colors[index].astype(np.float32) * shade, 0.0, 255.0).astype(np.uint8)
        for dy in range(-radius, radius + 1):
            yy = py + dy
            if yy < 0 or yy >= image_size:
                continue
            for dx in range(-radius, radius + 1):
                xx = px + dx
                if xx < 0 or xx >= image_size:
                    continue
                if depth >= float(depth_buffer[yy, xx]):
                    depth_buffer[yy, xx] = depth
                    rgb[yy, xx] = color
    return rgb, query_pixels, query_valid


def _load_dinov2_backend(
        *,
        model_name: str = "dinov2_vits14",
        device: str | torch.device = "cuda",
        image_size: int = 224,
):
    import torch.nn.functional as F

    model = torch.hub.load("facebookresearch/dinov2", str(model_name))
    device_obj = torch.device(device)
    model.eval().to(device_obj)
    patch_size = getattr(model, "patch_size", 14)
    if isinstance(patch_size, tuple):
        patch_size = int(patch_size[0])
    patch_size = int(patch_size)
    resolved_size = max(patch_size, (int(image_size) // patch_size) * patch_size)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device_obj).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device_obj).view(1, 3, 1, 1)

    def extract_patch_features(rgb: np.ndarray) -> dict:
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        tensor = torch.from_numpy(rgb_np).to(device=device_obj, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        tensor = F.interpolate(tensor, size=(resolved_size, resolved_size), mode="bilinear", align_corners=False)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                output = model.forward_features(tensor)
                patch_tokens = output.get("x_norm_patchtokens") if isinstance(output, dict) else output
                if patch_tokens is None and isinstance(output, dict):
                    patch_tokens = output.get("patch_tokens")
            else:
                patch_tokens = model(tensor)
        if patch_tokens is None:
            raise RuntimeError(f"DINOv2 model {model_name} did not return patch tokens.")
        patch_tokens = patch_tokens.detach().float().cpu().numpy()
        grid_h = resolved_size // patch_size
        grid_w = resolved_size // patch_size
        patch_tokens = patch_tokens.reshape(1, grid_h, grid_w, -1)[0]
        return {
            "features": patch_tokens.astype(np.float32, copy=False),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "image_h": int(rgb_np.shape[0]),
            "image_w": int(rgb_np.shape[1]),
        }

    return extract_patch_features


def _extract_pseudo_dinov2_query_features(
        *,
        dinov2_backend,
        support_points: np.ndarray,
        query_points: np.ndarray,
        render_views: int = 3,
        image_size: int = 224,
) -> np.ndarray:
    if dinov2_backend is None:
        raise RuntimeError("DINOv2 feature extraction requested but no DINOv2 backend was initialized.")

    query_np = np.asarray(query_points, dtype=np.float32).reshape(-1, 3)
    feature_sum = None
    valid_count = np.zeros((query_np.shape[0], 1), dtype=np.float32)
    for direction in _render_view_directions(int(render_views)):
        rgb, pixels, valid = _render_pseudo_rgb_view(
            support_points=support_points,
            query_points=query_np,
            view_direction=direction,
            image_size=int(image_size),
        )
        patch = dinov2_backend(rgb)
        patch_features = np.asarray(patch["features"], dtype=np.float32)
        if feature_sum is None:
            feature_sum = np.zeros((query_np.shape[0], patch_features.shape[-1]), dtype=np.float32)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.shape[0] == 0:
            continue
        px = pixels[valid_indices, 0] / max(float(patch["image_w"]), 1.0)
        py = pixels[valid_indices, 1] / max(float(patch["image_h"]), 1.0)
        patch_x = np.clip((px * int(patch["grid_w"])).astype(np.int64), 0, int(patch["grid_w"]) - 1)
        patch_y = np.clip((py * int(patch["grid_h"])).astype(np.int64), 0, int(patch["grid_h"]) - 1)
        feature_sum[valid_indices] += patch_features[patch_y, patch_x]
        valid_count[valid_indices, 0] += 1.0

    if feature_sum is None:
        raise RuntimeError("DINOv2 feature extraction did not produce any patch features.")
    valid_rows = valid_count[:, 0] > 0.0
    if np.any(valid_rows):
        feature_sum[valid_rows] /= valid_count[valid_rows]
        fallback = feature_sum[valid_rows].mean(axis=0, keepdims=True)
    else:
        fallback = np.zeros((1, feature_sum.shape[1]), dtype=np.float32)
    if np.any(~valid_rows):
        feature_sum[~valid_rows] = fallback
    return feature_sum.astype(np.float32, copy=False)


def _export_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    cloud.export(path)


def _export_joint_feature_pca_outputs(sample_payloads: list[dict]) -> dict[str, dict[str, int]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for payload in sample_payloads:
        method = str(payload.get("feature_method", "semantic"))
        if method not in FEATURE_METHODS:
            raise ValueError(f"Unsupported feature method: {method}")
        groups.setdefault(method, []).append(payload)

    summaries: dict[str, dict[str, int]] = {}
    for method, method_payloads in groups.items():
        projection = _fit_shared_pca_projection(
            [payload["feature_embeddings"] for payload in method_payloads]
        )
        file_key = _feature_joint_file_key(method)
        file_name = f"{file_key}.ply"
        for payload in method_payloads:
            output_dir = Path(payload["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            colors = _apply_shared_pca_projection_to_rgb(
                payload["feature_embeddings"],
                projection,
            )
            joint_path = output_dir / file_name
            _export_point_cloud(joint_path, payload["surface_points"], colors)
            payload["summary"]["files"][file_key] = str(joint_path.resolve())
            summary_path = payload.get("summary_path")
            if summary_path is not None:
                Path(summary_path).write_text(
                    json.dumps(payload["summary"], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        summaries[method] = {
            "num_samples": len(method_payloads),
            "embedding_dim": int(projection["embedding_dim"]),
            "num_points": int(projection["num_points"]),
        }
    return summaries


def _export_joint_semantic_pca_outputs(sample_payloads: list[dict]) -> dict[str, int]:
    feature_payloads = [
        {
            **payload,
            "feature_method": "semantic",
            "feature_embeddings": payload["sem_embeddings"],
        }
        for payload in sample_payloads
    ]
    summary = _export_joint_feature_pca_outputs(feature_payloads)
    return summary["semantic"]


def _labels_to_named_histogram(
        label_indices: np.ndarray,
        label_names: list[str],
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    if label_indices.size == 0:
        return histogram
    unique, counts = np.unique(label_indices, return_counts=True)
    for label_index, count in zip(unique, counts, strict=True):
        if 0 <= int(label_index) < len(label_names):
            histogram[label_names[int(label_index)]] = int(count)
    return histogram


def _mean_or_none(values: list[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return float(np.mean(valid_values))


def _resolve_point_cloud_query_surface_count(
        cli_args: argparse.Namespace,
        checkpoint_args: dict,
) -> int:
    if cli_args.num_query_surface_points is not None:
        return int(cli_args.num_query_surface_points)
    return 5000


def _resolve_point_cloud_paths(cli_args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    point_cloud_path = getattr(cli_args, "point_cloud_path", None)
    if point_cloud_path is not None:
        paths.append(Path(point_cloud_path))
    paths.extend(Path(path) for path in (getattr(cli_args, "point_cloud_paths", None) or []))

    mesh_path = getattr(cli_args, "mesh_path", None)
    if not paths and mesh_path is not None and Path(mesh_path).suffix.lower() == ".ply":
        paths.append(Path(mesh_path))
    return paths


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize predictions from a trained Utonia universal-field checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/semantic_field/best.pt"))
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=None,
        help="Direct mesh path to run instead of selecting a PartNext dataset sample.",
    )
    parser.add_argument(
        "--point-cloud-path",
        type=Path,
        default=None,
        help="Direct PLY point-cloud path to run semantic-only prediction.",
    )
    parser.add_argument(
        "--point-cloud-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Multiple direct PLY point-cloud paths to run in one shared semantic PCA space.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        default=True,
        help="Export visualizations for the full selected split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap when using --all-samples.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/PartNext_mesh"))

    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--alias-config", type=Path, default=None)
    parser.add_argument("--num-support-points", type=int, default=5000)
    parser.add_argument("--num-query-surface-points", type=int, default=5000)
    parser.add_argument("--num-query-occ-points", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations_universal_field"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--disable-global-sem-pca",
        action="store_true",
        help="Disable the shared semantic-embedding PCA export across the selected samples.",
    )
    parser.add_argument(
        "--feature-methods",
        nargs="+",
        default=["semantic"],
        help=(
            "Feature methods to PCA-color. Use one or more of: semantic, utonia, dinov2. "
            "Comma-separated values are accepted; each method gets an independent PCA."
        ),
    )
    parser.add_argument("--utonia-feature-knn", type=int, default=3)
    parser.add_argument("--utonia-feature-chunk-size", type=int, default=1024)
    parser.add_argument("--dinov2-model-name", default="dinov2_vits14")
    parser.add_argument("--dinov2-device", default="")
    parser.add_argument("--dinov2-image-size", type=int, default=224)
    parser.add_argument("--dinov2-render-views", type=int, default=3)
    return parser


def _build_dataset(
        checkpoint_args: dict,
        canonical_label_names: list[str],
        cli_args: argparse.Namespace,
) -> PartNextUniversalFieldDataset:
    categories = cli_args.categories
    if categories is None:
        categories = checkpoint_args.get("categories")

    alias_config = cli_args.alias_config
    if alias_config is None:
        alias_config = _resolve_path(checkpoint_args.get("alias_config"))

    dataset_root = cli_args.dataset_root or _resolve_path(checkpoint_args.get("dataset_root"))
    if dataset_root is None:
        raise ValueError("Could not resolve dataset_root from checkpoint or CLI.")

    return PartNextUniversalFieldDataset(
        dataset_root=dataset_root,
        split=cli_args.split,
        categories=categories,
        num_support_points=cli_args.num_support_points or int(checkpoint_args["num_support_points"]),
        num_query_surface_points=(
                cli_args.num_query_surface_points
                or int(checkpoint_args["num_query_surface_points"])
        ),
        num_query_occ_points=(
                cli_args.num_query_occ_points
                or int(checkpoint_args["num_query_occ_points"])
        ),
        balanced_sampling_ratio=float(checkpoint_args.get("balanced_sampling_ratio", 0.5)),
        query_sampling_ratio=float(checkpoint_args.get("query_sampling_ratio", 0.75)),
        near_surface_ratio=float(checkpoint_args.get("near_surface_ratio", 0.75)),
        near_surface_noise=float(checkpoint_args.get("near_surface_noise", 0.02)),
        uniform_padding=float(checkpoint_args.get("uniform_padding", 0.15)),
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
        grid_size=float(checkpoint_args.get("grid_size", 0.01)),
        val_ratio=float(checkpoint_args.get("val_ratio", 0.1)),
        test_ratio=float(checkpoint_args.get("test_ratio", 0.0)),
        split_seed=int(checkpoint_args.get("split_seed", 42)),
        limit_samples=None,
        label_level=str(checkpoint_args.get("label_level", "leaf")),
        alias_config_path=alias_config,
        canonical_label_names=canonical_label_names,
        ignore_labels=checkpoint_args.get("ignore_labels", ["Other"]),
        rotation_mode="none",
        jitter_std=0.0,
        second_view=False,
    )


def _build_utonia_transform(grid_size: float):
    return UTONIA.transform.Compose(
        [
            dict(
                type="GridSample",
                grid_size=float(grid_size),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
    )


def _sample_rows(array: np.ndarray, num_rows: int) -> np.ndarray:
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}.")
    if array.shape[0] == 0:
        raise ValueError("Cannot sample rows from an empty array.")
    replace = array.shape[0] < num_rows
    indices = np.random.choice(array.shape[0], size=num_rows, replace=replace)
    return array[indices]


def _fallback_normals(points: np.ndarray) -> np.ndarray:
    centered = points.astype(np.float32) - points.astype(np.float32).mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normals = np.zeros_like(centered, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    normals[valid] = centered[valid] / norms[valid]
    normals[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32)


def _normalize_colors(colors: np.ndarray | None, num_points: int) -> tuple[np.ndarray, bool]:
    if colors is None or len(colors) != num_points:
        return np.zeros((num_points, 3), dtype=np.float32), False
    colors_np = np.asarray(colors[:, :3], dtype=np.float32)
    if colors_np.size > 0 and float(colors_np.max()) <= 1.0:
        colors_np = colors_np * 255.0
    return np.clip(colors_np, 0.0, 255.0).astype(np.float32), True


def _normalize_normals(normals: np.ndarray | None, points: np.ndarray) -> tuple[np.ndarray, bool]:
    if normals is None or len(normals) != len(points):
        return _fallback_normals(points), False
    normals_np = np.asarray(normals[:, :3], dtype=np.float32)
    norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    if not valid.all():
        fallback = _fallback_normals(points)
        normals_np[~valid] = fallback[~valid]
        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
    normals_np[valid] = normals_np[valid] / norms[valid]
    return normals_np.astype(np.float32), True


def _load_point_cloud_arrays(point_cloud_path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    resolved_path = Path(point_cloud_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"point-cloud-path does not exist: {resolved_path}")

    geometry = trimesh.load(resolved_path, process=False)
    if isinstance(geometry, trimesh.Scene):
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray | None] = []
        normal_sets: list[np.ndarray | None] = []
        for item in geometry.geometry.values():
            vertices = np.asarray(getattr(item, "vertices", []), dtype=np.float32)
            if vertices.size == 0:
                continue
            point_sets.append(vertices.reshape(-1, 3))
            item_colors = getattr(item, "colors", None)
            if item_colors is None and hasattr(item, "visual"):
                item_colors = getattr(item.visual, "vertex_colors", None)
            color_sets.append(None if item_colors is None else np.asarray(item_colors))
            item_normals = getattr(item, "vertex_normals", None)
            normal_sets.append(None if item_normals is None else np.asarray(item_normals))
        if not point_sets:
            raise ValueError(f"No points found in point cloud: {resolved_path}")
        points = np.concatenate(point_sets, axis=0).astype(np.float32)
        colors_raw = (
            np.concatenate([item for item in color_sets if item is not None], axis=0)
            if all(item is not None for item in color_sets)
            else None
        )
        normals_raw = (
            np.concatenate([item for item in normal_sets if item is not None], axis=0)
            if all(item is not None for item in normal_sets)
            else None
        )
    else:
        points = np.asarray(getattr(geometry, "vertices", []), dtype=np.float32).reshape(-1, 3)
        colors_raw = getattr(geometry, "colors", None)
        if colors_raw is None and hasattr(geometry, "visual"):
            colors_raw = getattr(geometry.visual, "vertex_colors", None)
        normals_raw = getattr(geometry, "vertex_normals", None)

    if points.shape[0] == 0:
        raise ValueError(f"No points found in point cloud: {resolved_path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains non-finite coordinates: {resolved_path}")

    return (
        points.astype(np.float32),
        None if colors_raw is None else np.asarray(colors_raw),
        None if normals_raw is None else np.asarray(normals_raw),
    )


def _build_direct_point_cloud_sample(
        *,
        point_cloud_path: Path,
        num_support_points: int,
        num_query_surface_points: int,
        coord_scale: float,
        center_shift_z: bool,
        grid_size: float,
) -> dict:
    resolved_point_cloud_path = Path(point_cloud_path).expanduser().resolve()
    points_raw, raw_colors, raw_normals = _load_point_cloud_arrays(resolved_point_cloud_path)
    colors_raw, has_colors = _normalize_colors(raw_colors, points_raw.shape[0])
    normals_raw, has_normals = _normalize_normals(raw_normals, points_raw)

    support_indices = np.random.choice(
        points_raw.shape[0],
        size=int(num_support_points),
        replace=points_raw.shape[0] < int(num_support_points),
    )
    query_surface_raw = _sample_rows(points_raw, int(num_query_surface_points))
    support_raw = points_raw[support_indices]
    support_colors = colors_raw[support_indices]
    support_normals = normals_raw[support_indices]

    support_points, query_surface_points = _normalize_multiple_arrays(
        support_raw,
        query_surface_raw,
        coord_scale=float(coord_scale),
        center_shift_z=bool(center_shift_z),
    )
    utonia_transform = _build_utonia_transform(grid_size)
    surface_labels = np.full((query_surface_points.shape[0],), IGNORE_LABEL, dtype=np.int64)
    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_long = torch.zeros((0,), dtype=torch.long)

    return {
        "utonia_input_view1": utonia_transform(
            {
                "coord": support_points.copy(),
                "color": support_colors.copy(),
                "normal": support_normals.copy(),
            }
        ),
        "support_points_view1": torch.from_numpy(support_points).float(),
        "query_surface_points_view1": torch.from_numpy(query_surface_points).float(),
        "query_occ_points_view1": torch.from_numpy(empty_points.copy()).float(),
        "query_probe_points_view1": torch.from_numpy(empty_points.copy()).float(),
        "query_occ_points_canonical": torch.from_numpy(empty_points.copy()).float(),
        "query_probe_points_canonical": torch.from_numpy(empty_points.copy()).float(),
        "query_occ_correspondence": empty_long.clone(),
        "query_probe_correspondence": empty_long.clone(),
        "query_probe_labels": empty_long.clone(),
        "query_surface_points_canonical": torch.from_numpy(query_surface_points.copy()).float(),
        "query_surface_correspondence": torch.arange(query_surface_points.shape[0], dtype=torch.long),
        "query_surface_labels": torch.from_numpy(surface_labels).long(),
        "query_occ_labels": torch.zeros((0,), dtype=torch.float32),
        "category": "direct_point_cloud",
        "model_id": resolved_point_cloud_path.stem,
        "mesh_path": str(resolved_point_cloud_path),
        "point_cloud_has_colors": bool(has_colors),
        "point_cloud_has_normals": bool(has_normals),
    }


def _build_direct_mesh_sample(
        *,
        mesh_path: Path,
        num_support_points: int,
        num_query_surface_points: int,
        num_query_occ_points: int,
        num_query_probe_points: int,
        balanced_sampling_ratio: float,
        query_sampling_ratio: float,
        near_surface_ratio: float,
        near_surface_noise: float,
        uniform_padding: float,
        coord_scale: float,
        center_shift_z: bool,
        grid_size: float,
) -> dict:
    resolved_mesh_path = Path(mesh_path).expanduser().resolve()
    if not resolved_mesh_path.exists():
        raise FileNotFoundError(f"mesh-path does not exist: {resolved_mesh_path}")

    meshes = _load_geometry_meshes(resolved_mesh_path)
    mesh = _concatenate_meshes(meshes)
    face_labels = np.zeros((len(mesh.faces),), dtype=np.int64)

    support_raw, support_normals, support_colors, _ = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=int(num_support_points),
        balanced_sampling_ratio=float(balanced_sampling_ratio),
    )
    query_surface_raw, _, _, _ = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=int(num_query_surface_points),
        balanced_sampling_ratio=float(query_sampling_ratio),
    )
    query_occ_raw, query_occ_labels = _sample_queries(
        mesh=mesh,
        num_points=int(num_query_occ_points),
        near_surface_ratio=float(near_surface_ratio),
        near_surface_noise=float(near_surface_noise),
        uniform_padding=float(uniform_padding),
    )
    query_probe_raw, _ = _sample_queries(
        mesh=mesh,
        num_points=int(num_query_probe_points),
        near_surface_ratio=1.0,
        near_surface_noise=float(near_surface_noise),
        uniform_padding=float(uniform_padding),
    )

    (
        support_points,
        query_surface_points,
        query_occ_points,
        query_probe_points,
    ) = _normalize_multiple_arrays(
        support_raw,
        query_surface_raw,
        query_occ_raw,
        query_probe_raw,
        coord_scale=float(coord_scale),
        center_shift_z=bool(center_shift_z),
    )
    utonia_transform = _build_utonia_transform(grid_size)
    surface_labels = np.full((query_surface_points.shape[0],), IGNORE_LABEL, dtype=np.int64)
    probe_labels = np.full((query_probe_points.shape[0],), IGNORE_LABEL, dtype=np.int64)

    return {
        "utonia_input_view1": utonia_transform(
            {
                "coord": support_points.copy(),
                "color": support_colors.copy(),
                "normal": support_normals.copy(),
            }
        ),
        "support_points_view1": torch.from_numpy(support_points).float(),
        "query_surface_points_view1": torch.from_numpy(query_surface_points).float(),
        "query_occ_points_view1": torch.from_numpy(query_occ_points).float(),
        "query_probe_points_view1": torch.from_numpy(query_probe_points).float(),
        "query_occ_points_canonical": torch.from_numpy(query_occ_points.copy()).float(),
        "query_probe_points_canonical": torch.from_numpy(query_probe_points.copy()).float(),
        "query_occ_correspondence": torch.arange(query_occ_points.shape[0], dtype=torch.long),
        "query_probe_correspondence": torch.arange(query_probe_points.shape[0], dtype=torch.long),
        "query_probe_labels": torch.from_numpy(probe_labels).long(),
        "query_surface_points_canonical": torch.from_numpy(query_surface_points.copy()).float(),
        "query_surface_correspondence": torch.arange(query_surface_points.shape[0], dtype=torch.long),
        "query_surface_labels": torch.from_numpy(surface_labels).long(),
        "query_occ_labels": torch.from_numpy(query_occ_labels.astype(np.float32, copy=False)).float(),
        "category": "direct_mesh",
        "model_id": resolved_mesh_path.stem,
        "mesh_path": str(resolved_mesh_path),
    }


class _SingleSampleDataset:
    def __init__(self, sample: dict, split: str = "mesh_path") -> None:
        self._sample = sample
        self.split = split
        self.records = [
            SimpleNamespace(
                category=sample["category"],
                model_id=sample["model_id"],
            )
        ]

    def __getitem__(self, index: int) -> dict:
        if index != 0:
            raise IndexError("direct mesh dataset only contains one sample")
        return self._sample

    def __len__(self) -> int:
        return 1


def _load_model(
        checkpoint_state: dict,
        checkpoint_args: dict,
        canonical_label_names: list[str],
        device: torch.device,
) -> UtoniaUniversalFieldNet:
    model = UtoniaUniversalFieldNet(
        num_classes=len(canonical_label_names),
        utonia_checkpoint=checkpoint_args.get("utonia_checkpoint", "auto"),
        utonia_repo_id=checkpoint_args.get("utonia_repo_id", "Pointcept/Utonia"),
        utonia_upcast_levels=int(checkpoint_args.get("utonia_upcast_levels", 0)),
        freeze_utonia=True,
        adapter_hidden_dim=int(checkpoint_args.get("adapter_hidden_dim", 256)),
        branch_dim=int(checkpoint_args.get("branch_dim", 256)),
        sem_embedding_dim=int(checkpoint_args.get("sem_embedding_dim", 128)),
        geo_embedding_dim=int(checkpoint_args.get("geo_embedding_dim", 128)),
        num_anchors=int(checkpoint_args.get("num_anchors", 256)),
        rbf_sigma=float(checkpoint_args.get("rbf_sigma", 0.12)),
        triplane_resolution=int(checkpoint_args.get("triplane_resolution", 64)),
        triplane_padding=float(checkpoint_args.get("triplane_padding", 0.05)),
    )
    projector_state = checkpoint_state["projector_state_dict"]
    model.semantic_adapter.load_state_dict(projector_state["semantic_adapter"])
    model.geometric_adapter.load_state_dict(projector_state["geometric_adapter"])
    if "semantic_local_fusion" in projector_state:
        model.semantic_local_fusion.load_state_dict(projector_state["semantic_local_fusion"])
    if "geometric_local_fusion" in projector_state:
        model.geometric_local_fusion.load_state_dict(projector_state["geometric_local_fusion"])
    if "occupancy_local_fusion" in projector_state:
        model.occupancy_local_fusion.load_state_dict(projector_state["occupancy_local_fusion"])
    model.semantic_decoder.load_state_dict(projector_state["semantic_decoder"])
    model.geometric_decoder.load_state_dict(projector_state["geometric_decoder"])
    if "occupancy_decoder" in projector_state:
        model.occupancy_decoder.load_state_dict(projector_state["occupancy_decoder"])
    if "encoder_state_dict" in checkpoint_state:
        model.feature_extractor.encoder.load_state_dict(checkpoint_state["encoder_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def _select_sample_index(
        dataset: PartNextUniversalFieldDataset,
        sample_index: int,
        model_id: str | None,
) -> int:
    if model_id is None:
        if not (0 <= sample_index < len(dataset)):
            raise IndexError(f"sample-index {sample_index} is out of range for dataset size {len(dataset)}")
        return sample_index

    for index, record in enumerate(dataset.records):
        if record.model_id == model_id:
            return index
    raise ValueError(f"model-id {model_id} was not found in split={dataset.split}")


def _run_single_sample(
        *,
        dataset: PartNextUniversalFieldDataset,
        sample_index: int,
        model: UtoniaUniversalFieldNet,
        device: torch.device,
        canonical_label_names: list[str],
        checkpoint: Path,
        output_root: Path,
        semantic_only: bool = False,
        feature_methods: Sequence[str] | str | None = None,
        dinov2_backend=None,
        utonia_feature_knn: int = 3,
        utonia_feature_chunk_size: int = 1024,
        dinov2_render_views: int = 3,
        dinov2_image_size: int = 224,
) -> dict:
    requested_feature_methods = parse_feature_methods(feature_methods)
    sample = dataset[sample_index]
    batch = partnext_universal_field_collate_fn([sample])
    batch = move_to_device(batch, device)

    with torch.no_grad():
        outputs = model(batch, train_mode="semantic")["view1"] if semantic_only else model(batch)["view1"]
        sem_logits = outputs["sem_logits"][0].detach().cpu()
        sem_embeddings = outputs["sem_embeddings"][0].detach().cpu()
        geo_embeddings = (
            outputs["geo_embeddings"][0].detach().cpu()
            if "geo_embeddings" in outputs
            else None
        )
        occ_logits = (
            outputs["occ_logits"][0].detach().cpu()
            if "occ_logits" in outputs
            else None
        )

    surface_points = sample["query_surface_points_view1"].cpu().numpy()
    occ_points = sample["query_occ_points_view1"].cpu().numpy()
    support_points = sample["support_points_view1"].cpu().numpy()
    gt_labels = sample["query_surface_labels"].cpu().numpy()
    occ_gt = sample["query_occ_labels"].cpu().numpy()

    sem_probabilities = torch.softmax(sem_logits, dim=-1)
    sem_confidence, pred_labels = torch.max(sem_probabilities, dim=-1)
    sem_confidence_np = sem_confidence.numpy()
    pred_labels_np = pred_labels.numpy()
    sem_embeddings_np = sem_embeddings.numpy()
    geo_embeddings_np = geo_embeddings.numpy() if geo_embeddings is not None else None
    feature_embeddings_by_method: dict[str, np.ndarray] = {"semantic": sem_embeddings_np}
    if "utonia" in requested_feature_methods:
        feature_embeddings_by_method["utonia"] = _extract_raw_utonia_query_features(
            model=model,
            batch=batch,
            sample=sample,
            device=device,
            knn=int(utonia_feature_knn),
            chunk_size=int(utonia_feature_chunk_size),
        )
    if "dinov2" in requested_feature_methods:
        feature_embeddings_by_method["dinov2"] = _extract_pseudo_dinov2_query_features(
            dinov2_backend=dinov2_backend,
            support_points=sample["support_points_view1"].cpu().numpy(),
            query_points=surface_points,
            render_views=int(dinov2_render_views),
            image_size=int(dinov2_image_size),
        )

    occ_probability = torch.sigmoid(occ_logits).numpy() if occ_logits is not None else None
    occ_pred = (occ_probability >= 0.5).astype(np.int64) if occ_probability is not None else None

    palette = _build_palette(len(canonical_label_names))
    # palette = np.asarray([[0,0,255], [0,255,0], [255,0,0 ]],dtype=np.uint8)
    unlabeled_color = np.asarray([128, 128, 128], dtype=np.uint8)
    correct_color = np.asarray([255,0,0], dtype=np.uint8)
    wrong_color = np.asarray([230, 25, 75], dtype=np.uint8)
    support_color = np.asarray([180, 180, 180], dtype=np.uint8)
    occ_positive_color = np.asarray([255,0,0 ], dtype=np.uint8)
    occ_negative_color = np.asarray([0,0,255], dtype=np.uint8)

    pred_colors = palette[pred_labels_np]
    gt_colors = np.repeat(unlabeled_color[None, :], len(surface_points), axis=0)
    valid_mask = gt_labels != IGNORE_LABEL
    gt_colors[valid_mask] = palette[gt_labels[valid_mask]]

    error_colors = np.repeat(unlabeled_color[None, :], len(surface_points), axis=0)
    error_colors[valid_mask & (pred_labels_np == gt_labels)] = correct_color
    error_colors[valid_mask & (pred_labels_np != gt_labels)] = wrong_color

    confidence_colors = _probability_to_rgb(sem_confidence_np)
    sem_embedding_colors = _pca_to_rgb(sem_embeddings_np)
    geo_embedding_colors = _pca_to_rgb(geo_embeddings_np) if geo_embeddings_np is not None else None
    support_colors = np.repeat(support_color[None, :], len(support_points), axis=0)
    occ_probability_colors = (
        _probability_to_rgb(occ_probability)
        if occ_probability is not None
        else None
    )
    occ_binary_colors = None
    if occ_pred is not None:
        occ_binary_colors = np.repeat(occ_negative_color[None, :], len(occ_points), axis=0)
        occ_binary_colors[occ_pred.astype(bool)] = occ_positive_color

    run_stem = checkpoint.resolve().stem
    sample_tag = f"{dataset.records[sample_index].category}_{dataset.records[sample_index].model_id}"
    output_dir = output_root / run_stem / sample_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_point_cloud(output_dir / "surface_pred_labels.ply", surface_points, pred_colors)
    _export_point_cloud(output_dir / "surface_confidence_map.ply", surface_points, confidence_colors)
    _export_point_cloud(output_dir / "surface_sem_embedding_pca.ply", surface_points, sem_embedding_colors)
    _export_point_cloud(output_dir / "support_points.ply", support_points, support_colors)
    for method in requested_feature_methods:
        if method == "semantic":
            continue
        embeddings = feature_embeddings_by_method[method]
        _export_point_cloud(
            output_dir / f"{_feature_local_file_key(method)}.ply",
            surface_points,
            _pca_to_rgb(embeddings),
        )
    if not semantic_only:
        _export_point_cloud(output_dir / "surface_gt_labels.ply", surface_points, gt_colors)
        _export_point_cloud(output_dir / "surface_error_map.ply", surface_points, error_colors)
    if geo_embedding_colors is not None:
        _export_point_cloud(output_dir / "surface_geo_embedding_pca.ply", surface_points, geo_embedding_colors)
    if occ_probability_colors is not None and occ_binary_colors is not None:
        _export_point_cloud(output_dir / "occ_probability_map.ply", occ_points, occ_probability_colors)
        _export_point_cloud(output_dir / "occ_binary_map.ply", occ_points, occ_binary_colors)

    surface_accuracy = (
        float((pred_labels_np[valid_mask] == gt_labels[valid_mask]).mean())
        if int(valid_mask.sum()) > 0
        else None
    )
    occ_accuracy = (
        float((occ_pred == occ_gt).mean())
        if occ_pred is not None and len(occ_gt) > 0
        else None
    )
    files = {
        "surface_pred_labels": str((output_dir / "surface_pred_labels.ply").resolve()),
        "surface_confidence_map": str((output_dir / "surface_confidence_map.ply").resolve()),
        "surface_sem_embedding_pca": str((output_dir / "surface_sem_embedding_pca.ply").resolve()),
        "support_points": str((output_dir / "support_points.ply").resolve()),
    }
    for method in requested_feature_methods:
        if method == "semantic":
            continue
        file_key = _feature_local_file_key(method)
        files[file_key] = str((output_dir / f"{file_key}.ply").resolve())
    if not semantic_only:
        files["surface_gt_labels"] = str((output_dir / "surface_gt_labels.ply").resolve())
        files["surface_error_map"] = str((output_dir / "surface_error_map.ply").resolve())
    if geo_embedding_colors is not None:
        files["surface_geo_embedding_pca"] = str((output_dir / "surface_geo_embedding_pca.ply").resolve())
    if occ_probability_colors is not None and occ_binary_colors is not None:
        files["occ_probability_map"] = str((output_dir / "occ_probability_map.ply").resolve())
        files["occ_binary_map"] = str((output_dir / "occ_binary_map.ply").resolve())
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "input_mode": "point_cloud" if semantic_only else dataset.split,
        "split": dataset.split,
        "sample_index": sample_index,
        "category": dataset.records[sample_index].category,
        "model_id": dataset.records[sample_index].model_id,
        "mesh_path": sample["mesh_path"],
        "num_support_points": int(len(support_points)),
        "num_surface_points": int(len(surface_points)),
        "num_occ_points": int(len(occ_points)),
        "valid_surface_labels": int(valid_mask.sum()),
        "surface_accuracy": surface_accuracy,
        "mean_surface_confidence": float(sem_confidence_np.mean()),
        "occ_accuracy": occ_accuracy,
        "mean_occ_probability": (
            float(occ_probability.mean()) if occ_probability is not None else None
        ),
        "feature_methods": requested_feature_methods,
        "pred_histogram": _labels_to_named_histogram(pred_labels_np, canonical_label_names),
        "gt_histogram": _labels_to_named_histogram(gt_labels[valid_mask], canonical_label_names),
        "files": files,
    }
    if "point_cloud_has_colors" in sample:
        summary["point_cloud_has_colors"] = bool(sample["point_cloud_has_colors"])
    if "point_cloud_has_normals" in sample:
        summary["point_cloud_has_normals"] = bool(sample["point_cloud_has_normals"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "label_palette.json").write_text(
        json.dumps(
            {
                label_name: palette[index].tolist()
                for index, label_name in enumerate(canonical_label_names)
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[visualize] saved sample {sample_index}: {output_dir}")
    joint_sem_payload = {
        "output_dir": output_dir,
        "surface_points": surface_points,
        "sem_embeddings": sem_embeddings_np,
        "summary": summary,
        "summary_path": summary_path,
        "category": dataset.records[sample_index].category,
        "model_id": dataset.records[sample_index].model_id,
    }
    joint_feature_payloads = [
        {
            "feature_method": method,
            "output_dir": output_dir,
            "surface_points": surface_points,
            "feature_embeddings": feature_embeddings_by_method[method],
            "summary": summary,
            "summary_path": summary_path,
            "category": dataset.records[sample_index].category,
            "model_id": dataset.records[sample_index].model_id,
        }
        for method in requested_feature_methods
    ]
    return {
        "summary": summary,
        "joint_sem_pca_payload": joint_sem_payload,
        "joint_feature_pca_payloads": joint_feature_payloads,
    }


def _run_direct_point_cloud_samples(
        *,
        point_cloud_paths: list[Path],
        model: UtoniaUniversalFieldNet,
        device: torch.device,
        canonical_label_names: list[str],
        checkpoint: Path,
        output_root: Path,
        num_support_points: int,
        num_query_surface_points: int,
        coord_scale: float,
        center_shift_z: bool,
        grid_size: float,
        disable_global_sem_pca: bool,
        feature_methods: Sequence[str] | str | None = None,
        dinov2_backend=None,
        utonia_feature_knn: int = 3,
        utonia_feature_chunk_size: int = 1024,
        dinov2_render_views: int = 3,
        dinov2_image_size: int = 224,
) -> dict:
    if not point_cloud_paths:
        raise ValueError("At least one point-cloud path is required.")

    requested_feature_methods = parse_feature_methods(feature_methods)
    sample_results: list[dict] = []
    seen_model_ids: dict[str, int] = {}
    for point_cloud_path in point_cloud_paths:
        sample = _build_direct_point_cloud_sample(
            point_cloud_path=point_cloud_path,
            num_support_points=int(num_support_points),
            num_query_surface_points=int(num_query_surface_points),
            coord_scale=float(coord_scale),
            center_shift_z=bool(center_shift_z),
            grid_size=float(grid_size),
        )
        model_id = str(sample["model_id"])
        seen_model_ids[model_id] = seen_model_ids.get(model_id, 0) + 1
        if seen_model_ids[model_id] > 1:
            sample["model_id"] = f"{model_id}_{seen_model_ids[model_id]}"

        result = _run_single_sample(
            dataset=_SingleSampleDataset(sample, split="point_cloud"),
            sample_index=0,
            model=model,
            device=device,
            canonical_label_names=canonical_label_names,
            checkpoint=checkpoint,
            output_root=output_root,
            semantic_only=True,
            feature_methods=requested_feature_methods,
            dinov2_backend=dinov2_backend,
            utonia_feature_knn=int(utonia_feature_knn),
            utonia_feature_chunk_size=int(utonia_feature_chunk_size),
            dinov2_render_views=int(dinov2_render_views),
            dinov2_image_size=int(dinov2_image_size),
        )
        sample_results.append(result)

    summaries = [result["summary"] for result in sample_results]
    joint_sem_pca = None
    joint_feature_pca = None
    if not disable_global_sem_pca:
        if requested_feature_methods == ["semantic"]:
            joint_sem_pca = _export_joint_semantic_pca_outputs(
                [result["joint_sem_pca_payload"] for result in sample_results]
            )
        else:
            joint_feature_pca = _export_joint_feature_pca_outputs(
                [
                    payload
                    for result in sample_results
                    for payload in result["joint_feature_pca_payloads"]
                ]
            )
            joint_sem_pca = joint_feature_pca.get("semantic")

    aggregate = {
        "checkpoint": str(checkpoint.resolve()),
        "input_mode": "point_cloud_paths" if len(point_cloud_paths) > 1 else "point_cloud",
        "feature_methods": requested_feature_methods,
        "num_samples": len(summaries),
        "num_support_points": int(num_support_points),
        "num_query_surface_points": int(num_query_surface_points),
        "point_cloud_paths": [
            str(Path(point_cloud_path).expanduser().resolve())
            for point_cloud_path in point_cloud_paths
        ],
        "mean_surface_accuracy": _mean_or_none([item["surface_accuracy"] for item in summaries]),
        "mean_occ_accuracy": _mean_or_none([item["occ_accuracy"] for item in summaries]),
        "mean_surface_confidence": _mean_or_none([item["mean_surface_confidence"] for item in summaries]),
        "mean_occ_probability": _mean_or_none([item["mean_occ_probability"] for item in summaries]),
        "samples": summaries,
    }
    if joint_sem_pca is not None:
        aggregate["joint_sem_pca"] = joint_sem_pca
    if joint_feature_pca is not None:
        aggregate["joint_feature_pca"] = joint_feature_pca

    aggregate_name = "summary_point_clouds.json" if len(point_cloud_paths) > 1 else "summary_point_cloud.json"
    aggregate_path = output_root / checkpoint.resolve().stem / aggregate_name
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "sample_results": sample_results,
        "summary": aggregate,
        "summary_path": aggregate_path,
    }


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    feature_methods = parse_feature_methods(args.feature_methods)

    checkpoint_state = _load_checkpoint(args.checkpoint)
    checkpoint_args = checkpoint_state.get("args", {})
    canonical_label_names = list(checkpoint_state.get("canonical_label_names", []))
    if not canonical_label_names:
        labels_path = args.checkpoint.resolve().parent / "canonical_labels.json"
        canonical_label_names = json.loads(labels_path.read_text(encoding="utf-8"))

    device = torch.device(args.device)
    model = _load_model(
        checkpoint_state=checkpoint_state,
        checkpoint_args=checkpoint_args,
        canonical_label_names=canonical_label_names,
        device=device,
    )

    dinov2_backend = None
    if "dinov2" in feature_methods:
        dinov2_backend = _load_dinov2_backend(
            model_name=str(args.dinov2_model_name),
            device=str(args.dinov2_device or args.device),
            image_size=int(args.dinov2_image_size),
        )

    point_cloud_paths = _resolve_point_cloud_paths(args)
    mesh_path = args.mesh_path
    if mesh_path is not None and mesh_path.suffix.lower() == ".ply":
        mesh_path = None

    if point_cloud_paths:
        result = _run_direct_point_cloud_samples(
            point_cloud_paths=point_cloud_paths,
            model=model,
            device=device,
            canonical_label_names=canonical_label_names,
            checkpoint=args.checkpoint,
            output_root=args.output_dir,
            num_support_points=int(
                args.num_support_points
                or checkpoint_args.get("num_support_points", 5000)
            ),
            num_query_surface_points=_resolve_point_cloud_query_surface_count(args, checkpoint_args),
            coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
            center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
            grid_size=float(checkpoint_args.get("grid_size", 0.01)),
            disable_global_sem_pca=bool(args.disable_global_sem_pca),
            feature_methods=feature_methods,
            dinov2_backend=dinov2_backend,
            utonia_feature_knn=int(args.utonia_feature_knn),
            utonia_feature_chunk_size=int(args.utonia_feature_chunk_size),
            dinov2_render_views=int(args.dinov2_render_views),
            dinov2_image_size=int(args.dinov2_image_size),
        )
        if len(result["sample_results"]) == 1:
            sample_summary = result["sample_results"][0]["summary"]
            print(f"Visualization saved to: {sample_summary['files']['surface_pred_labels']}")
        else:
            print(f"[visualize] saved point-cloud batch summary: {result['summary_path']}")
        return

    if mesh_path is not None:
        sample = _build_direct_mesh_sample(
            mesh_path=mesh_path,
            num_support_points=int(
                args.num_support_points
                or checkpoint_args.get("num_support_points", 5000)
            ),
            num_query_surface_points=int(
                args.num_query_surface_points
                or checkpoint_args.get("num_query_surface_points", 2048)
            ),
            num_query_occ_points=int(
                args.num_query_occ_points
                or checkpoint_args.get("num_query_occ_points", 1536)
            ),
            num_query_probe_points=int(checkpoint_args.get("num_query_probe_points", 1024)),
            balanced_sampling_ratio=float(checkpoint_args.get("balanced_sampling_ratio", 0.5)),
            query_sampling_ratio=float(checkpoint_args.get("query_sampling_ratio", 0.75)),
            near_surface_ratio=float(checkpoint_args.get("near_surface_ratio", 0.75)),
            near_surface_noise=float(checkpoint_args.get("near_surface_noise", 0.02)),
            uniform_padding=float(checkpoint_args.get("uniform_padding", 0.15)),
            coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
            center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
            grid_size=float(checkpoint_args.get("grid_size", 0.01)),
        )
        result = _run_single_sample(
            dataset=_SingleSampleDataset(sample),
            sample_index=0,
            model=model,
            device=device,
            canonical_label_names=canonical_label_names,
            checkpoint=args.checkpoint,
            output_root=args.output_dir,
            feature_methods=feature_methods,
            dinov2_backend=dinov2_backend,
            utonia_feature_knn=int(args.utonia_feature_knn),
            utonia_feature_chunk_size=int(args.utonia_feature_chunk_size),
            dinov2_render_views=int(args.dinov2_render_views),
            dinov2_image_size=int(args.dinov2_image_size),
        )
        if not args.disable_global_sem_pca:
            if feature_methods == ["semantic"]:
                _export_joint_semantic_pca_outputs([result["joint_sem_pca_payload"]])
            else:
                _export_joint_feature_pca_outputs(result["joint_feature_pca_payloads"])
        print(f"Visualization saved to: {result['summary']['files']['surface_pred_labels']}")
        return

    dataset = _build_dataset(
        checkpoint_args=checkpoint_args,
        canonical_label_names=canonical_label_names,
        cli_args=args,
    )

    if args.all_samples:
        sample_indices = list(range(len(dataset)))
        if args.max_samples is not None:
            sample_indices = sample_indices[: args.max_samples]
        sample_results = [
            _run_single_sample(
                dataset=dataset,
                sample_index=sample_index,
                model=model,
                device=device,
                canonical_label_names=canonical_label_names,
                checkpoint=args.checkpoint,
                output_root=args.output_dir,
                feature_methods=feature_methods,
                dinov2_backend=dinov2_backend,
                utonia_feature_knn=int(args.utonia_feature_knn),
                utonia_feature_chunk_size=int(args.utonia_feature_chunk_size),
                dinov2_render_views=int(args.dinov2_render_views),
                dinov2_image_size=int(args.dinov2_image_size),
            )
            for sample_index in sample_indices
        ]
        summaries = [result["summary"] for result in sample_results]
        joint_sem_pca = None
        joint_feature_pca = None
        if not args.disable_global_sem_pca and sample_results:
            if feature_methods == ["semantic"]:
                joint_sem_pca = _export_joint_semantic_pca_outputs(
                    [result["joint_sem_pca_payload"] for result in sample_results]
                )
            else:
                joint_feature_pca = _export_joint_feature_pca_outputs(
                    [
                        payload
                        for result in sample_results
                        for payload in result["joint_feature_pca_payloads"]
                    ]
                )
                joint_sem_pca = joint_feature_pca.get("semantic")
        aggregate = {
            "checkpoint": str(args.checkpoint.resolve()),
            "split": dataset.split,
            "feature_methods": feature_methods,
            "num_samples": len(summaries),
            "mean_surface_accuracy": _mean_or_none([item["surface_accuracy"] for item in summaries]),
            "mean_occ_accuracy": _mean_or_none([item["occ_accuracy"] for item in summaries]),
            "mean_surface_confidence": _mean_or_none([item["mean_surface_confidence"] for item in summaries]),
            "mean_occ_probability": _mean_or_none([item["mean_occ_probability"] for item in summaries]),
            "samples": summaries,
        }
        if joint_sem_pca is not None:
            aggregate["joint_sem_pca"] = joint_sem_pca
        if joint_feature_pca is not None:
            aggregate["joint_feature_pca"] = joint_feature_pca
        aggregate_path = args.output_dir / args.checkpoint.resolve().stem / "summary_all.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[visualize] saved split summary: {aggregate_path}")
        return

    sample_index = _select_sample_index(
        dataset=dataset,
        sample_index=args.sample_index,
        model_id=args.model_id,
    )
    result = _run_single_sample(
        dataset=dataset,
        sample_index=sample_index,
        model=model,
        device=device,
        canonical_label_names=canonical_label_names,
        checkpoint=args.checkpoint,
        output_root=args.output_dir,
        feature_methods=feature_methods,
        dinov2_backend=dinov2_backend,
        utonia_feature_knn=int(args.utonia_feature_knn),
        utonia_feature_chunk_size=int(args.utonia_feature_chunk_size),
        dinov2_render_views=int(args.dinov2_render_views),
        dinov2_image_size=int(args.dinov2_image_size),
    )
    if not args.disable_global_sem_pca:
        if feature_methods == ["semantic"]:
            _export_joint_semantic_pca_outputs([result["joint_sem_pca_payload"]])
        else:
            _export_joint_feature_pca_outputs(result["joint_feature_pca_payloads"])
    print(f"Visualization saved to: {result['summary']['files']['surface_pred_labels']}")


if __name__ == "__main__":
    main()
