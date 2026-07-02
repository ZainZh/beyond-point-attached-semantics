from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from myutils.imports import ensure_utonia_importable

from .partnext_paths import resolve_partnext_mesh_path


UTONIA = ensure_utonia_importable()


@dataclass(frozen=True)
class PartNextRecord:
    category: str
    model_id: str
    mesh_path: Path


def list_partnext_categories(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "annotation.jsonl").exists()
    )


def _parse_categories(
    dataset_root: str | Path,
    categories: Iterable[str] | None,
) -> list[str]:
    available = list_partnext_categories(dataset_root)
    if not categories:
        return available

    parsed: list[str] = []
    for category in categories:
        parsed.extend(part.strip() for part in str(category).split(",") if part.strip())

    if not parsed or any(category.lower() == "all" for category in parsed):
        return available

    invalid = sorted(set(parsed) - set(available))
    if invalid:
        raise ValueError(
            f"Unknown PartNext categories: {invalid}. Available categories: {available}"
        )
    return sorted(set(parsed))


def _split_records(
    records: list[PartNextRecord],
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    split: str,
) -> list[PartNextRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    grouped: dict[str, list[PartNextRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)

    selected: list[PartNextRecord] = []
    for category, category_records in grouped.items():
        rng = random.Random(f"{split_seed}:{category}")
        category_records = list(category_records)
        rng.shuffle(category_records)

        total = len(category_records)
        num_test = int(round(total * test_ratio))
        num_val = int(round(total * val_ratio))

        if test_ratio > 0 and total >= 3:
            num_test = max(1, num_test)
        if val_ratio > 0 and total - num_test >= 2:
            num_val = max(1, num_val)

        if num_test + num_val >= total:
            overflow = num_test + num_val - total + 1
            num_val = max(0, num_val - overflow)

        train_end = total - num_val - num_test
        val_end = total - num_test

        if split == "train":
            selected.extend(category_records[:train_end])
        elif split == "val":
            selected.extend(category_records[train_end:val_end])
        else:
            selected.extend(category_records[val_end:])
    return selected


def _load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    geometry = trimesh.load(mesh_path, force="scene")
    if isinstance(geometry, trimesh.Scene):
        mesh = geometry.to_geometry()
    else:
        mesh = geometry
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported geometry type for {mesh_path}: {type(mesh)}")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh after processing: {mesh_path}")
    return mesh


def _sample_surface(
    mesh: trimesh.Trimesh,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord, face_index = trimesh.sample.sample_surface(mesh, num_points)
    coord = coord.astype(np.float32)
    normal = mesh.face_normals[face_index].astype(np.float32)

    if (
        hasattr(mesh.visual, "face_colors")
        and mesh.visual.face_colors is not None
        and len(mesh.visual.face_colors) == len(mesh.faces)
    ):
        color = np.asarray(mesh.visual.face_colors[face_index, :3], dtype=np.float32)
    elif (
        hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    ):
        color = np.asarray(
            mesh.visual.vertex_colors[mesh.faces[face_index], :3].mean(axis=1),
            dtype=np.float32,
        )
    else:
        color = np.zeros((num_points, 3), dtype=np.float32)
    return coord, normal, color


def _sample_queries(
    mesh: trimesh.Trimesh,
    num_points: int,
    near_surface_ratio: float,
    near_surface_noise: float,
    uniform_padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    bbox_min, bbox_max = mesh.bounds
    extent = float(np.max(bbox_max - bbox_min))
    extent = max(extent, 1e-3)

    num_near = int(num_points * near_surface_ratio)
    num_uniform = num_points - num_near
    num_near_small = num_near // 2
    num_near_large = num_near - num_near_small

    near_small, _, _ = _sample_surface(mesh, num_near_small)
    near_large, _, _ = _sample_surface(mesh, num_near_large)
    near_small += np.random.normal(
        loc=0.0,
        scale=extent * near_surface_noise,
        size=near_small.shape,
    ).astype(np.float32)
    near_large += np.random.normal(
        loc=0.0,
        scale=extent * near_surface_noise * 4.0,
        size=near_large.shape,
    ).astype(np.float32)

    padding = extent * uniform_padding
    uniform = np.random.uniform(
        low=bbox_min - padding,
        high=bbox_max + padding,
        size=(num_uniform, 3),
    ).astype(np.float32)

    query_points = np.concatenate([near_small, near_large, uniform], axis=0)
    occupancy = mesh.contains(query_points).astype(np.float32)

    permutation = np.random.permutation(len(query_points))
    return query_points[permutation], occupancy[permutation]


def compute_occupancy_normalization(
    support_points: np.ndarray,
    coord_scale: float,
    center_shift_z: bool,
) -> dict[str, np.ndarray | float]:
    centroid = np.mean(support_points, axis=0, keepdims=True)
    centered_support = support_points - centroid

    radius = np.linalg.norm(centered_support, axis=1).max()
    radius = max(float(radius), 1e-6)
    normalized_support = centered_support / radius
    shift = np.zeros(3, dtype=np.float32)

    if center_shift_z:
        x_min, y_min, z_min = normalized_support.min(axis=0)
        x_max, y_max, _ = normalized_support.max(axis=0)
        shift = np.asarray(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )

    return {
        "centroid": centroid.astype(np.float32),
        "radius": float(radius),
        "shift": shift.astype(np.float32),
        "coord_scale": float(coord_scale),
    }


def apply_occupancy_normalization(
    points: np.ndarray,
    normalization: dict[str, np.ndarray | float],
) -> np.ndarray:
    centroid = np.asarray(normalization["centroid"], dtype=np.float32)
    radius = max(float(normalization["radius"]), 1e-6)
    shift = np.asarray(normalization["shift"], dtype=np.float32)
    coord_scale = float(normalization["coord_scale"])
    normalized = (points - centroid) / radius
    normalized = normalized - shift
    normalized = normalized * coord_scale
    return normalized.astype(np.float32)


def normalize_occupancy_object_space(
    support_points: np.ndarray,
    query_points: np.ndarray,
    coord_scale: float,
    center_shift_z: bool,
    *,
    return_normalization: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    normalization = compute_occupancy_normalization(
        support_points=support_points,
        coord_scale=coord_scale,
        center_shift_z=center_shift_z,
    )
    support_points = apply_occupancy_normalization(support_points, normalization)
    query_points = apply_occupancy_normalization(query_points, normalization)
    if return_normalization:
        return support_points, query_points, normalization
    return support_points, query_points


def partnext_occupancy_collate_fn(batch: list[dict]) -> dict:
    return {
        "utonia_input": UTONIA.data.collate_fn([item["utonia_input"] for item in batch]),
        "support_points": torch.stack([item["support_points"] for item in batch], dim=0),
        "query_points": torch.stack([item["query_points"] for item in batch], dim=0),
        "occupancy": torch.stack([item["occupancy"] for item in batch], dim=0),
        "category": [item["category"] for item in batch],
        "model_id": [item["model_id"] for item in batch],
        "mesh_path": [item["mesh_path"] for item in batch],
    }


class PartNextOccupancyDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        categories: Iterable[str] | None = None,
        num_support_points: int = 2048,
        num_query_points: int = 4096,
        near_surface_ratio: float = 0.75,
        near_surface_noise: float = 0.02,
        uniform_padding: float = 0.15,
        coord_scale: float = 1.0,
        center_shift_z: bool = True,
        grid_size: float = 0.01,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        split_seed: int = 42,
        limit_samples: int | None = None,
        max_retries: int = 8,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.categories = _parse_categories(self.dataset_root, categories)
        self.num_support_points = num_support_points
        self.num_query_points = num_query_points
        self.near_surface_ratio = near_surface_ratio
        self.near_surface_noise = near_surface_noise
        self.uniform_padding = uniform_padding
        self.coord_scale = coord_scale
        self.center_shift_z = center_shift_z
        self.max_retries = max_retries

        records = self._build_records()
        records = _split_records(
            records=records,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            split=split,
        )
        if limit_samples is not None:
            records = records[:limit_samples]
        self.records = records

        self.utonia_transform = UTONIA.transform.Compose(
            [
                dict(
                    type="GridSample",
                    grid_size=grid_size,
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

    def _build_records(self) -> list[PartNextRecord]:
        records: list[PartNextRecord] = []
        for category in self.categories:
            annotation_path = self.dataset_root / category / "annotation.jsonl"
            with annotation_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    annotation = json.loads(line)
                    if not annotation.get("glb_exists", False):
                        continue
                    mesh_path = resolve_partnext_mesh_path(
                        annotation_path=annotation_path,
                        glb_dst=annotation["glb_dst"],
                        dataset_root=self.dataset_root,
                    )
                    if mesh_path is None:
                        continue
                    records.append(
                        PartNextRecord(
                            category=category,
                            model_id=annotation["model_id"],
                            mesh_path=mesh_path,
                        )
                    )
        if not records:
            raise RuntimeError(
                f"No valid PartNext samples found under {self.dataset_root} "
                f"for categories {self.categories}"
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        last_error: Exception | None = None
        for retry_index in range(self.max_retries):
            record = self.records[(index + retry_index) % len(self.records)]
            try:
                return self._load_sample(record)
            except Exception as error:  # pragma: no cover - retry path
                last_error = error
                warnings.warn(
                    f"Failed to load {record.mesh_path} ({type(error).__name__}: {error}). "
                    f"Retrying with the next sample.",
                    stacklevel=2,
                )
        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} retries") from last_error

    def _load_sample(self, record: PartNextRecord) -> dict:
        mesh = _load_mesh(record.mesh_path)
        support_points, support_normals, support_colors = _sample_surface(
            mesh,
            self.num_support_points,
        )
        query_points, occupancy = _sample_queries(
            mesh,
            self.num_query_points,
            near_surface_ratio=self.near_surface_ratio,
            near_surface_noise=self.near_surface_noise,
            uniform_padding=self.uniform_padding,
        )
        support_points, query_points = normalize_occupancy_object_space(
            support_points=support_points,
            query_points=query_points,
            coord_scale=self.coord_scale,
            center_shift_z=self.center_shift_z,
        )

        utonia_input = self.utonia_transform(
            {
                "coord": support_points.copy(),
                "color": support_colors.copy(),
                "normal": support_normals.copy(),
            }
        )

        return {
            "utonia_input": utonia_input,
            "support_points": torch.from_numpy(support_points).float(),
            "query_points": torch.from_numpy(query_points).float(),
            "occupancy": torch.from_numpy(occupancy).float(),
            "category": record.category,
            "model_id": record.model_id,
            "mesh_path": str(record.mesh_path),
        }
