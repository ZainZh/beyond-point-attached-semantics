from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .partnext_canonical_field import (
    IGNORE_LABEL,
    UTONIA,
    PartNextCanonicalRecord,
    _build_global_face_labels,
    _build_rotation_matrix,
    _concatenate_meshes,
    _load_alias_lookup,
    _load_geometry_meshes,
    _normalize_label_name,
    _parse_categories,
    _sample_surface_points,
    _split_records,
    build_partnext_canonical_label_space,
)
from .partnext_occupancy import _sample_queries
from .partnext_paths import resolve_partnext_mesh_path


def _apply_point_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    jitter_std: float,
) -> np.ndarray:
    transformed = points @ rotation.T
    if jitter_std > 0:
        transformed = transformed + np.random.normal(
            loc=0.0,
            scale=jitter_std,
            size=transformed.shape,
        ).astype(np.float32)
    return transformed.astype(np.float32)


def _apply_point_normal_transform(
    points: np.ndarray,
    normals: np.ndarray,
    rotation: np.ndarray,
    jitter_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _apply_point_transform(points, rotation, jitter_std),
        (normals @ rotation.T).astype(np.float32),
    )


def _normalize_multiple_arrays(
    support_points: np.ndarray,
    *query_arrays: np.ndarray,
    coord_scale: float,
    center_shift_z: bool,
) -> tuple[np.ndarray, ...]:
    centroid = np.mean(support_points, axis=0, keepdims=True)
    support = support_points - centroid
    transformed_queries = [points - centroid for points in query_arrays]

    radius = np.linalg.norm(support, axis=1).max()
    radius = max(float(radius), 1e-6)
    support = support / radius
    transformed_queries = [points / radius for points in transformed_queries]

    if center_shift_z:
        x_min, y_min, z_min = support.min(axis=0)
        x_max, y_max, _ = support.max(axis=0)
        shift = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
        support = support - shift
        transformed_queries = [points - shift for points in transformed_queries]

    support = (support * coord_scale).astype(np.float32)
    transformed_queries = [
        (points * coord_scale).astype(np.float32) for points in transformed_queries
    ]
    return (support, *transformed_queries)


def _assign_nearest_surface_labels(
    probe_points: np.ndarray,
    surface_points: np.ndarray,
    surface_labels: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    if probe_points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if surface_points.shape[0] == 0:
        return np.full((probe_points.shape[0],), IGNORE_LABEL, dtype=np.int64)

    distances = np.linalg.norm(
        probe_points[:, None, :] - surface_points[None, :, :],
        axis=-1,
    )
    nearest_indices = distances.argmin(axis=1)
    nearest_distances = distances[np.arange(probe_points.shape[0]), nearest_indices]
    labels = surface_labels[nearest_indices].astype(np.int64, copy=True)
    labels[nearest_distances > max(float(max_distance), 1e-6)] = IGNORE_LABEL
    return labels


def partnext_universal_field_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "utonia_input_view1": UTONIA.data.collate_fn(
            [item["utonia_input_view1"] for item in batch]
        ),
        "support_points_view1": torch.stack(
            [item["support_points_view1"] for item in batch],
            dim=0,
        ),
        "query_surface_points_view1": torch.stack(
            [item["query_surface_points_view1"] for item in batch],
            dim=0,
        ),
        "query_occ_points_view1": torch.stack(
            [item["query_occ_points_view1"] for item in batch],
            dim=0,
        ),
        "query_probe_points_view1": torch.stack(
            [item["query_probe_points_view1"] for item in batch],
            dim=0,
        ),
        "query_occ_points_canonical": torch.stack(
            [item["query_occ_points_canonical"] for item in batch],
            dim=0,
        ),
        "query_probe_points_canonical": torch.stack(
            [item["query_probe_points_canonical"] for item in batch],
            dim=0,
        ),
        "query_occ_correspondence": torch.stack(
            [item["query_occ_correspondence"] for item in batch],
            dim=0,
        ),
        "query_probe_correspondence": torch.stack(
            [item["query_probe_correspondence"] for item in batch],
            dim=0,
        ),
        "query_probe_labels": torch.stack(
            [item["query_probe_labels"] for item in batch],
            dim=0,
        ),
        "query_surface_points_canonical": torch.stack(
            [item["query_surface_points_canonical"] for item in batch],
            dim=0,
        ),
        "query_surface_correspondence": torch.stack(
            [item["query_surface_correspondence"] for item in batch],
            dim=0,
        ),
        "query_surface_labels": torch.stack(
            [item["query_surface_labels"] for item in batch],
            dim=0,
        ),
        "query_occ_labels": torch.stack(
            [item["query_occ_labels"] for item in batch],
            dim=0,
        ),
        "category": [item["category"] for item in batch],
        "model_id": [item["model_id"] for item in batch],
        "mesh_path": [item["mesh_path"] for item in batch],
    }
    if "utonia_input_view2" in batch[0]:
        collated["utonia_input_view2"] = UTONIA.data.collate_fn(
            [item["utonia_input_view2"] for item in batch]
        )
        collated["support_points_view2"] = torch.stack(
            [item["support_points_view2"] for item in batch],
            dim=0,
        )
        collated["query_surface_points_view2"] = torch.stack(
            [item["query_surface_points_view2"] for item in batch],
            dim=0,
        )
        collated["query_occ_points_view2"] = torch.stack(
            [item["query_occ_points_view2"] for item in batch],
            dim=0,
        )
        collated["query_probe_points_view2"] = torch.stack(
            [item["query_probe_points_view2"] for item in batch],
            dim=0,
        )
    return collated


class PartNextUniversalFieldDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        categories: Iterable[str] | None = None,
        num_support_points: int = 5000,
        num_query_surface_points: int = 2048,
        num_query_occ_points: int = 1536,
        num_query_probe_points: int = 1024,
        balanced_sampling_ratio: float = 0.5,
        query_sampling_ratio: float = 0.75,
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
        label_level: str = "leaf",
        alias_config_path: str | Path | None = None,
        canonical_label_names: list[str] | None = None,
        ignore_labels: Iterable[str] | None = ("Other",),
        rotation_mode: str = "so3",
        jitter_std: float = 0.005,
        second_view: bool = True,
        probe_label_radius: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.categories = _parse_categories(self.dataset_root, categories)
        self.num_support_points = int(num_support_points)
        self.num_query_surface_points = int(num_query_surface_points)
        self.num_query_occ_points = int(num_query_occ_points)
        self.num_query_probe_points = int(num_query_probe_points)
        self.balanced_sampling_ratio = float(balanced_sampling_ratio)
        self.query_sampling_ratio = float(query_sampling_ratio)
        self.near_surface_ratio = float(near_surface_ratio)
        self.near_surface_noise = float(near_surface_noise)
        self.uniform_padding = float(uniform_padding)
        self.coord_scale = float(coord_scale)
        self.center_shift_z = bool(center_shift_z)
        self.max_retries = int(max_retries)
        self.label_level = label_level
        self.alias_lookup = _load_alias_lookup(alias_config_path)
        self.ignore_names = {
            _normalize_label_name(label) for label in (ignore_labels or [])
        }
        self.rotation_mode = rotation_mode
        self.jitter_std = float(jitter_std)
        self.second_view = bool(second_view)
        self.probe_label_radius = float(probe_label_radius)

        self.canonical_label_names = (
            list(canonical_label_names)
            if canonical_label_names is not None
            else build_partnext_canonical_label_space(
                dataset_root=self.dataset_root,
                categories=self.categories,
                label_level=self.label_level,
                alias_config_path=alias_config_path,
                ignore_labels=ignore_labels,
            )
        )
        self.canonical_label_to_id = {
            label_name: index for index, label_name in enumerate(self.canonical_label_names)
        }

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

    @property
    def num_classes(self) -> int:
        return len(self.canonical_label_names)

    def _build_records(self) -> list[PartNextCanonicalRecord]:
        records: list[PartNextCanonicalRecord] = []
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
                        PartNextCanonicalRecord(
                            category=category,
                            model_id=annotation["model_id"],
                            mesh_path=mesh_path,
                            hierarchy_json=annotation["hierarchyList"],
                            masks_json=annotation["masks"],
                            mesh_face_num_json=annotation["mesh_face_num"],
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
        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} retries"
        ) from last_error

    def _load_sample(self, record: PartNextCanonicalRecord) -> dict:
        meshes = _load_geometry_meshes(record.mesh_path)
        mesh = _concatenate_meshes(meshes)
        face_labels = _build_global_face_labels(
            record=record,
            num_meshes=len(meshes),
            mesh_face_counts=[len(item.faces) for item in meshes],
            total_faces=len(mesh.faces),
            canonical_label_to_id=self.canonical_label_to_id,
            label_level=self.label_level,
            alias_lookup=self.alias_lookup,
            ignore_names=self.ignore_names,
        )

        support_points_raw, support_normals, support_colors, _ = _sample_surface_points(
            mesh=mesh,
            face_labels=face_labels,
            num_points=self.num_support_points,
            balanced_sampling_ratio=self.balanced_sampling_ratio,
        )
        query_surface_raw, _, _, query_surface_labels = _sample_surface_points(
            mesh=mesh,
            face_labels=face_labels,
            num_points=self.num_query_surface_points,
            balanced_sampling_ratio=self.query_sampling_ratio,
        )
        query_occ_raw, query_occ_labels = _sample_queries(
            mesh=mesh,
            num_points=self.num_query_occ_points,
            near_surface_ratio=self.near_surface_ratio,
            near_surface_noise=self.near_surface_noise,
            uniform_padding=self.uniform_padding,
        )
        query_probe_raw, _ = _sample_queries(
            mesh=mesh,
            num_points=self.num_query_probe_points,
            near_surface_ratio=1.0,
            near_surface_noise=self.near_surface_noise,
            uniform_padding=self.uniform_padding,
        )

        (
            support_points,
            query_surface_points,
            query_occ_points,
            query_probe_points,
        ) = _normalize_multiple_arrays(
            support_points_raw,
            query_surface_raw,
            query_occ_raw,
            query_probe_raw,
            coord_scale=self.coord_scale,
            center_shift_z=self.center_shift_z,
        )
        query_probe_labels = _assign_nearest_surface_labels(
            probe_points=query_probe_points,
            surface_points=query_surface_points,
            surface_labels=query_surface_labels,
            max_distance=self.probe_label_radius * self.coord_scale,
        )

        if self.split == "train":
            rotation_view1 = _build_rotation_matrix(self.rotation_mode)
            rotation_view2 = _build_rotation_matrix(self.rotation_mode)
            jitter_std = self.jitter_std
        else:
            rotation_view1 = np.eye(3, dtype=np.float32)
            rotation_view2 = np.eye(3, dtype=np.float32)
            jitter_std = 0.0

        view1_support_points, view1_support_normals = _apply_point_normal_transform(
            support_points,
            support_normals,
            rotation=rotation_view1,
            jitter_std=jitter_std,
        )
        view1_query_surface = _apply_point_transform(
            query_surface_points,
            rotation=rotation_view1,
            jitter_std=jitter_std,
        )
        view1_query_occ = _apply_point_transform(
            query_occ_points,
            rotation=rotation_view1,
            jitter_std=jitter_std,
        )
        view1_query_probe = _apply_point_transform(
            query_probe_points,
            rotation=rotation_view1,
            jitter_std=jitter_std,
        )

        sample = {
            "utonia_input_view1": self.utonia_transform(
                {
                    "coord": view1_support_points.copy(),
                    "color": support_colors.copy(),
                    "normal": view1_support_normals.copy(),
                }
            ),
            "support_points_view1": torch.from_numpy(view1_support_points).float(),
            "query_surface_points_view1": torch.from_numpy(view1_query_surface).float(),
            "query_occ_points_view1": torch.from_numpy(view1_query_occ).float(),
            "query_probe_points_view1": torch.from_numpy(view1_query_probe).float(),
            "query_occ_points_canonical": torch.from_numpy(query_occ_points.copy()).float(),
            "query_probe_points_canonical": torch.from_numpy(query_probe_points.copy()).float(),
            "query_occ_correspondence": torch.arange(
                query_occ_points.shape[0],
                dtype=torch.long,
            ),
            "query_probe_correspondence": torch.arange(
                query_probe_points.shape[0],
                dtype=torch.long,
            ),
            "query_surface_points_canonical": torch.from_numpy(query_surface_points.copy()).float(),
            "query_surface_correspondence": torch.arange(
                query_surface_points.shape[0],
                dtype=torch.long,
            ),
            "query_surface_labels": torch.from_numpy(query_surface_labels).long(),
            "query_probe_labels": torch.from_numpy(query_probe_labels).long(),
            "query_occ_labels": torch.from_numpy(query_occ_labels).float(),
            "category": record.category,
            "model_id": record.model_id,
            "mesh_path": str(record.mesh_path),
        }

        if self.second_view:
            view2_support_points, view2_support_normals = _apply_point_normal_transform(
                support_points,
                support_normals,
                rotation=rotation_view2,
                jitter_std=jitter_std,
            )
            sample["utonia_input_view2"] = self.utonia_transform(
                {
                    "coord": view2_support_points.copy(),
                    "color": support_colors.copy(),
                    "normal": view2_support_normals.copy(),
                }
            )
            sample["support_points_view2"] = torch.from_numpy(view2_support_points).float()
            sample["query_surface_points_view2"] = torch.from_numpy(
                _apply_point_transform(
                    query_surface_points,
                    rotation=rotation_view2,
                    jitter_std=jitter_std,
                )
            ).float()
            sample["query_occ_points_view2"] = torch.from_numpy(
                _apply_point_transform(
                    query_occ_points,
                    rotation=rotation_view2,
                    jitter_std=jitter_std,
                )
            ).float()
            sample["query_probe_points_view2"] = torch.from_numpy(
                _apply_point_transform(
                    query_probe_points,
                    rotation=rotation_view2,
                    jitter_std=jitter_std,
                )
            ).float()

        return sample
