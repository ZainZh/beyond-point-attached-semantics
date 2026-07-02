from __future__ import annotations

import json
import random
import re
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

IGNORE_LABEL = -1


@dataclass(frozen=True)
class PartNextCanonicalRecord:
    category: str
    model_id: str
    mesh_path: Path
    hierarchy_json: str
    masks_json: str
    mesh_face_num_json: str


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
    records: list[PartNextCanonicalRecord],
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    split: str,
) -> list[PartNextCanonicalRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    grouped: dict[str, list[PartNextCanonicalRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)

    selected: list[PartNextCanonicalRecord] = []
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


def _normalize_object_space(
    points: np.ndarray,
    coord_scale: float,
    center_shift_z: bool,
) -> np.ndarray:
    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid

    radius = np.linalg.norm(points, axis=1).max()
    radius = max(float(radius), 1e-6)
    points = points / radius

    if center_shift_z:
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, _ = points.max(axis=0)
        shift = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
        points = points - shift

    return (points * coord_scale).astype(np.float32)


def _normalize_label_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _load_alias_lookup(alias_config_path: str | Path | None) -> dict[str, str | None]:
    if alias_config_path is None:
        return {}

    path = Path(alias_config_path)
    alias_data = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, str | None] = {}

    if all(isinstance(value, (str, type(None))) for value in alias_data.values()):
        for alias, canonical in alias_data.items():
            lookup[_normalize_label_name(alias)] = (
                None if canonical in {None, "__ignore__"} else str(canonical)
            )
        return lookup

    for canonical, aliases in alias_data.items():
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            raise TypeError(
                "Alias config must be either alias->canonical strings or "
                "canonical->list[str] mappings."
            )
        canonical_name = None if canonical == "__ignore__" else str(canonical)
        for alias in [canonical, *aliases]:
            lookup[_normalize_label_name(str(alias))] = canonical_name
    return lookup


def _deduplicate_labels(labels: Iterable[str]) -> tuple[str, ...]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        deduplicated.append(label)
        seen.add(label)
    return tuple(deduplicated)


def _select_part_label(path: list[str], label_level: str) -> str:
    if label_level == "leaf":
        return path[-1]
    if label_level == "parent":
        return path[-2] if len(path) >= 2 else path[-1]
    if label_level == "path":
        return " > ".join(path)
    raise ValueError(f"Unsupported label_level: {label_level}")


def _select_part_label_candidates(path: list[str], label_level: str) -> tuple[str, ...]:
    if label_level == "config":
        candidates = [" > ".join(path)]
        if len(path) >= 2:
            candidates.append(path[-2])
        candidates.append(path[-1])
        return _deduplicate_labels(candidates)
    return (_select_part_label(path, label_level),)


def _build_mask_id_to_label_name(
    hierarchy: list[dict],
    label_level: str,
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}

    def walk(node: dict, path: list[str]) -> None:
        current_path = [*path, str(node["name"])]
        if "maskId" in node:
            mapping[str(node["maskId"])] = _select_part_label_candidates(
                current_path,
                label_level,
            )
        for child in node.get("children", []) or []:
            walk(child, current_path)

    for root in hierarchy:
        walk(root, [])
    return mapping


def _lookup_alias(
    raw_name: str,
    category: str,
    alias_lookup: dict[str, str | None],
) -> tuple[bool, str | None]:
    category_scoped = _normalize_label_name(f"{category}::{raw_name}")
    if category_scoped in alias_lookup:
        return True, alias_lookup[category_scoped]

    normalized = _normalize_label_name(raw_name)
    if normalized in alias_lookup:
        return True, alias_lookup[normalized]

    return False, None


def _resolve_canonical_name(
    raw_name: str | Iterable[str],
    category: str,
    alias_lookup: dict[str, str | None],
    ignore_names: set[str],
    strict_alias: bool = False,
) -> str | None:
    if isinstance(raw_name, str):
        candidates = (raw_name,)
    else:
        candidates = tuple(raw_name)

    if strict_alias:
        if not alias_lookup:
            raise ValueError("label_level='config' requires a non-empty alias config.")
        for candidate in candidates:
            matched, mapped = _lookup_alias(candidate, category, alias_lookup)
            if not matched:
                continue
            if mapped is None:
                return None
            return str(mapped)
        preview = candidates[0] if candidates else "<empty label>"
        raise ValueError(
            f"No alias-config match for {preview!r} in category {category!r}. "
            f"Config label-level checked candidates: {list(candidates)}"
        )

    raw_name = candidates[0]
    normalized = _normalize_label_name(raw_name)
    if normalized in ignore_names:
        return None

    matched, mapped = _lookup_alias(raw_name, category, alias_lookup)
    if matched:
        if mapped is None:
            return None
        return str(mapped)
    return raw_name


def build_partnext_canonical_label_space(
    dataset_root: str | Path,
    categories: Iterable[str] | None = None,
    label_level: str = "leaf",
    alias_config_path: str | Path | None = None,
    ignore_labels: Iterable[str] | None = ("Other",),
) -> list[str]:
    root = Path(dataset_root)
    selected_categories = _parse_categories(root, categories)
    alias_lookup = _load_alias_lookup(alias_config_path)
    ignore_names = {_normalize_label_name(label) for label in (ignore_labels or [])}
    strict_alias = label_level == "config"
    if strict_alias and not alias_lookup:
        raise ValueError("label_level='config' requires --alias-config.")

    canonical_names: set[str] = set()
    for category in selected_categories:
        annotation_path = root / category / "annotation.jsonl"
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                annotation = json.loads(line)
                hierarchy = json.loads(annotation["hierarchyList"])
                mask_id_to_label = _build_mask_id_to_label_name(hierarchy, label_level)
                for raw_name in mask_id_to_label.values():
                    canonical_name = _resolve_canonical_name(
                        raw_name=raw_name,
                        category=category,
                        alias_lookup=alias_lookup,
                        ignore_names=ignore_names,
                        strict_alias=strict_alias,
                    )
                    if canonical_name is not None:
                        canonical_names.add(canonical_name)
    if not canonical_names:
        raise RuntimeError(
            "No canonical labels were found. Check the selected categories, "
            "label level, alias config, and ignore-label settings."
        )
    return sorted(canonical_names)


def _sample_face_colors(mesh: trimesh.Trimesh, face_index: np.ndarray) -> np.ndarray:
    if (
        hasattr(mesh.visual, "face_colors")
        and mesh.visual.face_colors is not None
        and len(mesh.visual.face_colors) == len(mesh.faces)
    ):
        return np.asarray(mesh.visual.face_colors[face_index, :3], dtype=np.float32)
    if (
        hasattr(mesh.visual, "vertex_colors")
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    ):
        return np.asarray(
            mesh.visual.vertex_colors[mesh.faces[face_index], :3].mean(axis=1),
            dtype=np.float32,
        )
    return np.zeros((len(face_index), 3), dtype=np.float32)


def _build_rotation_matrix(mode: str) -> np.ndarray:
    if mode == "none":
        return np.eye(3, dtype=np.float32)
    if mode == "z":
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        return np.asarray(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if mode == "so3":
        u1, u2, u3 = np.random.rand(3)
        qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        return np.asarray(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported rotation mode: {mode}")


def _apply_view_augmentation(
    points: np.ndarray,
    normals: np.ndarray,
    split: str,
    rotation_mode: str,
    jitter_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if split != "train":
        return points.astype(np.float32), normals.astype(np.float32)

    rotation = _build_rotation_matrix(rotation_mode)
    aug_points = points @ rotation.T
    aug_normals = normals @ rotation.T

    if jitter_std > 0:
        aug_points = aug_points + np.random.normal(
            loc=0.0,
            scale=jitter_std,
            size=aug_points.shape,
        ).astype(np.float32)
    return aug_points.astype(np.float32), aug_normals.astype(np.float32)


def _distribute_counts(total: int, num_bins: int) -> list[int]:
    if num_bins <= 0 or total <= 0:
        return [0] * max(num_bins, 0)
    base = total // num_bins
    remainder = total % num_bins
    return [base + (1 if index < remainder else 0) for index in range(num_bins)]


def _load_geometry_meshes(mesh_path: str | Path) -> list[trimesh.Trimesh]:
    geometry = trimesh.load(mesh_path, force="scene")
    if isinstance(geometry, trimesh.Scene):
        dumped = geometry.dump(concatenate=False)
        if isinstance(dumped, trimesh.Trimesh):
            dumped = [dumped]
        # Scene.geometry stores local node meshes without graph transforms,
        # which can silently swap/offset axes for GLTF assets. Dumped meshes
        # preserve the instantiated world-space transforms.
        dumped_meshes = [
            mesh.copy()
            for mesh in dumped
            if isinstance(mesh, trimesh.Trimesh)
        ]
        geometry_order = {name: index for index, name in enumerate(geometry.geometry.keys())}
        meshes = sorted(
            dumped_meshes,
            key=lambda mesh: (
                geometry_order.get(str(mesh.metadata.get("name", "")), len(geometry_order)),
                str(mesh.metadata.get("name", "")),
            ),
        )
    elif isinstance(geometry, trimesh.Trimesh):
        meshes = [geometry.copy()]
    else:
        raise TypeError(f"Unsupported geometry type for {mesh_path}: {type(geometry)}")

    if not meshes:
        raise ValueError(f"No valid mesh geometry found for {mesh_path}")
    return meshes


def _concatenate_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    valid_meshes = [mesh for mesh in meshes if len(mesh.faces) > 0]
    if len(valid_meshes) == 1:
        mesh = valid_meshes[0].copy()
    else:
        mesh = trimesh.util.concatenate(valid_meshes)
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("Annotated mesh is empty after concatenation")
    return mesh


def _lcs_mesh_id_to_geometry_index(
    annotated_face_counts: dict[int, int],
    mesh_face_counts: list[int],
) -> tuple[dict[int, int], float]:
    annotated_items = sorted(annotated_face_counts.items())
    num_annotated = len(annotated_items)
    num_loaded = len(mesh_face_counts)
    dp = [[0] * (num_loaded + 1) for _ in range(num_annotated + 1)]

    for ann_index in range(num_annotated - 1, -1, -1):
        ann_count = annotated_items[ann_index][1]
        for geom_index in range(num_loaded - 1, -1, -1):
            if ann_count == mesh_face_counts[geom_index]:
                dp[ann_index][geom_index] = dp[ann_index + 1][geom_index + 1] + 1
            else:
                dp[ann_index][geom_index] = max(
                    dp[ann_index + 1][geom_index],
                    dp[ann_index][geom_index + 1],
                )

    mapping: dict[int, int] = {}
    ann_index = 0
    geom_index = 0
    while ann_index < num_annotated and geom_index < num_loaded:
        ann_mesh_id, ann_count = annotated_items[ann_index]
        if (
            ann_count == mesh_face_counts[geom_index]
            and dp[ann_index][geom_index] == dp[ann_index + 1][geom_index + 1] + 1
        ):
            mapping[ann_mesh_id] = geom_index
            ann_index += 1
            geom_index += 1
        elif dp[ann_index + 1][geom_index] >= dp[ann_index][geom_index + 1]:
            ann_index += 1
        else:
            geom_index += 1

    total_annotated_faces = float(sum(annotated_face_counts.values()))
    matched_faces = float(
        sum(annotated_face_counts[mesh_id] for mesh_id in mapping.keys())
    )
    coverage = matched_faces / total_annotated_faces if total_annotated_faces > 0 else 0.0
    return mapping, coverage


def _build_global_face_labels(
    record: PartNextCanonicalRecord,
    num_meshes: int,
    mesh_face_counts: list[int],
    total_faces: int,
    canonical_label_to_id: dict[str, int],
    label_level: str,
    alias_lookup: dict[str, str | None],
    ignore_names: set[str],
) -> np.ndarray:
    hierarchy = json.loads(record.hierarchy_json)
    masks = json.loads(record.masks_json)
    annotated_face_counts = {
        int(mesh_id): int(face_count)
        for mesh_id, face_count in json.loads(record.mesh_face_num_json).items()
    }
    strict_alias = label_level == "config"
    if strict_alias and not alias_lookup:
        raise ValueError("label_level='config' requires --alias-config.")
    mask_id_to_label_name = _build_mask_id_to_label_name(hierarchy, label_level)
    mesh_id_to_geometry_index, matched_face_coverage = _lcs_mesh_id_to_geometry_index(
        annotated_face_counts=annotated_face_counts,
        mesh_face_counts=mesh_face_counts,
    )
    if matched_face_coverage < 0.9:
        raise ValueError(
            f"Only recovered {matched_face_coverage:.2%} of annotated faces for {record.mesh_path}"
        )
    if matched_face_coverage < 0.999:
        warnings.warn(
            f"Recovered {matched_face_coverage:.2%} of annotated faces for {record.mesh_path} "
            f"after remapping annotation mesh ids to loaded geometry slots.",
            stacklevel=2,
        )

    face_offsets = np.cumsum([0, *mesh_face_counts[:-1]], dtype=np.int64)
    face_labels = np.full(total_faces, IGNORE_LABEL, dtype=np.int64)
    unmatched_mesh_ids: set[int] = set()
    overflow_mesh_ids: set[int] = set()

    for mask_id, mesh_to_faces in masks.items():
        raw_name = mask_id_to_label_name.get(str(mask_id))
        if raw_name is None:
            continue

        canonical_name = _resolve_canonical_name(
            raw_name=raw_name,
            category=record.category,
            alias_lookup=alias_lookup,
            ignore_names=ignore_names,
            strict_alias=strict_alias,
        )
        if canonical_name is None:
            continue

        label_id = canonical_label_to_id.get(canonical_name)
        if label_id is None:
            continue

        for mesh_id_text, local_faces in mesh_to_faces.items():
            mesh_id = int(mesh_id_text)
            geometry_index = mesh_id_to_geometry_index.get(mesh_id)
            if geometry_index is None:
                unmatched_mesh_ids.add(mesh_id)
                continue
            local_faces_np = np.asarray(local_faces, dtype=np.int64)
            geometry_face_count = mesh_face_counts[geometry_index]
            if local_faces_np.size == 0:
                continue
            if int(local_faces_np.max()) >= geometry_face_count:
                overflow_mesh_ids.add(mesh_id)
                continue
            global_faces = face_offsets[geometry_index] + local_faces_np
            face_labels[global_faces] = label_id

    if unmatched_mesh_ids:
        preview = sorted(unmatched_mesh_ids)[:12]
        warnings.warn(
            f"Skipped {len(unmatched_mesh_ids)} annotation mesh ids that could not be "
            f"matched to loaded geometry slots in {record.mesh_path}. "
            f"Examples: {preview}",
            stacklevel=2,
        )
    if overflow_mesh_ids:
        preview = sorted(overflow_mesh_ids)[:12]
        warnings.warn(
            f"Skipped {len(overflow_mesh_ids)} annotation mesh ids whose local face indices "
            f"exceeded the matched geometry face count in {record.mesh_path}. "
            f"Examples: {preview}",
            stacklevel=2,
        )

    return face_labels


def _sample_surface_points(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    num_points: int,
    balanced_sampling_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    balanced_sampling_ratio = float(np.clip(balanced_sampling_ratio, 0.0, 1.0))
    num_balanced = int(round(num_points * balanced_sampling_ratio))
    num_uniform = max(0, num_points - num_balanced)

    sampled_points: list[np.ndarray] = []
    sampled_normals: list[np.ndarray] = []
    sampled_colors: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []

    def append_samples(face_index: np.ndarray, points: np.ndarray) -> None:
        sampled_points.append(points.astype(np.float32))
        sampled_normals.append(mesh.face_normals[face_index].astype(np.float32))
        sampled_colors.append(_sample_face_colors(mesh, face_index))
        sampled_labels.append(face_labels[face_index].astype(np.int64))

    if num_uniform > 0:
        points, face_index = trimesh.sample.sample_surface(mesh, num_uniform)
        append_samples(face_index=face_index, points=points)

    valid_face_labels = sorted(int(label) for label in np.unique(face_labels) if label >= 0)
    if num_balanced > 0 and valid_face_labels:
        area_weights = np.asarray(mesh.area_faces, dtype=np.float64)
        label_counts = _distribute_counts(num_balanced, len(valid_face_labels))
        for label_id, label_count in zip(valid_face_labels, label_counts):
            if label_count <= 0:
                continue
            face_weight = np.where(face_labels == label_id, area_weights, 0.0)
            if float(face_weight.sum()) <= 0:
                continue
            points, face_index = trimesh.sample.sample_surface(
                mesh,
                label_count,
                face_weight=face_weight,
            )
            append_samples(face_index=face_index, points=points)

    if not sampled_points:
        raise RuntimeError("Failed to sample any labeled surface points from the mesh")

    points = np.concatenate(sampled_points, axis=0)
    normals = np.concatenate(sampled_normals, axis=0)
    colors = np.concatenate(sampled_colors, axis=0)
    labels = np.concatenate(sampled_labels, axis=0)

    if len(points) > num_points:
        selection = np.random.permutation(len(points))[:num_points]
        points = points[selection]
        normals = normals[selection]
        colors = colors[selection]
        labels = labels[selection]
    elif len(points) < num_points:
        shortfall = num_points - len(points)
        selection = np.random.choice(len(points), size=shortfall, replace=True)
        points = np.concatenate([points, points[selection]], axis=0)
        normals = np.concatenate([normals, normals[selection]], axis=0)
        colors = np.concatenate([colors, colors[selection]], axis=0)
        labels = np.concatenate([labels, labels[selection]], axis=0)

    permutation = np.random.permutation(len(points))
    return (
        points[permutation].astype(np.float32),
        normals[permutation].astype(np.float32),
        colors[permutation].astype(np.float32),
        labels[permutation].astype(np.int64),
    )


def partnext_canonical_field_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "utonia_input_view1": UTONIA.data.collate_fn(
            [item["utonia_input_view1"] for item in batch]
        ),
        "support_points_view1": torch.stack(
            [item["support_points_view1"] for item in batch],
            dim=0,
        ),
        "point_labels": torch.stack([item["point_labels"] for item in batch], dim=0),
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
    return collated


class PartNextCanonicalFieldDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        categories: Iterable[str] | None = None,
        num_points: int = 2048,
        balanced_sampling_ratio: float = 0.5,
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
        jitter_std: float = 0.0,
        second_view: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.categories = _parse_categories(self.dataset_root, categories)
        self.num_points = num_points
        self.balanced_sampling_ratio = balanced_sampling_ratio
        self.coord_scale = coord_scale
        self.center_shift_z = center_shift_z
        self.max_retries = max_retries
        self.label_level = label_level
        self.alias_lookup = _load_alias_lookup(alias_config_path)
        self.ignore_names = {_normalize_label_name(label) for label in (ignore_labels or [])}
        self.rotation_mode = rotation_mode
        self.jitter_std = jitter_std
        self.second_view = second_view

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
        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} retries") from last_error

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

        points, normals, colors, point_labels = _sample_surface_points(
            mesh=mesh,
            face_labels=face_labels,
            num_points=self.num_points,
            balanced_sampling_ratio=self.balanced_sampling_ratio,
        )
        points = _normalize_object_space(
            points=points,
            coord_scale=self.coord_scale,
            center_shift_z=self.center_shift_z,
        )

        view1_points, view1_normals = _apply_view_augmentation(
            points=points,
            normals=normals,
            split=self.split,
            rotation_mode=self.rotation_mode,
            jitter_std=self.jitter_std,
        )
        sample = {
            "utonia_input_view1": self.utonia_transform(
                {
                    "coord": view1_points.copy(),
                    "color": colors.copy(),
                    "normal": view1_normals.copy(),
                }
            ),
            "support_points_view1": torch.from_numpy(view1_points).float(),
            "point_labels": torch.from_numpy(point_labels).long(),
            "category": record.category,
            "model_id": record.model_id,
            "mesh_path": str(record.mesh_path),
        }

        if self.second_view:
            view2_points, view2_normals = _apply_view_augmentation(
                points=points,
                normals=normals,
                split=self.split,
                rotation_mode=self.rotation_mode,
                jitter_std=self.jitter_std,
            )
            sample["utonia_input_view2"] = self.utonia_transform(
                {
                    "coord": view2_points.copy(),
                    "color": colors.copy(),
                    "normal": view2_normals.copy(),
                }
            )
            sample["support_points_view2"] = torch.from_numpy(view2_points).float()

        return sample
