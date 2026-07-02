from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from my_datasets.partnext_canonical_field import UTONIA
from models import UtoniaUniversalFieldNet
from myutils import move_to_device


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _load_point_cloud(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        cloud = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        key = "point_cloud" if "point_cloud" in payload else "points"
        cloud = payload[key]
    else:
        geometry = trimesh.load(path, process=False)
        if isinstance(geometry, trimesh.Scene):
            geometries = [g for g in geometry.dump() if hasattr(g, "vertices")]
            if not geometries:
                raise ValueError(f"No point or mesh geometry found in {path}")
            vertices = np.concatenate([np.asarray(g.vertices) for g in geometries], axis=0)
            colors = []
            for geom in geometries:
                rgba = getattr(getattr(geom, "visual", None), "vertex_colors", None)
                if rgba is None or len(rgba) != len(geom.vertices):
                    rgba = np.zeros((len(geom.vertices), 4), dtype=np.uint8)
                colors.append(np.asarray(rgba)[:, :3])
            cloud = np.concatenate([vertices, np.concatenate(colors, axis=0)], axis=1)
        else:
            vertices = np.asarray(geometry.vertices)
            rgba = getattr(getattr(geometry, "visual", None), "vertex_colors", None)
            if rgba is None or len(rgba) != len(vertices):
                rgba = np.zeros((len(vertices), 4), dtype=np.uint8)
            cloud = np.concatenate([vertices, np.asarray(rgba)[:, :3]], axis=1)

    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Expected point cloud with shape [N, >=3], got {cloud.shape}")
    if cloud.shape[1] < 6:
        padded = np.zeros((cloud.shape[0], 6), dtype=np.float32)
        padded[:, : cloud.shape[1]] = cloud
        cloud = padded
    return cloud[:, :6].astype(np.float32, copy=False)


def _strip_zero_points(point_cloud: np.ndarray) -> np.ndarray:
    xyz = np.asarray(point_cloud, dtype=np.float32)[:, :3]
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-8)
    return np.asarray(point_cloud, dtype=np.float32)[valid]


def _sample_rows(point_cloud: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(point_cloud) == 0:
        return np.zeros((int(num_points), point_cloud.shape[1]), dtype=np.float32)
    replace = len(point_cloud) < int(num_points)
    indices = rng.choice(len(point_cloud), size=int(num_points), replace=replace)
    return point_cloud[indices].astype(np.float32)


def _fallback_normals(points: np.ndarray) -> np.ndarray:
    centered = points.astype(np.float32) - points.astype(np.float32).mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normals = np.zeros_like(centered, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    normals[valid] = centered[valid] / norms[valid]
    return normals


def _normalize_support_and_query(
    support_points: np.ndarray,
    query_points: np.ndarray,
    *,
    coord_scale: float,
    center_shift_z: bool,
) -> tuple[np.ndarray, np.ndarray]:
    centroid = support_points.mean(axis=0, keepdims=True)
    support = support_points - centroid
    query = query_points - centroid
    radius = max(float(np.linalg.norm(support, axis=1).max()), 1e-6)
    support = support / radius
    query = query / radius

    if center_shift_z:
        x_min, y_min, z_min = support.min(axis=0)
        x_max, y_max, _ = support.max(axis=0)
        shift = np.asarray([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min], dtype=np.float32)
        support = support - shift
        query = query - shift

    return (support * float(coord_scale)).astype(np.float32), (query * float(coord_scale)).astype(np.float32)


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


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[UtoniaUniversalFieldNet, dict[str, Any], list[str]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    args = dict(checkpoint.get("args", {}))
    label_names = list(checkpoint.get("canonical_label_names", []))
    if not label_names:
        labels_path = checkpoint_path.parent / "canonical_labels.json"
        label_names = json.loads(labels_path.read_text(encoding="utf-8"))

    model = UtoniaUniversalFieldNet(
        num_classes=len(label_names),
        utonia_checkpoint=args.get("utonia_checkpoint", "auto"),
        utonia_repo_id=args.get("utonia_repo_id", "Pointcept/Utonia"),
        utonia_upcast_levels=int(args.get("utonia_upcast_levels", 0)),
        freeze_utonia=True,
        adapter_hidden_dim=int(args.get("adapter_hidden_dim", 256)),
        branch_dim=int(args.get("branch_dim", 256)),
        sem_embedding_dim=int(args.get("sem_embedding_dim", 128)),
        geo_embedding_dim=int(args.get("geo_embedding_dim", 128)),
        num_anchors=int(args.get("num_anchors", 256)),
        rbf_sigma=float(args.get("rbf_sigma", 0.12)),
        triplane_resolution=int(args.get("triplane_resolution", 64)),
        triplane_padding=float(args.get("triplane_padding", 0.05)),
    )
    projector_state = checkpoint["projector_state_dict"]
    model.semantic_adapter.load_state_dict(projector_state["semantic_adapter"])
    if "semantic_local_fusion" in projector_state:
        model.semantic_local_fusion.load_state_dict(projector_state["semantic_local_fusion"])
    model.semantic_decoder.load_state_dict(projector_state["semantic_decoder"])
    if "encoder_state_dict" in checkpoint:
        model.feature_extractor.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.to(device).eval()
    return model, args, label_names


@torch.no_grad()
def export_semantic_point_cloud(
    *,
    checkpoint: Path,
    input_point_cloud: Path,
    output_npz: Path,
    num_query_points: int,
    device: torch.device,
    seed: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    model, checkpoint_args, label_names = _load_model(checkpoint, device)
    cloud = _strip_zero_points(_load_point_cloud(input_point_cloud))
    if len(cloud) == 0:
        raise ValueError("Input point cloud contains no valid xyz points.")

    query_cloud = _sample_rows(cloud, int(num_query_points), rng)
    support_xyz = cloud[:, :3].astype(np.float32)
    query_world_xyz = query_cloud[:, :3].astype(np.float32)
    support_rgb = cloud[:, 3:6].astype(np.float32)
    support_normals = _fallback_normals(support_xyz)

    support_normed, query_normed = _normalize_support_and_query(
        support_xyz,
        query_world_xyz,
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
    )

    transform = _build_utonia_transform(float(checkpoint_args.get("grid_size", 0.01)))
    utonia_input = transform(
        {
            "coord": support_normed.copy(),
            "color": support_rgb.copy(),
            "normal": support_normals.copy(),
        }
    )
    utonia_input = UTONIA.data.collate_fn([utonia_input])
    utonia_input = move_to_device(utonia_input, device)

    support_t = torch.from_numpy(support_normed[None]).float().to(device)
    query_t = torch.from_numpy(query_normed[None]).float().to(device)
    cache = model.encode_support(utonia_input=utonia_input, support_points=support_t)
    output = model.query_semantic(cache=cache, query_points=query_t)

    embeddings = output["embedding"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    logits = output["logits"].squeeze(0).detach().cpu()
    probabilities = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
    confidence = probabilities.max(axis=1)
    pred_labels = probabilities.argmax(axis=1).astype(np.int64)
    semantic_point_cloud = np.concatenate([query_world_xyz, embeddings], axis=1).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        semantic_point_cloud=semantic_point_cloud,
        query_xyz=query_world_xyz,
        sem_embeddings=embeddings,
        sem_logits=logits.numpy().astype(np.float32),
        sem_probabilities=probabilities,
        pred_labels=pred_labels,
        confidence=confidence,
        label_names=np.asarray(label_names, dtype=object),
    )
    print(f"Wrote semantic point cloud: {output_npz}")
    print(f"Shape: {semantic_point_cloud.shape} = [xyz(3) + embedding({embeddings.shape[1]})]")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export xyz + semantic embedding point clouds from a trained semantic field.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--input-point-cloud", type=Path, required=True, help="Input .npy/.npz/.ply/.pcd point cloud.")
    parser.add_argument("--output-npz", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--num-query-points", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    export_semantic_point_cloud(
        checkpoint=args.checkpoint,
        input_point_cloud=args.input_point_cloud,
        output_npz=args.output_npz,
        num_query_points=args.num_query_points,
        device=torch.device(args.device),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
