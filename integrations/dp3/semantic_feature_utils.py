import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from object_pointcloud_utils import resample_point_cloud, strip_zero_points


REPO_ROOT = Path(__file__).resolve().parents[3]
SEM_ROOT = REPO_ROOT / "include" / "3d_semantic_train"
if SEM_ROOT.exists() and str(SEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SEM_ROOT))

from my_datasets.partnext_canonical_field import UTONIA  # noqa: E402
from models.universal_field.utonia_universal_field import UtoniaUniversalFieldNet  # noqa: E402


DEBUG_PLACEHOLDER_COLORS_RGB = {
    "{A}": np.asarray([255.0, 48.0, 48.0], dtype=np.float32),
    "A": np.asarray([255.0, 48.0, 48.0], dtype=np.float32),
    "{B}": np.asarray([48.0, 96.0, 255.0], dtype=np.float32),
    "B": np.asarray([48.0, 96.0, 255.0], dtype=np.float32),
}
DEFAULT_DEBUG_PLACEHOLDER_COLOR_RGB = np.asarray([180.0, 180.0, 180.0], dtype=np.float32)
SEMANTIC_INPUT_COLOR_MODES = {"debug_placeholder", "stored_scaled", "stored"}
SEMANTIC_FORWARD_MODES = {"reference", "dp3"}


def ensure_point_cloud_channels(point_cloud: np.ndarray, *, channels: int = 6) -> np.ndarray:
    cloud = np.asarray(point_cloud, dtype=np.float32)
    if cloud.ndim != 2:
        raise ValueError(f"Expected point cloud with shape [N,C], got {cloud.shape!r}")
    if cloud.shape[1] >= int(channels):
        return cloud[:, : int(channels)].astype(np.float32, copy=False)
    padded = np.zeros((cloud.shape[0], int(channels)), dtype=np.float32)
    padded[:, : cloud.shape[1]] = cloud
    return padded


def prepare_semantic_input_point_cloud(
    object_point_cloud: np.ndarray,
    *,
    placeholder: str = "",
    color_mode: str = "debug_placeholder",
) -> np.ndarray:
    cloud = ensure_point_cloud_channels(object_point_cloud, channels=6).astype(np.float32, copy=True)
    mode = str(color_mode)
    if mode == "stored":
        return cloud
    if mode == "stored_scaled":
        if cloud.shape[0] > 0 and float(np.nanmax(cloud[:, 3:6])) <= 1.0:
            cloud[:, 3:6] *= 255.0
        return cloud
    if mode == "debug_placeholder":
        color = DEBUG_PLACEHOLDER_COLORS_RGB.get(str(placeholder), DEFAULT_DEBUG_PLACEHOLDER_COLOR_RGB)
        cloud[:, 3:6] = color[None, :]
        return cloud
    raise ValueError(f"Unsupported semantic input color mode: {color_mode}")


def semantic_forward_options(
    *,
    semantic_forward_mode: str = "reference",
    normal_mode: str | None = None,
    query_sample_mode: str | None = None,
) -> tuple[str, str]:
    mode = str(semantic_forward_mode)
    if mode not in SEMANTIC_FORWARD_MODES:
        raise ValueError(f"Unsupported semantic_forward_mode: {semantic_forward_mode}")
    default_normal_mode, default_query_sample_mode = (
        ("fallback", "random") if mode == "reference" else ("estimated", "fps")
    )
    return (
        default_normal_mode if normal_mode is None else str(normal_mode),
        default_query_sample_mode if query_sample_mode is None else str(query_sample_mode),
    )


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


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


def _normalize_support_and_query(
    support_points: np.ndarray,
    query_points: np.ndarray,
    *,
    coord_scale: float,
    center_shift_z: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(support_points, axis=0, keepdims=True)
    support = support_points - centroid
    query = query_points - centroid

    radius = np.linalg.norm(support, axis=1).max()
    radius = max(float(radius), 1e-6)
    support = support / radius
    query = query / radius

    if center_shift_z:
        x_min, y_min, z_min = support.min(axis=0)
        x_max, y_max, _ = support.max(axis=0)
        shift = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
        support = support - shift
        query = query - shift

    return (support * float(coord_scale)).astype(np.float32), (query * float(coord_scale)).astype(np.float32)


def _estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return np.zeros_like(points, dtype=np.float32)

    k = max(3, min(int(k), len(points) - 1))
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    neighbor_idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]

    normals = np.zeros_like(points, dtype=np.float32)
    for idx in range(len(points)):
        neighbors = points[neighbor_idx[idx]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(neighbors), 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        normal = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        normals[idx] = normal
    return normals.astype(np.float32)


def _fallback_normals(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float32) - np.asarray(points, dtype=np.float32).mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normals = np.zeros_like(centered, dtype=np.float32)
    valid = norms[:, 0] > 1e-8
    normals[valid] = centered[valid] / norms[valid]
    return normals.astype(np.float32)


def _sample_rows(point_cloud: np.ndarray, target_num_points: int) -> np.ndarray:
    point_cloud = strip_zero_points(point_cloud)
    if int(target_num_points) <= 0:
        return point_cloud.astype(np.float32)
    if len(point_cloud) == 0:
        return np.zeros((int(target_num_points), 6), dtype=np.float32)
    replace = len(point_cloud) < int(target_num_points)
    indices = np.random.choice(len(point_cloud), size=int(target_num_points), replace=replace)
    return point_cloud[indices].astype(np.float32)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def load_semantic_model(
    checkpoint: str,
    *,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_state = _load_checkpoint(checkpoint_path)
    checkpoint_args = dict(checkpoint_state.get("args", {}))
    canonical_label_names = list(checkpoint_state.get("canonical_label_names", []))
    if not canonical_label_names:
        labels_path = checkpoint_path.parent / "canonical_labels.json"
        canonical_label_names = json.loads(labels_path.read_text(encoding="utf-8"))

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
    if "semantic_local_fusion" in projector_state:
        model.semantic_local_fusion.load_state_dict(projector_state["semantic_local_fusion"])
    model.semantic_decoder.load_state_dict(projector_state["semantic_decoder"])
    if "encoder_state_dict" in checkpoint_state:
        model.feature_extractor.encoder.load_state_dict(checkpoint_state["encoder_state_dict"])
    model = model.to(device)
    model.eval()

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_args": checkpoint_args,
        "canonical_label_names": canonical_label_names,
        "model": model,
        "device": device,
        "utonia_transform": _build_utonia_transform(float(checkpoint_args.get("grid_size", 0.01))),
        "sem_embedding_dim": int(checkpoint_args.get("sem_embedding_dim", 128)),
    }


@torch.no_grad()
def _compute_semantic_query(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    *,
    target_num_points: int,
    normal_mode: str = "fallback",
    query_sample_mode: str = "random",
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    point_cloud = strip_zero_points(np.asarray(object_point_cloud, dtype=np.float32))
    if len(point_cloud) == 0:
        raise ValueError("Cannot query semantic field from an empty object point cloud.")

    support_pc = point_cloud.astype(np.float32)
    if str(query_sample_mode) == "random":
        query_pc = _sample_rows(point_cloud, int(target_num_points))
    elif str(query_sample_mode) == "fps":
        query_pc = resample_point_cloud(point_cloud, int(target_num_points))
    else:
        raise ValueError(f"Unsupported semantic query_sample_mode: {query_sample_mode}")

    support_xyz = support_pc[:, :3].astype(np.float32)
    support_rgb = support_pc[:, 3:6].astype(np.float32) if support_pc.shape[1] >= 6 else np.zeros((len(support_pc), 3), dtype=np.float32)
    if str(normal_mode) == "fallback":
        support_normals = _fallback_normals(support_xyz)
    elif str(normal_mode) == "estimated":
        support_normals = _estimate_normals(support_xyz)
    else:
        raise ValueError(f"Unsupported semantic normal_mode: {normal_mode}")
    query_world_xyz = query_pc[:, :3].astype(np.float32)

    checkpoint_args = artifacts["checkpoint_args"]
    support_normed, query_normed = _normalize_support_and_query(
        support_xyz,
        query_world_xyz,
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
    )

    utonia_input = artifacts["utonia_transform"](
        {
            "coord": support_normed.copy(),
            "color": support_rgb.copy(),
            "normal": support_normals.copy(),
        }
    )
    utonia_input = UTONIA.data.collate_fn([utonia_input])
    device = artifacts["device"]
    utonia_input = _move_to_device(utonia_input, device)

    support_t = torch.from_numpy(support_normed[None, ...]).float().to(device)
    query_t = torch.from_numpy(query_normed[None, ...]).float().to(device)
    model = artifacts["model"]
    cache = model.encode_support(
        utonia_input=utonia_input,
        support_points=support_t,
    )
    semantic_output = model.query_semantic(
        cache=cache,
        query_points=query_t,
    )
    return query_world_xyz.astype(np.float32), semantic_output


@torch.no_grad()
def compute_semantic_pointwise_cloud(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    *,
    target_num_points: int,
    placeholder: str = "",
    semantic_input_color_mode: str = "debug_placeholder",
    semantic_forward_mode: str = "reference",
    normal_mode: str | None = None,
    query_sample_mode: str | None = None,
) -> np.ndarray:
    point_cloud = prepare_semantic_input_point_cloud(
        object_point_cloud,
        placeholder=str(placeholder),
        color_mode=str(semantic_input_color_mode),
    )
    point_cloud = strip_zero_points(point_cloud)
    sem_dim = int(artifacts["sem_embedding_dim"])
    if len(point_cloud) == 0:
        return np.zeros((int(target_num_points), 3 + sem_dim), dtype=np.float32)
    resolved_normal_mode, resolved_query_sample_mode = semantic_forward_options(
        semantic_forward_mode=str(semantic_forward_mode),
        normal_mode=normal_mode,
        query_sample_mode=query_sample_mode,
    )

    query_world_xyz, semantic_output = _compute_semantic_query(
        artifacts,
        point_cloud,
        target_num_points=int(target_num_points),
        normal_mode=resolved_normal_mode,
        query_sample_mode=resolved_query_sample_mode,
    )
    embeddings = semantic_output["embedding"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.concatenate([query_world_xyz, embeddings], axis=1)


@torch.no_grad()
def compute_semantic_pointwise_prediction(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    *,
    target_num_points: int,
    placeholder: str = "",
    semantic_input_color_mode: str = "debug_placeholder",
    semantic_forward_mode: str = "reference",
    normal_mode: str | None = None,
    query_sample_mode: str | None = None,
) -> dict[str, np.ndarray | list[str]]:
    point_cloud = prepare_semantic_input_point_cloud(
        object_point_cloud,
        placeholder=str(placeholder),
        color_mode=str(semantic_input_color_mode),
    )
    point_cloud = strip_zero_points(point_cloud)
    sem_dim = int(artifacts["sem_embedding_dim"])
    if len(point_cloud) == 0:
        return {
            "point_cloud": np.zeros((int(target_num_points), 3 + sem_dim), dtype=np.float32),
            "pred_labels": np.full((int(target_num_points),), -1, dtype=np.int64),
            "confidence": np.zeros((int(target_num_points),), dtype=np.float32),
            "label_names": list(artifacts.get("canonical_label_names", [])),
        }
    resolved_normal_mode, resolved_query_sample_mode = semantic_forward_options(
        semantic_forward_mode=str(semantic_forward_mode),
        normal_mode=normal_mode,
        query_sample_mode=query_sample_mode,
    )

    query_world_xyz, semantic_output = _compute_semantic_query(
        artifacts,
        point_cloud,
        target_num_points=int(target_num_points),
        normal_mode=resolved_normal_mode,
        query_sample_mode=resolved_query_sample_mode,
    )
    embeddings = semantic_output["embedding"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    logits = semantic_output.get("logits")
    if logits is None:
        raise RuntimeError("Semantic model output does not include logits; predicted-label visualization is unavailable.")
    probabilities = torch.softmax(logits.squeeze(0), dim=-1)
    confidence, pred_labels = torch.max(probabilities, dim=-1)
    return {
        "point_cloud": np.concatenate([query_world_xyz, embeddings], axis=1),
        "pred_labels": pred_labels.detach().cpu().numpy().astype(np.int64),
        "confidence": confidence.detach().cpu().numpy().astype(np.float32),
        "label_names": list(artifacts.get("canonical_label_names", [])),
    }
