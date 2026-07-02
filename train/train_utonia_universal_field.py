from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import wandb

from my_datasets import (
    PartNextUniversalFieldDataset,
    build_partnext_canonical_label_space,
    list_partnext_categories,
    partnext_universal_field_collate_fn,
)
from models import (
    UtoniaUniversalFieldNet,
    compute_universal_field_metrics,
    dense_correspondence_contrastive_loss,
    embedding_consistency_loss,
    geometric_metric_loss,
    occupancy_bce_loss,
    semantic_cross_entropy_loss,
    universal_supervised_contrastive_loss,
)
from myutils import MetricTracker, move_to_device, set_seed


def _parse_category_args(categories: list[str] | None) -> list[str] | None:
    if not categories:
        return None
    parsed: list[str] = []
    for category in categories:
        parsed.extend(part.strip() for part in category.split(",") if part.strip())
    return parsed or None


def _to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _namespace_from_checkpoint_args(
    defaults: argparse.Namespace,
    checkpoint_args: dict | None,
) -> argparse.Namespace:
    merged = vars(defaults).copy()
    if not checkpoint_args:
        return argparse.Namespace(**merged)

    path_keys = {"dataset_root", "output_dir", "alias_config"}
    for key, value in checkpoint_args.items():
        if key not in merged:
            continue
        if key in path_keys and value is not None:
            merged[key] = Path(value)
        else:
            merged[key] = value
    return argparse.Namespace(**merged)


def _merge_resume_args(
    defaults: argparse.Namespace,
    cli_args: argparse.Namespace,
    checkpoint_args: dict | None,
) -> argparse.Namespace:
    merged = vars(_namespace_from_checkpoint_args(defaults, checkpoint_args)).copy()
    default_values = vars(defaults)
    cli_values = vars(cli_args)

    for key, value in cli_values.items():
        if key == "resume":
            merged[key] = value
            continue
        if key not in default_values:
            continue
        if value != default_values[key]:
            merged[key] = value
    return argparse.Namespace(**merged)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a universal continuous semantic-geometric field on top of Utonia features.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/PartNext_mesh"))
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["all"],
        help="PartNext categories to use. Example: --categories Hammer Spoon Mug",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--num-support-points", type=int, default=5000)
    parser.add_argument("--num-query-surface-points", type=int, default=2048)
    parser.add_argument("--num-query-occ-points", type=int, default=1536)
    parser.add_argument("--balanced-sampling-ratio", type=float, default=0.0)
    parser.add_argument("--query-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--near-surface-ratio", type=float, default=0.75)
    parser.add_argument("--near-surface-noise", type=float, default=0.02)
    parser.add_argument("--uniform-padding", type=float, default=0.15)
    parser.add_argument("--coord-scale", type=float, default=1.0)
    parser.add_argument("--disable-z-shift", action="store_true")
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument(
        "--label-level",
        type=str,
        choices=["leaf", "parent", "path", "config"],
        default="leaf",
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=None,
        help="Optional JSON file that aliases category-specific part names into shared canonical names.",
    )
    parser.add_argument(
        "--ignore-labels",
        nargs="*",
        default=["Other"],
        help="Raw part labels to ignore before canonical remapping.",
    )
    parser.add_argument(
        "--rotation-mode",
        type=str,
        choices=["none", "z", "so3"],
        default="so3",
    )
    parser.add_argument("--jitter-std", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["semantic", "pose", "joint"],
        default="joint",
        help="Train semantic branch, pose branch, or the legacy joint setup.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--branch-dim", type=int, default=256)
    parser.add_argument("--sem-embedding-dim", type=int, default=128)
    parser.add_argument("--geo-embedding-dim", type=int, default=128)
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=256,
        help="Deprecated legacy RBF argument kept only for CLI/checkpoint compatibility.",
    )
    parser.add_argument(
        "--rbf-sigma",
        type=float,
        default=0.12,
        help="Deprecated legacy RBF argument kept only for CLI/checkpoint compatibility.",
    )
    parser.add_argument("--triplane-resolution", type=int, default=64)
    parser.add_argument("--triplane-padding", type=float, default=0.05)
    parser.add_argument("--sem-ce-weight", type=float, default=1.0)
    parser.add_argument("--sem-contrastive-weight", type=float, default=0.2)
    parser.add_argument("--sem-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--sem-contrastive-max-points", type=int, default=2048)
    parser.add_argument("--sem-label-smoothing", type=float, default=0.0)
    parser.add_argument("--geo-dense-correspondence-weight", type=float, default=1.0)
    parser.add_argument("--geo-dense-correspondence-temperature", type=float, default=0.1)
    parser.add_argument("--geo-dense-correspondence-max-points", type=int, default=1024)
    parser.add_argument("--geo-spatial-tolerance-radius", type=float, default=0.03)
    parser.add_argument("--geo-metric-weight", type=float, default=0.2)
    parser.add_argument("--geo-k", type=int, default=16)
    parser.add_argument("--geo-same-part-scale", type=float, default=0.15)
    parser.add_argument("--geo-negative-margin", type=float, default=0.2)
    parser.add_argument("--geo-consistency-weight", type=float, default=0.2)
    parser.add_argument("--sem-consistency-weight", type=float, default=0.1)
    parser.add_argument("--occ-weight", type=float, default=0.1)
    parser.add_argument("--best-sem-occ-weight", type=float, default=0.1)
    parser.add_argument("--best-geo-metric-weight", type=float, default=0.1)
    parser.add_argument("--best-joint-geo-dense-weight", type=float, default=0.5)
    parser.add_argument("--best-joint-geo-metric-weight", type=float, default=0.1)
    parser.add_argument("--best-joint-geo-weight", type=float, default=0.5)
    parser.add_argument("--utonia-checkpoint", type=str, default="auto")
    parser.add_argument("--utonia-repo-id", type=str, default="Pointcept/Utonia")
    parser.add_argument("--utonia-upcast-levels", type=int, default=0)
    parser.add_argument("--finetune-utonia", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_field"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="semantic-field")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved checkpoint such as last.pt or best.pt",
    )
    parser.add_argument(
        "--resume-additional-epochs",
        type=int,
        default=0,
        help="When resuming, extend training by N additional epochs beyond the checkpoint epoch",
    )
    return parser


def build_dataset(
    args: argparse.Namespace,
    split: str,
    canonical_label_names: list[str],
) -> PartNextUniversalFieldDataset:
    if args.train_mode == "semantic":
        second_view = args.sem_consistency_weight > 0
    elif args.train_mode == "pose":
        second_view = (
            args.geo_dense_correspondence_weight > 0
            or args.geo_consistency_weight > 0
        )
    else:
        second_view = (
            args.geo_dense_correspondence_weight > 0
            or args.geo_consistency_weight > 0
            or args.sem_consistency_weight > 0
        )
    return PartNextUniversalFieldDataset(
        dataset_root=args.dataset_root,
        split=split,
        categories=_parse_category_args(args.categories),
        num_support_points=args.num_support_points,
        num_query_surface_points=args.num_query_surface_points,
        num_query_occ_points=args.num_query_occ_points,
        balanced_sampling_ratio=args.balanced_sampling_ratio,
        query_sampling_ratio=args.query_sampling_ratio,
        near_surface_ratio=args.near_surface_ratio,
        near_surface_noise=args.near_surface_noise,
        uniform_padding=args.uniform_padding,
        coord_scale=args.coord_scale,
        center_shift_z=not args.disable_z_shift,
        grid_size=args.grid_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        limit_samples=(
            args.limit_train_samples if split == "train" else args.limit_val_samples
        ),
        label_level=args.label_level,
        alias_config_path=args.alias_config,
        canonical_label_names=canonical_label_names,
        ignore_labels=args.ignore_labels,
        rotation_mode=args.rotation_mode,
        jitter_std=args.jitter_std,
        second_view=second_view,
    )


def build_dataloader(
    dataset: PartNextUniversalFieldDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=partnext_universal_field_collate_fn,
    )


def _configure_trainable_parameters(
    model: UtoniaUniversalFieldNet,
    args: argparse.Namespace,
) -> list[torch.nn.Parameter]:
    if args.train_mode == "semantic":
        modules = [
            model.semantic_adapter,
            model.semantic_local_fusion,
            model.semantic_decoder,
        ]
    elif args.train_mode == "pose":
        modules = [
            model.pose_adapter,
            model.pose_local_fusion,
            model.pose_decoder,
            model.occupancy_local_fusion,
            model.occupancy_decoder,
        ]
    else:
        modules = [model]

    selected_parameters = set()
    if args.train_mode == "joint":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
            selected_parameters.add(id(parameter))

    if args.finetune_utonia:
        for parameter in model.feature_extractor.encoder.parameters():
            parameter.requires_grad = True
            selected_parameters.add(id(parameter))

    return [parameter for parameter in model.parameters() if id(parameter) in selected_parameters]


def build_optimizer(model: UtoniaUniversalFieldNet, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameters = _configure_trainable_parameters(model, args)
    return torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def save_checkpoint(
    path: Path,
    *,
    model: UtoniaUniversalFieldNet,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_sem_score: float,
    best_pose_score: float,
    best_joint_score: float,
    args: argparse.Namespace,
    canonical_label_names: list[str],
) -> None:
    state = {
        "projector_state_dict": {
            "semantic_adapter": model.semantic_adapter.state_dict(),
            "pose_adapter": model.pose_adapter.state_dict(),
            "geometric_adapter": model.pose_adapter.state_dict(),
            "semantic_local_fusion": model.semantic_local_fusion.state_dict(),
            "pose_local_fusion": model.pose_local_fusion.state_dict(),
            "geometric_local_fusion": model.pose_local_fusion.state_dict(),
            "occupancy_local_fusion": model.occupancy_local_fusion.state_dict(),
            "semantic_decoder": model.semantic_decoder.state_dict(),
            "pose_decoder": model.pose_decoder.state_dict(),
            "geometric_decoder": model.pose_decoder.state_dict(),
            "occupancy_decoder": model.occupancy_decoder.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_score": best_sem_score,
        "best_sem_score": best_sem_score,
        "best_pose_score": best_pose_score,
        "best_geo_score": best_pose_score,
        "best_joint_score": best_joint_score,
        "args": _to_serializable(vars(args)),
        "canonical_label_names": canonical_label_names,
    }
    if not model.feature_extractor.freeze_encoder:
        state["encoder_state_dict"] = model.feature_extractor.encoder.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def restore_checkpoint(
    checkpoint: dict,
    *,
    model: UtoniaUniversalFieldNet,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
) -> tuple[int, int, dict[str, float]]:
    projector_state = checkpoint.get("projector_state_dict", {})
    model.semantic_adapter.load_state_dict(projector_state["semantic_adapter"])
    pose_adapter_state = projector_state.get("pose_adapter", projector_state["geometric_adapter"])
    model.pose_adapter.load_state_dict(pose_adapter_state)
    if "semantic_local_fusion" in projector_state:
        model.semantic_local_fusion.load_state_dict(projector_state["semantic_local_fusion"])
    pose_local_fusion_state = projector_state.get(
        "pose_local_fusion",
        projector_state.get("geometric_local_fusion"),
    )
    if pose_local_fusion_state is not None:
        model.pose_local_fusion.load_state_dict(pose_local_fusion_state)
    if "occupancy_local_fusion" in projector_state:
        model.occupancy_local_fusion.load_state_dict(projector_state["occupancy_local_fusion"])
    model.semantic_decoder.load_state_dict(projector_state["semantic_decoder"])
    pose_decoder_state = projector_state.get("pose_decoder", projector_state["geometric_decoder"])
    model.pose_decoder.load_state_dict(pose_decoder_state)
    if "occupancy_decoder" in projector_state:
        model.occupancy_decoder.load_state_dict(projector_state["occupancy_decoder"])
    if "encoder_state_dict" in checkpoint:
        model.feature_extractor.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    best_scores = {
        "sem": float(checkpoint.get("best_sem_score", checkpoint.get("best_val_score", float("-inf")))),
        "pose": float(checkpoint.get("best_pose_score", checkpoint.get("best_geo_score", float("-inf")))),
        "joint": float(checkpoint.get("best_joint_score", float("-inf"))),
    }
    return epoch, global_step, best_scores


def _compute_loss_terms(
    outputs: dict[str, dict[str, torch.Tensor]],
    batch: dict,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    labels = batch["query_surface_labels"]
    canonical_surface_points = batch["query_surface_points_canonical"]
    surface_correspondence = batch["query_surface_correspondence"]
    mode = args.train_mode
    base_tensor = batch["query_surface_points_view1"]
    zero = base_tensor.new_zeros(())

    sem_ce = zero
    sem_contrastive = zero
    sem_consistency = zero
    pose_dense_correspondence = zero
    pose_metric = zero
    pose_consistency = zero
    occ = zero

    if mode in {"semantic", "joint"}:
        sem_logits_view1 = outputs["view1"]["sem_logits"]
        sem_embeddings_view1 = outputs["view1"]["sem_embeddings"]
        sem_ce_losses = [
            semantic_cross_entropy_loss(
                sem_logits_view1,
                labels,
                label_smoothing=args.sem_label_smoothing,
            )
        ]
        sem_contrastive_losses = [
            universal_supervised_contrastive_loss(
                sem_embeddings_view1,
                labels,
                temperature=args.sem_contrastive_temperature,
                max_points=args.sem_contrastive_max_points,
            )
        ]
        if "view2" in outputs and "sem_embeddings" in outputs["view2"]:
            sem_logits_view2 = outputs["view2"]["sem_logits"]
            sem_embeddings_view2 = outputs["view2"]["sem_embeddings"]
            sem_ce_losses.append(
                semantic_cross_entropy_loss(
                    sem_logits_view2,
                    labels,
                    label_smoothing=args.sem_label_smoothing,
                )
            )
            sem_contrastive_losses.append(
                universal_supervised_contrastive_loss(
                    sem_embeddings_view2,
                    labels,
                    temperature=args.sem_contrastive_temperature,
                    max_points=args.sem_contrastive_max_points,
                )
            )
            sem_consistency = embedding_consistency_loss(
                sem_embeddings_view1,
                sem_embeddings_view2,
            )
        sem_ce = torch.stack(sem_ce_losses).mean()
        sem_contrastive = torch.stack(sem_contrastive_losses).mean()

    if mode in {"pose", "joint"}:
        pose_embeddings_view1 = outputs["view1"].get("pose_embeddings", outputs["view1"]["geo_embeddings"])
        occ_logits_view1 = outputs["view1"]["occ_logits"]
        occ_labels = batch["query_occ_labels"]
        pose_metric_losses = [
            geometric_metric_loss(
                pose_embeddings_view1,
                batch["query_surface_points_view1"],
                labels,
                k=args.geo_k,
                same_part_scale=args.geo_same_part_scale,
                negative_margin=args.geo_negative_margin,
            )
        ]
        occ_losses = [
            occupancy_bce_loss(
                occ_logits_view1,
                occ_labels,
            )
        ]
        if "view2" in outputs and "pose_embeddings" in outputs["view2"]:
            pose_embeddings_view2 = outputs["view2"].get("pose_embeddings", outputs["view2"]["geo_embeddings"])
            occ_logits_view2 = outputs["view2"]["occ_logits"]
            pose_metric_losses.append(
                geometric_metric_loss(
                    pose_embeddings_view2,
                    batch["query_surface_points_view2"],
                    labels,
                    k=args.geo_k,
                    same_part_scale=args.geo_same_part_scale,
                    negative_margin=args.geo_negative_margin,
                )
            )
            occ_losses.append(
                occupancy_bce_loss(
                    occ_logits_view2,
                    occ_labels,
                )
            )
            pose_consistency = embedding_consistency_loss(
                pose_embeddings_view1,
                pose_embeddings_view2,
            )
            pose_dense_correspondence = dense_correspondence_contrastive_loss(
                pose_embeddings_view1,
                pose_embeddings_view2,
                canonical_points=canonical_surface_points,
                labels=labels,
                correspondences=surface_correspondence,
                temperature=args.geo_dense_correspondence_temperature,
                max_anchors=args.geo_dense_correspondence_max_points,
                spatial_tolerance_radius=args.geo_spatial_tolerance_radius,
            )
        pose_metric = torch.stack(pose_metric_losses).mean()
        occ = torch.stack(occ_losses).mean()

    total = zero
    if mode in {"semantic", "joint"}:
        total = total + (
            args.sem_ce_weight * sem_ce
            + args.sem_contrastive_weight * sem_contrastive
            + args.sem_consistency_weight * sem_consistency
        )
    if mode in {"pose", "joint"}:
        total = total + (
            args.geo_dense_correspondence_weight * pose_dense_correspondence
            + args.geo_metric_weight * pose_metric
            + args.geo_consistency_weight * pose_consistency
            + args.occ_weight * occ
        )
    return {
        "loss": total,
        "sem_ce": sem_ce,
        "sem_contrastive": sem_contrastive,
        "pose_dense_correspondence": pose_dense_correspondence,
        "pose_metric": pose_metric,
        "pose_consistency": pose_consistency,
        "geo_dense_correspondence": pose_dense_correspondence,
        "geo_metric": pose_metric,
        "geo_consistency": pose_consistency,
        "sem_consistency": sem_consistency,
        "occ": occ,
    }


def run_epoch(
    *,
    model: UtoniaUniversalFieldNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    max_batches: int | None,
    use_amp: bool,
    wandb_run,
    global_step: int,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    tracker = MetricTracker()
    mode = "train" if is_train else "val"
    if is_train:
        model.train()
    else:
        model.eval()

    autocast_enabled = use_amp and device.type == "cuda"

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = move_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(batch, train_mode=args.train_mode)
                loss_terms = _compute_loss_terms(outputs, batch, args)
                loss = loss_terms["loss"]

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        metrics = compute_universal_field_metrics(
            sem_logits=(
                outputs["view1"]["sem_logits"].detach()
                if "sem_logits" in outputs["view1"]
                else None
            ),
            sem_labels=batch["query_surface_labels"] if "sem_logits" in outputs["view1"] else None,
            occ_logits=(
                outputs["view1"]["occ_logits"].detach()
                if "occ_logits" in outputs["view1"]
                else None
            ),
            occ_labels=batch["query_occ_labels"] if "occ_logits" in outputs["view1"] else None,
        )
        metrics["loss"] = float(loss_terms["loss"].item())
        metrics["sem_ce"] = float(loss_terms["sem_ce"].item())
        metrics["sem_contrastive"] = float(loss_terms["sem_contrastive"].item())
        metrics["pose_dense_correspondence"] = float(
            loss_terms["pose_dense_correspondence"].item()
        )
        metrics["pose_metric"] = float(loss_terms["pose_metric"].item())
        metrics["pose_consistency"] = float(loss_terms["pose_consistency"].item())
        metrics["geo_dense_correspondence"] = metrics["pose_dense_correspondence"]
        metrics["geo_metric"] = metrics["pose_metric"]
        metrics["geo_consistency"] = metrics["pose_consistency"]
        metrics["sem_consistency"] = float(loss_terms["sem_consistency"].item())
        metrics["occ"] = float(loss_terms["occ"].item())
        tracker.update(metrics, weight=batch["query_surface_labels"].shape[0])

        if is_train:
            global_step += 1
            if wandb_run is not None:
                wandb_run.log(
                    {f"{mode}/{key}": value for key, value in metrics.items()},
                    step=global_step,
                )

    epoch_metrics = tracker.averages()
    if args.train_mode == "semantic":
        print(
            f"[{mode}] epoch={epoch:03d} "
            f"loss={epoch_metrics['loss']:.4f} "
            f"sem_acc={epoch_metrics['sem_acc']:.4f} "
            f"sem_ce={epoch_metrics['sem_ce']:.4f} "
            f"sem_contrastive={epoch_metrics['sem_contrastive']:.4f} "
            f"sem_consistency={epoch_metrics['sem_consistency']:.4f}"
        )
    elif args.train_mode == "pose":
        print(
            f"[{mode}] epoch={epoch:03d} "
            f"loss={epoch_metrics['loss']:.4f} "
            f"occ_acc={epoch_metrics['occ_acc']:.4f} "
            f"pose_dense={epoch_metrics['pose_dense_correspondence']:.4f} "
            f"pose_metric={epoch_metrics['pose_metric']:.4f} "
            f"pose_consistency={epoch_metrics['pose_consistency']:.4f}"
        )
    else:
        print(
            f"[{mode}] epoch={epoch:03d} "
            f"loss={epoch_metrics['loss']:.4f} "
            f"sem_acc={epoch_metrics['sem_acc']:.4f} "
            f"occ_acc={epoch_metrics['occ_acc']:.4f} "
            f"sem_ce={epoch_metrics['sem_ce']:.4f} "
            f"pose_dense={epoch_metrics['pose_dense_correspondence']:.4f} "
            f"pose_metric={epoch_metrics['pose_metric']:.4f}"
        )
    return epoch_metrics, global_step


def main() -> None:
    parser = build_argparser()
    default_args = parser.parse_args([])
    cli_args = parser.parse_args()

    resume_state = None
    if cli_args.resume is not None:
        resume_state = _load_checkpoint(cli_args.resume)
        args = _merge_resume_args(default_args, cli_args, resume_state.get("args"))
    else:
        args = cli_args

    set_seed(args.seed)
    device = torch.device(args.device)

    categories = _parse_category_args(args.categories)
    if categories is None or any(category.lower() == "all" for category in categories):
        categories = list_partnext_categories(args.dataset_root)

    canonical_label_names = build_partnext_canonical_label_space(
        dataset_root=args.dataset_root,
        categories=categories,
        label_level=args.label_level,
        alias_config_path=args.alias_config,
        ignore_labels=args.ignore_labels,
    )

    if resume_state is not None and "canonical_label_names" in resume_state:
        saved_label_names = list(resume_state["canonical_label_names"])
        if saved_label_names != canonical_label_names:
            raise RuntimeError(
                "Canonical label names do not match the checkpoint. "
                "Keep categories/alias config consistent when resuming."
            )

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"utonia_field_{args.train_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = (
        args.resume.resolve().parent
        if args.resume is not None
        else (args.output_dir / run_name)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _to_serializable(
                {
                    **vars(args),
                    "categories": categories,
                    "canonical_label_names": canonical_label_names,
                }
            ),
            handle,
            indent=2,
        )

    with (run_dir / "canonical_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(canonical_label_names, handle, indent=2)

    print(f"Training categories ({len(categories)}): {categories}")
    print(f"Canonical labels ({len(canonical_label_names)}): {canonical_label_names}")

    train_dataset = build_dataset(args, split="train", canonical_label_names=canonical_label_names)
    val_dataset = build_dataset(args, split="val", canonical_label_names=canonical_label_names)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = UtoniaUniversalFieldNet(
        num_classes=len(canonical_label_names),
        utonia_checkpoint=args.utonia_checkpoint,
        utonia_repo_id=args.utonia_repo_id,
        utonia_upcast_levels=args.utonia_upcast_levels,
        freeze_utonia=not args.finetune_utonia,
        adapter_hidden_dim=args.adapter_hidden_dim,
        branch_dim=args.branch_dim,
        sem_embedding_dim=args.sem_embedding_dim,
        geo_embedding_dim=args.geo_embedding_dim,
        num_anchors=args.num_anchors,
        rbf_sigma=args.rbf_sigma,
        triplane_resolution=args.triplane_resolution,
        triplane_padding=args.triplane_padding,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )

    start_epoch = 1
    global_step = 0
    best_scores = {
        "sem": float("-inf"),
        "pose": float("-inf"),
        "joint": float("-inf"),
    }

    if resume_state is not None:
        start_epoch, global_step, best_scores = restore_checkpoint(
            resume_state,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        start_epoch += 1
        if args.resume_additional_epochs > 0:
            args.epochs = max(args.epochs, start_epoch - 1 + args.resume_additional_epochs)
        print(
            f"Resumed from {args.resume} at epoch={start_epoch - 1}, "
            f"global_step={global_step}, best_sem_score={best_scores['sem']:.4f}, "
            f"best_pose_score={best_scores['pose']:.4f}, best_joint_score={best_scores['joint']:.4f}"
        )

    wandb_run = None
    if args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            config=_to_serializable(
                {
                    **vars(args),
                    "categories": categories,
                    "canonical_label_names": canonical_label_names,
                }
            ),
            dir=str(run_dir),
            tags=args.wandb_tags,
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                args=args,
                max_batches=args.max_train_batches,
                use_amp=args.amp,
                wandb_run=wandb_run,
                global_step=global_step,
            )
            val_metrics, global_step = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                epoch=epoch,
                args=args,
                max_batches=args.max_val_batches,
                use_amp=args.amp,
                wandb_run=None,
                global_step=global_step,
            )

            if wandb_run is not None:
                sem_score = val_metrics["sem_acc"] + args.best_sem_occ_weight * val_metrics["occ_acc"]
                pose_score = -(
                    val_metrics["pose_dense_correspondence"]
                    + args.best_geo_metric_weight * val_metrics["pose_metric"]
                )
                if args.train_mode == "semantic":
                    joint_score = sem_score
                elif args.train_mode == "pose":
                    joint_score = pose_score
                else:
                    joint_score = (
                        sem_score
                        - args.best_joint_geo_dense_weight * val_metrics["pose_dense_correspondence"]
                        - args.best_joint_geo_metric_weight * val_metrics["pose_metric"]
                    )
                wandb_run.log(
                    {
                        **{f"train_epoch/{key}": value for key, value in train_metrics.items()},
                        **{f"val_epoch/{key}": value for key, value in val_metrics.items()},
                        "val_epoch/sem_score": sem_score,
                        "val_epoch/pose_score": pose_score,
                        "val_epoch/geo_score": pose_score,
                        "val_epoch/joint_score": joint_score,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            else:
                sem_score = val_metrics["sem_acc"] + args.best_sem_occ_weight * val_metrics["occ_acc"]
                pose_score = -(
                    val_metrics["pose_dense_correspondence"]
                    + args.best_geo_metric_weight * val_metrics["pose_metric"]
                )
                if args.train_mode == "semantic":
                    joint_score = sem_score
                elif args.train_mode == "pose":
                    joint_score = pose_score
                else:
                    joint_score = (
                        sem_score
                        - args.best_joint_geo_dense_weight * val_metrics["pose_dense_correspondence"]
                        - args.best_joint_geo_metric_weight * val_metrics["pose_metric"]
                    )
            save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_sem_score=max(best_scores["sem"], sem_score),
                best_pose_score=max(best_scores["pose"], pose_score),
                best_joint_score=max(best_scores["joint"], joint_score),
                args=args,
                canonical_label_names=canonical_label_names,
            )
            if args.train_mode in {"semantic", "joint"} and sem_score > best_scores["sem"]:
                best_scores["sem"] = sem_score
                save_checkpoint(
                    run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
                save_checkpoint(
                    run_dir / "best_sem.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
                print(f"[checkpoint] Saved new best semantic model at epoch={epoch} score={best_scores['sem']:.4f}")

            if args.train_mode in {"pose", "joint"} and pose_score > best_scores["pose"]:
                best_scores["pose"] = pose_score
                save_checkpoint(
                    run_dir / "best_pose.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
                save_checkpoint(
                    run_dir / "best_geo.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
                if args.train_mode == "pose":
                    save_checkpoint(
                        run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_sem_score=best_scores["sem"],
                        best_pose_score=best_scores["pose"],
                        best_joint_score=best_scores["joint"],
                        args=args,
                        canonical_label_names=canonical_label_names,
                    )
                print(f"[checkpoint] Saved new best pose model at epoch={epoch} score={best_scores['pose']:.4f}")

            if args.train_mode == "joint" and joint_score > best_scores["joint"]:
                best_scores["joint"] = joint_score
                save_checkpoint(
                    run_dir / "best_joint.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
                print(f"[checkpoint] Saved new best joint model at epoch={epoch} score={best_scores['joint']:.4f}")

            if epoch % args.save_every == 0:
                save_checkpoint(
                    run_dir / f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_sem_score=best_scores["sem"],
                    best_pose_score=best_scores["pose"],
                    best_joint_score=best_scores["joint"],
                    args=args,
                    canonical_label_names=canonical_label_names,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
