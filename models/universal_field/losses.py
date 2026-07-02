from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_LABEL = -1


def _flatten_valid_points(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != ignore_index
    if tensor.ndim == 3:
        flat_tensor = tensor.reshape(-1, tensor.shape[-1])
        return flat_tensor[valid_mask], flat_labels[valid_mask]
    return tensor.reshape(-1)[valid_mask], flat_labels[valid_mask]


def semantic_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid_logits, valid_labels = _flatten_valid_points(logits, labels, ignore_index)
    if valid_labels.numel() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(
        valid_logits,
        valid_labels,
        label_smoothing=label_smoothing,
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    max_points: int = 2048,
    ignore_index: int = IGNORE_LABEL,
) -> torch.Tensor:
    valid_embeddings, valid_labels = _flatten_valid_points(embeddings, labels, ignore_index)
    if valid_labels.numel() <= 1:
        return embeddings.new_zeros(())

    if valid_labels.numel() > max_points:
        selection = torch.randperm(valid_labels.numel(), device=valid_labels.device)[:max_points]
        valid_embeddings = valid_embeddings[selection]
        valid_labels = valid_labels[selection]

    similarity = valid_embeddings @ valid_embeddings.T / temperature
    logits_mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_similarity = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)

    positive_mask = (valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)) & logits_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchor_mask = positive_count > 0
    if not valid_anchor_mask.any():
        return embeddings.new_zeros(())

    mean_log_prob_pos = (
        (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_log_prob_pos[valid_anchor_mask].mean()


def embedding_consistency_loss(
    embeddings_view1: torch.Tensor,
    embeddings_view2: torch.Tensor,
) -> torch.Tensor:
    if embeddings_view1.numel() == 0 or embeddings_view2.numel() == 0:
        return embeddings_view1.new_zeros(())
    return (1.0 - (embeddings_view1 * embeddings_view2).sum(dim=-1)).mean()


def _directional_dense_correspondence_loss(
    anchor_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    anchor_points: torch.Tensor,
    candidate_points: torch.Tensor,
    positive_targets: torch.Tensor,
    temperature: float,
    spatial_tolerance_radius: float,
) -> torch.Tensor:
    logits = anchor_embeddings @ candidate_embeddings.T / max(temperature, 1e-6)

    if spatial_tolerance_radius > 0:
        spatial_distances = torch.cdist(anchor_points, candidate_points, p=2)
        tolerance_mask = spatial_distances <= spatial_tolerance_radius
        positive_mask = torch.zeros_like(tolerance_mask)
        positive_mask[
            torch.arange(positive_targets.shape[0], device=positive_targets.device),
            positive_targets,
        ] = True
        logits = logits.masked_fill(tolerance_mask & ~positive_mask, -1e4)

    return F.cross_entropy(logits, positive_targets)


def dense_correspondence_contrastive_loss(
    embeddings_view1: torch.Tensor,
    embeddings_view2: torch.Tensor,
    canonical_points: torch.Tensor,
    labels: torch.Tensor,
    correspondences: torch.Tensor | None = None,
    temperature: float = 0.1,
    max_anchors: int = 1024,
    spatial_tolerance_radius: float = 0.03,
    ignore_index: int = IGNORE_LABEL,
) -> torch.Tensor:
    batch_losses: list[torch.Tensor] = []

    for batch_index in range(labels.shape[0]):
        valid_mask = labels[batch_index] != ignore_index
        num_valid = int(valid_mask.sum().item())
        if num_valid <= 1:
            continue

        valid_embeddings_view1 = embeddings_view1[batch_index][valid_mask]
        valid_embeddings_view2 = embeddings_view2[batch_index][valid_mask]
        valid_points = canonical_points[batch_index][valid_mask]

        if correspondences is None:
            positive_targets = torch.arange(num_valid, device=labels.device)
        else:
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            candidate_lookup = torch.full(
                (labels.shape[1],),
                -1,
                dtype=torch.long,
                device=labels.device,
            )
            candidate_lookup[valid_indices] = torch.arange(num_valid, device=labels.device)
            positive_targets = candidate_lookup[correspondences[batch_index][valid_mask]]
            keep_mask = positive_targets >= 0
            if int(keep_mask.sum().item()) <= 1:
                continue
            valid_embeddings_view1 = valid_embeddings_view1[keep_mask]
            valid_embeddings_view2 = valid_embeddings_view2[keep_mask]
            valid_points = valid_points[keep_mask]
            positive_targets = positive_targets[keep_mask]

        if valid_embeddings_view1.shape[0] > max_anchors:
            selection = torch.randperm(valid_embeddings_view1.shape[0], device=labels.device)[
                :max_anchors
            ]
            valid_embeddings_view1 = valid_embeddings_view1[selection]
            valid_embeddings_view2 = valid_embeddings_view2[selection]
            valid_points = valid_points[selection]
            positive_targets = positive_targets[selection]

        forward_loss = _directional_dense_correspondence_loss(
            anchor_embeddings=valid_embeddings_view1,
            candidate_embeddings=embeddings_view2[batch_index][valid_mask],
            anchor_points=valid_points,
            candidate_points=canonical_points[batch_index][valid_mask],
            positive_targets=positive_targets,
            temperature=temperature,
            spatial_tolerance_radius=spatial_tolerance_radius,
        )
        backward_loss = _directional_dense_correspondence_loss(
            anchor_embeddings=valid_embeddings_view2,
            candidate_embeddings=embeddings_view1[batch_index][valid_mask],
            anchor_points=valid_points,
            candidate_points=canonical_points[batch_index][valid_mask],
            positive_targets=positive_targets,
            temperature=temperature,
            spatial_tolerance_radius=spatial_tolerance_radius,
        )
        batch_losses.append(0.5 * (forward_loss + backward_loss))

    if not batch_losses:
        return embeddings_view1.new_zeros(())
    return torch.stack(batch_losses).mean()


def geometric_metric_loss(
    embeddings: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
    k: int = 16,
    same_part_scale: float = 0.15,
    negative_margin: float = 0.2,
    ignore_index: int = IGNORE_LABEL,
) -> torch.Tensor:
    batch_losses: list[torch.Tensor] = []
    for batch_index in range(labels.shape[0]):
        valid_mask = labels[batch_index] != ignore_index
        if valid_mask.sum().item() <= 1:
            continue

        cur_points = points[batch_index][valid_mask]
        cur_embeddings = embeddings[batch_index][valid_mask]
        cur_labels = labels[batch_index][valid_mask]

        effective_k = min(k + 1, cur_points.shape[0])
        if effective_k <= 1:
            continue

        distances = torch.cdist(cur_points, cur_points, p=2)
        knn_indices = torch.topk(
            distances,
            k=effective_k,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices[:, 1:]

        anchor_embeddings = cur_embeddings.unsqueeze(1)
        neighbor_embeddings = cur_embeddings[knn_indices]
        cosine_similarity = (anchor_embeddings * neighbor_embeddings).sum(dim=-1)
        pair_distances = distances.gather(dim=1, index=knn_indices)

        anchor_labels = cur_labels.unsqueeze(1)
        neighbor_labels = cur_labels[knn_indices]
        same_mask = anchor_labels == neighbor_labels
        diff_mask = ~same_mask

        same_targets = torch.clamp(pair_distances / max(same_part_scale, 1e-6), 0.0, 1.0)
        same_loss = (
            F.mse_loss(1.0 - cosine_similarity[same_mask], same_targets[same_mask])
            if same_mask.any()
            else embeddings.new_zeros(())
        )
        diff_loss = (
            F.relu(cosine_similarity[diff_mask] - negative_margin).mean()
            if diff_mask.any()
            else embeddings.new_zeros(())
        )
        batch_losses.append(same_loss + diff_loss)

    if not batch_losses:
        return embeddings.new_zeros(())
    return torch.stack(batch_losses).mean()


def occupancy_bce_loss(
    occ_logits: torch.Tensor,
    occ_labels: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(occ_logits, occ_labels)


def compute_universal_field_metrics(
    sem_logits: torch.Tensor | None,
    sem_labels: torch.Tensor | None,
    occ_logits: torch.Tensor | None,
    occ_labels: torch.Tensor | None,
    ignore_index: int = IGNORE_LABEL,
) -> dict[str, float]:
    if sem_logits is None or sem_labels is None:
        sem_acc = 0.0
        valid_ratio = 0.0
    else:
        flat_logits, flat_labels = _flatten_valid_points(sem_logits, sem_labels, ignore_index)
        if flat_labels.numel() == 0:
            sem_acc = 0.0
            valid_ratio = 0.0
        else:
            sem_predictions = flat_logits.argmax(dim=-1)
            sem_acc = float((sem_predictions == flat_labels).float().mean().item())
            valid_ratio = float((sem_labels != ignore_index).float().mean().item())

    if occ_logits is None or occ_labels is None:
        occ_acc = 0.0
    else:
        occ_predictions = (torch.sigmoid(occ_logits) >= 0.5).float()
        occ_acc = float((occ_predictions == occ_labels).float().mean().item())
    return {
        "sem_acc": sem_acc,
        "occ_acc": occ_acc,
        "valid_ratio": valid_ratio,
    }
