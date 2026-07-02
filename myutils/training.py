from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    if isinstance(data, Mapping):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(move_to_device(item, device) for item in data)
    return data


class MetricTracker:
    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._weight = 0.0

    def update(self, metrics: dict[str, float], weight: float = 1.0) -> None:
        self._weight += weight
        for key, value in metrics.items():
            self._totals[key] = self._totals.get(key, 0.0) + float(value) * weight

    def averages(self) -> dict[str, float]:
        if self._weight == 0:
            return {}
        return {key: value / self._weight for key, value in self._totals.items()}


def compute_occupancy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    preds = logits >= 0
    targets_bool = targets >= 0.5

    accuracy = (preds == targets_bool).float().mean()
    intersection = (preds & targets_bool).sum(dim=1).float()
    union = (preds | targets_bool).sum(dim=1).float()
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    positive_ratio = targets.float().mean()

    return {
        "acc": float(accuracy.item()),
        "iou": float(iou.mean().item()),
        "pos_ratio": float(positive_ratio.item()),
    }
