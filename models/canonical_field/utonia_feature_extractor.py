from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from myutils.imports import ensure_utonia_importable


class UtoniaFeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint: str = "auto",
        repo_id: str = "Pointcept/Utonia",
        upcast_levels: int = 0,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        if upcast_levels < 0:
            raise ValueError("upcast_levels must be >= 0")

        self.utonia = ensure_utonia_importable()
        self.repo_id = repo_id
        self.upcast_levels = upcast_levels
        self.freeze_encoder = freeze_encoder
        self.checkpoint = self._resolve_checkpoint(checkpoint)
        self.encoder = self._load_encoder()

        if self.freeze_encoder:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        if checkpoint != "auto":
            return checkpoint
        local_checkpoint = Path.home() / ".cache" / "utonia" / "ckpt" / "utonia.pth"
        if local_checkpoint.exists():
            return str(local_checkpoint)
        return "utonia"

    def _load_encoder(self) -> nn.Module:
        custom_config = None
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            custom_config = {
                "enable_flash": False,
                "enc_patch_size": [1024] * 5,
                "dec_patch_size": [1024] * 4,
            }

        return self.utonia.model.load(
            self.checkpoint,
            repo_id=self.repo_id,
            custom_config=custom_config,
        )

    def train(self, mode: bool = True) -> "UtoniaFeatureExtractor":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _restore_point_features(self, point) -> torch.Tensor:
        for _ in range(self.upcast_levels):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point.feat[point.inverse]

    def forward(self, utonia_input: dict, num_support_points: int) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        if self.freeze_encoder:
            self.encoder.eval()

        with context:
            point = self.encoder(utonia_input)
            point_features = self._restore_point_features(point)

        batch_size = int(utonia_input["offset"].shape[0])
        return point_features.reshape(batch_size, num_support_points, -1)
