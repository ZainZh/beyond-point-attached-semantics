from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.canonical_field.utonia_feature_extractor import UtoniaFeatureExtractor


def _normalize_points_to_bounds(
    points: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
) -> torch.Tensor:
    span = (bounds_max - bounds_min).clamp_min(1e-4)
    normalized = (points - bounds_min.unsqueeze(1)) / span.unsqueeze(1)
    return normalized.mul(2.0).sub(1.0).clamp_(-1.0, 1.0)


class PointAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        first_linear: nn.Module
        if input_dim is None:
            first_linear = nn.LazyLinear(hidden_dim)
        else:
            first_linear = nn.Linear(input_dim, hidden_dim)
        self.net = nn.Sequential(
            first_linear,
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TriPlaneProjector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        resolution: int = 64,
        padding: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.resolution = int(resolution)
        self.padding = float(padding)

    def _compute_bounds(self, support_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounds_min = support_points.min(dim=1).values
        bounds_max = support_points.max(dim=1).values
        center = 0.5 * (bounds_min + bounds_max)
        half_extent = 0.5 * (bounds_max - bounds_min).clamp_min(1e-3)
        half_extent = half_extent * (1.0 + self.padding)
        return center - half_extent, center + half_extent

    def _coords_to_grid(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resolution = float(self.resolution - 1)
        x = (coords[..., 0] + 1.0) * 0.5 * resolution
        y = (coords[..., 1] + 1.0) * 0.5 * resolution
        return x, y

    def _splat_plane(
        self,
        plane_coords: torch.Tensor,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, feature_dim = support_features.shape
        resolution = self.resolution
        plane = support_features.new_zeros(batch_size, feature_dim, resolution * resolution)
        plane_weight = support_features.new_zeros(batch_size, 1, resolution * resolution)

        grid_x, grid_y = self._coords_to_grid(plane_coords)
        x0 = torch.floor(grid_x).long().clamp_(0, resolution - 1)
        y0 = torch.floor(grid_y).long().clamp_(0, resolution - 1)
        x1 = (x0 + 1).clamp_(0, resolution - 1)
        y1 = (y0 + 1).clamp_(0, resolution - 1)

        dx = (grid_x - x0.float()).clamp_(0.0, 1.0)
        dy = (grid_y - y0.float()).clamp_(0.0, 1.0)

        corners = (
            (x0, y0, (1.0 - dx) * (1.0 - dy)),
            (x1, y0, dx * (1.0 - dy)),
            (x0, y1, (1.0 - dx) * dy),
            (x1, y1, dx * dy),
        )

        for batch_index in range(batch_size):
            feature_bank = plane[batch_index]
            weight_bank = plane_weight[batch_index]
            feature_slice = support_features[batch_index]
            for x_idx, y_idx, weight in corners:
                flat_index = (y_idx[batch_index] * resolution + x_idx[batch_index]).unsqueeze(0)
                weighted_features = (feature_slice * weight[batch_index].unsqueeze(-1)).transpose(0, 1)
                feature_bank.scatter_add_(1, flat_index.expand(feature_dim, -1), weighted_features)
                weight_bank.scatter_add_(1, flat_index, weight[batch_index].unsqueeze(0))

        plane = plane / plane_weight.clamp_min(1e-6)
        return plane.view(batch_size, feature_dim, resolution, resolution)

    def build(
        self,
        support_points: torch.Tensor,
        support_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bounds_min, bounds_max = self._compute_bounds(support_points)
        normalized_points = _normalize_points_to_bounds(
            support_points,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        plane_xy = self._splat_plane(normalized_points[..., [0, 1]], support_features)
        plane_xz = self._splat_plane(normalized_points[..., [0, 2]], support_features)
        plane_yz = self._splat_plane(normalized_points[..., [1, 2]], support_features)
        return {
            "plane_xy": plane_xy,
            "plane_xz": plane_xz,
            "plane_yz": plane_yz,
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "global_features": support_features.mean(dim=1),
        }

    def sample(
        self,
        cache: dict[str, torch.Tensor],
        query_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_queries = _normalize_points_to_bounds(
            query_points,
            bounds_min=cache["bounds_min"],
            bounds_max=cache["bounds_max"],
        )
        grid_xy = normalized_queries[..., [0, 1]].unsqueeze(2)
        grid_xz = normalized_queries[..., [0, 2]].unsqueeze(2)
        grid_yz = normalized_queries[..., [1, 2]].unsqueeze(2)

        xy_features = F.grid_sample(
            cache["plane_xy"],
            grid_xy,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        xz_features = F.grid_sample(
            cache["plane_xz"],
            grid_xz,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        yz_features = F.grid_sample(
            cache["plane_yz"],
            grid_yz,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        local_features = torch.cat([xy_features, xz_features, yz_features], dim=-1)
        global_features = cache["global_features"].unsqueeze(1).expand(-1, query_points.shape[1], -1)
        return local_features, global_features


class QueryBranchDecoder(nn.Module):
    def __init__(
        self,
        branch_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        out_logits: int | None = None,
    ) -> None:
        super().__init__()
        input_dim = branch_dim * 2 + 3
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.logit_head = nn.Linear(embedding_dim, out_logits) if out_logits is not None else None

    def forward(
        self,
        query_points: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([query_points, local_features, global_features], dim=-1))
        embedding = F.normalize(self.embedding_head(hidden), dim=-1)
        output = {
            "embedding": embedding,
            "hidden": hidden,
        }
        if self.logit_head is not None:
            output["logits"] = self.logit_head(embedding)
        return output


class OccupancyDecoder(nn.Module):
    def __init__(
        self,
        branch_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_dim = branch_dim * 2 + 3
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.occ_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_points: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([query_points, local_features, global_features], dim=-1))
        return {
            "hidden": hidden,
            "occ_logits": self.occ_head(hidden).squeeze(-1),
        }


class UtoniaUniversalFieldNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        utonia_checkpoint: str = "auto",
        utonia_repo_id: str = "Pointcept/Utonia",
        utonia_upcast_levels: int = 0,
        freeze_utonia: bool = True,
        adapter_input_dim: int | None = None,
        adapter_hidden_dim: int = 256,
        branch_dim: int = 256,
        sem_embedding_dim: int = 128,
        geo_embedding_dim: int = 128,
        num_anchors: int = 256,
        rbf_sigma: float = 0.12,
        triplane_resolution: int = 64,
        triplane_padding: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_extractor = UtoniaFeatureExtractor(
            checkpoint=utonia_checkpoint,
            repo_id=utonia_repo_id,
            upcast_levels=utonia_upcast_levels,
            freeze_encoder=freeze_utonia,
        )
        self.semantic_adapter = PointAdapter(
            input_dim=adapter_input_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=branch_dim,
        )
        self.pose_adapter = PointAdapter(
            input_dim=adapter_input_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=branch_dim,
        )
        self.semantic_projector = TriPlaneProjector(
            feature_dim=branch_dim,
            resolution=triplane_resolution,
            padding=triplane_padding,
        )
        self.pose_projector = TriPlaneProjector(
            feature_dim=branch_dim,
            resolution=triplane_resolution,
            padding=triplane_padding,
        )
        self.semantic_local_fusion = nn.Sequential(
            nn.Linear(branch_dim * 3, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )
        self.pose_local_fusion = nn.Sequential(
            nn.Linear(branch_dim * 3, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )
        self.semantic_decoder = QueryBranchDecoder(
            branch_dim=branch_dim,
            hidden_dim=adapter_hidden_dim,
            embedding_dim=sem_embedding_dim,
            out_logits=num_classes,
        )
        self.pose_decoder = QueryBranchDecoder(
            branch_dim=branch_dim,
            hidden_dim=adapter_hidden_dim,
            embedding_dim=geo_embedding_dim,
            out_logits=None,
        )
        self.occupancy_local_fusion = nn.Sequential(
            nn.Linear(branch_dim * 3, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )
        self.occupancy_decoder = OccupancyDecoder(
            branch_dim=branch_dim,
            hidden_dim=adapter_hidden_dim,
        )
        self.num_anchors = int(num_anchors)
        self.rbf_sigma = float(rbf_sigma)
        self.triplane_resolution = int(triplane_resolution)
        self.triplane_padding = float(triplane_padding)
        # Backward-compatible aliases for older tooling and checkpoints.
        self.geometric_adapter = self.pose_adapter
        self.geometric_projector = self.pose_projector
        self.geometric_local_fusion = self.pose_local_fusion
        self.geometric_decoder = self.pose_decoder

    def encode_support(
        self,
        utonia_input: dict,
        support_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_support_points = int(support_points.shape[1])
        support_features = self.feature_extractor(
            utonia_input,
            num_support_points=num_support_points,
        )
        semantic_support = self.semantic_adapter(support_features)
        pose_support = self.pose_adapter(support_features)
        semantic_cache = self.semantic_projector.build(
            support_points=support_points,
            support_features=semantic_support,
        )
        pose_cache = self.pose_projector.build(
            support_points=support_points,
            support_features=pose_support,
        )
        return {
            "semantic": semantic_cache,
            "pose": pose_cache,
            "geometric": pose_cache,
        }

    def query_semantic(
        self,
        cache: dict[str, torch.Tensor],
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_features, global_features = self.semantic_projector.sample(
            cache=cache["semantic"],
            query_points=query_points,
        )
        local_features = self.semantic_local_fusion(local_features)
        return self.semantic_decoder(
            query_points=query_points,
            local_features=local_features,
            global_features=global_features,
        )

    def query_pose(
        self,
        cache: dict[str, torch.Tensor],
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_features, global_features = self.pose_projector.sample(
            cache=cache["pose"],
            query_points=query_points,
        )
        local_features = self.pose_local_fusion(local_features)
        return self.pose_decoder(
            query_points=query_points,
            local_features=local_features,
            global_features=global_features,
        )

    def query_geometric(
        self,
        cache: dict[str, torch.Tensor],
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.query_pose(cache=cache, query_points=query_points)

    def query_occupancy(
        self,
        cache: dict[str, torch.Tensor],
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_features, global_features = self.pose_projector.sample(
            cache=cache["pose"],
            query_points=query_points,
        )
        local_features = self.occupancy_local_fusion(local_features)
        return self.occupancy_decoder(
            query_points=query_points,
            local_features=local_features,
            global_features=global_features,
        )

    def encode_view(
        self,
        utonia_input: dict,
        support_points: torch.Tensor,
        query_surface_points: torch.Tensor,
        query_occ_points: torch.Tensor,
        *,
        compute_semantic: bool = True,
        compute_pose: bool = True,
        compute_occupancy: bool = True,
    ) -> dict[str, torch.Tensor]:
        cache = self.encode_support(
            utonia_input=utonia_input,
            support_points=support_points,
        )
        output: dict[str, torch.Tensor] = {}
        if compute_semantic:
            semantic_output = self.query_semantic(
                cache=cache,
                query_points=query_surface_points,
            )
            output["sem_embeddings"] = semantic_output["embedding"]
            output["sem_logits"] = semantic_output["logits"]
        if compute_pose:
            pose_surface_output = self.query_pose(
                cache=cache,
                query_points=query_surface_points,
            )
            output["pose_embeddings"] = pose_surface_output["embedding"]
            output["geo_embeddings"] = pose_surface_output["embedding"]
        if compute_occupancy:
            occupancy_output = self.query_occupancy(
                cache=cache,
                query_points=query_occ_points,
            )
            output["occ_logits"] = occupancy_output["occ_logits"]
        return output

    def forward(
        self,
        batch: dict,
        train_mode: str = "joint",
    ) -> dict[str, dict[str, torch.Tensor]]:
        compute_semantic = train_mode in {"semantic", "joint"}
        compute_pose = train_mode in {"pose", "joint"}
        compute_occupancy = train_mode in {"pose", "joint"}
        outputs = {
            "view1": self.encode_view(
                utonia_input=batch["utonia_input_view1"],
                support_points=batch["support_points_view1"],
                query_surface_points=batch["query_surface_points_view1"],
                query_occ_points=batch["query_occ_points_view1"],
                compute_semantic=compute_semantic,
                compute_pose=compute_pose,
                compute_occupancy=compute_occupancy,
            )
        }
        if "utonia_input_view2" in batch:
            outputs["view2"] = self.encode_view(
                utonia_input=batch["utonia_input_view2"],
                support_points=batch["support_points_view2"],
                query_surface_points=batch["query_surface_points_view2"],
                query_occ_points=batch["query_occ_points_view2"],
                compute_semantic=compute_semantic,
                compute_pose=compute_pose,
                compute_occupancy=compute_occupancy,
            )
        return outputs
