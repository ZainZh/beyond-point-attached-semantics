from typing import Mapping, Sequence

import numpy as np

from object_pointcloud_utils import merge_object_point_clouds


def resolve_context_placeholders(
    placeholders: Sequence[str],
    feature_placeholders: Sequence[str],
    *,
    keep_feature_placeholders_in_context: bool = False,
) -> list[str]:
    if keep_feature_placeholders_in_context:
        return [str(placeholder) for placeholder in placeholders]
    feature_set = {str(placeholder) for placeholder in feature_placeholders}
    return [str(placeholder) for placeholder in placeholders if str(placeholder) not in feature_set]


def build_context_point_cloud(
    per_placeholder_point_clouds: Mapping[str, np.ndarray],
    *,
    placeholders: Sequence[str],
    feature_placeholders: Sequence[str],
    target_num_points: int,
    keep_feature_placeholders_in_context: bool = False,
) -> tuple[np.ndarray, list[str]]:
    context_placeholders = resolve_context_placeholders(
        placeholders,
        feature_placeholders,
        keep_feature_placeholders_in_context=keep_feature_placeholders_in_context,
    )
    context_clouds = [
        per_placeholder_point_clouds[placeholder]
        for placeholder in context_placeholders
        if placeholder in per_placeholder_point_clouds
    ]
    if len(context_clouds) == 0:
        return np.zeros((int(target_num_points), 6), dtype=np.float32), context_placeholders
    return (
        merge_object_point_clouds(context_clouds, target_num_points=int(target_num_points)).astype(np.float32),
        context_placeholders,
    )
