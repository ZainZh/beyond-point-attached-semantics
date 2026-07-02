from __future__ import annotations

from pathlib import Path


def resolve_partnext_mesh_path(
    annotation_path: str | Path,
    glb_dst: str | Path,
    dataset_root: str | Path | None = None,
) -> Path | None:
    annotation_path = Path(annotation_path)
    mesh_path = Path(glb_dst)

    if mesh_path.is_absolute():
        return mesh_path if mesh_path.exists() else None

    candidates = [annotation_path.parent / mesh_path]
    if dataset_root is not None:
        candidates.append(Path(dataset_root) / mesh_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
