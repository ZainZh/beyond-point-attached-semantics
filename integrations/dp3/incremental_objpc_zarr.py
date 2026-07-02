import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import zarr


DP3_DIFFUSION_ROOT = Path(__file__).resolve().parents[1] / "3D-Diffusion-Policy"
if str(DP3_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_DIFFUSION_ROOT))

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


DEFAULT_CHUNK_LENGTH = 100
DEFAULT_COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)


def _is_valid_replay_buffer_root(root) -> bool:
    try:
        return "data" in root and "meta" in root and "episode_ends" in root["meta"]
    except Exception:
        return False


def open_or_reset_replay_buffer(zarr_path: str) -> Tuple[ReplayBuffer, int]:
    path = Path(zarr_path).resolve()
    if path.exists():
        try:
            root = zarr.open(str(path), mode="a")
            if _is_valid_replay_buffer_root(root):
                buffer = ReplayBuffer.create_from_group(root)
                return buffer, int(buffer.n_episodes)
        except Exception:
            pass
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(path))
    buffer = ReplayBuffer.create_empty_zarr(root=root)
    return buffer, 0


def build_episode_chunks(episode_data: Dict[str, np.ndarray], chunk_length: int = DEFAULT_CHUNK_LENGTH) -> Dict[str, tuple]:
    chunks = {}
    for key, value in episode_data.items():
        value = np.asarray(value)
        chunks[key] = (int(chunk_length),) + tuple(value.shape[1:])
    return chunks


def append_episode_to_buffer(
    buffer: ReplayBuffer,
    episode_data: Dict[str, np.ndarray],
    *,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
):
    normalized = {key: np.asarray(value) for key, value in episode_data.items()}
    chunks = build_episode_chunks(normalized, chunk_length=int(chunk_length))
    compressors = {key: DEFAULT_COMPRESSOR for key in normalized.keys()}
    buffer.add_episode(normalized, chunks=chunks, compressors=compressors)


__all__ = [
    "append_episode_to_buffer",
    "build_episode_chunks",
    "open_or_reset_replay_buffer",
]
