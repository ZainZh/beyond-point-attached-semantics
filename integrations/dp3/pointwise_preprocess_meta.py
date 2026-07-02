import json
import os


def load_or_init_meta(meta_path: str, *, task_name: str, task_config: str, expert_data_num: int):
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("episodes", [])
        return meta
    return {
        "task_name": task_name,
        "task_config": task_config,
        "expert_data_num": int(expert_data_num),
        "episodes": [],
    }


def write_meta(meta_path: str, meta: dict):
    tmp_path = f"{meta_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def reconcile_episode_stats(stats, *, start_episode: int):
    episode_stats = list(stats or [])
    if len(episode_stats) > int(start_episode):
        episode_stats = episode_stats[: int(start_episode)]
    while len(episode_stats) < int(start_episode):
        episode_stats.append(
            {
                "episode": len(episode_stats),
                "recovered_without_stats": True,
            }
        )
    return episode_stats


__all__ = [
    "load_or_init_meta",
    "reconcile_episode_stats",
    "write_meta",
]
