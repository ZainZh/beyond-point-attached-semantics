from __future__ import annotations

import sys
from pathlib import Path


def ensure_utonia_importable():
    try:
        import utonia

        return utonia
    except ImportError as error:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "include" / "Utonia"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            import utonia

            return utonia
        raise ImportError(
            "Failed to import Utonia. Install the package in your environment or "
            "ensure include/Utonia points to a valid checkout."
        ) from error
