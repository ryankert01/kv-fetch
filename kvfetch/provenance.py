"""Provenance block attached to every result file (spec §10 conventions)."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("kvfetch", "scripts", "configs", "tests")


def git_commit() -> dict:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    commit = run("git", "rev-parse", "HEAD") or "unknown"
    # untracked code counts as dirty: results must be reproducible from the commit
    dirty = bool(run("git", "status", "--porcelain", "--", *CODE_DIRS))
    return {"commit": commit, "dirty": dirty}


def provenance(cfg: dict, **fields: Any) -> dict:
    """Required fields per spec: model, context length, page size, w, git commit.

    Callers pass ``model`` and ``context_length`` (or None when not applicable);
    page size and w are read from the config so they cannot drift from what ran.
    """
    try:  # the trace-size estimator should run without torch
        import torch
    except ImportError:
        torch = None
    has_gpu = torch is not None and torch.cuda.is_available()

    defs = cfg["definitions"]
    out = {
        "model": fields.pop("model"),
        "context_length": fields.pop("context_length"),
        "page_size": defs["page_size"],
        "w": defs["w"],
        "git": git_commit(),
        "config_path": cfg.get("_meta", {}).get("config_path"),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": getattr(torch, "__version__", None),
        "gpu": torch.cuda.get_device_name() if has_gpu else None,
    }
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:
        pass
    out.update(fields)
    return out


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
