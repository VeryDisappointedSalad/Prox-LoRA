import bz2
import datetime
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
"""Root of project (containing .git, pyproject.toml, src/)"""


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that encodes datetime objects as isoformat() str."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime.datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)


def load_pickle(path: Path) -> Any:
    """Load a pickle, auto-detect compression from file extension."""

    open_f: Any
    if path.suffix == ".bz2":
        open_f = bz2.open
    elif path.suffix == ".gz":
        open_f = gzip.open
    elif path.suffix in (".pickle", ".pkl"):
        open_f = open

    with open_f(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Dump data to a pickle file, auto-detect compression from file extension."""

    open_f: Any
    if path.suffix == ".bz2":
        open_f = bz2.open
    elif path.suffix == ".gz":
        open_f = gzip.open
    elif path.suffix in (".pickle", ".pkl"):
        open_f = open

    with open_f(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json_atomic(data: Any, path: Path) -> None:
    """
    Save a JSON file atomically (write to temp file, then rename).

    Any ongoing reads will see the old file (since they read the old inode).
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4, cls=DateTimeEncoder)
    temp_path.rename(path)


def find_latest_checkpoint(base_run_dir: Path) -> Path | None:
    if not base_run_dir.exists():
        return None
    checkpoints = list(base_run_dir.glob("**/checkpoints/last.ckpt"))
    if not checkpoints:
        checkpoints = list(base_run_dir.glob("**/*.ckpt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ckpt_path = checkpoints[0]
    config_path = ckpt_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found for checkpoint: {config_path}")
    return ckpt_path
