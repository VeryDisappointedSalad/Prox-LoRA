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
