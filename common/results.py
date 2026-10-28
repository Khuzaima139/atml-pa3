"""Atomic result-JSON writing that never clobbers another run's file."""
import json
import os
import tempfile


class ResultOverwriteError(RuntimeError):
    pass


def write_result(path, data):
    """Atomically write `data` (must contain "run_id") to `path` as JSON.

    Rewriting a file from the same run_id is allowed (periodic logging);
    a file holding a different or missing run_id is never overwritten.
    """
    run_id = data.get("run_id")
    if not run_id:
        raise ValueError("result data must contain a non-empty 'run_id'")
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f).get("run_id")
        except (OSError, ValueError, AttributeError):
            existing = None
        if existing != run_id:
            raise ResultOverwriteError(
                f"{path} belongs to run_id={existing!r}, refusing to overwrite with {run_id!r}")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def read_result(path):
    with open(path) as f:
        return json.load(f)
