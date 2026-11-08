"""sha256 helpers for files and JSON-serialisable content."""
import hashlib
import json


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    """Key-sorted, compact, UTF-8 JSON bytes: the form every content hash is taken over."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj)).hexdigest()
