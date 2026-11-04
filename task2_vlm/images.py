"""COCO image download: parallel, with retries; every file must open as an image."""
import concurrent.futures
import json
import os
import time
import urllib.request

from PIL import Image


def load_image_table(path):
    with open(path) as f:
        return json.load(f)


def image_path(root, split, file_name):
    return os.path.join(root, split, file_name)


def open_rgb(path):
    """Fully decode an image and convert it to RGB (raises if the file is not a readable image)."""
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def _fetch(url, dest, retries, timeout):
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            open_rgb(tmp)
            os.replace(tmp, dest)
            return attempt + 1
        except Exception as e:  # network or decode failure: retry with backoff
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last!r}")


def download_images(table, root, workers=16, retries=5, timeout=30):
    """Download every image of an images_{split}.json table into root/<split>/; files already present and
    readable are kept. Returns counts and wall-clock seconds; raises if any image cannot be fetched."""
    split = table["split"]
    os.makedirs(os.path.join(root, split), exist_ok=True)
    todo, present = [], 0
    for r in table["images"]:
        p = image_path(root, split, r["file_name"])
        if os.path.exists(p):
            try:
                open_rgb(p)
                present += 1
                continue
            except Exception:
                os.remove(p)
        todo.append((r["url"], p))
    t0 = time.time()
    attempts = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(lambda a: _fetch(a[0], a[1], retries, timeout), todo):
            attempts += n
    return {"split": split, "images": len(table["images"]), "already_present": present,
            "downloaded": len(todo), "attempts": attempts, "seconds": time.time() - t0}
