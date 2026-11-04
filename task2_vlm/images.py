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


def download_images(table, root, workers=16, retries=5, timeout=30, report_after=None, pending_total=None):
    """Download every image of an images_{split}.json table into root/<split>/; files already present and
    readable are kept. Returns counts and wall-clock seconds; raises if any image cannot be fetched.
    After `report_after` downloads, prints the rate and the projected time for `pending_total` images (default:
    this table's missing images) and returns it under "progress"."""
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
    attempts, done, progress = 0, 0, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch, u, p, retries, timeout) for u, p in todo]
        for f in concurrent.futures.as_completed(futs):
            attempts += f.result()
            done += 1
            if report_after and done == report_after:
                dt = time.time() - t0
                total = pending_total or len(todo)
                progress = {"images": done, "seconds": dt, "images_per_s": done / dt, "pending_total": total,
                            "projected_total_s": total * dt / done}
                print(f"download: first {done} images in {dt:.1f}s = {done / dt:.2f} images/s; projected "
                      f"{total} images: {total * dt / done / 60:.1f} min", flush=True)
    return {"split": split, "images": len(table["images"]), "already_present": present,
            "downloaded": len(todo), "attempts": attempts, "seconds": time.time() - t0, "progress": progress}
