"""Label-check sheets for the grounding dev and test pairs (manual L501-502: inspect labels independently of
model predictions). Not committed: written to notes/task2/label_check/.

Each sheet shows `pairs_per_sheet` pairs, one per row: the two images side by side (fitted into
image_px x image_px, no boxes), and under each image the pair id, the question and its label. A pair with a
wrong label goes into flags.json (a list of pair ids), which task2_vlm.apply_label_flags consumes.

Run: python -m task2_vlm.label_sheets
"""
import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

from task2_vlm.images import download_images, image_path, open_rgb
from task2_vlm.select_data import CONFIG, load_config

PAD = 12
TEXT_H = 66
FONT_PX = 13


def pair_rows(split_dir):
    rows = []
    for s in ("dev", "test"):
        with open(os.path.join(split_dir, f"grounding_{s}.json")) as f:
            rows += [(s, p) for p in json.load(f)["pairs"]]
    return rows


def draw_sheet(rows, files, px, font):
    w = 2 * px + 3 * PAD
    row_h = px + TEXT_H + PAD
    sheet = Image.new("RGB", (w, PAD + len(rows) * row_h), "white")
    d = ImageDraw.Draw(sheet)
    for r, (split, p) in enumerate(rows):
        y = PAD + r * row_h
        for slot in range(2):
            x = PAD + slot * (px + PAD)
            im = open_rgb(files[p["image_ids"][slot]])
            im.thumbnail((px, px))
            sheet.paste(im, (x + (px - im.width) // 2, y + (px - im.height) // 2))
            d.rectangle([x - 1, y - 1, x + px, y + px], outline="#999999")
            lines = [f"{split} {p['pair_id']}"] + textwrap.wrap(p["question"], 44) + [
                f"label: {p['answers'][slot]}   (non-crowd {p['category']}: {p['counts'][slot]})"]
            for i, line in enumerate(lines[:4]):
                d.text((x, y + px + 4 + i * (FONT_PX + 3)), line, fill="black", font=font)
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    lc, im = cfg["label_check"], cfg["images"]
    with open(os.path.join(cfg["out_dir"], "images_val.json")) as f:
        table = json.load(f)
    rows = pair_rows(cfg["out_dir"])
    need = {i for _, p in rows for i in p["image_ids"]}
    sub = {"split": table["split"], "images": [r for r in table["images"] if r["image_id"] in need]}
    print(download_images(sub, im["dir"], im["workers"], im["retries"], im["timeout_s"]))
    files = {r["image_id"]: image_path(im["dir"], sub["split"], r["file_name"]) for r in sub["images"]}
    os.makedirs(lc["out_dir"], exist_ok=True)
    font = ImageFont.load_default(size=FONT_PX)
    k = lc["pairs_per_sheet"]
    for n, i in enumerate(range(0, len(rows), k), 1):
        path = os.path.join(lc["out_dir"], f"sheet_{n:02d}.png")
        draw_sheet(rows[i:i + k], files, lc["image_px"], font).save(path)
        print(path, [p["pair_id"] for _, p in rows[i:i + k]])
    flags = os.path.join(lc["out_dir"], "flags.json")
    if not os.path.exists(flags):  # never clobber flags already entered
        with open(flags, "w") as f:
            f.write("[]\n")
    print(f"{len(rows)} pairs on {n} sheets; flags file {flags}")


if __name__ == "__main__":
    main()
