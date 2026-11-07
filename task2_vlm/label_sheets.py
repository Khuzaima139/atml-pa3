"""Label-check sheets for the grounding dev and test pairs (manual L501-502: inspect labels independently of
model predictions). Not committed: written to notes/task2/label_check/.

Each sheet shows `pairs_per_sheet` pairs, one per row: the two images side by side (fitted into
image_px x image_px, no boxes), and under each image the pair id, the question and its label. A pair with a
wrong label goes into flags.json (a list of pair ids), which task2_vlm.apply_label_flags consumes.

--preview draws the candidates of preview.json (task2_vlm.apply_label_flags --preview) as preview_NN.png:
3 pairs per sheet, each image scaled to 430 px high (full image, no boxes), with pair id, question and labels.

Run: python -m task2_vlm.label_sheets [--preview]
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


def draw_preview_sheet(rows, files, height, font):
    ims = [[open_rgb(files[p["image_ids"][s]]) for s in range(2)] for p in rows]
    ims = [[im.resize((round(im.width * height / im.height), height), Image.BICUBIC) for im in r] for r in ims]
    w = max(3 * PAD + a.width + b.width for a, b in ims)
    row_h = height + TEXT_H + PAD
    sheet = Image.new("RGB", (w, PAD + len(rows) * row_h), "white")
    d = ImageDraw.Draw(sheet)
    for r, (p, pair_ims) in enumerate(zip(rows, ims)):
        y, x = PAD + r * row_h, PAD
        for slot, im in enumerate(pair_ims):
            sheet.paste(im, (x, y))
            lines = [f"{p['pair_id']}  ({p['set']} {p['type']} #{p['stream_position']})", p["question"],
                     f"label: {p['answers'][slot]}   (non-crowd {p['category']}: {p['counts'][slot]})"]
            for i, line in enumerate(lines):
                d.text((x, y + height + 4 + i * (FONT_PX + 5)), line, fill="black", font=font)
            x += im.width + PAD
    return sheet


def preview_sheets(cfg, per_sheet, height):
    lc, im = cfg["label_check"], cfg["images"]
    with open(os.path.join(lc["out_dir"], "preview.json")) as f:
        rows = json.load(f)["pairs"]
    ids = sorted({i for p in rows for i in p["image_ids"]})
    names = {i: f"COCO_val2014_{i:012d}.jpg" for i in ids}
    table = {"split": "val2014", "images": [{"image_id": i, "file_name": names[i],
                                             "url": cfg["image_url"].format(split="val2014", file_name=names[i])}
                                            for i in ids]}
    print(download_images(table, im["dir"], im["workers"], im["retries"], im["timeout_s"]))
    files = {i: image_path(im["dir"], "val2014", names[i]) for i in ids}
    font = ImageFont.load_default(size=FONT_PX + 2)
    for n, i in enumerate(range(0, len(rows), per_sheet), 1):
        path = os.path.join(lc["out_dir"], f"preview_{n:02d}.png")
        draw_preview_sheet(rows[i:i + per_sheet], files, height, font).save(path)
        print(path, [p["pair_id"] for p in rows[i:i + per_sheet]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--per_sheet", type=int, default=3)
    ap.add_argument("--height", type=int, default=430)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.preview:
        preview_sheets(cfg, args.per_sheet, args.height)
        return
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
