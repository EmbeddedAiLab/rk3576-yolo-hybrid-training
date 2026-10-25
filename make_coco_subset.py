#!/usr/bin/env python3
"""Build a larger YOLO-format subset of COCO for: keyboard, mouse, bottle, cup, person.

Usage:
  python3 make_coco_subset.py instances_train2017.json dataset_big [images_per_class=1000] [min_box=0.04]

- instances_train2017.json comes from http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- Images are downloaded one by one from their coco_url (no need to download all of COCO).
- Output layout: out/images/{train,val}, out/labels/{train,val}, out/data.yaml (class order below).
- Objects smaller than min_box (fraction of image width or height) are dropped as too small for 128x128 input.
"""
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

NAMES = ["keyboard", "mouse", "bottle", "cup", "person"]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ann_path, out = sys.argv[1], sys.argv[2]
    per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    min_box = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04
    random.seed(0)

    print("Loading annotations...")
    d = json.load(open(ann_path))
    cat_id = {c["name"]: c["id"] for c in d["categories"]}
    idx_of = {cat_id[n]: i for i, n in enumerate(NAMES)}
    imgs = {im["id"]: im for im in d["images"]}

    by_img = defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] not in idx_of or a.get("iscrowd", 0):
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        if w / W < min_box or h / H < min_box:
            continue
        by_img[a["image_id"]].append((idx_of[a["category_id"]], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))

    chosen = set()
    for ci, name in enumerate(NAMES):
        cand = [i for i, b in by_img.items() if any(t[0] == ci for t in b) and i not in chosen]
        random.shuffle(cand)
        chosen.update(cand[:per_class])
        print(f"  {name}: {len(cand)} candidate images, taking up to {per_class}")

    ids = sorted(chosen)
    random.shuffle(ids)
    nval = max(1, len(ids) // 10)
    split = {i: ("val" if k < nval else "train") for k, i in enumerate(ids)}
    for s in ("train", "val"):
        os.makedirs(f"{out}/images/{s}", exist_ok=True)
        os.makedirs(f"{out}/labels/{s}", exist_ok=True)

    def work(i):
        im, s = imgs[i], split[i]
        stem = f"{i:012d}"
        ip = f"{out}/images/{s}/{stem}.jpg"
        if not os.path.exists(ip):
            try:
                urllib.request.urlretrieve(im["coco_url"], ip)
            except Exception as e:  # skip images that fail to download
                print("download failed:", im["coco_url"], e)
                return False
        with open(f"{out}/labels/{s}/{stem}.txt", "w") as f:
            for c, cx, cy, w, h in by_img[i]:
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        return True

    print(f"Downloading {len(ids)} images...")
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(work, ids))

    with open(f"{out}/data.yaml", "w") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")

    counts = {s: [0] * len(NAMES) for s in ("train", "val")}
    nimg = {"train": 0, "val": 0}
    for i, good in zip(ids, ok):
        if not good:
            continue
        nimg[split[i]] += 1
        for c, *_ in by_img[i]:
            counts[split[i]][c] += 1
    print(f"Done: {nimg['train']} train / {nimg['val']} val images")
    for s in ("train", "val"):
        print(f"  {s} objects:", ", ".join(f"{n} {c}" for n, c in zip(NAMES, counts[s])))


if __name__ == "__main__":
    main()
