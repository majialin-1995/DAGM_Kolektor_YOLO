import os
import json
import random
import argparse
import shutil
from pathlib import Path

import cv2


def masks_to_boxes_all(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h <= 4: continue
        xc,yc = x + w/2, y + h/2
        boxes.append([xc,yc,w,h])
    return boxes


def load_label_mapping(croot: Path):
    """Return mapping from image filename (lowercase) to mask path.

    The DAGM Kaggle upload stores binary masks inside ``Label`` folders with
    a companion ``Labels.txt`` file. Each entry has the format::

        <index> <has_defect> <image_name> <unused> <mask_name or 0>

    ``mask_name`` is ``0`` when the image has no defect. We normalise keys to
    lowercase for robust lookups regardless of filename casing.
    """

    mapping = {}
    for label_dir in sorted(p for p in croot.rglob("*") if p.is_dir() and p.name.lower() == "label"):
        label_file = label_dir / "Labels.txt"
        if not label_file.exists():
            continue
        with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                fname = parts[2].strip().lower()
                mask_name = parts[4].strip()
                if mask_name != "0":
                    mapping[fname] = label_dir / mask_name
                else:
                    mapping[fname] = None
    return mapping


def write_label(txt_path, boxes_xywh, cls_id, W, H):
    with open(txt_path, 'w') as f:
        for (xc, yc, w, h) in boxes_xywh:
            f.write(f"{cls_id} {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n")
def discover_images(croot):
    croot = Path(croot)
    files = []
    for p in croot.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "label" not in p.stem.lower():
            files.append(str(p))
    return sorted(set(files))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    for d in ['images/train','images/val','images/test','labels/train','labels/val','labels/test']:
        (out/d).mkdir(parents=True, exist_ok=True)
    classes = [d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))]
    classes = sorted(classes)
    if not classes: raise RuntimeError("No class folders found under --root")
    samples = []
    for ci, cname in enumerate(classes):
        croot = Path(args.root)/cname
        img_files = discover_images(croot)
        label_map = load_label_mapping(croot)
        for im in img_files:
            base = os.path.splitext(im)[0]
            name_key = Path(im).name.lower()
            mask_from_labels = label_map.get(name_key)
            m_cands = [base+"_mask.png", base+"_mask.PNG", base+"_Label.png", base+"_Label.PNG",
                       base.replace("Images","Labels")+".png", base.replace("Images","Labels")+".PNG"]
            mask_path = None
            if mask_from_labels and mask_from_labels.exists():
                mask_path = str(mask_from_labels)
            else:
                for m in m_cands:
                    if os.path.exists(m):
                        mask_path = m
                        break
            samples.append((im, mask_path, ci))
    if not samples: raise RuntimeError("No images found; check your DAGM directory structure.")
    random.shuffle(samples)
    n = len(samples)
    n_test = int(n*args.test_ratio); n_val = int(n*args.val_ratio)
    splits = {'test': samples[:n_test], 'val': samples[n_test:n_test+n_val], 'train': samples[n_test+n_val:]}
    for split, items in splits.items():
        for im_path, mask_path, ci in items:
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if img is None: continue
            H, W = img.shape[:2]
            boxes = []
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    boxes = masks_to_boxes_all(mask)
            dst_img = out / f"images/{split}/{Path(im_path).name}"
            shutil.copy2(im_path, dst_img)
            dst_lbl = out / f"labels/{split}/{Path(im_path).with_suffix('.txt').name}"
            if boxes:
                write_label(dst_lbl, boxes, ci, W, H)
            else:
                open(dst_lbl, 'w').close()
    data_yaml = {'path': str(out.resolve()), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': classes}
    with open(out/'data.yaml','w') as f: f.write(json.dumps(data_yaml, indent=2))
    print(f"Done. YOLO dataset at: {out.resolve()}"); print(f"Classes: {classes}")


if __name__ == "__main__":
    main()
