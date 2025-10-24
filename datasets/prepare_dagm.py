import os, json, random, argparse, shutil, glob
from pathlib import Path
import cv2
import numpy as np
def masks_to_boxes_all(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h <= 4: continue
        xc,yc = x + w/2, y + h/2
        boxes.append([xc,yc,w,h])
    return boxes
def write_label(txt_path, boxes_xywh, cls_id, W, H):
    with open(txt_path, 'w') as f:
        for (xc, yc, w, h) in boxes_xywh:
            f.write(f"{cls_id} {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n")
def discover_images(croot):
    pats = ["Train","train","Training","Images","image","img","Test","test"]
    files = []
    for p in pats:
        files += glob.glob(str(Path(croot)/p/"*.png"))
        files += glob.glob(str(Path(croot)/p/"*.jpg"))
        files += glob.glob(str(Path(croot)/p/"*.bmp"))
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
        for im in img_files:
            base = os.path.splitext(im)[0]
            m_cands = [base+"_mask.png", base+"_Label.png", base.replace("Images","Labels")+".png"]
            mask_path = None
            for m in m_cands:
                if os.path.exists(m): mask_path = m; break
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
if __name__ == "__main__": main()