import os, json, random, argparse, shutil, glob
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    for d in ['images/train','images/val','images/test','labels/train','labels/val','labels/test']:
        (out/d).mkdir(parents=True, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(args.root, 'images', '*.png')) + glob.glob(os.path.join(args.root, 'images', '*.jpg')))
    if not img_files: raise RuntimeError("No images found under images/*.png or *.jpg")
    pairs = []
    for im in img_files:
        base = os.path.splitext(os.path.basename(im))[0]
        m1 = os.path.join(args.root, 'gt', base + '.png')
        m2 = os.path.join(args.root, 'masks', base + '.png')
        mask_path = m1 if os.path.exists(m1) else (m2 if os.path.exists(m2) else None)
        pairs.append((im, mask_path))
    random.shuffle(pairs)
    n = len(pairs); n_test = int(n*args.test_ratio); n_val = int(n*args.val_ratio)
    splits = {'test': pairs[:n_test], 'val': pairs[n_test:n_test+n_val], 'train': pairs[n_test+n_val:]}
    for split, items in splits.items():
        for im_path, mask_path in items:
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if img is None: continue
            H, W = img.shape[:2]
            boxes = []
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                boxes = masks_to_boxes_all(mask)
            dst_img = out / f'images/{split}/{Path(im_path).name}'
            shutil.copy2(im_path, dst_img)
            dst_lbl = out / f'labels/{split}/{Path(im_path).with_suffix(".txt").name}'
            with open(dst_lbl, 'w') as f:
                for (xc,yc,w,h) in boxes:
                    f.write(f"0 {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\\n")
    data_yaml = {'path': str(out.resolve()), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': ['defect']}
    with open(out / 'data.yaml', 'w') as f: f.write(json.dumps(data_yaml, indent=2))
    print(f"Done. YOLO dataset at: {out.resolve()}")
if __name__ == "__main__": main()
