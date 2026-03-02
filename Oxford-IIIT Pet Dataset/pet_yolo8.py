from ultralytics import YOLO
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import re

def build_yolov8(model_name="yolov8n.pt"):
    return YOLO(model_name)

def _stem_to_breed(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)

def _pseudo_bbox_from_image(rgb: np.ndarray):
    gray = (0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]).astype(np.float32)
    g = (gray - gray.min()) / (np.ptp(gray) + 1e-6)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    edge = gx + gy
    thr = np.quantile(edge, 0.90)
    mask = edge >= thr
    ys, xs = np.where(mask)
    h, w = g.shape
    if ys.size < 50:
        return int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    pad_x = int(0.05 * (x2 - x1 + 1))
    pad_y = int(0.05 * (y2 - y1 + 1))
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w-1, x2 + pad_x); y2 = min(h-1, y2 + pad_y)
    if (x2-x1) < w*0.2 or (y2-y1) < h*0.2:
        return int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)
    return int(x1), int(y1), int(x2), int(y2)

def export_pets_images_only_to_yolo(pets_root, out_root, breeds_5, train_idx, val_idx, test_idx, img_size=384):
    pets_root = Path(pets_root)
    out_root = Path(out_root)

    images_dir = pets_root / "images"
    all_imgs = sorted([p for p in images_dir.glob("*.jpg")])

    breed_to_class = {b: i for i, b in enumerate(breeds_5)}

    items = []
    for p in all_imgs:
        stem = p.stem
        breed = _stem_to_breed(stem)
        if breed in breed_to_class:
            items.append((p.name, stem, breed))

    if len(items) == 0:
        raise RuntimeError("No matching images for chosen breeds. Check names vs filenames.")

    if out_root.exists():
        shutil.rmtree(out_root)

    def write_split(split_name, indices):
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for i in indices:
            fname, stem, breed = items[i]
            img = Image.open(images_dir / fname).convert("RGB").resize((img_size, img_size))
            img_np = np.array(img, dtype=np.uint8)

            x1, y1, x2, y2 = _pseudo_bbox_from_image(img_np)
            w = h = img_size
            xc = ((x1 + x2) / 2.0) / w
            yc = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            cls = breed_to_class[breed]

            out_img = out_root / "images" / split_name / fname
            out_lbl = out_root / "labels" / split_name / f"{stem}.txt"
            img.save(out_img)
            out_lbl.write_text(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    write_split("train", train_idx)
    write_split("val", val_idx)
    write_split("test", test_idx)

    data_yaml = out_root / "data.yaml"
    lines = [
        f"path: {out_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, b in enumerate(breeds_5):
        lines.append(f"  {i}: {b}")
    data_yaml.write_text("\n".join(lines))

    return str(data_yaml), len(items)

