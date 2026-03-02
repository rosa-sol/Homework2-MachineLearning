from ultralytics import YOLO
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


def build_yolov8(model_name="yolov8n.pt"):
    # pretrained weights => transfer learning
    return YOLO(model_name)


def _mask_to_boxes(mask: np.ndarray):
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]
    boxes = []
    h, w = mask.shape
    for oid in obj_ids:
        ys, xs = np.where(mask == oid)
        if ys.size == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        boxes.append((x1, y1, x2, y2, w, h))
    return boxes


def _xyxy_to_yolo(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def export_pennfudan_to_yolo(
    penn_root: str,
    out_root: str,
    train_idx,
    val_idx,
    test_idx,
    img_size: int = 384
):
    """
    Converts Penn-Fudan (PNGImages + PedMasks) into YOLO format:
      out_root/
        images/train|val|test/*.png
        labels/train|val|test/*.txt
        data.yaml

    Class mapping: 0 = person
    """
    penn_root = Path(penn_root)
    out_root = Path(out_root)

    img_dir = penn_root / "PNGImages"
    mask_dir = penn_root / "PedMasks"
    imgs = sorted(img_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))
    assert len(imgs) == len(masks), "Penn-Fudan images/masks mismatch"

    if out_root.exists():
        shutil.rmtree(out_root)

    def write_split(split_name, indices):
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for i in indices:
            img = Image.open(imgs[i]).convert("RGB").resize((img_size, img_size))
            mask = np.array(Image.open(masks[i]), dtype=np.uint8)
            mask = Image.fromarray(mask).resize((img_size, img_size), resample=Image.NEAREST)
            mask = np.array(mask, dtype=np.uint8)

            out_img = out_root / "images" / split_name / imgs[i].name
            out_lbl = out_root / "labels" / split_name / (imgs[i].stem + ".txt")

            img.save(out_img)

            boxes = _mask_to_boxes(mask)
            lines = []
            for x1, y1, x2, y2, w, h in boxes:
                xc, yc, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, w, h)
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            out_lbl.write_text("\n".join(lines))

    write_split("train", train_idx)
    write_split("val", val_idx)
    write_split("test", test_idx)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {out_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: person"
        ])
    )

    return str(data_yaml)
