import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import collate_fn
from metrics_timing import match_detections, average_precision, precision_recall


@torch.no_grad()
def eval_frcnn_map50(model, ds, device="cpu", score_thresh=0.4, iou_thresh=0.5, num_workers=0):
    """
    Lightweight mAP@0.5 (VOC-style AP from PR curve), plus Precision/Recall and inference speed.
    """
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda")
    )

    model.to(device)
    model.eval()

    all_tp, all_fp = [], []
    total_gt = 0
    tp_total = fp_total = fn_total = 0

    t0 = time.time()
    n_images = 0

    for images, targets in tqdm(loader, desc="FRCNN eval", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        out = outputs[0]
        gt = targets[0]
        n_images += 1

        scores = out["scores"].detach().cpu().numpy()
        boxes = out["boxes"].detach().cpu().numpy()
        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]

        gt_boxes = gt["boxes"].detach().cpu().numpy()

        tp, fp, n_gt = match_detections(boxes, scores, gt_boxes, iou_thresh=iou_thresh)
        all_tp.extend(tp)
        all_fp.extend(fp)
        total_gt += n_gt

        tp_sum = int(np.sum(tp)) if len(tp) else 0
        fp_sum = int(np.sum(fp)) if len(fp) else 0
        fn_sum = int(n_gt - tp_sum)

        tp_total += tp_sum
        fp_total += fp_sum
        fn_total += fn_sum

    dt = time.time() - t0
    ips = n_images / (dt + 1e-9)

    ap50 = average_precision(all_tp, all_fp, total_gt)
    p, r = precision_recall(tp_total, fp_total, fn_total)
    return ap50, p, r, ips


def _extract_ultralytics_metrics(metrics_obj):
    """
    Ultralytics versions differ. Try common attribute paths.
    Returns (map50, precision, recall) or (None, None, None) if not found.
    """
    for box_attr in ("box", "bbox"):
        box = getattr(metrics_obj, box_attr, None)
        if box is not None:
            map50 = getattr(box, "map50", None)
            mp = getattr(box, "mp", None)
            mr = getattr(box, "mr", None)
            if map50 is not None and mp is not None and mr is not None:
                return float(map50), float(mp), float(mr)

    rd = getattr(metrics_obj, "results_dict", None)
    if isinstance(rd, dict):
        candidates_map50 = ["metrics/mAP50(B)", "metrics/mAP50", "map50"]
        candidates_p = ["metrics/precision(B)", "metrics/precision", "precision"]
        candidates_r = ["metrics/recall(B)", "metrics/recall", "recall"]

        map50 = next((rd.get(k) for k in candidates_map50 if k in rd), None)
        p = next((rd.get(k) for k in candidates_p if k in rd), None)
        r = next((rd.get(k) for k in candidates_r if k in rd), None)

        if map50 is not None and p is not None and r is not None:
            return float(map50), float(p), float(r)

    return None, None, None


def eval_yolov8(yolo_model, data_yaml, imgsz=384, batch=8, device="cpu"):
    """
    Uses Ultralytics built-in metrics for mAP50/P/R and measures inference speed (img/s) on test set.
    Works on CPU or GPU.
    """
    yolo_device = 0 if device == "cuda" and torch.cuda.is_available() else "cpu"
    workers = 2 if yolo_device != "cpu" else 0

    # validation metrics
    metrics = yolo_model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=yolo_device,
        workers=workers,
        verbose=False
    )
    map50, p, r = _extract_ultralytics_metrics(metrics)

    # inference speed on test folder
    from pathlib import Path
    outroot = Path(data_yaml).parent
    test_folder = outroot / "images" / "test"

    t0 = time.time()
    preds = yolo_model.predict(
        source=str(test_folder),
        imgsz=imgsz,
        device=yolo_device,
        verbose=False
    )
    preds = list(preds)
    dt = time.time() - t0
    ips = len(preds) / (dt + 1e-9)

    return map50, p, r, ips, metrics
