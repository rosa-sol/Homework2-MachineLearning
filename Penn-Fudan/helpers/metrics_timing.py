import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def timed():
    t0 = time.time()
    yield lambda: time.time() - t0

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def match_detections(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        return [], [], len(gt_boxes)

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    used = np.zeros(len(gt_boxes), dtype=bool)
    tp, fp = [], []

    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if used[j]:
                continue
            v = iou_xyxy(pb, gb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and best_j >= 0:
            used[best_j] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    return tp, fp, len(gt_boxes)

def average_precision(tp, fp, total_gt):
    if total_gt == 0:
        return 0.0
    tp = np.array(tp, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32)
    if tp.size == 0:
        return 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / (total_gt + 1e-9)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap

def precision_recall(tp_total, fp_total, fn_total):
    p = tp_total / (tp_total + fp_total + 1e-9)
    r = tp_total / (tp_total + fn_total + 1e-9)
    return float(p), float(r)
