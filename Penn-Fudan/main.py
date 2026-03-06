import os
import torch

from utils_seed_split import set_seed, split_indices
from dataset import PennFudanDataset
from rcnn import build_frcnn
from yolov8 import build_yolov8, export_pennfudan_to_yolo
from train import train_frcnn, train_yolov8
from eval import eval_frcnn_map50, eval_yolov8


def main():
    # -----------------------------
    # SETTINGS
    # -----------------------------
    seed = 42
    data_root = "data/PennFudanPed"

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Penn-Fudan requirement: 10–15 epochs
    frcnn_epochs = 10
    yolo_epochs = 15

    # You can usually afford a larger image size on GPU
    img_size = 512 if device == "cuda" else 384
    yolo_imgsz = 512 if device == "cuda" else 384

    # Larger batch sizes on GPU
    frcnn_batch = 4 if device == "cuda" else 2
    yolo_batch = 16 if device == "cuda" else 8

    early_patience = 3

    os.makedirs("outputs", exist_ok=True)
    set_seed(seed)

    # -----------------------------
    # SPLIT
    # -----------------------------
    tmp = PennFudanDataset(data_root, img_size=img_size, cache=True)
    n = len(tmp.imgs)
    train_idx, val_idx, test_idx = split_indices(n, 0.7, 0.15, 0.15, seed=seed)

    train_ds = PennFudanDataset(data_root, indices=train_idx, img_size=img_size, cache=True)
    val_ds   = PennFudanDataset(data_root, indices=val_idx,   img_size=img_size, cache=True)
    test_ds  = PennFudanDataset(data_root, indices=test_idx,  img_size=img_size, cache=True)

    # -----------------------------
    # MODEL 1: Faster R-CNN
    # -----------------------------
    frcnn = build_frcnn(
        num_classes=2,
        cpu_fast=(device == "cpu"),
        trainable_backbone_layers=3 if device == "cuda" else 0
    )
    frcnn = frcnn.to(device)

    frcnn_train_time, frcnn_best_val = train_frcnn(
        frcnn,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        batch_size=frcnn_batch,
        epochs=frcnn_epochs,
        early_stopping=True,
        patience=early_patience,
        min_delta=0.001,
        score_thresh=0.4
    )

    frcnn_map50, frcnn_p, frcnn_r, frcnn_ips = eval_frcnn_map50(
        frcnn, test_ds, device=device, score_thresh=0.4
    )

    # -----------------------------
    # MODEL 2: YOLOv8n
    # -----------------------------
    yolo_export_root = "outputs/yolo_pennfudan"
    data_yaml = export_pennfudan_to_yolo(
        penn_root=data_root,
        out_root=yolo_export_root,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        img_size=yolo_imgsz
    )

    yolo = build_yolov8("yolov8n.pt")

    # If your train/eval helpers support a device argument, pass it through.
    # If not, you should update those helper functions too.
    yolo_train_time = train_yolov8(
    yolo,
    data_yaml,
    epochs=yolo_epochs,
    imgsz=yolo_imgsz,
    batch=yolo_batch,
    patience=early_patience,
    project="outputs",
    name="yolov8n_gpu" if device == "cuda" else "yolov8n_cpu",
    device=device
    )

    yolo_map50, yolo_p, yolo_r, yolo_ips, yolo_metrics_obj = eval_yolov8(
    yolo,
    data_yaml,
    imgsz=yolo_imgsz,
    batch=yolo_batch,
    device=device
    )

    # -----------------------------
    # PRINT RESULTS TABLE
    # -----------------------------
    print("\n==================== FINAL RESULTS (Penn-Fudan) ====================")
    print("Dataset\t\tModel\t\tmAP@0.5\tPrecision\tRecall\t\tTrain Time(s)\tInf Speed(img/s)")
    print(f"PennFudan\tFRCNN\t\t{frcnn_map50:.4f}\t{frcnn_p:.4f}\t\t{frcnn_r:.4f}\t\t{frcnn_train_time:.1f}\t\t{frcnn_ips:.2f}")

    if yolo_map50 is None or yolo_p is None or yolo_r is None:
        print(f"PennFudan\tYOLOv8n\t\t(N/A)\t\t(N/A)\t\t(N/A)\t\t{yolo_train_time:.1f}\t\t{yolo_ips:.2f}")
        print("\n[YOLO] Could not auto-extract mAP/P/R from this ultralytics version. Here is the metrics object:")
        print(yolo_metrics_obj)
    else:
        print(f"PennFudan\tYOLOv8n\t\t{yolo_map50:.4f}\t{yolo_p:.4f}\t\t{yolo_r:.4f}\t\t{yolo_train_time:.1f}\t\t{yolo_ips:.2f}")

    print("=====================================================================\n")


if __name__ == "__main__":
    main()
