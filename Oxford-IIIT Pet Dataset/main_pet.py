import os
import torch

from utils_seed_split import set_seed, split_indices
from pet_dataset import OxfordPetsImagesOnlyDetectionSubset
from rcnn import build_frcnn
from pet_yolov8 import build_yolov8, export_pets_images_only_to_yolo
from pet_train import train_frcnn, train_yolov8  
from eval import eval_frcnn_map50, eval_yolov8  


def main():
    os.makedirs("outputs", exist_ok=True)

    seed = 42
    set_seed(seed)

    # GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pets_root = r"C:\Users\solisrv\OneDrive - beloit.edu\Desktop\Homework2\Oxford-IIIT Pet"

    # choose 5 breeds that appear in your filenames
    BREEDS_5 = ["Abyssinian", "american_bulldog", "beagle", "Bengal", "Birman"]

    # Use 512 for GPU, smaller for CPU fallback
    img_size = 512 if device == "cuda" else 384

    # Pets subset: 15–20 epochs
    max_epochs = 20
    patience = 3

    # Batch sizes: higher on GPU, conservative on CPU
    frcnn_batch = 4 if device == "cuda" else 2
    yolo_batch = 16 if device == "cuda" else 8

    # Caching helps CPU, not needed on GPU
    cache_ds = (device == "cpu")

    tmp = OxfordPetsImagesOnlyDetectionSubset(pets_root, breeds=BREEDS_5, img_size=img_size, cache=cache_ds)
    n = len(tmp.items)
    train_idx, val_idx, test_idx = split_indices(n, 0.7, 0.15, 0.15, seed=seed)

    train_ds = OxfordPetsImagesOnlyDetectionSubset(pets_root, BREEDS_5, indices=train_idx, img_size=img_size, cache=cache_ds)
    val_ds   = OxfordPetsImagesOnlyDetectionSubset(pets_root, BREEDS_5, indices=val_idx,   img_size=img_size, cache=cache_ds)
    test_ds  = OxfordPetsImagesOnlyDetectionSubset(pets_root, BREEDS_5, indices=test_idx,  img_size=img_size, cache=cache_ds)

    # Faster R-CNN: 5 breeds + background => 6
    # cpu_fast=True is only useful on CPU; disable on GPU
    frcnn = build_frcnn(
        num_classes=len(BREEDS_5) + 1,
        cpu_fast=(device == "cpu"),
        trainable_backbone_layers=2 if device == "cuda" else 0
    )

    frcnn_time, frcnn_best_val = train_frcnn(
        frcnn, train_ds, val_ds,
        device=device,
        batch_size=frcnn_batch,
        epochs=max_epochs,
        early_stopping=True,
        patience=patience,
        score_thresh=0.4,
        mixed_precision=(device == "cuda")  # <-- add this flag in train_frcnn
    )
    frcnn_map50, frcnn_p, frcnn_r, frcnn_ips = eval_frcnn_map50(
        frcnn, test_ds, device=device, score_thresh=0.4
    )

    # YOLO export + train + eval
    yolo_export_root = "outputs/yolo_pets_images_only_5"
    data_yaml, total_items = export_pets_images_only_to_yolo(
        pets_root, yolo_export_root, BREEDS_5,
        train_idx, val_idx, test_idx,
        img_size=img_size
    )

    yolo = build_yolov8("yolov8n.pt")

    # Train YOLO on GPU if available
    yolo_time = train_yolov8(
        yolo,
        data_yaml,
        epochs=max_epochs,
        imgsz=img_size,
        batch=yolo_batch,
        patience=patience,
        device=device,           # <-- add device param in train_yolov8
        project="outputs",
        name="yolov8n_pets_images_only",
        amp=(device == "cuda")   # <-- enable mixed precision on GPU
    )

    yolo_map50, yolo_p, yolo_r, yolo_ips, yolo_metrics_obj = eval_yolov8(
        yolo,
        data_yaml,
        imgsz=img_size,
        batch=yolo_batch,
        device=device            # <-- add device param in eval_yolov8
    )

    print("\n==================== FINAL RESULTS (Pets images-only: pseudo boxes) ====================")
    print("Dataset\tModel\t\tmAP@0.5\tPrecision\tRecall\t\tTrain Time(s)\tInf Speed(img/s)")
    print(f"Pets-5\tFRCNN\t\t{frcnn_map50:.4f}\t{frcnn_p:.4f}\t\t{frcnn_r:.4f}\t\t{frcnn_time:.1f}\t\t{frcnn_ips:.2f}")

    if yolo_map50 is None or yolo_p is None or yolo_r is None:
        print(f"Pets-5\tYOLOv8n\t\t(N/A)\t(N/A)\t\t(N/A)\t\t{yolo_time:.1f}\t\t{yolo_ips:.2f}")
        print("\n[YOLO] Metrics object:")
        print(yolo_metrics_obj)
    else:
        print(f"Pets-5\tYOLOv8n\t\t{yolo_map50:.4f}\t{yolo_p:.4f}\t\t{yolo_r:.4f}\t\t{yolo_time:.1f}\t\t{yolo_ips:.2f}")
    print("========================================================================================\n")


if __name__ == "__main__":
    main()
