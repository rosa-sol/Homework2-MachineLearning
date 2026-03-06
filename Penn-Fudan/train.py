import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import collate_fn


def train_frcnn(
    model,
    train_ds,
    val_ds,
    device="cpu",
    batch_size=2,
    epochs=10,              # Penn-Fudan: 10–15 max
    lr=0.005,
    num_workers=0,
    early_stopping=True,
    patience=3,
    min_delta=0.001,
    score_thresh=0.4,
):
    """
    Faster R-CNN training:
    - works on CPU or GPU
    - validate each epoch, early stop on val mAP@0.5
    """
    from eval import eval_frcnn_map50  # local import to avoid circular imports

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda")
    )

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    best_map = -1.0
    bad_epochs = 0
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for images, targets in tqdm(train_loader, desc=f"FRCNN train e{epoch+1}", leave=False):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())

        avg_loss = loss_sum / max(1, len(train_loader))
        val_map50, val_p, val_r, _ = eval_frcnn_map50(
            model, val_ds, device=device, score_thresh=score_thresh
        )

        print(
            f"[FRCNN] epoch {epoch+1}/{epochs} "
            f"loss={avg_loss:.4f} val_mAP50={val_map50:.4f} P={val_p:.4f} R={val_r:.4f}"
        )

        if val_map50 > best_map + min_delta:
            best_map = val_map50
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= patience:
                print(f"[FRCNN] Early stopping. Best val mAP50={best_map:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    return train_time, best_map


def train_yolov8(
    yolo_model,
    data_yaml,
    epochs=15,          # Penn-Fudan: 10–15 max
    imgsz=384,
    batch=8,
    patience=3,
    project="outputs",
    name="yolov8n",
    device="cpu"
):
    """
    Ultralytics YOLOv8 training on CPU or GPU with pretrained weights.
    """
    # Ultralytics expects 0 / 1 / ... for GPU, or "cpu"
    yolo_device = 0 if device == "cuda" and torch.cuda.is_available() else "cpu"

    t0 = time.time()
    yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=yolo_device,
        workers=2 if yolo_device != "cpu" else 0,
        pretrained=True,
        patience=patience,
        project=project,
        name=name,
        verbose=False
    )
    return time.time() - t0
