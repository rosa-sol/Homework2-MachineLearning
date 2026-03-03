# Project Setup & Structure

## Repository Structure
```
Homework2/
├─ Penn-Fudan/
│  ├─ dataset.py            # PennFudanDataset + collate_fn (mask → boxes)
│  ├─ train.py              # Training loops for Faster R-CNN + YOLOv8n
│  ├─ eval.py               # Evaluation (mAP@0.5, precision, recall, speed)
│  ├─ main.py               # Runs Penn-Fudan experiment end-to-end
│  ├─ rcnn.py               # Faster R-CNN model builder
│  ├─ yolov8.py             # YOLOv8 model builder + Penn-Fudan export
│  ├─ metrics_timing.py     # IoU matching + AP/PR utilities
│  └─ utils_seed_split.py   # Seed + train/val/test split
│
├─ Oxford-IIIT Pet Dataset/
│  ├─ pet_dataset.py        # Images-only dataset + pseudo boxes
│  ├─ pet_train.py          # Training loops for pet experiment
│  ├─ pet_yolo8.py          # YOLOv8 model builder + Pets export
│  ├─ main_pet.py           # Runs Pets experiment
│  ├─ rcnn.py               # Faster R-CNN model builder (shared)
│  ├─ eval.py               # Evaluation (shared)
│  ├─ metrics_timing.py     # IoU/AP utilities (shared)
│  └─ utils_seed_split.py   # Seed + split (shared)
│
├─ data/                    # Not in repo, but on device running code
│  ├─ PennFudanPed/
│  │  ├─ PNGImages/
│  │  └─ PedMasks/
│  └─ Oxford-IIIT Pet/
│     └─ images/

```

---

# Requirements

- Python 3.10+
- PyTorch
- Torchvision
- Ultralytics (YOLOv8)
- NumPy
- tqdm
- Pillow

## Install Dependencies

```bash
pip install torch torchvision ultralytics numpy tqdm pillow
```
---

# Dataset Setup

## 1) Penn-Fudan

Place the dataset at:

```
data/PennFudanPed/
├─ PNGImages/
└─ PedMasks/
```

---

## 2) Oxford-IIIT Pet (Images-Only)

Place the dataset at:

```
data/Oxford-IIIT Pet/
└─ images/
```

If your pets folder is located elsewhere (e.g., OneDrive), update `pets_root` inside `main_pet.py`.

---# Training Setup

- GPU acceleration (primary experiments)
- Image size: 512×512
- CPU fallback: 384×384
- Transfer learning enabled
- Early stopping (patience = 3)

### Penn-Fudan
- 10–15 epochs

### Pets Subset
- 15–20 epochs


# Running Experiments

## Penn-Fudan (Pedestrian Detection)

From the `Homework2` directory:

```bash
python Penn-Fudan/main.py
```

---

## Oxford-IIIT Pets (5-Breed Subset Detection)

From the `Homework2` directory:

```bash
python "Oxford-IIIT Pet Dataset/main_pet.py"
```

---

### Final results are printed as tables including:

mAP@0.5

Precision

Recall

Training time (seconds)

Inference speed (images/second)

## Common Issues
Import/module not found

Make sure the filename matches the import exactly.
Example: if the file is pet_dataset.py, imports must use from pet_dataset import ....

Penn-Fudan “cache” argument error

If your PennFudanDataset does not accept cache, remove cache=True from main.py or add cache support to the dataset class.

NumPy .ptp() error

## If you see:
AttributeError: 'numpy.ndarray' object has no attribute 'ptp'
replace:
gray.ptp() with np.ptp(gray) in the pseudo-box code.
'''
