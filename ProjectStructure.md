# Project Setup & Structure

## Repository Structure

"""
Homework2/
├─ Penn-Fudan/
│ ├─ dataset.py # PennFudanDataset + collate_fn (mask → boxes)
│ ├─ train.py # training loops for Faster R-CNN + YOLOv8n
│ ├─ eval.py # evaluation (mAP@0.5, precision, recall, speed)
│ ├─ main.py # runs Penn-Fudan experiment end-to-end
│ ├─ rcnn.py # Faster R-CNN model builder (transfer learning)
│ ├─ yolov8.py # YOLOv8 model builder + Penn-Fudan → YOLO export
│ ├─ metrics_timing.py # IoU matching + AP/PR utilities
│ └─ utils_seed_split.py # set_seed + train/val/test split
│
├─ Oxford-IIIT Pet Dataset/
│ ├─ pet_dataset.py # images-only dataset + pseudo boxes + collate_fn
│ ├─ pet_train.py # training loops for pet experiment
│ ├─ pet_yolo8.py # YOLOv8 model builder + Pets → YOLO export
│ ├─ main_pet.py # runs Pets (5-breed subset) experiment end-to-end
│ ├─ rcnn.py # Faster R-CNN model builder (shared)
│ ├─ eval.py # evaluation (shared)
│ ├─ metrics_timing.py # IoU/AP utilities (shared)
│ └─ utils_seed_split.py # seed + split (shared)

---
"""
## Requirements

- Python 3.10+ (tested with Python 3.12)
- PyTorch
- Torchvision
- Ultralytics (YOLOv8)
- NumPy
- tqdm
- Pillow

### Install dependencies:

```bash
pip install torch torchvision ultralytics numpy tqdm pillow
Dataset Setup
1) Penn-Fudan

Place the dataset at:

data/PennFudanPed/
├─ PNGImages/
└─ PedMasks/
2) Oxford-IIIT Pet (images-only)

Place the dataset at:

data/Oxford-IIIT Pet/
└─ images/

If your pets folder is elsewhere (e.g., OneDrive path), update pets_root in main_pet.py.

Running Experiments
Penn-Fudan (Pedestrian Detection)

From the Homework2 directory:

python Penn-Fudan/main.py
Oxford-IIIT Pets (5-Breed Subset Detection)

From the Homework2 directory:

python Oxford-IIIT\ Pet\ Dataset/main_pet.py
GPU vs CPU Notes

Primary experiments were run on GPU at 512×512 resolution.

The pipeline can run on CPU, but for speed/memory reasons the image size is typically reduced (e.g., 384×384).

If you are CPU-only, reduce:

img_size in main.py / main_pet.py

YOLO imgsz argument in training

Outputs

During YOLO training/export, YOLO-formatted datasets and training artifacts are written to:

outputs/

Final results are printed as tables including:

mAP@0.5

Precision

Recall

Training time (seconds)

Inference speed (images/second)

Common Issues
Import/module not found

Make sure the filename matches the import exactly.
Example: if the file is pet_dataset.py, imports must use from pet_dataset import ....

Penn-Fudan “cache” argument error

If your PennFudanDataset does not accept cache, remove cache=True from main.py or add cache support to the dataset class.

NumPy .ptp() error

If you see:
AttributeError: 'numpy.ndarray' object has no attribute 'ptp'
replace:
gray.ptp() with np.ptp(gray) in the pseudo-box code.
'''
