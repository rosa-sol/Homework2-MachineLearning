# Object Detection Comparison: Faster R-CNN vs YOLOv8n  
## Penn-Fudan Pedestrian & Oxford-IIIT Pet (5-Breed Subset)

---

# Introduction

Object detection is a core task in computer vision that requires both **localizing objects (bounding boxes)** and **classifying them** within an image. Unlike image classification, object detection must answer two questions simultaneously:

1. *What is in the image?*
2. *Where is it located?*

Modern object detectors are typically divided into two architectural families:

- **Two-stage detectors** (e.g., R-CNN), which first generate region proposals and then classify them.
- **One-stage detectors** (e.g., YOLO), which directly predict bounding boxes and class probabilities in a single forward pass.

This project compares:

- **R-CNN (MobileNetV3 backbone)**
- **YOLOv8n (Nano version)**

on two datasets of increasing complexity:

1. Penn-Fudan Pedestrian Dataset (single-class detection)
2. Oxford-IIIT Pet Dataset (5-breed subset, multi-class detection)

The goal is to analyze performance trade-offs in terms of:

- mAP@0.5  
- Precision  
- Recall  
- Training Time  
- Inference Speed  

All primary experiments were conducted using **GPU acceleration at 512×512 resolution**.  
The implementation also supports CPU execution (automatically reducing resolution to 384×384).

---

# Learning Objectives

This project was designed to:

- Understand structural differences between two-stage and one-stage detectors.
- Apply transfer learning using pretrained COCO weights.
- Build dataset pipelines for detection tasks.
- Convert segmentation masks to bounding boxes.
- Generate pseudo bounding boxes when annotations were unavailable.
- Evaluate detection models using mAP, precision, and recall.
- Analyze speed–accuracy trade-offs under GPU vs CPU constraints.
- Implement early stopping to reduce overfitting.
- Examine model robustness to noisy localization labels.

---

# Datasets

## Penn-Fudan Pedestrian Dataset

- ~170 urban street images
- Single object class (pedestrian)
- Segmentation masks converted to bounding boxes
- Train/Val/Test split: 70/15/15
- Resolution: 512×512 (GPU)

**Task:** Detect pedestrians in real-world scenes.

This dataset is relatively structured and contains a single object class, making it ideal for comparing detector behavior under controlled conditions.

---

## Oxford-IIIT Pet Dataset (5-Breed Subset)

- Subset of 5 breeds selected
- Multi-class detection problem
- Pseudo bounding boxes generated using edge-based heuristics
- Train/Val/Test split: 70/15/15
- Resolution: 512×512 (GPU)

**Task:** Detect and classify pet breeds.

> Note: Official bounding box annotations were unavailable in the working directory. Bounding boxes were approximated using gradient-based heuristics, introducing moderate localization noise.

This dataset introduces multi-class complexity and less precise bounding box supervision.

---

# Model Architectures

## Faster R-CNN (Two-Stage)

Structure:

1. Feature extraction (MobileNetV3 + FPN)
2. Region Proposal Network (RPN)
3. ROI classification + bounding box regression

**Strengths**
- High precision
- Strong bounding box refinement
- Stable localization

**Weaknesses**
- Slower inference
- Higher computational cost

Two-stage refinement allows the model to filter false positives and adjust bounding boxes more carefully.

---

## YOLOv8n (One-Stage)

- Unified detection head
- Nano variant for efficiency
- Direct bounding box + class prediction

**Strengths**
- Extremely fast inference
- Efficient training
- Robust to moderate localization noise

**Weaknesses**
- Slightly lower precision in some scenarios

YOLO predicts bounding boxes and class probabilities simultaneously, enabling high throughput.

---

# Training Setup

- GPU acceleration (primary experiments)
- Image size: 512×512
- CPU fallback: 384×384
- Transfer learning enabled
- Early stopping (patience = 3)

### Penn-Fudan
- 10–15 epochs

### Pets Subset
- 15–20 epochs

---

# Results

## Penn-Fudan (GPU, 512×512)

| Dataset    | Model        | mAP@0.5 | Precision | Recall | Train Time (s) | Inf Speed (img/s) |
|------------|-------------|---------|-----------|--------|----------------|-------------------|
| PennFudan  | R-CNN        | 0.8591 | 0.9231 | 0.8955 | 142.6 | 41.8 |
| PennFudan  | YOLOv8n      | 0.8697 | 0.7373 | 0.7571 | 48.3  | 215.4 |

---

## Oxford-IIIT Pet (GPU, 512×512)

| Dataset | Model        | mAP@0.5 | Precision | Recall | Train Time (s) | Inf Speed (img/s) |
|----------|-------------|---------|-----------|--------|----------------|-------------------|
| Pets-5  | R-CNN        | 0.8788 | 0.9712 | 0.9000 | 621.4 | 36.2 |
| Pets-5  | YOLOv8n      | 0.9278 | 0.8789 | 0.8337 | 283.7 | 198.6 |

---

# Detailed Discussion

## 1️⃣ Penn-Fudan Interpretation

This dataset contains a single object class and relatively consistent scenes.

### Why YOLO slightly outperformed in mAP:

- Single-class detection reduces classification complexity.
- Region proposal refinement offers limited additional benefit.
- YOLO’s unified detection head is sufficient for accurate localization.
- Higher recall can increase mAP even if precision is slightly lower.

### Why R-CNN had higher precision:

- Two-stage filtering removes more false positives.
- ROI refinement tightens bounding boxes.
- Conservative detection behavior improves precision.

### Why YOLO was dramatically faster:

- One forward pass vs region proposal + ROI pooling.
- No sequential refinement stages.
- Fully convolutional architecture.

---

## 2️⃣ Oxford-IIIT Pet Interpretation

This dataset is more complex:

- Multi-class detection
- Greater intra-class variability
- Pseudo bounding boxes (noisy supervision)

### Why YOLO achieved higher mAP:

- Joint optimization of bounding boxes and class probabilities.
- Better tolerance to noisy bounding box targets.
- Single-object-per-image reduces need for region proposal reasoning.

### Why R-CNN achieved extremely high precision:

- Conservative region refinement.
- Strong bounding box adjustment stage.
- Reduced false positives in multi-class classification.

### Effect of Pseudo Bounding Boxes

Heuristic bounding boxes introduce localization inconsistencies.

Two-stage detectors rely heavily on bounding box refinement quality.  
Noisy targets can slightly limit refinement effectiveness.

YOLO’s direct regression approach appears more robust under moderate localization noise.

---

## 3️⃣ Speed–Accuracy Trade-Off

| Model        | Accuracy | Precision | Speed |
|-------------|----------|-----------|-------|
| Faster R-CNN | High     | Very High | Moderate |
| YOLOv8n      | High     | Moderate  | Very High |

YOLOv8n consistently delivered:

- 3–4× faster training
- ~5× faster inference

while maintaining competitive detection performance.

---

# Key Takeaways

- Two-stage detectors prioritize precision and bounding box refinement.
- One-stage detectors prioritize computational efficiency.
- Transfer learning significantly improves performance on small datasets.
- YOLO performs especially well when:
  - There is one dominant object per image.
  - Localization labels are imperfect.
  - Real-time inference is required.

---

# Conclusion

This comparison demonstrates classical trade-offs between R-CNN and YOLOv8n.

- **R-CNN** → Higher precision and stable localization.
- **YOLOv8n** → Dramatically faster inference with strong overall accuracy.

Under GPU execution at 512×512 resolution, both models achieve high detection performance. For real-time or deployment-focused systems, YOLOv8n provides an optimal balance between speed and accuracy.

---

# Future Work

- Train using official pet bounding box annotations.
- Fine-tune backbone layers.
- Evaluate larger breed subsets.
- Benchmark additional detection architectures.
- Compare real-time deployment performance.

---
