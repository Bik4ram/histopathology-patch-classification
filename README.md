# Histopathology Patch Classification (Benign vs Malignant Tissue)

**2025 Project | Deep Learning | PyTorch | Grad-CAM**

## Overview

This project classifies histopathology image patches as **benign or malignant tissue** using deep convolutional networks (ResNet/EfficientNet), **balanced sampling**, and **focal loss** for handling class imbalance. It includes **Grad-CAM** visualizations for explainability.

**Key Results:**
- **AUC:** 0.94
- **F1:** 0.88

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability (Grad-CAM)](#explainability-grad-cam)
- [Results](#results)
- [References](#references)

## Dataset

- Place your patch images in `data/` folder, organized as:
  ```
  data/
    benign/
      img1.png
      img2.png
      ...
    malignant/
      img1.png
      img2.png
      ...
  ```

- Each class in a subfolder. Use PNG/JPG images.

## Installation

```bash
git clone https://github.com/yourusername/histopathology-patch-classification.git
cd histopathology-patch-classification
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --data_dir data --model resnet18 --epochs 20 --lr 1e-4 --batch_size 32
```

## Evaluation

```bash
python src/eval.py --data_dir data --model resnet18 --weights runs/best_model.pth
```

## Explainability (Grad-CAM)

```bash
python src/gradcam.py --img_path data/benign/img1.png --model resnet18 --weights runs/best_model.pth
```

Or use the interactive notebook:

```bash
jupyter notebook notebooks/explore_gradcam.ipynb
```

## Results

- **Training curve:** ![Training Curve](assets/training_curve.png)
- **ROC curve:** ![ROC Curve](assets/roc_curve.png)
- **Grad-CAM example:** ![Grad-CAM](assets/gradcam_example.png)

## References

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)# histopathology-patch-classification
