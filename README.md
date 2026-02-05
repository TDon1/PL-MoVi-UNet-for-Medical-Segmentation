# PL-MoVi-UNet:A Parallel Lightweight MobileNet-ViT Network with Heterogeneous Feature Alignment Fusion for Medical Segmentation

## Introduction

**PL-MoVi-UNet** is a lightweight, high-precision medical image segmentation model designed for resource-constrained clinical environments (e.g., mobile terminals, bedside devices). It features a parallel dual-stream architecture that synergizes the local inductive bias of **CNNs** (MobileNetV3) and the global long-range dependencies of **Transformers** (MobileViT).

**Key Contributions:**

* **Parallel Dual-Stream:** Simultaneously extracts local and global features.
* **DCAF Module:** Dual-stream Channel-Spatial Adaptive Fusion for effective feature alignment.
* **MLSP Module:** Multilevel Spatial Pyramid for capturing small lesions and fine details.
* **Extreme Efficiency:** Achieves **38x parameter reduction** compared to TransUNet, with only **2.45M Params** and **0.94G FLOPs**.

![*(Please replace `docs/architecture.png` with your Figure 1 architecture diagram)*](file:///C:/Users/tjl/Pictures/Typedown/607cc134-cea8-4b40-8400-e2f979a9bb3c.png)

---



## Datasets

This project uses the following datasets for training and evaluation. Please download them from the official links and organize them as described in the [Project Structure](#-project-structure) section.

### 1. LiTS17 (Liver Tumor Segmentation)

- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/andrewmvd/lits17-challenge-dataset) or [CodaLab](https://competitions.codalab.org/competitions/17094)
- **Description**: 3D CT scans for liver and tumor segmentation.
- **Preprocessing**: Run `python preprocess/preprocess_lits.py` (if you have this script) to convert raw NIfTI files to the required format.

### 2. QaTa-COV19v2 (COVID-19 X-ray)

- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)
- **Description**: Chest X-ray images with ground truth masks for COVID-19 infection areas.

### 3. AHJU-LCPS (Lung CT)

- **Note**: This is an in-house dataset used for lung parenchyma segmentation evaluation. Due to privacy regulations, this dataset is not publicly available. 

## Project Structure (项目结构)

Based on the provided file list, the repository is organized as follows:

```text
PL-MoVi-UNet/
├── Dataset/                  # Dataset configurations
├── models/                   # Model definitions (PL-MoVi-UNet, DCAF, MLSP)
├── preprocess/               # Data preprocessing scripts
├── results/                  # Segmentation masks and evaluation logs
├── runs/                     # Tensorboard logs and checkpoints
├── src/                      # Core training logic and config
├── utils/                    # Helper functions (metrics, logger, etc.)
├── DCAF_weight.py            # Script to analyze/visualize DCAF module weights
├── dataset.py                # Generic dataset loader
├── datasetAHJN-LCPS.py       # Specific loader for AHJN-LCPS dataset
├── get_param_flops.py        # Script to calculate model complexity (Params/FLOPs)
├── test_PL_MoViTUNet_LiTS.py # Inference script for LiTS17 dataset
├── train_PL_MoViTUNet_LiTS.py# Training script for LiTS17 dataset
└── visualize_stage.py        # Visualization of intermediate feature maps

```

---

## 🚀 Getting Started (快速开始)

### 1. Requirements

* Python >= 3.8
* PyTorch >= 1.10
* torchvision
* numpy
* scipy
* medpy (for medical metrics)

Install dependencies:

```bash
pip install -r requirements.txt

```

### 2. Data Preparation

Please organize your dataset (e.g., LiTS17) as follows:

```text
└── Dataset\
     └── LiTS\
           └── LiTS\
               ├── 0\              
               ├── 1\              
               └── 2\              

```

### 3. Training

To train PL-MoVi-UNet on the LiTS17 dataset:

```bash
python train_PL_MoViTUNet_LiTS.py --batch_size 16 --epochs 200 --lr 0.001

```

### 4. Evaluation & Inference

To test the model and generate segmentation results:

```bash
python test_PL_MoViTUNet_LiTS.py

```

### 5. Efficiency Analysis

Check the Parameters and FLOPs of the model:

```bash
python get_param_flops.py
```

### 6. Interpretability & Visualization

To visualize the DCAF weights or intermediate feature maps (as discussed in the paper's interpretability section):

```bash
# Analyze DCAF weight evolution
python DCAF_weight.py

# Visualize stage-wise feature maps
python visualize_stage.py 

```

![6c3b6a39-270f-4397-8a06-6c875e9bc523](file:///C:/Users/tjl/Pictures/Typedown/6c3b6a39-270f-4397-8a06-6c875e9bc523.png)
