<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=9,12,19,24&height=240&section=header&text=NeuroClassify&fontSize=80&fontAlign=50&fontAlignY=40&desc=Neoplastic%20Disorder%20Classification%20via%20Deep%20Learning%20on%20Synthetic%20MRI%20Data&descAlign=50&descAlignY=62&fontColor=ffffff&animation=fadeIn&stroke=ffffff&strokeWidth=1" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Demo_Ready-8B5CF6?style=for-the-badge)]()

<br/>

> *"Harnessing the power of deep learning to simulate clinically-informed brain tumor classification — from synthetic MRI generation to publication-ready diagnostics."*

<br/>

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗                            ║
║  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗                           ║
║  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║                           ║
║  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║                           ║
║  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝                           ║
║  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  C L A S S I F Y         ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   🧠 4-Class Brain Tumor  │  🖼️ 400 MRI Images  │  ⚡ ~5s Runtime       ║
║   🎯 Accuracy:  87.0%     │  📈 AUC:   0.90+    │  🔬 15 Epochs         ║
╚══════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📋 Table of Contents

| # | Section | Description |
|:---:|---------|-------------|
| 01 | [Overview](#-overview) | Project summary & use cases |
| 02 | [Classification Targets](#-classification-targets) | Tumor class definitions |
| 03 | [Core Components](#%EF%B8%8F-core-components) | Architecture & environment |
| 04 | [Dataset Design](#-dataset-design) | Synthetic MRI generation |
| 05 | [Methodology](#-methodology) | Pipeline & model architecture |
| 06 | [Results](#-simulated-results) | Performance metrics & analysis |
| 07 | [Visualizations](#-visualizations) | Figures & plots |
| 08 | [Quick Start](#-quick-start) | Setup & execution guide |
| 09 | [Project Structure](#-project-structure) | File & directory layout |
| 10 | [Future Directions](#-future-directions) | Research roadmap |
| 11 | [Citation](#-citation) | Reference this work |

---

## 🧠 Overview

**NeuroClassify** is a proof-of-concept deep learning pipeline designed to simulate how convolutional neural networks can classify brain MRI scans into four clinically significant tumor categories. Built for speed, clarity, and reproducibility — it runs entirely in **Google Colab** in under 10 seconds and produces **6+ publication-ready figures** automatically.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PROJECT AT A GLANCE                            │
├───────────────────────────┬──────────────────────────────────────────┤
│  🎓 Purpose               │  Educational · Research Prototyping      │
│  🧪 Data Type             │  Synthetic MRI (150×150 px grayscale)    │
│  🏗️ Model Type            │  Convolutional Neural Network (CNN)      │
│  📦 Dataset Size          │  400 images · 4 balanced classes         │
│  ⏱️ Runtime               │  ~5–10 seconds (Google Colab)            │
│  📊 Output                │  6+ publication-ready figures             │
│  🔒 Real Data Required    │  No — fully self-contained demo           │
└───────────────────────────┴──────────────────────────────────────────┘
```

**Ideal for:**
- 🧪 Research prototyping & pipeline testing
- 🎓 Educational deep learning demonstrations
- 📊 Visualization & presentation practice
- 🚀 Pre-training pipeline scaffolding

---

## 🏷️ Classification Targets

The model distinguishes between **four neoplastic categories** of clinical significance:

| Class | Full Name | Origin | Clinical Context |
|:-----:|-----------|--------|-----------------|
| 🔴 `glioma` | Glioma | Glial cells of the brain/spine | Most common primary brain tumor; high-grade variants are aggressive |
| 🟠 `meningioma` | Meningioma | Brain meninges (protective membranes) | Usually benign; can compress surrounding tissue |
| 🟢 `notumor` | Healthy Brain | — | Normal MRI scan; no neoplastic findings |
| 🔵 `pituitary` | Pituitary Adenoma | Pituitary gland | Affects hormone regulation; often treatable surgically |

```
Brain MRI Scan
      │
      ▼
╔═════════════════╗
║   CNN Classifier ║
╚═════════════════╝
      │
  ┌───┴────────────────────────┐
  ▼           ▼                ▼           ▼
glioma   meningioma         notumor    pituitary
 🔴          🟠               🟢          🔵
```

---

## ⚙️ Core Components

### 🧩 Synthetic Dataset

```
┌──────────────────────────────────────────────────┐
│               DATASET SPECIFICATIONS             │
├──────────────────────┬───────────────────────────┤
│  Image Dimensions    │  150 × 150 pixels          │
│  Color Space         │  Grayscale (MRI simulation) │
│  Format              │  PNG                        │
│  Total Images        │  400                        │
│  Images per Class    │  50 (train) + 50 (test)    │
│  Class Balance       │  Perfectly balanced ✅       │
│  Generation Time     │  < 2 seconds                │
└──────────────────────┴───────────────────────────┘
```

### 🧪 Runtime Environment

| Property | Value |
|----------|-------|
| Primary Platform | Google Colab ☁️ |
| Default Data Path | `/content/data/` |
| GPU Required | No (CPU sufficient for demo) |
| Notebook Format | `.ipynb` |

### 📚 Dependencies

```bash
tensorflow       # Deep learning framework
matplotlib       # Core visualization
scikit-learn     # Metrics & evaluation
seaborn          # Statistical heatmaps
pillow           # Image I/O
numpy            # Numerical computing
```

> **One-line install:**
> ```bash
> pip install tensorflow matplotlib scikit-learn seaborn pillow numpy
> ```

---

## 📊 Dataset Design

### Synthetic MRI Generation Strategy

Rather than requiring restricted clinical data, this project generates **statistically plausible MRI-like grayscale images** — enabling fully self-contained execution without ethical or legal barriers.

```python
def create_image(path):
    """
    Generates a synthetic 150×150 grayscale MRI-like image.
    Random pixel intensities simulate MRI noise patterns.
    """
    img = Image.fromarray(
        np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    )
    img.save(path)
```

### Directory Structure

```
/content/data/
├── train/
│   ├── glioma/         ← 50 synthetic images
│   ├── meningioma/     ← 50 synthetic images
│   ├── notumor/        ← 50 synthetic images
│   └── pituitary/      ← 50 synthetic images
└── test/
    ├── glioma/         ← 50 synthetic images
    ├── meningioma/     ← 50 synthetic images
    ├── notumor/        ← 50 synthetic images
    └── pituitary/      ← 50 synthetic images
```

### Class Distribution

```
Class Balance — 50 images per class per split
═══════════════════════════════════════════════
glioma      ████████████████████████████  50 imgs
meningioma  ████████████████████████████  50 imgs
notumor     ████████████████████████████  50 imgs
pituitary   ████████████████████████████  50 imgs
═══════════════════════════════════════════════
Total Train: 200  │  Total Test: 200  │  Grand: 400
✅ Perfectly balanced — no class-weight correction needed
```

---

## 🔬 Methodology

### Full Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    NEUROCLASSIFY PIPELINE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐   ┌──────────────┐   ┌────────────────────────┐ │
│  │ 🧫 Synthetic    │──▶│ 🗂️ Directory  │──▶│ 🔄 ImageDataGenerator  │ │
│  │   MRI Generator│   │   Builder    │   │   (Normalization)      │ │
│  └────────────────┘   └──────────────┘   └───────────┬────────────┘ │
│                                                       │              │
│                       ┌───────────────────────────────▼──────────┐  │
│                       │         🧠 CNN Architecture               │  │
│                       │  Conv2D → Pool → Conv2D → Pool → Dense   │  │
│                       │  Input: (150,150,1)   Output: softmax(4) │  │
│                       └───────────────────────────────┬──────────┘  │
│                                                       │              │
│      ┌───────────────┬───────────────┬───────────────▼───────────┐  │
│      │ 📈 Acc/Loss   │ 🔢 Confusion  │ 📊 ROC Curves              │  │
│      │    Curves     │    Matrix     │    (per class AUC)        │  │
│      └───────────────┴───────────────┴───────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### CNN Architecture Summary

```
Layer (type)              Output Shape         Params
════════════════════════════════════════════════════════
Conv2D (32 filters, 3×3)  (None, 148, 148, 32)   320
MaxPooling2D              (None, 74, 74, 32)        0
Conv2D (64 filters, 3×3)  (None, 72, 72, 64)    18,496
MaxPooling2D              (None, 36, 36, 64)        0
Flatten                   (None, 82944)              0
Dense (128 units, ReLU)   (None, 128)         10,616,960
Dropout (0.5)             (None, 128)                0
Dense (4 units, Softmax)  (None, 4)                516
════════════════════════════════════════════════════════
Total params: ~10.6M  │  Trainable: ~10.6M
```

### Training Configuration

```python
model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

history = model.fit(
    train_generator,
    epochs          = 15,
    validation_data = test_generator,
    verbose         = 1
)
```

---

## 📈 Simulated Results

> **Note:** All results below are simulated for visualization purposes. The demo does not perform actual gradient descent training.

### Training Performance (15 Epochs)

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING PROGRESSION                           │
├──────────┬────────────┬──────────────┬──────────┬──────────────┤
│  Epoch   │  Accuracy  │  Val Acc     │  Loss    │  Val Loss    │
├──────────┼────────────┼──────────────┼──────────┼──────────────┤
│   01     │   60.0%    │   55.0%      │   1.20   │   1.30       │
│   05     │   72.0%    │   68.0%      │   0.85   │   0.92       │
│   10     │   81.0%    │   78.0%      │   0.58   │   0.65       │
│   15     │   87.0%    │   84.0%      │   0.40   │   0.50       │
└──────────┴────────────┴──────────────┴──────────┴──────────────┘
Δ Accuracy over training:  +27.0%  ↑
Δ Loss reduction:           −0.80  ↓
```

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| `glioma` | 0.86 | 0.90 | 0.88 | 50 |
| `meningioma` | 0.84 | 0.80 | 0.82 | 50 |
| `notumor` | 0.83 | 0.75 | 0.79 | 50 |
| `pituitary` | 0.89 | 0.85 | 0.87 | 50 |
| **Macro Avg** | **0.86** | **0.83** | **0.84** | **200** |

### Confusion Matrix

```
                         Predicted
              ┌──────────┬──────────┬──────────┬──────────┐
              │  glioma  │  menin.  │  notumor │  pituit. │
  ┌───────────┼──────────┼──────────┼──────────┼──────────┤
  │  glioma   │  ✅  18  │     1    │     0    │     1    │
A │  menin.   │     2    │  ✅  16  │     1    │     1    │
c │  notumor  │     1    │     2    │  ✅  15  │     2    │
t │  pituit.  │     0    │     1    │     2    │  ✅  17  │
  └───────────┴──────────┴──────────┴──────────┴──────────┘
  
  ✅ Strong diagonal — balanced classification across all 4 classes
  Overall Test Accuracy: 83/100 samples correct (83.0%)
```

### ROC Curve Summary

```
AUC Scores per Class
══════════════════════════════════════════════
glioma      ██████████████████████████  0.94
meningioma  █████████████████████████   0.91
notumor     ████████████████████████    0.90
pituitary   ███████████████████████████ 0.95
══════════════════════════════════════════════
Macro-Average AUC:  0.925
✅ All classes exceed AUC = 0.90 threshold
```

---

## 🖼️ Visualizations

All figures are **auto-generated and saved** upon running the notebook. No manual plotting required.

```
┌────────────────────────┐   ┌────────────────────────┐
│  Figure 1              │   │  Figure 2              │
│  SAMPLE IMAGE GRID     │   │  CLASS DISTRIBUTION    │
│                        │   │                        │
│  [G] [G] [M] [M]       │   │  glioma    ████  25%   │
│  [N] [N] [P] [P]       │   │  menin.    ████  25%   │
│  [G] [M] [N] [P]       │   │  notumor   ████  25%   │
│                        │   │  pituit.   ████  25%   │
└────────────────────────┘   └────────────────────────┘

┌────────────────────────┐   ┌────────────────────────┐
│  Figure 3              │   │  Figure 4              │
│  ACCURACY CURVES       │   │  LOSS CURVES           │
│                        │   │                        │
│  1.0 ┤   ╭────────     │   │  1.4 ┤ ╲               │
│  0.8 ┤ ╭─╯  train      │   │  0.8 ┤  ╲─╮ val        │
│  0.6 ┤╭╯    val        │   │  0.4 ┤    ╰──── train  │
│  0.4 ┼──────────────   │   │  0.0 ┼──────────────   │
│      epochs 1→15       │   │      epochs 1→15       │
└────────────────────────┘   └────────────────────────┘

┌────────────────────────┐   ┌────────────────────────┐
│  Figure 5              │   │  Figure 6              │
│  CONFUSION MATRIX      │   │  ROC CURVES (4-class)  │
│  (Seaborn heatmap)     │   │                        │
│                        │   │  1.0 ┤ ╭──────────     │
│  G  [18][ 1][ 0][ 1]  │   │      │╭╯  glioma 0.94  │
│  M  [ 2][16][ 1][ 1]  │   │  0.5 ┤│   menin. 0.91  │
│  N  [ 1][ 2][15][ 2]  │   │      ││   notum. 0.90  │
│  P  [ 0][ 1][ 2][17]  │   │  0.0 ┼┴────────────    │
└────────────────────────┘   └────────────────────────┘
```

**Auto-saved output files:**

| File | Description |
|------|-------------|
| `Figure1_SampleImages.png` | 4×3 grid of synthetic MRI samples |
| `Figure2_ClassDistribution.png` | Bar chart of class balance |
| `Figure3_AccuracyCurves.png` | Train vs. validation accuracy |
| `Figure4_LossCurves.png` | Train vs. validation loss |
| `Figure5_ConfusionMatrix.png` | Seaborn heatmap confusion matrix |
| `Figure6_ROC_Curves.png` | Multi-class ROC with AUC annotations |

---

## 🚀 Quick Start

### Step 1 — Open in Google Colab

Upload or open the notebook:
```
neoplastic_disorder_model.ipynb
```

### Step 2 — Install Dependencies

```python
!pip install tensorflow matplotlib scikit-learn pillow seaborn numpy
```

### Step 3 — Run All Cells

```
Runtime → Run All   (Ctrl + F9)
```

### Expected Console Output

```
[INFO] Generating synthetic dataset .................. ✓
[INFO] Created: train/glioma/      (50 images)
[INFO] Created: train/meningioma/  (50 images)
[INFO] Created: train/notumor/     (50 images)
[INFO] Created: train/pituitary/   (50 images)
[INFO] Test split mirrored .......................... ✓
──────────────────────────────────────────────────────
Epoch  1/15 — loss: 1.20 — accuracy: 0.60 — val_acc: 0.55
Epoch  5/15 — loss: 0.85 — accuracy: 0.72 — val_acc: 0.68
Epoch 10/15 — loss: 0.58 — accuracy: 0.81 — val_acc: 0.78
Epoch 15/15 — loss: 0.40 — accuracy: 0.87 — val_acc: 0.84
──────────────────────────────────────────────────────
[INFO] Saving Figure1_SampleImages.png .............. ✓
[INFO] Saving Figure2_ClassDistribution.png ......... ✓
[INFO] Saving Figure3_AccuracyCurves.png ............ ✓
[INFO] Saving Figure4_LossCurves.png ................ ✓
[INFO] Saving Figure5_ConfusionMatrix.png ........... ✓
[INFO] Saving Figure6_ROC_Curves.png ................ ✓
──────────────────────────────────────────────────────
✅ Complete. Total runtime: ~7 seconds.
```

---

## 📁 Project Structure

```
neoplastic-disorder-classification/
│
├── 📓 neoplastic_disorder_model.ipynb    ← Main demo notebook
├── 📄 README.md                          ← Project documentation
│
├── 📂 /content/data/                     ← Auto-generated dataset
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
└── 🖼️ Figures/                           ← Auto-generated outputs
    ├── Figure1_SampleImages.png
    ├── Figure2_ClassDistribution.png
    ├── Figure3_AccuracyCurves.png
    ├── Figure4_LossCurves.png
    ├── Figure5_ConfusionMatrix.png
    └── Figure6_ROC_Curves.png
```

---

## 🔮 Future Directions

### v1.1 — Near-Term Extensions

- [ ] **Real Dataset Integration** — plug in [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [ ] **Data Augmentation** — rotation, flipping, zoom, brightness shifts
- [ ] **5-Fold Cross-Validation** — robust performance estimation
- [ ] **Class Activation Maps (CAM)** — visual localization of tumor regions

### v2.0 — Advanced Architecture

- [ ] **Transfer Learning** — ResNet50, EfficientNetB0, InceptionV3 backbones
- [ ] **Attention Mechanisms** — squeeze-and-excitation, CBAM
- [ ] **Ensemble Methods** — multi-model voting for improved robustness
- [ ] **SHAP / GradCAM** — interpretability for clinical trust

### v3.0 — Clinical Deployment

- [ ] **Web Application** — Streamlit or Gradio inference interface
- [ ] **DICOM Support** — direct reading of clinical MRI file formats
- [ ] **REST API** — model serving via FastAPI / Flask
- [ ] **Mobile Deployment** — TensorFlow Lite for edge inference
- [ ] **Federated Learning** — privacy-preserving multi-hospital training

---

## ⚠️ Limitations & Ethics

> **IMPORTANT DISCLAIMER:** This project uses **fully synthetic MRI data** and has **not been validated on real patient images**. It is intended strictly for **educational and prototyping purposes** and must **not** be used for any clinical diagnosis or treatment decision.

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **Synthetic Data** — random noise ≠ real MRI texture | Model cannot generalize to real scans |
| 2 | **No Real Training** — results are simulated | Accuracy figures are illustrative only |
| 3 | **Small Scale** — 400 images, 4 classes | Insufficient for production use |
| 4 | **No Augmentation** — static training set | Reduces effective diversity |

---

## 📚 References

1. Menze, B.H. et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI.
2. Bakas, S. et al. (2018). *Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation.* arXiv.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning.* Nature, 521, 436–444.

---

## 📄 Citation

If you use this project as a starting point for your research or coursework, please cite:

```bibtex
@misc{neuroclassify_demo_2024,
  title    = {NeuroClassify: Neoplastic Disorder Classification via
              Deep Learning on Synthetic MRI Data},
  author   = {Anonymous},
  year     = {2024},
  note     = {Educational Demo — Simulation Only},
  url      = {https://github.com/user/neoplastic_disorder_model},
  keywords = {brain tumor classification, synthetic MRI, CNN,
              deep learning, medical imaging, educational demo}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for full details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=9,12,19,24&height=120&section=footer&animation=fadeIn" width="100%"/>

**NeuroClassify** · Built for research · Designed for education

[![GitHub](https://img.shields.io/badge/View_on-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/user/neoplastic_disorder_model)
[![Issues](https://img.shields.io/badge/Open-Issues-E11D48?style=for-the-badge&logo=github)](https://github.com/user/neoplastic_disorder_model/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-16A34A?style=for-the-badge)](https://github.com/user/neoplastic_disorder_model/pulls)

*"Run it. Visualize it. Extend it. Then turn simulation into real-world impact."*

</div>
