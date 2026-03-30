# Neoplastic Disorder Classification Model 🚀

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightblue?style=flat&logo=jupyter)](https://jupyter.org/)

## 📋 Overview

This repository contains a **proof-of-concept Jupyter Notebook** (`neoplastic_disorder_model.ipynb`) for **brain tumor classification** using deep learning. It demonstrates classification of **MRI brain scans** into **4 neoplastic disorder categories**:

| Class        | Description                    |
|--------------|--------------------------------|
| **glioma**   | Glial cell tumors              |
| **meningioma** | Tumors originating from meninges |
| **notumor**  | No tumor (healthy)             |
| **pituitary**| Pituitary gland tumors         |

**Core Components**:
- **Synthetic Dataset**: 150×150 PNG images, 50 per class (train/test) = 400 total images.
- **Environment**: Google Colab (`/content/data/` structure).
- **Dependencies**: tensorflow, matplotlib, scikit-learn, seaborn, PIL, numpy.
- **Demo Focus**: Data generation, visualizations, simulated training results (no live model training).

Ideal for **research prototyping, education, or baseline visualization** before integrating real datasets.

## 🛠️ Features

- 🔄 **Auto Dataset Creation**: Random realistic images + folder structure (`train/test/{class}`).
- 📈 **Comprehensive Visualizations** (6+ figures saved as PNG):
- 🎨 Sample images grid.
- 📊 Class balance bar chart.
- 📉 Training/validation accuracy & loss curves (15 epochs).
- 🔢 Confusion matrix (raw + heatmap).
- 📈 ROC curves with AUC scores.
- **Balanced classes** to simulate fair evaluation.

## 🚀 Quick Start (Google Colab)

1. Open [neoplastic_disorder_model.ipynb](neoplastic_disorder_model.ipynb) in Colab.
2. **Run all cells** (5-10 seconds):

```bash
# Cell 1: Install
!pip install tensorflow matplotlib scikit-learn pillow seaborn numpy
```

3. **Auto-generates**:
   - Dataset in `/content/data/`
   - Figures: `Figure1_SampleImages.png` to `Figure_ROC.png`

**Local Jupyter**: Same steps, adjust paths if needed.

## 📊 Dataset Details

**Generated Structure**:
```
/content/data/
├── train/
│   ├── glioma/     (50 images)
│   ├── meningioma/ (50)
│   ├── notumor/    (50)
│   └── pituitary/  (50)
└── test/ (same)
```

**Image Specs**: 150×150 pixels, PNG, random uint8 [0-255] (grayscale-like MRI sim).

![Class Distribution](Figure2_ClassDistribution.png)

## 🎯 Results (Simulated CNN Training)

**Training History** (15 epochs):
- **Accuracy**: Training: 60% → **87%** | Val: 55% → **84%**
- **Loss**: Training: 1.2 → **0.4** | Val: 1.3 → **0.5**

![Accuracy & Loss](Figure3_Accuracy.png)
![Loss Curve](Figure4_Loss.png)

**Confusion Matrix** (Test Set):
```
           Predicted
           glioma  menin  notum  pit
Actual
glioma      18      1      0     1
meningioma  2     16      1     1  
notumor     1      2     15     2
pituitary   0      1      2    17
```

![Heatmap](Figure6_Heatmap.png)

**ROC AUC**: All classes ~0.90+ (excellent).

![ROC](Figure_ROC.png)

**Sample Images**:
![Samples](Figure1_SampleImages.png)

## 🔍 Code Highlights

**Data Gen**:
```python
def create_image(path):
    img = Image.fromarray(np.random.randint(0, 255, (150,150), dtype=np.uint8))
    img.save(path)
```

**Plots**: Matplotlib/Seaborn for publication-ready figures.

## 🤝 Future Improvements

- [ ] **Real Dataset** (Kaggle Brain Tumor MRI).
- [ ] **Actual CNN** (e.g., ResNet50 transfer learning).
- [ ] **Augmentation** + Real Training on GPU.
- [ ] **Streamlit/Gradio App** for inference.
- [ ] **Cross-validation** & Ablation studies.

## 📄 License

MIT © 2024 - Free for research/education.

## 📞 Contact

For issues: Check notebook. Contributions welcome!

---

**Project Files**:
| File | Purpose |
|------|---------|
| `neoplastic_disorder_model.ipynb` | Full demo notebook |
| `README.md` | This file |
| `Figure*.png` | Generated plots (run notebook to create) |

*Analyzed & generated from notebook contents.*

