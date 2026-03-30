# 🧠✨ Neoplastic Disorder Classification Model 🚀

> A visually rich, beginner-friendly deep learning demo for brain tumor classification using synthetic MRI data.

---

## 📌 Overview

Welcome to the **Neoplastic Disorder Classification Model** — a proof-of-concept Jupyter Notebook designed to simulate how deep learning can classify brain MRI scans into **four major tumor categories**:

| 🏷️ Class      | 🧾 Description             |
| -------------- | -------------------------- |
| **glioma**     | Tumors from glial cells    |
| **meningioma** | Tumors from brain meninges |
| **notumor**    | Healthy brain (no tumor)   |
| **pituitary**  | Tumors in pituitary gland  |

💡 This project is ideal for:

* 🧪 Research prototyping
* 🎓 Educational demos
* 📊 Visualization practice
* 🚀 Pre-training pipeline testing

---

## ⚙️ Core Components

### 🧩 Synthetic Dataset

* 📐 Image Size: **150 × 150 px**
* 🖼️ Format: PNG (grayscale-style MRI simulation)
* 📦 Total Images: **400**
* ⚖️ Balanced Classes: 50 images per class (train/test)

### 🧪 Environment

* Optimized for **Google Colab**
* Default path: `/content/data/`

### 📚 Dependencies

```bash
tensorflow
matplotlib
scikit-learn
seaborn
pillow
numpy
```

### 🎯 Focus

> ⚠️ This is a **simulation-only demo**
> No real training — only synthetic data + mock results for visualization.

---

## ✨ Features

### 🔄 Automatic Dataset Generation

* Creates realistic random MRI-like images
* Builds directory structure automatically:

```
train/test/{glioma, meningioma, notumor, pituitary}
```

### 📊 Stunning Visualizations (Auto-Saved)

Includes **6+ publication-ready figures**:

* 🎨 Sample image grid
* 📊 Class distribution chart
* 📉 Training vs Validation accuracy
* 📉 Loss curves
* 🔢 Confusion matrix (raw + heatmap)
* 📈 ROC curves with AUC scores

---

## 🚀 Quick Start (Google Colab)

### 1️⃣ Open Notebook

Upload or open:

```
neoplastic_disorder_model.ipynb
```

### 2️⃣ Install Dependencies

```python
!pip install tensorflow matplotlib scikit-learn pillow seaborn numpy
```

### 3️⃣ Run All Cells ⏱️

* Execution time: **~5–10 seconds**
* Outputs generated automatically:

📁 Dataset:

```
/content/data/
```

🖼️ Figures:

```
Figure1_SampleImages.png → Figure_ROC.png
```

---

## 📂 Dataset Structure

```
/content/data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── test/
    └── (same structure)
```

📌 Each folder contains **50 images per class**

---

## 📈 Simulated Results

### 🎯 Training Performance (15 Epochs)

| Metric       | Start | End |
| ------------ | ----- | --- |
| Accuracy     | 60%   | 87% |
| Val Accuracy | 55%   | 84% |
| Loss         | 1.2   | 0.4 |
| Val Loss     | 1.3   | 0.5 |

---

### 🔢 Confusion Matrix

```
              Predicted
           glioma menin notum pit
Actual
glioma       18     1     0     1
meningioma    2    16     1     1  
notumor       1     2    15     2
pituitary     0     1     2    17
```

✔️ Balanced predictions
✔️ Strong diagonal performance

---

### 📈 ROC Curve

* 🔥 AUC Score: **~0.90+ for all classes**
* Indicates excellent classification capability (simulated)

---

## 💡 Code Highlights

### 🧪 Synthetic Image Generator

```python
def create_image(path):
    img = Image.fromarray(
        np.random.randint(0, 255, (150,150), dtype=np.uint8)
    )
    img.save(path)
```

### 📊 Visualization Stack

* Matplotlib 📉
* Seaborn 🎨
* Clean, research-ready plots

---

## 🔮 Future Improvements

Take this project to the next level:

* 📥 Integrate real datasets (e.g., Kaggle Brain MRI)
* 🧠 Train actual CNN models (ResNet50, EfficientNet)
* 🔄 Apply data augmentation
* ⚡ GPU training for performance
* 🌐 Build web apps (Streamlit / Gradio)
* 📊 Cross-validation & ablation studies

---

## 📁 Project Files

| File                              | Description              |
| --------------------------------- | ------------------------ |
| `neoplastic_disorder_model.ipynb` | Main demo notebook       |
| `README.md`                       | Project documentation    |
| `Figure*.png`                     | Generated visualizations |

---

## 🌟 Final Note

This project is a **clean, fast, and visually appealing starting point** for anyone exploring medical imaging with deep learning.

> 💬 *Run it. Visualize it. Extend it.*
> 🚀 Then turn simulation into real-world impact.

---
