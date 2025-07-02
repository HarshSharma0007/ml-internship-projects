# 🐾 Animal Image Classification – Version 2

> A deep learning pipeline for fine-grained, multi-class image classification using ResNet, EfficientNet, and ensemble learning. Built from scratch with modular design and a strong emphasis on performance, explainability, and ML engineering principles.

---

## 🔧 Key Features
- 📦 **Transfer Learning** with ResNet and EfficientNet backbones  
- 🔄 **Data Augmentation** using Albumentations  
- 🧠 **Ensemble Modeling** via soft voting  
- 🔍 **Grad-CAM Explainability** for interpretability  
- 📊 **Stratified Splitting & Summary Reports** to audit data distribution  
- 🧪 **MLflow** for experiment tracking

---

## 📂 Project Structure

```
├── data/
│   ├── raw/         # Original dataset (class-wise folders)
│   ├── processed/   # Train / Val / Test splits after stratification
│   └── splits/      # (Optional) CSVs tracking split assignments
├── notebooks/
│   └── 01_data_split_and_explore.ipynb  # Stratified split + distribution plots
├── reports/
│   └── class_distribution_summary.csv   # Count of images per class per split
│    
├── src/
│   ├── config.py
│   ├── dataloader.py
│   ├── models/
│   ├── train.py
│   └── ...
└── requirements.txt
```

---

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/animal-classifier-v2.git
   cd animal-classifier-v2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the preprocessing notebook:
   ```bash
   jupyter notebook notebooks/01_data_split_and_explore.ipynb
   ```
