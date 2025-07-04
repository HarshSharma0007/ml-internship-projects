# 🐾 Animal Image Classification – Version 2

> A full-stack deep learning solution for multi-class animal image classification using ResNet18, Captum explainability, and beautiful interactive frontends in **Streamlit** and **Gradio**. Built for performance, transparency, and deployment-readiness. Built from scratch with modular design and a strong emphasis on performance, explainability, and ML engineering principles.

---

## 🌟 Project Highlights

- 🧠 **ResNet18 Transfer Learning** with fine-tuned final layers  
- 🖼️ **15 Animal Classes** including 🐯 Tiger, 🐼 Panda, 🐶 Dog, 🐘 Elephant, and more  
- 📈 **93.15% Accuracy** on validation set  
- ✨ **Explainability** via **Captum** (Integrated Gradients + Saliency)  
- 🚀 **Dual Frontends**:
  - `Streamlit` for local dashboard presentation  
  - `Gradio` for fast hosted demo or colab-style interface  
- 📁 **Modular Design**: train/test/infer pipelines decoupled and reusable  
- 📊 **Tracked Evaluation**: class-wise metrics, heatmaps, Top-3 probabilities  

---

## 🔧 Key Features
- 📦 **Transfer Learning** with ResNet and EfficientNet backbones  
- 🔄 **Data Augmentation** using Albumentations  
- 🧠 **Ensemble Modeling** via soft voting  
- 🔍 **Grad-CAM Explainability** for interpretability  
- 📊 **Stratified Splitting & Summary Reports** to audit data distribution  
- 🧪 **MLflow** for experiment tracking

---

## 🧱 Tech Stack

| Type            | Tools / Libraries                        |
|------------------|-------------------------------------------|
| Language         | Python 3.10+                              |
| Frameworks       | PyTorch (ResNet18), TorchVision           |
| Explainability   | Captum (IG, Saliency), Matplotlib         |
| Image Handling   | Pillow, OpenCV, Albumentations            |
| UI / UX          | Streamlit, Gradio                         |
| Data Utils       | scikit-learn, Pandas, Pillow              |
| DevOps           | Git, Github                                       |
| Tracking         | MLflow, CSV summaries                     |
| Deployment       | Gradio share link or Streamlit local app  |

---

## 📂 Project Structure

```
├── data/
│   ├── raw/          # Original dataset (class-wise folders)
│   ├── processed/    # Train / Val / Test splits after stratification
│   │    ├── test/
│   │    ├── train/
│   │    └── val/   
│   └── splits/       # CSVs tracking split assignments
│
├── models/           # Saved model weights (.pt)
│
├── notebooks/
│   ├── 01_data_split_and_explore.ipynb       # Stratified split + distribution plots
│   ├── 02_dataloader_test.ipynb
│   ├── 03_train_in_notebook.ipynb
│   ├── 04_evaluate_model.ipynb
│   ├── 05_project_summary.ipynb
│   ├── 06_two_models_evaluation.ipynb
│   └── 07_captum_explainability.ipynb
│
├── reports/
│   ├──  explainability/                      # Captum Attribution images per class
│   ├──  Model1/                              # Model with high accuracy
│        ├── metrics.txt
│        ├── per_class_metrics.csv
│        └── class_distribution_summary.csv   # Count of images per class per split
│   ├──  metrics.txt
│   ├──  per_class_metrics.csv
│   └── class_distribution_summary.csv        # Count of images per class per split
│    
├── src/
│   ├── interpret/
│       └── captum_visualizer.py
│   ├── models/ 
│       ├── effnet.py
│       └── resnet.py
│   ├── utils/
│       └── helpers.py
│   ├── init.py
│   ├── config.py
│   ├── dataloader.py
│   ├── ensemble.py
│   ├── evaluate.py
│   ├── inference.py
│   └── train.py
│
├── streamlit_app/
│   ├── assets/
│       └── sample_images/
│   ├── models/ 
│   ├── utils/
│       ├── inference.py     # Prediction wrapper
│       └── captum_utils.py  # Attribution backend
│   ├── Home.py              # Streamlit UI
│   └── requirements.txt
│
├── gradio_app/
│   ├── examples/
│       └── sample_images/
│   ├── models/ 
│   ├── utils/
│       └── infer_example.py     # Gradio backend interface
│   └── app.py                   # Gradio frontend launcher
│   
├── requirements.lock
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

- 📚 Architecture: ResNet18 with final layer adapted to 15-class softmax  
- 🏋️‍♂️ Training: 20 epochs with ImageNet-sized inputs (224×224)  
- 📈 Achieved 93.15% validation accuracy  
- 🧠 Saved model: `models/resnet18_20250702_155546.pt`  
- 🔎 Loss: CrossEntropyLoss | Optimizer: Adam | Scheduler: StepLR  
- 🔄 Augmentations via **Albumentations** during training



## 🧪 Captum-Based Explainability

- Implemented using `captum.attr` and custom renderer `plot_attributions`
- Attribution methods:
  - **Integrated Gradients** – smooth, interpretable
  - **Saliency Maps** – fast, gradient-based
- Works across both Streamlit and Gradio interfaces



## 📊 Streamlit Dashboard

Launch locally for a highly polished dashboard experience:

### ▶️ Start the app:

```bash
cd streamlit_app
streamlit run Home.py
```

### Features:
- File uploader + sample selector
- Prediction panel with emoji-tagged Top-3 classes
- Class-wise confidence bars
- Attribution heatmap side-by-side
- Integrated / Saliency toggle via sidebar
- Reset + minimal UX design



## 🌐 Gradio Interface

For quick sharing or integration with Hugging Face / notebooks / colab:

### ▶️ Start the app:

```bash
python app.py
```

### Features:
- Input: Upload or pick sample image
- Attribution method toggle (IG / Saliency)
- Outputs: Top-3 confidence map + Explanation heatmap
- Compatible with Hugging Face Spaces deployment




## 🏁 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/HarshSharma0007/ml-internship-projects.git
cd Project_1_Image_Classification_of_Animals_v2
```

### 2. Create environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

---