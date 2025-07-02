# src/generate_test_report.py

import os
import torch
import mlflow
import mlflow.pytorch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data_loader import get_test_loader
from src.utils import get_class_names
from src.config import DEVICE, TEST_DATA_DIR

def generate_report(model_uri="models:/AnimalClassifierModel/1", save_dir="reports"):
    os.makedirs(save_dir, exist_ok=True)
    class_names = get_class_names()
    model = mlflow.pytorch.load_model(model_uri)
    model.to(DEVICE)
    model.eval()

    # Load test data
    test_loader = get_test_loader(data_dir=TEST_DATA_DIR)
    y_true, y_pred = [], []

    # Run inference
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # === Metrics ===
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Save misclassified samples
    misclassified = [{"True Label": class_names[t], "Predicted Label": class_names[p]}
                     for t, p in zip(y_true, y_pred) if t != p]
    pd.DataFrame(misclassified).to_csv(os.path.join(save_dir, "misclassifications.csv"), index=False)

    # Log to MLflow
    with mlflow.start_run(run_name="Final Test Report"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_artifact(os.path.join(save_dir, "classification_report.txt"))
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(os.path.join(save_dir, "misclassifications.csv"))

    print(f"[✓] Accuracy: {acc:.4f}")
    print(f"[✓] Report saved to: {save_dir}/")

if __name__ == "__main__":
    generate_report()
