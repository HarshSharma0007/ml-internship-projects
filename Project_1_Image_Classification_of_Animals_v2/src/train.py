import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import mlflow

from src.dataloader import get_dataloaders
from src.config import NUM_CLASSES, NUM_EPOCHS, DEVICE, LEARNING_RATE, SAVE_BEST_ONLY, USE_MLFLOW, MODEL_DIR, BATCH_SIZE

# === Step 1: Model Setup === #
def get_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)
# === Step 2: Training Function === #
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct = 0.0, 0

    for images, labels in tqdm(loader, desc="🔁 Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct = 0.0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="🧪 Validation", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / len(loader.dataset)
    return val_loss, val_acc

from datetime import datetime
import os
os.makedirs(MODEL_DIR, exist_ok=True)
# === Step 3: Training Loop === #
def train():
    model = get_model()
    train_loader, val_loader, _ = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"resnet18_{timestamp}.pt"
    model_path = os.path.join(MODEL_DIR, model_name)

    if USE_MLFLOW:
        mlflow.set_experiment("animal_classifier_v2")
        mlflow.start_run()
        mlflow.log_params({
            "model": "resnet18",
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE
        }
        )

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n🔁 Epoch {epoch}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if USE_MLFLOW:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"📦 Saved new best model to: {model_path}")

    if USE_MLFLOW:
        mlflow.log_artifact(model_path)
        mlflow.end_run()

if __name__ == "__main__":
    train()
