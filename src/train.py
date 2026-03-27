"""
Training Script
================
End-to-end training and evaluation loop for the
Delhi NCR Land Cover Classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RGB_DIR
from data_preprocessing import prepare_data
from dataset import DelhiLandCoverDataset
from model import build_model, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model and return accuracy (%)."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return (correct / total) * 100


def main():
    """Main training pipeline."""
    # ---- Data Preparation ----
    X_train, X_test, y_train, y_test = prepare_data()

    train_dataset = DelhiLandCoverDataset(X_train, y_train, RGB_DIR)
    test_dataset = DelhiLandCoverDataset(X_test, y_test, RGB_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model Setup ----
    device = get_device()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n{'='*50}")
    print(f"Training on: {device}")
    print(f"Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print(f"{'='*50}\n")

    # ---- Training Loop ----
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    # ---- Save Model ----
    save_path = "land_cover_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
