# train_cnn.py
# Fine-tune ResNet18 on DF20 images for binary edibility classification
#
# Loads the 45-species filtered DF20 dataset from dataset.py
# replaces ResNet18's final fully-connected layer with a binary classifier
# fine-tunes on MPS.
#
#
# Outputs saved to results/cnn/:
#   best_resnet18.pth — best checkpoint by val loss
#   cnn_training_history.png — loss + accuracy curves
#   cnn_test_metrics.json — overall test set metrics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

from src.vision.dataset import build_dataloaders

# use mps for my laptop
def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# Load ResNet18 pretrained on ImageNet — downloads weights on first run
# The final FC layer (fc) is replaced with a binary classifier
# This maps ResNet18's 512-dim feature vector to a single edibility probability
# The original fc was Linear(512 → 1000) for ImageNet's 1000 classes.
def build_model(device: str, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freeze_backbone=True: only the new FC layer is trained. Faster but less accurate.
    # freeze_backbone=False: all layers fine-tuned. Slower but better accuracy.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        # Dropout(0.3)
        nn.Dropout(0.3),
        # Linear(512 → 1)
        nn.Linear(in_features, 1),
        # Sigmoid
        nn.Sigmoid()
    )

    return model.to(device)

# One full pass through the training set
def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple:
    # model.train() enables Dropout and BatchNorm training behaviour
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        # unsqueeze(1) reshapes labels from [batch] to [batch, 1] to match the model's output shape from the single-output FC layer
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # total_loss is accumulated as loss * batch_size so we can average correctly across batches of different sizes
        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5) == labels.bool()).sum().item()
        total += len(labels)

    return total_loss/total, correct/total

# Evaluate on val or test set
# @torch.no_grad() disables gradient tracking for the entire function
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    # model.eval() disables Dropout and uses BatchNorm running stats
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # The _ discards species names
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        preds = model(imgs)
        loss = criterion(preds, labels)

        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5) == labels.bool()).sum().item()
        total += len(labels)

    return total_loss/total, correct/total

# Two-panel training curve — same structure as the MLP history plot in evaluate.py. Left: BCE loss. Right: accuracy.
def plot_training_history(history: dict, save_dir: str) -> None:
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss'])+1)

    ax1.plot(epochs, history['train_loss'], label='Train', color='#2196F3', lw=2)
    ax1.plot(epochs, history['val_loss'], label='Val', color='#F44336', lw=2)
    ax1.set_title('ResNet18 — Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', color='#2196F3', lw=2)
    ax2.plot(epochs, history['val_acc'], label='Val', color='#F44336', lw=2)
    ax2.set_title('ResNet18 — Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_training_history.png", dpi=150)
    plt.close()
    print("Training history saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on DF20 edibility")
    parser.add_argument('--metadata', type=str, default='data/DF20-train_metadata_PROD-2.csv')
    parser.add_argument('--image-root', type=str, default='data/DF20_300')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=224)
    # --no-save: useful during debugging so disk isn't filled with checkpoints
    parser.add_argument('--no-save', action='store_true', help='Skip saving model checkpoint')
    # --freeze: trains only the FC head — good for a 2-3 min sanity check
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone, train FC only')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    save_dir = "results/cnn"
    os.makedirs(save_dir, exist_ok=True)

    # Build all three DataLoaders from the DF20 metadata CSV and image folder
    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv=args.metadata,
        image_root=args.image_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Build model — downloads ImageNet weights on first run
    model = build_model(device, freeze_backbone=args.freeze)
    criterion = nn.BCELoss()

    # filter(lambda p: p.requires_grad, ...) ensures frozen parameters are excluded from the optimizer 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    # Halve LR after 3 epochs of no val_loss improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet18")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Backbone frozen: {args.freeze}")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_path = f"{save_dir}/best_resnet18.pth"
    t0 = time.time()

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'LR':>8}")
    print("-"*60)

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>9.4f} {val_acc:>8.4f} {current_lr:>8.2e}")

        # Save checkpoint only when val_loss improves
        if val_loss < best_val_loss and not args.no_save:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Load best checkpoint for final evaluation 
    if not args.no_save and Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint from '{best_path}'")

    # Overall test set metrics
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest set — Loss: {test_loss:.4f}  Accuracy: {test_acc:.4f}")

    # Save metrics JSON
    metrics = {
        'test_loss': round(test_loss, 4),
        'test_accuracy': round(test_acc, 4),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': device,
        'train_time_min': round(elapsed/60, 2),
                }
    with open(f"{save_dir}/cnn_test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to '{save_dir}/cnn_test_metrics.json'")

    plot_training_history(history, save_dir)

    print(f"\nCNN training complete. Results in '{save_dir}/'")


if __name__ == '__main__':
    main()