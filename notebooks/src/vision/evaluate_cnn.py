# evaluate_cnn.py — Evaluation plots and metrics for the trained ResNet18 model
# Loads best_resnet18.pth and generates:
#   cnn_confusion_matrix.png — confusion matrix on test set
#   cnn_roc_curve.png — ROC curve with AUC
#   cnn_test_metrics.json — accuracy, AUC, false negatives, false positives
#
# All outputs overwrite existing files in results/cnn/.
#
# Used in:
#   python src/vision/evaluate_cnn.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score)

from src.vision.dataset import build_dataloaders

# mps for my laptop
def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# Rebuild the same architecture used in train_cnn.py and load saved weights
def load_model(checkpoint_path: str, device: str) -> nn.Module:
    # weights=None skips the ImageNet download since weights loaded from checkpoint
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Run full test set through model and collect predictions
# Returns y_true, y_pred (thresholded at 0.5), y_proba (raw probabilities)
@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    all_true, all_proba = [], []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        proba = model(imgs).squeeze(1).cpu().numpy()
        all_true.extend(labels.numpy().astype(int))
        all_proba.extend(proba.tolist())

    y_true = np.array(all_true)
    y_proba = np.array(all_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba

# Confusion matrix with false negatives annotated in the x-axis label.
# FN (bottom-left cell) is poisonous predicted as edible 
def plot_confusion_matrix(y_true, y_pred, save_dir: str) -> None:
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Edible', 'Not Safe'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('ResNet18 — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    fn = cm[1][0]
    ax.set_xlabel(f"Predicted label",fontsize=11, color='black')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: cnn_confusion_matrix.png")

# ROC curve with AUC score in the legend
def plot_roc_curve(y_true, y_proba, save_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#1565C0', lw=2, label=f'ResNet18 (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ResNet18 — ROC Curve (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cnn_roc_curve.png", dpi=150)
    plt.close()
    print("Saved: cnn_roc_curve.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet18 on DF20 test set")
    parser.add_argument('--checkpoint', default='results/cnn/best_resnet18.pth')
    parser.add_argument('--metadata', default='data/DF20-train_metadata_PROD-2.csv')
    parser.add_argument('--image-root', default='data/DF20_300')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    save_dir = "results/cnn"

    print("CNN EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Run: python src/vision/train_cnn.py --epochs 20 first.")

    _, _, test_loader = build_dataloaders(
        metadata_csv=args.metadata,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    print("\nRunning inference on test set...")
    y_true, y_pred, y_proba = run_inference(model, test_loader, device)

    accuracy = (y_true == y_pred).mean()
    auc = roc_auc_score(y_true, y_proba)
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    print(f"\n Overall Test Metrics ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\n Classification Report ")
    print(classification_report(y_true, y_pred, target_names=['Edible', 'Poisonous'], digits=4))
    print(f"False negatives (poisonous predicted edible): {fn}")

    metrics = {
        'test_accuracy': round(float(accuracy), 4),
        'test_roc_auc': round(float(auc), 4),
        'test_precision': round(float(precision_score(y_true, y_pred)), 4),
        'test_recall': round(float(recall_score(y_true, y_pred)), 4),
        'test_f1': round(float(f1_score(y_true, y_pred)), 4),
        'n_test_samples': int(len(y_true)),
        'false_negatives': fn,
        'false_positives': fp,
    }
    metrics_path = f"{save_dir}/cnn_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred, save_dir)
    plot_roc_curve(y_true, y_proba, save_dir)

    print(f"\nCNN evaluation complete. Results in '{save_dir}/'")


if __name__ == '__main__':
    main()