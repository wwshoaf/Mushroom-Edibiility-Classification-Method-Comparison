# evaluate.py
# Evaluation metrics and plots for tabular classification models.
#
# Called by main.py after training
# Handles sklearn models and the PyTorch MLP.
#
# Functions:
#   evaluate_model — compute metrics dict for one model
#   evaluate_all — run all models on test set, return sorted DataFrame
#   plot_confusion_matrices — grid of confusion matrices for all models
#   plot_roc_curves — all ROC curves on one figure
#   plot_metric_comparison — grouped bar chart of all metrics
#   plot_mlp_history — MLP training/validation loss and accuracy curves
#   plot_feature_importance — horizontal bar chart for tree-based models
#   save_results_csv 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Used to convert binary labels back to readable strings in plot axis labels
LABEL_MAP = {0: 'Edible', 1: 'Poisonous'}


# helpers 
# Extract probability scores from any sklearn model
# Most models have predict_proba — returns P(class=1) from the second column
# SVM with probability=False uses decision_function instead, normalise it to [0,1], so ROC-AUC can still be computed
# the 1e-9 prevents division by zero when all scores are identical
def _get_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return model.predict(X).astype(float)

# Extract probability scores from the PyTorch MLP
# model.eval() disables Dropout so predictions are deterministic.=
# torch.no_grad() disables gradient tracking — saves memory during inference
# .cpu().numpy() moves the result off MPS/CUDA back to CPU for sklearn metrics
def _get_proba_mlp(model, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.FloatTensor(X).to(device)
        return model(xt).cpu().numpy()


# Core evaluation 
def evaluate_model(name: str, y_true: np.ndarray,
                   y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    # Compute all five metrics for one model and return as a dict
    # zero_division=0 prevents warnings when a class has no predicted samples
    # recall here is for the poisonous class (label=1)
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }

# Run all fitted models on the held-out test set and return a DataFrame sorted by F1 descending
# mlp_model is None if --no-mlp was passed
def evaluate_all(fitted_models: dict, mlp_model,
                 X_test: np.ndarray, y_test: np.ndarray,
                 device: str = 'cpu') -> pd.DataFrame:
    rows = []

    for name, (model, _) in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = _get_proba_sklearn(model, X_test)
        rows.append(evaluate_model(name, y_test, y_pred, y_proba))

    if mlp_model is not None:
        y_proba = _get_proba_mlp(mlp_model, X_test, device)
        y_pred = (y_proba >= 0.5).astype(int)
        rows.append(evaluate_model('MLP (Neural Net)', y_test, y_pred, y_proba))

    df = pd.DataFrame(rows).sort_values('f1', ascending=False).reset_index(drop=True)
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


# Plots 
# Grid of confusion matrices
# shows TP, FP, TN, FN counts in a 2x2 grid.
# poisonous predicted as edible is bottom left cell
# Unused subplots (when n_models % 3 != 0) are hidden
def plot_confusion_matrices(fitted_models: dict, mlp_model,
                            X_test: np.ndarray, y_test: np.ndarray,
                            save_dir: str = "results/figures",
                            device: str = 'cpu') -> None:
    os.makedirs(save_dir, exist_ok=True)

    all_models = {name: model for name, (model, _) in fitted_models.items()}
    if mlp_model is not None:
        all_models['MLP (Neural Net)'] = mlp_model

    n = len(all_models)
    # models are laid out in rows of 3
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*4))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, all_models.items()):
        if name == 'MLP (Neural Net)':
            y_proba = _get_proba_mlp(model, X_test, device)
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Edible', 'Poisonous'],
                    yticklabels=['Edible', 'Poisonous'],
                    linewidths=0.5, cbar=False)
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAcc={acc:.4f}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrices saved.")


# All ROC curves on one figure
# True Positive Rate vs False Positive Rate at every classification threshold
# 1.0 = perfect, 0.5 = random
# The dashed diagonal line represents a random classifier (AUC=0.5).
def plot_roc_curves(fitted_models: dict, mlp_model,
                    X_test: np.ndarray, y_test: np.ndarray,
                    save_dir: str = "results/figures",
                    device: str = 'cpu') -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap('tab10')

    all_models = {name: model for name, (model, _) in fitted_models.items()}
    if mlp_model is not None:
        all_models['MLP (Neural Net)'] = mlp_model

    for i, (name, model) in enumerate(all_models.items()):
        if name == 'MLP (Neural Net)':
            y_proba = _get_proba_mlp(model, X_test, device)
        else:
            y_proba = _get_proba_sklearn(model, X_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=cmap(i), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png", dpi=150)
    plt.close()
    print("ROC curves saved.")

# Grouped bar chart showing all 5 metrics side by side for every model
def plot_metric_comparison(results_df: pd.DataFrame,
                           save_dir: str = "results/figures") -> None:
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    n_metrics = len(metrics)
    models = results_df['model'].tolist()
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.15
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fig, ax = plt.subplots(figsize=(max(12, n_models*1.5), 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = results_df[metric].values
        ax.bar(x + i*width, vals, width, label=metric.upper(),
               color=color, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x + width*(n_metrics-1)/2)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Test Set Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metric_comparison.png", dpi=150)
    plt.close()
    print("Metric comparison plot saved.")

# Two-panel plot of MLP training progress
# Left: BCE loss — should decrease each epoch
# Right: accuracy — should increase
# The LR scheduler kicking in should be visible as a steeper drop in loss around epochs 10-15.
def plot_mlp_history(history: dict, save_dir: str = "results/figures") -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss'])+1)

    ax1.plot(epochs, history['train_loss'], label='Train', color='#2196F3', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val', color='#F44336', linewidth=2)
    ax1.set_title('MLP — Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', color='#2196F3', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val', color='#F44336', linewidth=2)
    ax2.set_title('MLP — Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.85, 1.01)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/mlp_training_history.png", dpi=150)
    plt.close()
    print("MLP training history saved.")

# Horizontal bar chart of top_n most important features for RF and XGBoost
# Features are sorted descending so the most important is at the top
# Reversed for barh (which plots bottom-to-top) so top = most important
def plot_feature_importance(fitted_models: dict, feature_names: list,
                            top_n: int = 20,
                            save_dir: str = "results/figures") -> None:
    os.makedirs(save_dir, exist_ok=True)

    tree_models = {name: model for name, (model, _) in fitted_models.items()
                   if hasattr(model, 'feature_importances_')}

    from sklearn.pipeline import Pipeline as SKPipeline
    for name, (model, _) in fitted_models.items():
        if isinstance(model, SKPipeline) and hasattr(model[-1], 'feature_importances_'):
            tree_models[name] = model[-1]

    if not tree_models:
        print("No tree-based models found for feature importance.")
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7*len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        ax.barh(range(top_n), top_importances[::-1], color='#1976D2',
                edgecolor='black', linewidth=0.4)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=8)
        ax.set_title(f"{name}\nTop {top_n} Features", fontweight='bold')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance plot saved.")

# save files
def save_results_csv(results_df: pd.DataFrame,
                     save_dir: str = "results") -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/test_metrics.csv"
    results_df.to_csv(path, index=False)
    print(f"Results saved to '{path}'")