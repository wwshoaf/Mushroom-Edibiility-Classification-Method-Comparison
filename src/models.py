"""
Model definitions, training, and cross-validation for mushroom classification.
Covers: Logistic Regression, Naive Bayes, Decision Tree, SVM, k-NN,
        Random Forest, XGBoost, and MLP (PyTorch).
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Install with: pip install xgboost")

# ── PyTorch MLP ───────────────────────────────────────────────────────────────

class MushroomMLP(nn.Module):
    """
    Feedforward neural network for binary mushroom classification.
    Architecture: Input → 256 → 128 → 64 → 1 (sigmoid)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              input_dim: int, epochs: int = 50, batch_size: int = 128,
              lr: float = 1e-3, device: str = None) -> tuple:
    """
    Train the MLP and return (model, history_dict, train_time_seconds).
    history_dict keys: train_loss, val_loss, train_acc, val_acc
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tensors
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model = MushroomMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(epochs):
        # ── train ──
        model.train()
        batch_losses, batch_correct, batch_total = [], 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_correct += ((preds >= 0.5) == yb.bool()).sum().item()
            batch_total += len(yb)
        train_loss = np.mean(batch_losses)
        train_acc = batch_correct / batch_total

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_preds = model(Xv)
            val_loss = criterion(val_preds, yv).item()
            val_acc = ((val_preds >= 0.5) == yv.bool()).float().mean().item()

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    train_time = time.time() - t0
    return model, history, train_time


def get_sklearn_models() -> dict:
    """
    Return a dict of name → sklearn Pipeline (with scaler where appropriate).
    """
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Naive Bayes': BernoulliNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'k-NN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            eval_metric='logloss', random_state=42, n_jobs=-1)
    return models


def cross_validate_models(models: dict, X_train: np.ndarray,
                          y_train: np.ndarray, cv: int = 5) -> dict:
    """
    Run stratified k-fold CV on all sklearn models.
    Returns dict: name → {mean_accuracy, std_accuracy, mean_f1, std_f1, fit_time}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    cv_results = {}

    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'Acc':>7} {'±':>5} {'F1':>7} {'AUC':>7}")
    print(f"{'='*60}")

    for name, model in models.items():
        t0 = time.time()
        res = cross_validate(model, X_train, y_train, cv=skf,
                             scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        cv_results[name] = {
            'mean_accuracy': res['test_accuracy'].mean(),
            'std_accuracy':  res['test_accuracy'].std(),
            'mean_f1':       res['test_f1'].mean(),
            'std_f1':        res['test_f1'].std(),
            'mean_auc':      res['test_roc_auc'].mean(),
            'std_auc':       res['test_roc_auc'].std(),
            'mean_precision':res['test_precision'].mean(),
            'mean_recall':   res['test_recall'].mean(),
            'cv_time_s':     elapsed,
        }
        r = cv_results[name]
        print(f"{name:<25} {r['mean_accuracy']:>6.4f} {r['std_accuracy']:>6.4f} "
              f"{r['mean_f1']:>7.4f} {r['mean_auc']:>7.4f}")

    return cv_results


def train_all_sklearn(models: dict, X_train: np.ndarray,
                      y_train: np.ndarray) -> dict:
    """
    Fit all sklearn models on the full training set.
    Returns dict: name → (fitted_model, train_time_seconds)
    """
    fitted = {}
    print("\nFitting models on full training set...")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        fitted[name] = (model, elapsed)
        print(f"  {name:<25} fit in {elapsed:.2f}s")
    return fitted