
# models.py — Model definitions, training, and cross-validation.
# models included: Logistic Regression, SVM, kNN, Random Forest, XGBoost, and PyTorch MLP

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# PyTorch MLP 
# feedforward neural network
# input layer: 118 dimensions
# hidden layers: up to 256, then 128, then 64
# output layer: sigmoid output 1
# BatchNorm1d to normalize acitvations
# Dropout 0.3 to prevent overfitting
# output probability in range [0,1]
class MushroomMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: expand to 256 features
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: compress to 128 features
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: compress to 64 features
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output: probability in range[0,1]
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # squeeze(1) converts shape to match label shape
        return self.net(x).squeeze(1)

# train mlp
# return model, history dictionary, training time
# penalise large weights
# binary cross-entropy loss for binary classification
# ReduceLROnPlateau to prevent overfitting
# validation set for monitoring
# Params: 
# Parameters:
# X_train, y_train: training data (numpy arrays)
# X_val, y_val: validation data for loss monitoring and LR scheduling
# input_dim: number of input features
# epochs: number of passes through the training set
# batch_size: samples per gradient update
# lr: initial learning rate
# device: 'mps', best processing time for my mac
def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              input_dim: int, epochs: int = 50, batch_size: int = 128,
              lr: float = 1e-3, device: str = None) -> tuple:
    
    # Auto-detect best device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            print("mps not available")
            return 

    # Convert numpy arrays to PyTorch tensors and move to device
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    # DataLoader handles batching and shuffling each epoch
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model = MushroomMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Halve LR after 5 epochs of no val_loss improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t0 = time.time()

    for epoch in range(epochs):

        # Training pass
        # enables Dropout and BatchNorm training behaviour
        model.train()  
        batch_losses, batch_correct, batch_total = [], 0, 0

        for xb, yb in loader:
            # clear gradients from previous batch
            optimizer.zero_grad()
            # forward pass
            preds = model(xb)
            # compute BCE loss
            loss  = criterion(preds, yb)
            # backpropagate gradients
            loss.backward()
            # update weights
            optimizer.step()

            batch_losses.append(loss.item())
            batch_correct += ((preds >= 0.5) == yb.bool()).sum().item()
            batch_total   += len(yb)

        train_loss = np.mean(batch_losses)
        train_acc  = batch_correct / batch_total

        # Validation pass 
        # disables Dropout; uses running stats for BatchNorm
        model.eval()

        # no gradients needed — saves memory and compute
        with torch.no_grad():
            val_preds = model(Xv)
            val_loss = criterion(val_preds, yv).item()
            val_acc = ((val_preds >= 0.5) == yv.bool()).float().mean().item()

        # Step scheduler — reduces LR if val_loss plateaus
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    train_time = time.time() - t0
    return model, history, train_time

# returns a dictionary of model names to fitted sklearn estimators
# LR, SVM, kNN need StandardScaler
# RF, XGBoost don't
def get_sklearn_models() -> dict:
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                # should converge
                max_iter=5000,
                # prevents dominating weights
                C=0.1,
                # handles sparse data well
                solver='saga',   
                random_state=26))
        ]),

        # prob=True for AUC-ROC scores
        # rbf kernel to map to infinite dimensions
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=26))
        ]),

        'k-NN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ]),

        'Random Forest': RandomForestClassifier(
            # 200 trees
            n_estimators=200,
            # fully grown trees, shouldn't overfit anyway
            max_depth=None,   
            random_state=26,
            # use all cores
            n_jobs=-1),

        # default settings
        'XGBoost': XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            eval_metric='logloss',
            random_state=26,
            n_jobs=-1)
    }

    return models

# stratified k-fold cross-validation on training set
# returns cv_results whihc is a dict mapping model name to performance metrics
def cross_validate_models(models: dict, X_train: np.ndarray,
                          y_train: np.ndarray, cv: int = 5) -> dict:
    
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    cv_results = {}

    print(f"{'Model':<25} {'Acc':>7} {'±':>5} {'F1':>7} {'AUC':>7}")
    print(f"{'='*60}")

    for name, model in models.items():
        t0  = time.time()
        res = cross_validate(model, X_train, y_train, cv=skf,
                             scoring=scoring, n_jobs=1)
        elapsed = time.time() - t0

        cv_results[name] = {
            'mean_accuracy': res['test_accuracy'].mean(),
            'std_accuracy': res['test_accuracy'].std(),
            'mean_f1': res['test_f1'].mean(),
            'std_f1': res['test_f1'].std(),
            'mean_auc': res['test_roc_auc'].mean(),
            'std_auc': res['test_roc_auc'].std(),
            'mean_precision': res['test_precision'].mean(),
            'mean_recall': res['test_recall'].mean(),
            'cv_time_s': elapsed,
        }

        r = cv_results[name]
        print(f"{name:<25} {r['mean_accuracy']:>6.4f} {r['std_accuracy']:>6.4f} "
              f"{r['mean_f1']:>7.4f} {r['mean_auc']:>7.4f}")

    return cv_results

# fit models on full training set
# returns dict mapping model name to (fitted_model, train_time_seconds)
def train_all_sklearn(models: dict, X_train: np.ndarray,
                      y_train: np.ndarray) -> dict:

    fitted = {}
    print("\nFitting models on full training set...")

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        fitted[name] = (model, elapsed)
        print(f"  {name:<25} fit in {elapsed:.2f}s")

    return fitted