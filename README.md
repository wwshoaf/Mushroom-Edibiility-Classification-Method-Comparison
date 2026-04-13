# 🍄 Mushroom Classification — ML/DL Comparison
### Masters-Level Pattern Recognition Project

Compares classical ML, ensemble, and deep learning methods on the
[UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
(8,124 samples · 22 categorical features · binary: edible vs. poisonous).

---

## Models Compared

| Category        | Model                          |
|-----------------|-------------------------------|
| Baseline        | Logistic Regression, Naive Bayes |
| Classical ML    | Decision Tree, SVM (RBF), k-NN  |
| Ensemble        | Random Forest, Gradient Boosting, XGBoost |
| Deep Learning   | MLP — PyTorch (256→128→64→1)    |

---

## Project Structure

```
mushroom_project/
├── main.py               ← Full pipeline runner
├── requirements.txt
├── src/
│   ├── preprocess.py     ← EDA, encoding, train/val/test split
│   ├── models.py         ← sklearn models + PyTorch MLP
│   └── evaluate.py       ← metrics, plots, CSV export
└── results/
    ├── test_metrics.csv
    └── figures/
        ├── class_distribution.png
        ├── feature_distributions.png
        ├── correlation_heatmap.png
        ├── confusion_matrices.png
        ├── roc_curves.png
        ├── metric_comparison.png
        ├── mlp_training_history.png
        └── feature_importance.png
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

The dataset is fetched **automatically** via `ucimlrepo` (id=73) — no manual download needed.

### Options
```
--epochs N      MLP training epochs (default: 60)
--cv N          Cross-validation folds (default: 5)
--no-eda        Skip EDA plots
--no-mlp        Skip MLP training
```

---

## Pipeline Steps

1. **Load** — parse UCI format, replace `?` missing values
2. **EDA** — class balance, missing values, feature distributions, correlations
3. **Preprocess** — one-hot encoding, 70/15/15 train/val/test split (stratified)
4. **Cross-Validate** — stratified k-fold on sklearn models (accuracy, F1, AUC)
5. **Train** — fit all sklearn models on full training set
6. **Train MLP** — PyTorch feedforward net with BatchNorm, Dropout, LR scheduler
7. **Evaluate** — test-set metrics for all models
8. **Visualize** — confusion matrices, ROC curves, metric bars, feature importance

---

## Key Design Decisions

- **One-Hot Encoding** chosen over label encoding to avoid spurious ordinal relationships in categorical features.
- **`stalk-root`** missing values (~30%) imputed with mode (the feature is retained as it has signal).
- **`veil-type`** dropped — only one unique value, zero information.
- **Safety note**: False negatives (predicting edible when poisonous) are critically dangerous — pay close attention to recall for the poisonous class.

---

## Evaluation Metrics

| Metric     | Why it matters here |
|------------|---------------------|
| Accuracy   | Overall correctness |
| Precision  | Of predicted poisonous, how many really are? |
| Recall     | Of actual poisonous, how many do we catch? ⚠️ |
| F1         | Harmonic mean — balanced measure |
| ROC-AUC    | Discrimination across all thresholds |