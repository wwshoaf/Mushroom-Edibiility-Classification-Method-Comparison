## 📄 Documentation
- [Project Report (PDF)](BDEProject_Report.pdf)
- [Presentation Slides (PDF)](MushroomClassificationPRESENETATION_Shoaf.pdf)
  
# Mushroom Edibility Classification: Method Comparison

Comparing tabular machine learning classifiers against CNN-based image classification for predicting mushroom edibility, restricted in both cases to features a forager could actually observe in the field.

## Research Question
How does CNN image classification compare to tabular classification when both are restricted to visually observable features?

## Datasets
- **Secondary Mushroom Dataset** (UCI ML Repository, id=848) — tabular data, fetched live via `ucimlrepo`. Only visually observable features are used (e.g. cap shape/color/surface, gill attachment/spacing/color, stem measurements/color/surface, ring type); non-visible features like spore-print color, stem-root, and habitat are excluded.
- **DF20 (Danish Fungi 2020)** — image dataset, filtered to the **45 species** that overlap with the Secondary dataset and have an unambiguous edibility label, with a minimum of 50 images per species.

Binary target for both: **edible** vs. **poisonous** (poisonous includes deadly, toxic, and conditionally-edible-only-when-cooked species).

## Pipeline

### 1. Tabular models (`main.py`)
- Loads and explores the Secondary dataset (class balance, missing values, feature distributions, correlation heatmap).
- Preprocesses: drops zero-variance and >50%-missing columns, imputes remaining missing values (median for continuous, mode for categorical), scales continuous features, one-hot encodes categorical features.
- Stratified 70/15/15 train/val/test split.
- Trains and 5-fold cross-validates: **Logistic Regression, SVM (RBF), k-NN, Random Forest, XGBoost**, plus a **PyTorch MLP** (3 hidden layers, BatchNorm, Dropout).
- Evaluates all models on the held-out test set (accuracy, precision, recall, F1, ROC-AUC), and saves confusion matrices, ROC curves, and feature importances.

### 2. CNN (`src/vision/`)
- `dataset.py` builds stratified train/val/test PyTorch `DataLoader`s from DF20 images, filtered to the 45 shared species, with class-balanced sampling and data augmentation (crop, flip, color jitter, rotation) on the training split.
- `train_cnn.py` fine-tunes a **ResNet18** (ImageNet-pretrained) with a new binary classification head, using BCE loss, Adam, and LR scheduling on plateau. Best checkpoint (by validation loss) is saved.
- `evaluate_cnn.py` reloads the best checkpoint and computes test-set metrics, confusion matrix, and ROC curve.

### 3. Comparison (`comparison.py`)
Combines the tabular test metrics and CNN test metrics into a single table and bar chart, sorted by accuracy, to directly compare tabular vs. image-based classification.

## Results

| Model | Type | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | Tabular | 0.9999 | 0.9999 | 1.0000 |
| MLP (Neural Net) | Tabular | 0.9999 | 0.9999 | 1.0000 |
| k-NN (k=5) | Tabular | 0.9997 | 0.9997 | 1.0000 |
| SVM (RBF) | Tabular | 0.9953 | 0.9958 | 0.9995 |
| XGBoost | Tabular | 0.9897 | 0.9907 | 0.9991 |
| **ResNet18 (CNN)** | Image | **0.8337** | **0.8408** | **0.9188** |
| Logistic Regression | Tabular | 0.7679 | 0.7900 | 0.8361 |

**Key finding:** tabular models trained on structured, human-labeled visual features substantially outperform the CNN working directly from raw images — the tree-based and neural tabular models are effectively at ceiling, while ResNet18 (tested on 3,217 held-out images) trails by roughly 15+ accuracy points. `stem-width`, `stem-height`, and `cap-diameter` are consistently the most important features across the tree-based tabular models.

Full metrics, confusion matrices, ROC curves, and feature-importance plots are saved under `results/tabular/`, `results/cnn/`, and `results/comparison/`.

## Requirements
```
numpy>=1.24
pandas>=1.5
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
torch>=2.0
xgboost>=1.7
ucimlrepo>=0.0.3
```
Also requires `torchvision` and `PIL` (Pillow) for the CNN pipeline. GPU/MPS acceleration is auto-detected (falls back to CPU).

## Usage
```bash
# 1. Run all tabular models, EDA, and evaluation
python main.py

# 2. Train and evaluate the CNN (requires DF20 metadata CSV + image folder — not included in this repo)
python src/vision/train_cnn.py --metadata data/DF20-train_metadata_PROD-2.csv --image-root data/DF20_300
python src/vision/evaluate_cnn.py

# 3. Compare tabular vs. CNN results
python comparison.py
```

Optional flags for `main.py`:
- `--epochs N` — MLP training epochs (default 60)
- `--cv N` — cross-validation folds (default 5)
- `--no-eda` — skip EDA plots
- `--no-mlp` — skip MLP training

## Project Structure
```
main.py                     # tabular pipeline entry point
comparison.py               # tabular vs. CNN comparison
src/
  preprocess.py              # loading, EDA, preprocessing
  models.py                  # sklearn model defs + PyTorch MLP
  evaluate.py                # metrics, plots, CSV export
  vision/
    dataset.py                # DF20 PyTorch Dataset + DataLoaders
    edibility_map.py           # species -> edible/poisonous label lookup
    train_cnn.py                # ResNet18 fine-tuning
    evaluate_cnn.py              # CNN test-set evaluation
results/
  tabular/                    # test metrics, feature importances
  cnn/                        # checkpoint, training curves, test metrics
  comparison/                 # combined table + comparison plot
  figures/                    # EDA plots
```

## Notes
- The DF20 image data and metadata CSV are **not included** in this repo due to size — see the [DF20 dataset page](https://sites.google.com/view/danish-fungi-dataset) to obtain them.
- Edibility labels for the CNN's 45 species are manually curated in `edibility_map.py`, sourced from MycoBank, GBIF toxicity records, Roger Phillips (2006), and First Nature.
````
