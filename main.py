"""
main.py — Full pipeline runner for Mushroom Classification project.

Usage:
    python main.py
    python main.py --epochs 80 --cv 10

Dataset is fetched automatically via ucimlrepo (id=73).
Install with: pip install ucimlrepo
"""

import argparse
import os
import torch

from src.preprocess import load_data, run_eda, preprocess
from src.models import get_sklearn_models, cross_validate_models, train_all_sklearn, train_mlp
from src.evaluate import (
    evaluate_all, plot_confusion_matrices, plot_roc_curves,
    plot_metric_comparison, plot_mlp_history,
    plot_feature_importance, save_results_csv
)


def parse_args():
    parser = argparse.ArgumentParser(description="Mushroom Classification Pipeline")
    parser.add_argument('--epochs',  type=int, default=60,
                        help='MLP training epochs (default: 60)')
    parser.add_argument('--cv',      type=int, default=5,
                        help='Cross-validation folds (default: 5)')
    parser.add_argument('--no-eda',  action='store_true',
                        help='Skip EDA plots')
    parser.add_argument('--no-mlp',  action='store_true',
                        help='Skip MLP training')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs("results/figures", exist_ok=True)

    # ── 1. Load data via ucimlrepo ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Loading Data (ucimlrepo id=73)")
    print("="*60)
    X, y = load_data()
    print(f"\nFeatures preview:\n{X.head()}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not args.no_eda:
        print("\n" + "="*60)
        print("STEP 2: Exploratory Data Analysis")
        print("="*60)
        run_eda(X, y)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Preprocessing")
    print("="*60)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess(X, y)
    print(f"Training features: {X_train.shape}")

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 4: {args.cv}-Fold Cross-Validation (sklearn models)")
    print("="*60)
    sklearn_models = get_sklearn_models()
    cv_results = cross_validate_models(sklearn_models, X_train, y_train, cv=args.cv)

    # ── 5. Train on full training set ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Training All Models on Full Training Set")
    print("="*60)
    fitted_models = train_all_sklearn(sklearn_models, X_train, y_train)

    # ── 6. Train MLP ──────────────────────────────────────────────────────────
    mlp_model, mlp_history = None, None
    if not args.no_mlp:
        print("\n" + "="*60)
        print("STEP 6: Training MLP (PyTorch)")
        print("="*60)
        mlp_model, mlp_history, mlp_time = train_mlp(
            X_train, y_train, X_val, y_val,
            input_dim=X_train.shape[1],
            epochs=args.epochs, device=device
        )
        print(f"  MLP trained in {mlp_time:.2f}s")
        plot_mlp_history(mlp_history)

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: Evaluating on Test Set")
    print("="*60)
    results_df = evaluate_all(fitted_models, mlp_model, X_test, y_test, device=device)
    save_results_csv(results_df)

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8: Generating Evaluation Plots")
    print("="*60)
    plot_confusion_matrices(fitted_models, mlp_model, X_test, y_test, device=device)
    plot_roc_curves(fitted_models, mlp_model, X_test, y_test, device=device)
    plot_metric_comparison(results_df)
    plot_feature_importance(fitted_models, feature_names)

    print("\n✅ Pipeline complete. All results saved to results/")


if __name__ == '__main__':
    main()