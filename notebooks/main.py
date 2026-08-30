
# main.py - runs tabular dataset models

import argparse
import os
import warnings
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from src.preprocess import load_data, run_eda, preprocess
from src.models import (get_sklearn_models, cross_validate_models,
                        train_all_sklearn, train_mlp)
from src.evaluate import (evaluate_all, plot_confusion_matrices,
                           plot_roc_curves, plot_metric_comparison,
                           plot_mlp_history, plot_feature_importance,
                           save_results_csv)

# processing optimization for my mac
def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# savee feature importance to csv and overwrite existing files
def save_feature_importances(fitted_models: dict,
                              feature_names: list,
                              save_dir: str = "results/tabular") -> None:
    from sklearn.pipeline import Pipeline

    saved = []
    for name, (model, _) in fitted_models.items():
        clf = model[-1] if isinstance(model, Pipeline) else model
        if not hasattr(clf, 'feature_importances_'):
            continue
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_,
        }).sort_values('importance', ascending=False)

        fname = (name.lower()
                 .replace(' ', '_')
                 .replace('(', '').replace(')', ''))
        path = f"{save_dir}/{fname}_feature_importance.csv"
        df.to_csv(path, index=False)
        saved.append(path)
        print(f"Saved: {path}")

    if not saved:
        print("Feature importances not saved.")

# options for running file
def parse_args():
    parser = argparse.ArgumentParser(
        description="Secondary Mushroom Classification Pipeline")
    # number of epochs to train MLP
    parser.add_argument('--epochs', type=int, default=60,
                        help='MLP training epochs (default: 60)')
    # number of folds for k-fold cross-validation
    parser.add_argument('--cv', type=int, default=5,
                        help='Cross-validation folds (default: 5)')
    # flag to skip making plots for exploring data if rerunning
    parser.add_argument('--no-eda', action='store_true',
                        help='Skip EDA plots')
    # flah to skip running mlp again
    parser.add_argument('--no-mlp', action='store_true',
                        help='Skip MLP training')
    return parser.parse_args()

# full tabular classification
def main():
    args = parse_args()
    device = get_device()

    os.makedirs("results/tabular", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load
    print("STEP 1: Loading Data")
    print("="*60)
    X, y = load_data()
    print(f"\nFeatures preview:\n{X.head()}")

    # 2. Data exploration
    if not args.no_eda:
        print("STEP 2: Data Exploration")
        print("="*60)
        run_eda(X, y)

    # 3. Preprocessing 
    print("STEP 3: Preprocessing")
    print("="*60)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, preprocessor) = preprocess(X, y)
    print(f"Training features: {X_train.shape}")

    # 4. Cross-validation
    print(f"STEP 4: {args.cv}-Fold Cross-Validation for sklearn models")
    print("="*60)
    sklearn_models = get_sklearn_models()
    cv_results = cross_validate_models(
        sklearn_models, X_train, y_train, cv=args.cv)

    # 5. Training
    print("STEP 5: Training")
    print("="*60)
    fitted_models = train_all_sklearn(sklearn_models, X_train, y_train)

    # 6. Train MLP 
    mlp_model, mlp_history = None, None
    if not args.no_mlp:
        print("STEP 6: Training MLP (PyTorch)")
        print("="*60)
        mlp_model, mlp_history, mlp_time = train_mlp(
            X_train, y_train, X_val, y_val,
            input_dim=X_train.shape[1],
            epochs=args.epochs, device=device,
        )
        print(f"MLP trained in {mlp_time:.2f}s")
        # overwrites results/figures/mlp_training_history.png
        plot_mlp_history(mlp_history)   

    # 7. Evaluate on test set 
    print("STEP 7: Evaluating on Test Set")
    print("="*60)
    results_df = evaluate_all(
        fitted_models, mlp_model, X_test, y_test, device=device)
    # save/overwrite file
    save_results_csv(results_df, save_dir="results/tabular") 

    # 8. Plots 
    print("STEP 8: Generating Evaluation Plots")
    print("="*60)

    # generate and save all plots
    plot_confusion_matrices(
        fitted_models, mlp_model, X_test, y_test, device=device)
    plot_roc_curves(
        fitted_models, mlp_model, X_test, y_test, device=device)
    plot_metric_comparison(results_df)
    plot_feature_importance(fitted_models, feature_names)

    # 9. Feature importances 
    print("STEP 9: Saving Feature Importances")
    print("="*60)
    save_feature_importances(fitted_models, feature_names)

    print("\nTabular section complete")


if __name__ == '__main__':
    main()