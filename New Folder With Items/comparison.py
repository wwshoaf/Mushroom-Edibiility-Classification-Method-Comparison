# comparison.py — Compare tabular model performance vs CNN image classification.
# Research question:
#   How does CNN image classification compare to tabular classification when both are restricted to visually observable features?
#
# The tabular models are trained on the Secondary dataset with unobservable features removed
# The CNN is trained on DF20 images
#
# Inputs:
#   results/tabular/test_metrics.csv
#   results/cnn/cnn_test_metrics.json
#
# Outputs saved to results/comparison/:
#   comparison_table.csv — all models + CNN side by side
#   comparison_plot.png — bar chart comparing accuracy and AUC

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
warnings.filterwarnings('ignore')

# Load tabular metrics CSV and CNN metrics JSON.
def load_results() -> tuple:
    
    tabular_path = "results/tabular/test_metrics.csv"
    cnn_path = "results/cnn/cnn_test_metrics.json"

    for p in [tabular_path, cnn_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun main.py and evaluate_cnn.py first.")

    tabular_df = pd.read_csv(tabular_path)

    with open(cnn_path) as f:
        cnn_metrics = json.load(f)

    # Build CNN row in same format as tabular metrics
    cnn_row = pd.DataFrame([{
    'model': 'ResNet18 (CNN)',
    'accuracy': cnn_metrics['test_accuracy'],
    'precision': cnn_metrics['test_precision'],
    'recall': cnn_metrics['test_recall'],
    'f1': cnn_metrics['test_f1'],
    'roc_auc': cnn_metrics['test_roc_auc'],
    }])

    return tabular_df, cnn_row

# Combine tabular and CNN results into one table.
# Add a 'type' column to distinguish tabular from image models.
def build_comparison_table(tabular_df: pd.DataFrame,
                           cnn_row: pd.DataFrame) -> pd.DataFrame:
    tabular_df = tabular_df.copy()
    tabular_df['type'] = 'Tabular'
    cnn_row = cnn_row.copy()
    cnn_row['type'] = 'Image (CNN)'

    df = pd.concat([tabular_df, cnn_row], ignore_index=True)
    df = df[['model', 'type', 'accuracy', 'f1', 'roc_auc', 'precision', 'recall']]
    return df.sort_values('accuracy', ascending=False).reset_index(drop=True)

# accuracy and AUC for every model.
# sorted by accuracy descending.
def plot_comparison(comparison_df: pd.DataFrame, save_dir: str) -> None:
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    models = comparison_df['model'].tolist()
    accuracy = comparison_df['accuracy'].tolist()
    auc = comparison_df['roc_auc'].tolist()
    types = comparison_df['type'].tolist()

    # Colour: teal for tabular, orange for CNN
    acc_colors = ['#E07B28' if t == 'Image (CNN)' else '#028090' for t in types]
    auc_colors = ['#C05A1A' if t == 'Image (CNN)' else '#014F5A' for t in types]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.4), 6))

    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy',
                   color=acc_colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, auc, width, label='ROC-AUC',
                   color=auc_colors, edgecolor='black', linewidth=0.5, alpha=0.9)

    # Annotate each bar with its value
    for bar in bars1:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0.75, 1.04)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Tabular Models vs CNN — Accuracy and AUC', fontsize=14, fontweight='bold')

    tabular_patch = mpatches.Patch(color='#028090', label='Tabular model')
    cnn_patch = mpatches.Patch(color='#E07B28', label='Image CNN')
    ax.legend(handles=[tabular_patch, cnn_patch], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_plot.png", dpi=150)
    plt.close()
    print("Saved: comparison_plot.png")


def main():
    save_dir = "results/comparison"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("TABULAR VS CNN COMPARISON")
    print("="*60)

    tabular_df, cnn_row = load_results()
    comparison_df = build_comparison_table(tabular_df, cnn_row)

    # Print comparison table
    print("\n Results ")
    print(comparison_df.to_string(index=False,
          float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "  N/A"))

    # Save CSV
    csv_path = f"{save_dir}/comparison_table.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nTable saved: {csv_path}")

    # Key finding
    best_tabular_acc = tabular_df['accuracy'].max()
    cnn_acc = cnn_row['accuracy'].iloc[0]
    gap = best_tabular_acc - cnn_acc
    print(f"\n Key Finding ")
    print(f"Best tabular accuracy: {best_tabular_acc:.4f}")
    print(f"CNN accuracy: {cnn_acc:.4f}")
    print(f"Accuracy gap: {gap:.4f} ({gap*100:.1f}%)")

    plot_comparison(comparison_df, save_dir)
    print(f"\nComparison complete. Results in '{save_dir}/'")


if __name__ == '__main__':
    main()