"""
Preprocessing pipeline for UCI Mushroom Dataset.
Handles loading via ucimlrepo, EDA, encoding, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch the UCI Mushroom dataset (id=73) via ucimlrepo.

    Returns
    -------
    X : pd.DataFrame  — feature columns (22 categorical features)
    y : pd.DataFrame  — target column  ('poisonous': 'e' or 'p')
    """
    from ucimlrepo import fetch_ucirepo

    print("Fetching UCI Mushroom dataset (id=73)...")
    mushroom = fetch_ucirepo(id=73)

    X = mushroom.data.features
    y = mushroom.data.targets

    print(f"\n── Metadata ──────────────────────────────────────────")
    print(mushroom.metadata)
    print(f"\n── Variable Info ──────────────────────────────────────")
    print(mushroom.variables)
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance:\n{y.iloc[:, 0].value_counts()}")

    return X, y


def run_eda(X: pd.DataFrame, y: pd.DataFrame,
            save_dir: str = "results/figures") -> None:
    """
    Generate and save EDA plots.

    Parameters
    ----------
    X : feature DataFrame from ucimlrepo
    y : target DataFrame from ucimlrepo
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Flatten target to a Series; ucimlrepo returns a single-column DataFrame
    y_series = y.iloc[:, 0]
    target_col = y_series.name

    # Build a combined df for cross-tab plots
    df_plot = X.copy()
    df_plot[target_col] = y_series.values

    # ── 1. Class distribution ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_series.value_counts()
    labels = ['Edible' if c == 'e' else 'Poisonous' for c in counts.index]
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val}\n({val/len(y_series)*100:.1f}%)', ha='center', fontsize=11)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_distribution.png", dpi=150)
    plt.close()

    # ── 2. Missing values ─────────────────────────────────────────────────────
    missing = X.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        missing.plot(kind='bar', ax=ax, color='#FF7043', edgecolor='black')
        ax.set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/missing_values.png", dpi=150)
        plt.close()
        print(f"Missing values found:\n{missing}")
    else:
        print("No missing values found in features.")

    # ── 3. Top feature distributions by class ─────────────────────────────────
    # Use whichever of these exist in the fetched feature set
    candidate_features = ['odor', 'gill-color', 'spore-print-color',
                          'stalk-surface-above-ring', 'ring-type']
    top_features = [f for f in candidate_features if f in X.columns][:5]

    fig, axes = plt.subplots(1, len(top_features), figsize=(20, 5))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_features):
        ct = pd.crosstab(df_plot[feat], df_plot[target_col])
        # Rename columns to readable labels
        ct.columns = ['Edible' if c == 'e' else 'Poisonous' for c in ct.columns]
        ct.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'],
                edgecolor='black', linewidth=0.5)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle('Feature Distributions by Class (Top Discriminative Features)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_distributions.png", dpi=150)
    plt.close()

    # ── 4. Correlation heatmap (label-encoded) ────────────────────────────────
    df_enc = df_plot.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df_enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdYlGn', center=0,
                linewidths=0.3, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"EDA plots saved to '{save_dir}'")


def preprocess(X: pd.DataFrame, y: pd.DataFrame,
               test_size: float = 0.15, val_size: float = 0.15,
               random_state: int = 42):
    """
    Encode features and split into train / val / test sets.

    Parameters
    ----------
    X : feature DataFrame from ucimlrepo
    y : target DataFrame from ucimlrepo

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  (one-hot encoded)
    y_train, y_val, y_test : np.ndarray  (0 = edible, 1 = poisonous)
    feature_names          : list[str]
    """
    X = X.copy()
    y_series = y.iloc[:, 0]

    # Drop zero-variance columns (e.g. 'veil-type' has only one unique value)
    nunique = X.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        print(f"Dropping zero-variance columns: {drop_cols}")
        X.drop(columns=drop_cols, inplace=True)

    # Impute missing values with column mode (affects 'stalk-root' ~30%)
    for col in X.columns:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"  Imputed '{col}' missing values with mode='{mode_val}'")

    # Encode target: 0 = edible, 1 = poisonous
    y_enc = (y_series == 'p').astype(int).values

    # One-hot encode all categorical features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = ohe.fit_transform(X)
    feature_names = ohe.get_feature_names_out(X.columns).tolist()

    # Stratified splits: 70 / 15 / 15
    val_frac_of_trainval = val_size / (1 - test_size)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_encoded, y_enc, test_size=test_size,
        random_state=random_state, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_of_trainval,
        random_state=random_state, stratify=y_tv)

    print(f"\nSplit sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Feature dimensions after OHE: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names