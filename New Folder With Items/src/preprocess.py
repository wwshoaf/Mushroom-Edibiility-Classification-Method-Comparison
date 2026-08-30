
# preprocess.py — Data loading, exploring, and preprocessing 

# Secondary Mushroom Dataset (id=848)
# feature types — using only visible features
# 1. load_data() - fetch from UCI via ucimlrepo
# 2. run_eda()- generate exploratory plots
# 3. preprocess() — preprocess and split into train/val/test

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Feature type definitions 
# Physical measurements — numeric, require StandardScaler
CONTINUOUS_FEATURES = [
    # cap width in cm
    'cap-diameter',
    # stem height in cm
    'stem-height',
    # stem width in mm
    'stem-width',
]

# All remaining features are categorical — require OneHotEncoder.
# not including non-visible features: stem-root, spore-print-color, does-bruise-or-bleed, habitat, season
CATEGORICAL_FEATURES = [
    'cap-shape', 'cap-surface', 'cap-color', 
    'gill-attachment', 'gill-spacing', 'gill-color', 'stem-surface', 'stem-color',
    'veil-type', 'veil-color', 'has-ring', 'ring-type',
]

# load data
# fetch dataset from repository
# print metadata
def load_data() -> tuple:
    from ucimlrepo import fetch_ucirepo

    print("Fetching Secondary Mushroom dataset (id=848)...")
    mushroom = fetch_ucirepo(id=848)

    X = mushroom.data.features
    y = mushroom.data.targets

    # Print metadata for transparency and reproducibility
    print(f"\n Metadata")
    print("="*60)
    print(f"Name: {mushroom.metadata.get('name', 'N/A')}")
    print(f"Samples:{X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")

    print(f"\n Variable Info")
    print("="*60)
    print(mushroom.variables[['name', 'role', 'type', 'missing_values']].to_string())

    # Class balance — important to report since 55.5/44.5 is slightly imbalanced
    print(f"\n Class Balance ")
    print("="*60)
    target_col = y.iloc[:, 0]
    vc = target_col.value_counts()
    for val, count in vc.items():
        label = 'Edible' if val == 'e' else 'Poisonous'
        print(f"{label} ({val}): {count:,} ({count/len(target_col)*100:.1f}%)")

    return X, y

# exploratory plots
def run_eda(X: pd.DataFrame, y: pd.DataFrame,
            save_dir: str = "results/figures") -> None:
    
    os.makedirs(save_dir, exist_ok=True)

    # Convert target to binary int for plotting (0=edible, 1=poisonous)
    y_series = y.iloc[:, 0]
    y_bin = (y_series == 'p').astype(int)
    df_plot = X.copy()
    # temporary column for grouping
    df_plot['_label'] = y_bin.values
    label_map = {0: 'Edible', 1: 'Poisonous'}

    # Plot 1: Class distribution 
    # Shows how many edible vs poisonous samples exist.
    # Shows if stratified splits are needed
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_series.value_counts()
    labels = ['Edible' if c == 'e' else 'Poisonous' for c in counts.index]
    bars = ax.bar(labels, counts.values,
                  color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}\n({val/len(y_series)*100:.1f}%)',
                ha='center', fontsize=11)
    ax.set_title('Class Distribution — Secondary Mushroom Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Missing values 
    # Shows features with missing data and how much.
    # Shows if need median for continuous, or mode for categorical.
    # veil-type is 94.8% missing so dropped
    missing = X.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        pct = (missing / len(X) * 100)
        pct.plot(kind='bar', ax=ax, color='#FF7043', edgecolor='black')
        ax.set_title('Missing Values per Feature (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Missing %')
        for i, (col, val) in enumerate(pct.items()):
            ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/missing_values.png", dpi=150)
        plt.close()
        print(f"Missing values:\n{missing.to_string()}")
    else:
        print("No missing values found.")

    # Plot 3: Continuous feature distributions by class 
    # Overlapping histograms for cap-diameter, stem-height, stem-width.
    # density=True normalises so classes with different sample counts are comparable
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X.columns]
    if cont_cols:
        fig, axes = plt.subplots(1, len(cont_cols),
                                  figsize=(5 * len(cont_cols), 4))
        if len(cont_cols) == 1:
            axes = [axes]
        colors = {0: '#4CAF50', 1: '#F44336'}
        for ax, col in zip(axes, cont_cols):
            for lbl, grp in df_plot.groupby('_label'):
                grp[col].dropna().hist(
                    ax=ax, bins=40, alpha=0.6,
                    color=colors[lbl], label=label_map[lbl], density=True)
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
        fig.suptitle('Continuous Feature Distributions by Class',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/continuous_distributions.png", dpi=150)
        plt.close()

    # Plot 4: Categorical feature distributions by class
    # Grouped bar charts for most visually informative categorical features.
    # pd.crosstab counts how many edible/poisonous samples fall into each
    top_cat = ['cap-color', 'gill-color', 'ring-type', 'gill-spacing', 'stem-surface']
    top_cat = [c for c in top_cat if c in X.columns][:5]
    if top_cat:
        fig, axes = plt.subplots(1, len(top_cat), figsize=(20, 5))
        if len(top_cat) == 1:
            axes = [axes]
        for ax, feat in zip(axes, top_cat):
            ct = pd.crosstab(df_plot[feat], df_plot['_label'])
            ct.columns = [label_map[c] for c in ct.columns]
            ct.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'],
                    edgecolor='black', linewidth=0.5)
            ax.set_title(feat, fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.legend(fontsize=7)
        fig.suptitle('Categorical Feature Distributions by Class',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/categorical_distributions.png", dpi=150)
        plt.close()

    # Plot 5: Correlation heatmap 
    # LabelEncoder converts all features to integers
    # The heatmap uses a lower triangle mask to avoid duplicate information.
    # Strong correlations between features would suggest redundancy
    df_enc = df_plot.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(
            df_enc[col].astype(str).fillna('missing'))
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df_enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdYlGn', center=0,
                linewidths=0.3, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"EDA complete — all plots saved to '{save_dir}'")

# preprocess dataset and stratified split into training/test sets
# Continuous features use StandardScaler (zero mean, unit variance)
# Categorical features use OneHotEncoder (binary columns per category)
# Train 70%: used for model fitting and CV
# Val 15%: used for MLP early stopping and LR scheduling
# Test 15%: held-out final evaluation
def preprocess(X: pd.DataFrame, y: pd.DataFrame,
               test_size: float = 0.15, val_size: float = 0.15,
               random_state: int = 26):
   
    X = X.copy()
    y_series = y.iloc[:, 0]

    # Identify which features from the definition lists are present in this fetch
    # check in case ucimlrepo returns a slightly different column set
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    
    print(f"Continuous features ({len(cont_cols)}): {cont_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # Drop zero-variance columns 
    for col in cat_cols[:]: 
        if X[col].nunique(dropna=True) <= 1:
            print(f"Dropping zero-variance column: '{col}'")
            X.drop(columns=[col], inplace=True)
            cat_cols.remove(col)

    # Drop columns with mostly missing values
    DROP_THRESHOLD = 0.50
    for col in cat_cols[:]:
        missing_frac = X[col].isnull().mean()
        if missing_frac > DROP_THRESHOLD:
            print(f"Dropping '{col}' ({missing_frac*100:.1f}% missing)")
            X.drop(columns=[col], inplace=True)
            cat_cols.remove(col)

    # Impute remaining continuous missing values with median
    for col in cont_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"Imputed '{col}' with median={median_val:.2f}")

    # Impute remaining categorical missing values with mode
    for col in cat_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
            print(f"Imputed '{col}' with mode='{mode_val}'")

    # Encode target: 'e' → 0 (edible), 'p' → 1 (poisonous) for sklearn
    y_enc = (y_series == 'p').astype(int).values

    # ColumnTransformer applies different transformations to different columns.
    # remainder='drop' discards any columns not explicitly listed 
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(with_mean=True, with_std=True), cont_cols),
        ('cat', OneHotEncoder(sparse_output=False,
                              handle_unknown='ignore'), cat_cols),
    ], remainder='drop')

    X_encoded = preprocessor.fit_transform(X)

    # Clip encoded values to [-10, 10] to prevent outliers in LR
    X_encoded = np.clip(X_encoded, -10, 10)

    # Recover feature names for plots
    # OHE creates one column per category value, named 'feature_value'
    num_names = cont_cols
    cat_names = preprocessor.named_transformers_['cat']\
                        .get_feature_names_out(cat_cols).tolist()
    feature_names = num_names + cat_names

    # Two-step stratified split to get train / val / test.
    # carve out test set (15%)
    # split remainder into train (70%) and val (15%)
    # val_frac_of_trainval converts the desired 15% of total into the fraction of the 85% remainder: 0.15 / 0.85 ≈ 0.176
    val_frac_of_trainval = val_size / (1 - test_size)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_encoded, y_enc, test_size=test_size,
        random_state=random_state, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_of_trainval,
        random_state=random_state, stratify=y_tv)

    print(f"\nSplit sizes — Train: {len(X_train):,}  "
          f"Val: {len(X_val):,} Test: {len(X_test):,}")
    print(f"Feature dimensions after encoding: {X_train.shape[1]}")
    print(f"\nFinal features used ({len(feature_names)}):")
    print(f"Continuous: {cont_cols}")
    print(f"Categorical: {cat_cols}")   

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor

