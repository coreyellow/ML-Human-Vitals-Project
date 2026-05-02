import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

output_dir = 'PCA_output'
clustering_dir = 'Clustering_data'
os.makedirs(output_dir, exist_ok=True)

numerical_features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Age', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI',
    'Derived_MAP'
]

datasets = {
    'male':   'Datasets/male_dataset.csv',
    'female': 'Datasets/female_dataset.csv',
}

for gender, path in datasets.items():
    print("\n" + "="*60)
    print(f"PCA ANALYSIS — {gender.upper()} DATASET")
    print("="*60)

    # Load dataset
    df = pd.read_csv(path)
    print(f"Original dataset shape: {df.shape}")

    # Use only features present in this dataset
    available_features = [f for f in numerical_features if f in df.columns]
    X = df[available_features].copy()

    print(f"Features used ({len(available_features)}): {available_features}")

    # Handle missing values
    print(f"Missing values before cleaning:\n{X.isnull().sum()}")
    X = X.dropna()
    print(f"Rows after removing NaN: {len(X):,}")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA — retain 7 components (~88% variance explained)
    n_components = 7
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Save scaler and PCA
    scaler_path = os.path.join(clustering_dir, f'scaler_{gender}.pkl')
    pca_path = os.path.join(clustering_dir, f'pca_{gender}.pkl')
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    print(f"\nPCA Results:")
    print(f"  Components: {pca.n_components_}")
    print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  Variance per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var:.4f} ({var*100:.2f}%)")

    # ── Variance by component bar chart ────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA Variance by Component — {gender.capitalize()}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    var_plot_path = os.path.join(output_dir, f'PCA_VAR_Analysis_{gender}.png')
    plt.savefig(var_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVariance plot saved: {var_plot_path}")

    # ── Build output DataFrame ──────────────────────────────────────────────
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

    # Save CSV
    output_csv = os.path.join(output_dir, f'HVS_PCA_{gender}.csv')
    df_pca.to_csv(output_csv, index=False)
    print(f"PCA dataset saved:    {output_csv}")
    print(f"Shape: {df_pca.shape}  ({len(available_features)} features → {n_components} components)")

    # ── Feature loadings ────────────────────────────────────────────────────
    print(f"\nPCA Component Loadings:")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=pca_columns,
        index=available_features
    )
    print(loadings.round(4))

print("\n✓ PCA analysis complete for male and female datasets!")
