import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('human_vital_signs_dataset_2024.csv')

print("Original dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Separate numerical and categorical features
# Exclude Patient ID, Timestamp, and Risk Category from PCA
numerical_features = [
    'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Weight (kg)',
    'Height (m)', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI',
    'Derived_MAP'
]

# Extract numerical data
X = df[numerical_features].copy()

print(f"\nNumerical features for PCA: {len(numerical_features)}")
print("Features:", numerical_features)

# Handle any missing values
print(f"\nMissing values before cleaning:\n{X.isnull().sum()}")
X = X.dropna()

print(f"Rows after removing NaN: {len(X)}")

# Standardize the features (essential for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPerforming PCA...")

# Apply PCA to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Results:")
print(f"Original number of features: {len(numerical_features)}")
print(f"Reduced number of components: {pca.n_components_}")
print(f"Variance explained by {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.4f}")
print(f"\nVariance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

# Create visualization of cumulative variance explained
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(12, 5))

# Plot 1: Cumulative explained variance
plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Individual variance explained
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Explained Variance by Component')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_VAR_Analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'PCA_VAR_Analysis.png'")

# Create a DataFrame with PCA components
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

# Add back the categorical features from original dataframe (excluding rows with NaN)
df_filtered = df.loc[X.index]
df_pca['Patient ID'] = df_filtered['Patient ID'].values
df_pca['Gender'] = df_filtered['Gender'].values
df_pca['Risk Category'] = df_filtered['Risk Category'].values

# Save the PCA-reduced dataset
output_file = 'HVS_PCA.csv'
df_pca.to_csv(output_file, index=False)
print(f"\nPCA-reduced dataset saved to '{output_file}'")
print(f"New dataset shape: {df_pca.shape}")
print(f"Dimensionality reduction: {len(numerical_features)} → {pca.n_components_} features")
print(f"Data points: {len(df)} → {len(df_pca)}")

# Print feature loadings (which original features contribute most to each PC)
print("\n" + "="*60)
print("PCA Component Loadings (Feature Contributions)")
print("="*60)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=numerical_features
)
print(loadings.round(4))

# Visualize loadings for the first two components
if pca.n_components_ >= 2:
    plt.figure(figsize=(16, 14))
    for i, feature in enumerate(numerical_features):
        plt.arrow(0, 0, loadings[f'PC1'].iloc[i], loadings[f'PC2'].iloc[i],
                 head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.7, linewidth=2)
        plt.text(loadings[f'PC1'].iloc[i]*1.25, loadings[f'PC2'].iloc[i]*1.25, feature,
                fontsize=12, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=14, weight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=14, weight='bold')
    plt.title('PCA Biplot: Feature Loadings on First Two Components', fontsize=16, weight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    plt.tight_layout()
    plt.savefig('PCA_BIPLOT.png', dpi=300, bbox_inches='tight')
    print("\nBiplot saved as 'PCA_BIPLOT.png'")

print("\n✓ PCA analysis complete!")
