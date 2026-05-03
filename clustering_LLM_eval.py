import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

def summarize_clusters(pca, scaler, labels, pca_csv_path, feature_cols, gender):
    print("\n" + "="*70)
    print(f"{gender.upper()} CLUSTER STATISTICS")
    print("="*70)

    # Load PCA dataset
    df_pca = pd.read_csv(pca_csv_path)

    # Reconstruct original feature space
    X_pca = df_pca.values
    X_scaled = pca.inverse_transform(X_pca)
    X_original = scaler.inverse_transform(X_scaled)

    df_original = pd.DataFrame(X_original, columns=feature_cols)

    # Attach cluster labels
    df_original["cluster"] = labels

    # Group by cluster
    grouped = df_original.groupby("cluster")

    for cluster_id, group in grouped:
        print(f"\n--- Cluster {cluster_id} (n={len(group)}) ---")

        stats = pd.DataFrame({
            "mean": group[feature_cols].mean(),
            "min": group[feature_cols].min(),
            "max": group[feature_cols].max(),
            "std": group[feature_cols].std(),
        })

        print(stats.round(3))

if __name__ == "__main__":
    # Which clustering hyperparameters to evaluate
    linkage = "ward"
    clusters = 20
    
    scaler_male = joblib.load("Clustering_data/scaler_male.pkl")
    scaler_female = joblib.load("Clustering_data/scaler_female.pkl")
    
    pca_male = joblib.load("Clustering_data/pca_male.pkl")
    pca_female = joblib.load("Clustering_data/pca_female.pkl")
    
    labels_male = joblib.load(f"Clustering_data/male_clustering_labels_{linkage}_{clusters}.joblib")
    labels_female = joblib.load(f"Clustering_data/female_clustering_labels_{linkage}_{clusters}.joblib")

    FEATURE_COLS = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Age",
        "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP",
    ]
    
    summarize_clusters(
        pca=pca_male,
        scaler=scaler_male,
        labels=labels_male,
        pca_csv_path="PCA_output/HVS_PCA_male.csv",
        feature_cols=FEATURE_COLS,
        gender="male"
    )
    
    summarize_clusters(
        pca=pca_female,
        scaler=scaler_female,
        labels=labels_female,
        pca_csv_path="PCA_output/HVS_PCA_female.csv",
        feature_cols=FEATURE_COLS,
        gender="female"
    )
    
    # Load API key
    ROOT = Path(__file__).parent
    load_dotenv(ROOT / ".env")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    