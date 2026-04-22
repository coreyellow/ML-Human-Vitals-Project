import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler

def get_largest_clusters_centers(labels, features, n_clusters=10):
    cluster_sizes = pd.Series(labels).value_counts().nlargest(n_clusters)
    clusters_to_print = cluster_sizes.index.tolist()
    centers_unscaled = []
    for idx, cluster_id in enumerate(clusters_to_print):
        cluster_mask = labels == cluster_id
        center = features[cluster_mask].mean()
        centers_unscaled.append(center)
    return centers_unscaled

def print_cluster_summary(label, features, title):
    cluster_centers = get_largest_clusters_centers(label, features)
    centers_df = pd.DataFrame(cluster_centers)

    print("\n" + "="*80)
    print(f"{title} CLUSTER CENTERS SUMMARY (Count: {len(cluster_centers)})")
    print("="*80)
    print(centers_df.to_string())

if __name__ == "__main__":
    df_male = pd.read_csv('Datasets/male_dataset.csv')
    df_female = pd.read_csv('Datasets/female_dataset.csv')
    print("CSV files loaded successfully.")
    print(f"Male dataset shape: {df_male.shape}")
    print(f"Female dataset shape: {df_female.shape}")

    # Standardize features
    male_features = df_male[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP']]
    female_features = df_female[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP']]
    print("Features extracted successfully.")

    male_risk = df_male[['Risk Category']].to_numpy()
    female_risk = df_female[['Risk Category']].to_numpy()
    print("Risk categories extracted successfully.")

    scaler_male = StandardScaler()
    scaler_female = StandardScaler()

    male_features_scaled = scaler_male.fit_transform(male_features)
    female_features_scaled = scaler_female.fit_transform(female_features)
    print("Features standardized successfully.")

    # Sample limit
    limit = 1000
    male_features_scaled = male_features_scaled[:limit]
    female_features_scaled = female_features_scaled[:limit]
    male_features = male_features.iloc[:limit]
    female_features = female_features.iloc[:limit]

    male_risk = male_risk[:limit]
    female_risk = female_risk[:limit]
    print(f"Data limited to {limit} samples for clustering.")

    # Perform agglomerative clustering
    # Smaller distance_threshold = more clusters, larger distance_threshold = fewer clusters
    distance_threshold = 10

    male_agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    female_agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    print("Agglomerative clustering initialized successfully.")

    male_labels = male_agglo.fit_predict(male_features_scaled)
    female_labels = female_agglo.fit_predict(female_features_scaled)
    print("Agglomerative clustering performed successfully.")
    print(f"Male clusters formed: {len(set(male_labels))}")
    print(f"Female clusters formed: {len(set(female_labels))}")

    # Print cluster centers summary
    print_cluster_summary(male_labels, male_features, "MALE")
    print_cluster_summary(female_labels, female_features, "FEMALE")

    # Evaluate clustering performance
    silhouette_male = silhouette_score(male_features_scaled, male_labels)
    silhouette_female = silhouette_score(female_features_scaled, female_labels)
    print(f"\nSilhouette Score for Male Clustering: {silhouette_male:.4f}")
    print(f"Silhouette Score for Female Clustering: {silhouette_female:.4f}")

    fowlkes_mallows_male = fowlkes_mallows_score(male_risk.flatten(), male_labels)
    fowlkes_mallows_female = fowlkes_mallows_score(female_risk.flatten(), female_labels)
    print(f"Fowlkes-Mallows Score for Male Clustering: {fowlkes_mallows_male:.4f}")
    print(f"Fowlkes-Mallows Score for Female Clustering: {fowlkes_mallows_female:.4f}")

