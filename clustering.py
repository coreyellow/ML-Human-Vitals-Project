import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score, adjusted_rand_score
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

    # Perform agglomerative clustering for each linkage type and threshold
    # Smaller distance_threshold = more clusters, larger distance_threshold = fewer clusters
    linkage_types = ['ward', 'complete', 'average', 'single']
    distance_thresholds = [1, 3, 5, 10, 20, 40]
    labels = {}

    for linkage in linkage_types:
        print(f"\nStarting agglomerative clustering for linkage='{linkage}'...")
        labels[linkage] = {}
        for distance_threshold in distance_thresholds:
            print(f"Performing agglomerative clustering with linkage={linkage}, distance_threshold={distance_threshold}...")
            male_agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage)
            female_agglo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage)
            
            male_labels = male_agglo.fit_predict(male_features_scaled)
            female_labels = female_agglo.fit_predict(female_features_scaled)

            labels[linkage][distance_threshold] = {
                'male': male_labels,
                'female': female_labels
            }

            print(f"Agglomerative clustering completed for linkage={linkage}, distance_threshold={distance_threshold}.")
            print(f"Male clusters formed: {len(set(male_labels))}")
            print(f"Female clusters formed: {len(set(female_labels))}")

    # Save directory for evaluation and labels
    save_directory = "Clustering_data/"

    # Evaluate clustering performance
    evaluation_rows = []
    for linkage in linkage_types:
        for distance_threshold in distance_thresholds:
            print(f"\nEvaluating clustering performance for linkage={linkage}, distance_threshold={distance_threshold}...")
            male_labels = labels[linkage][distance_threshold]['male']
            female_labels = labels[linkage][distance_threshold]['female']

            if len(set(male_labels)) > 1:
                silhouette_male = silhouette_score(male_features_scaled, male_labels)
            else:
                silhouette_male = None

            if len(set(female_labels)) > 1:
                silhouette_female = silhouette_score(female_features_scaled, female_labels)
            else:
                silhouette_female = None

            adjusted_rand_male = adjusted_rand_score(male_risk.flatten(), male_labels)
            adjusted_rand_female = adjusted_rand_score(female_risk.flatten(), female_labels)

            fowlkes_mallows_male = fowlkes_mallows_score(male_risk.flatten(), male_labels)
            fowlkes_mallows_female = fowlkes_mallows_score(female_risk.flatten(), female_labels)

            evaluation_rows.append({
                'linkage': linkage,
                'distance_threshold': distance_threshold,
                'gender': 'male',
                'clusters': len(set(male_labels)),
                'silhouette_score': silhouette_male,
                'adjusted_rand_score': adjusted_rand_male,
                'fowlkes_mallows_score': fowlkes_mallows_male
            })
            evaluation_rows.append({
                'linkage': linkage,
                'distance_threshold': distance_threshold,
                'gender': 'female',
                'clusters': len(set(female_labels)),
                'silhouette_score': silhouette_female,
                'adjusted_rand_score': adjusted_rand_female,
                'fowlkes_mallows_score': fowlkes_mallows_female
            })

    # Save evaluation results to CSV
    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_filepath = f"{save_directory}clustering_evaluation.csv"
    evaluation_df.to_csv(evaluation_filepath, index=False)
    print(f"Clustering evaluation metrics saved to {evaluation_filepath}.")
    
    # Save labels
    save_linkage = 'ward'
    save_distance_threshold = 20

    male_labels = labels[save_linkage][save_distance_threshold]['male']
    female_labels = labels[save_linkage][save_distance_threshold]['female']
    joblib.dump(male_labels, f"{save_directory}male_clustering_labels_{save_linkage}_{save_distance_threshold}.joblib")
    joblib.dump(female_labels, f"{save_directory}female_clustering_labels_{save_linkage}_{save_distance_threshold}.joblib")
    print(f"Saved male/female labels for linkage={save_linkage}, distance_threshold={save_distance_threshold}.")

    # Save the scalers for future processing
    joblib.dump(scaler_male, f"{save_directory}scaler_male.joblib")
    joblib.dump(scaler_female, f"{save_directory}scaler_female.joblib")
    print("Scalers saved successfully.")
    

