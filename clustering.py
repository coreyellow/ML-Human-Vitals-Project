import pandas as pd
import joblib
from joblib import Memory
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os

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
    df_male = pd.read_csv('PCA_output/HVS_PCA_male.csv')
    df_female = pd.read_csv('PCA_output/HVS_PCA_female.csv')
    print("CSV files loaded successfully.")
    print(f"Male dataset shape: {df_male.shape}")
    print(f"Female dataset shape: {df_female.shape}")

    # Sample limit
    limit = 10000
    male_features = df_male.values[:limit]
    female_features = df_female.values[:limit]
    print(f"Data limited to {limit} samples for clustering.")

    # Setup memory caching for distance computations
    cache_dir_male = "Clustering_data/cache_male"
    cache_dir_female = "Clustering_data/cache_female"
    os.makedirs(cache_dir_male, exist_ok=True)
    os.makedirs(cache_dir_female, exist_ok=True)
    
    memory_male = Memory(cache_dir_male, verbose=1)
    memory_female = Memory(cache_dir_female, verbose=1)

    # Perform agglomerative clustering for each linkage type and threshold
    linkage_types = ['ward', 'complete', 'average', 'single']
    k_clusters = [10, 15, 20, 30, 50]
    labels = {}

    # Precompute distance matrices for efficient silhouette score evaluation
    print("\nPrecomputing distance matrices for faster evaluation...")
    male_distance_matrix = squareform(pdist(male_features, metric='euclidean'))
    female_distance_matrix = squareform(pdist(female_features, metric='euclidean'))

    for linkage in linkage_types:
        print(f"\nStarting agglomerative clustering for linkage='{linkage}'...")
        labels[linkage] = {}
        for k in k_clusters:
            male_agglo = AgglomerativeClustering(n_clusters=k, linkage=linkage, memory=memory_male)
            female_agglo = AgglomerativeClustering(n_clusters=k, linkage=linkage, memory=memory_female)

            male_labels = male_agglo.fit_predict(male_features)
            female_labels = female_agglo.fit_predict(female_features)

            labels[linkage][k] = {
                'male': male_labels,
                'female': female_labels
            }

            print(f"Agglomerative clustering completed for linkage={linkage}, n_clusters={k}.")
    # Save directory for evaluation and labels
    save_directory = "Clustering_data/"

    # Evaluate clustering performance
    evaluation_rows = []
    print()
    for linkage in linkage_types:
        for k in k_clusters:
            print(f"Evaluating clustering performance for linkage={linkage}, n_clusters={k}...")
            male_labels = labels[linkage][k]['male']
            female_labels = labels[linkage][k]['female']

            if len(set(male_labels)) > 1:
                silhouette_male = silhouette_score(male_distance_matrix, male_labels, metric='precomputed')
            else:
                silhouette_male = None

            if len(set(female_labels)) > 1:
                silhouette_female = silhouette_score(female_distance_matrix, female_labels, metric='precomputed')
            else:
                silhouette_female = None

            evaluation_rows.append({
                'linkage': linkage,
                'clusters': k,
                'gender': 'male',
                'silhouette_score': silhouette_male
            })
            evaluation_rows.append({
                'linkage': linkage,
                'clusters': k,
                'gender': 'female',
                'silhouette_score': silhouette_female
            })

    # Save evaluation results to CSV
    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_filepath = f"{save_directory}clustering_evaluation.csv"
    evaluation_df.to_csv(evaluation_filepath, index=False)
    print(f"Clustering evaluation metrics saved to {evaluation_filepath}.")
    
    # Save labels
    save_linkage = 'ward'
    save_k = 50

    male_labels = labels[save_linkage][save_k]['male']
    female_labels = labels[save_linkage][save_k]['female']
    joblib.dump(male_labels, f"{save_directory}male_clustering_labels_{save_linkage}_{save_k}.joblib")
    joblib.dump(female_labels, f"{save_directory}female_clustering_labels_{save_linkage}_{save_k}.joblib")
    print(f"Saved male/female labels for linkage={save_linkage}, n_clusters={save_k}.")
    

