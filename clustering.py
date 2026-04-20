import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()

male_features_scaled = scaler.fit_transform(male_features)
female_features_scaled = scaler.fit_transform(female_features)
print("Features standardized successfully.")

# Sample limit
limit = 1000
male_features_scaled = male_features_scaled[:limit]
female_features_scaled = female_features_scaled[:limit]

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
