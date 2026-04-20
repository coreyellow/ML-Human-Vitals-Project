import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler

df_male = pd.read_csv('Datasets/male_dataset.csv')
df_female = pd.read_csv('Datasets/female_dataset.csv')
print("CSV files loaded successfully.")

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

# TODO: agglomerative clustering with distance threshold
