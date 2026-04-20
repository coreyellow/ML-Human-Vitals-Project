import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn.preprocessing import StandardScaler

df_male = pd.read_csv('Datasets/male_scaled_dataset.csv')
df_female = pd.read_csv('Datasets/female_scaled_dataset.csv')

# Standardize features
male_features = df_male[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP', 'Risk Category']]
female_features = df_female[['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP', 'Risk Category']]

scaler = StandardScaler()

male_features_scaled = scaler.fit_transform(male_features)
female_features_scaled = scaler.fit_transform(female_features)

# TODO: agglomerative clustering with distance threshold
