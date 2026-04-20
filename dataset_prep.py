import pandas as pd
from sklearn.preprocessing import StandardScaler

full_dataset = 'Datasets/full_dataset.csv'

df = pd.read_csv(full_dataset)

# Drop columns not needed for clustering
df = df.drop(columns=[
    'Patient ID',
    'Timestamp',
    'Systolic Blood Pressure',
    'Diastolic Blood Pressure',
    'Weight (kg)',
    'Height (m)'
])

print("Reduced dataset shape: ", df.shape)

df.to_csv('Datasets/reduced_dataset.csv', index=False)

# Separate datasets by gender
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

df_male = df_male.drop(columns=['Gender'])
df_female = df_female.drop(columns=['Gender'])

print("\nMale dataset shape: ", df_male.shape)
print("Female dataset shape: ", df_female.shape)

df_male.to_csv('Datasets/male_dataset.csv', index=False)
df_female.to_csv('Datasets/female_dataset.csv', index=False)

# Standardize the datasets
scaler = StandardScaler().set_output(transform="pandas")

df_male = df_male.drop(columns=['Risk Category'])
df_female = df_female.drop(columns=['Risk Category'])

df_male_scaled = scaler.fit_transform(df_male)
df_female_scaled = scaler.fit_transform(df_female)

print("\nScaled male dataset shape:", df_male_scaled.shape)
print("Scaled female dataset shape:", df_female_scaled.shape)

df_male_scaled.to_csv('Datasets/male_scaled_dataset.csv', index=False)
df_female_scaled.to_csv('Datasets/female_scaled_dataset.csv', index=False)