import pandas as pd

full_dataset = 'Datasets/full_dataset.csv'

df = pd.read_csv(full_dataset)
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
    
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

df_male = df_male.drop(columns=['Gender'])
df_female = df_female.drop(columns=['Gender'])

print("\nMale dataset shape: ", df_male.shape)
print("Female dataset shape: ", df_female.shape)

df_male.to_csv('Datasets/male_dataset.csv', index=False)
df_female.to_csv('Datasets/female_dataset.csv', index=False)