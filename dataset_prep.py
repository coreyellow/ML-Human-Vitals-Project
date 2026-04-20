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

df.to_csv('Datasets/reduced_dataset.csv', index=False)
    