import pandas as pd
import csv
import glob

data_dir = 'Datasets'

if __name__ == "__main__":
    track_names = [
        'Solar8000/ART_DBP',
        'Solar8000/ART_SBP',
        'Solar8000/BT',
        'Solar8000/HR',
        'Solar8000/PLETH_SPO2',
        'Solar8000/RR' 
    ]
    
    # Create full dataset by combining all individual case CSV files
    # Column names:
    # Case ID, Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation, Age, Gender, Derived_HRV, Derived_Pulse_Pressure, Derived_BMI, Derived_MAP
    # Derived_HRV = Standard deviation of heart rate over 1 minute window / Mean heart rate over 1 minute window
    # Derived_Pulse_Pressure = Systolic BP - Diastolic BP
    # Derived_MAP = (Systolic BP + 2 * Diastolic BP) / 3
    cases_csv = f'{data_dir}/cases.csv'
    cases_df = pd.read_csv(cases_csv)

    combined_data = []
    
    for file in glob.glob(f'{data_dir}/lab_data/*.csv'):
        case_id = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file, header=None)
        
        # Metadata
        case_metadata = cases_df[cases_df['Case ID'] == case_id]
        age = case_metadata['age'].values[0]
        gender = case_metadata['sex'].values[0]
        bmi = case_metadata['bmi'].values[0]
        
        # Select 3 random samples from each case
        valid_rows = 0
        while valid_rows < 3:
            row_df = df.sample(n=1)
            row_number = row_df.index[0]
            row = row_df.values[0]

            # Skip rows with missing values
            if any(pd.isnull(row)):
                continue  
            
            diastolic_bp = row[0]           # Solar8000/ART_DBP
            systolic_bp = row[1]            # Solar8000/ART_SBP
            body_temp = row[2]              # Solar8000/BT
            heart_rate = row[3]             # Solar8000/HR
            spo2 = row[4]                   # Solar8000/PLETH_SPO2
            respiratory_rate = row[5]       # Solar8000/RR
            
            # Calculate HRV by looking back at the previous 1 minute of heart rate data
            df_window = df[(df.index >= row_number - 30) & (df.index < row_number)]
            hrv = df_window['Solar8000/HR'].std() / df_window['Solar8000/HR'].mean()
            
            # Remove noisy data points based on physiological plausibility
            if not (10 <= diastolic_bp <= 120
                    and 50 <= systolic_bp <= 200
                    and 30 <= body_temp <= 42
                    and 40 <= heart_rate <= 200
                    and 80 <= spo2 <= 100
                    and 6 <= respiratory_rate <= 30
                    and 0 <= hrv <= 0.5):
                continue

            combined_data.append({
                'Case ID': case_id,
                'Heart Rate': row[1],
                'Respiratory Rate': row[2],
                'Body Temperature': row[3],
                'Oxygen Saturation': row[4],
                'Age': age,
                'Gender': gender,
                'Derived_HRV': hrv,
                'Derived_Pulse_Pressure': systolic_bp - diastolic_bp,
                'Derived_BMI': bmi,
                'Derived_MAP': (systolic_bp + 2 * diastolic_bp) / 3
            })
            
            valid_rows += 1

    # Save combined dataset
    pd.DataFrame(combined_data).to_csv(f'{data_dir}/full_dataset.csv', index=False)
