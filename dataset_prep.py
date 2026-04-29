import pandas as pd
import glob

data_dir = 'Datasets'

def prepare_full_dataset():
    # Create full dataset by combining all individual case CSV files
    # Column names:
    # Case ID, Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation, Age, Gender, Derived_HRV, Derived_Pulse_Pressure, Derived_BMI, Derived_MAP
    # Derived_HRV = Standard deviation of heart rate over 1 minute window / Mean heart rate over 1 minute window
    # Derived_Pulse_Pressure = Systolic BP - Diastolic BP
    # Derived_MAP = (Systolic BP + 2 * Diastolic BP) / 3
    cases_csv = f'{data_dir}/cases.csv'
    cases_df = pd.read_csv(cases_csv)

    combined_data = []
    problematic_cases = []
    
    for file in glob.glob(f'{data_dir}/lab_data/*.csv'):
        case_id = int(file.split('\\')[-1].split('.')[0])
        df = pd.read_csv(file)
        
        # Metadata
        case_metadata = cases_df[cases_df['caseid'] == case_id]
        age = int(case_metadata['age'].values[0])
        gender = case_metadata['sex'].values[0]
        bmi = float(case_metadata['bmi'].values[0])
        
        # Select 3 random samples from each case
        valid_rows = 0
        attempts = 0
        
        while valid_rows < 3:
            # Avoid case with too many missing values
            if attempts > 5000:
                print(f"Too many attempts for case {case_id}, moving to next case.")
                problematic_cases.append(case_id)
                break
            
            print(f"Processing case {case_id}, attempt {attempts + 1}...", end='\r')
            attempts += 1
            
            row_df = df.sample(n=1)
            row_number = row_df.index[0]
            row = row_df.values[0]

            # Skip rows with missing values
            if any(pd.isnull(row)):
                continue  
            
            diastolic_bp = float(row[0])           # Solar8000/ART_DBP
            systolic_bp = float(row[1])            # Solar8000/ART_SBP
            body_temp = float(row[2])              # Solar8000/BT
            heart_rate = float(row[3])             # Solar8000/HR
            spo2 = float(row[4])                   # Solar8000/PLETH_SPO2
            respiratory_rate = float(row[5])       # Solar8000/RR
            
            # Calculate HRV by looking back at previous heart rate values
            if row_number < 100:
                continue  # Skip if there aren't enough previous samples for HRV calculation
            
            df_window = df[(df.index >= row_number - 100) & (df.index < row_number)]
            hr_window = df_window["Solar8000/HR"].astype(float)
            hrv = hr_window.std() / hr_window.mean()
            
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
            print(f"Processed case {case_id}, samples: {valid_rows}/3")

    # Save combined dataset
    print(f"Total combined samples: {len(combined_data)}")
    print(f"Problematic cases: {len(problematic_cases)}")
    pd.DataFrame(combined_data).to_csv(f'{data_dir}/full_dataset.csv', index=False)

def split_dataset_by_gender():
    df = pd.read_csv(f'{data_dir}/full_dataset.csv')
    df.drop(columns=['Case ID'], inplace=True)  # Drop Case ID as it's not needed for modeling
    
    male_df = df[df['Gender'] == 'M'].drop(columns=['Gender'])
    female_df = df[df['Gender'] == 'F'].drop(columns=['Gender'])

    print(f"Male samples: {len(male_df)}")
    print(f"Female samples: {len(female_df)}")

    male_df.to_csv(f'{data_dir}/male_dataset.csv', index=False)
    female_df.to_csv(f'{data_dir}/female_dataset.csv', index=False)

if __name__ == "__main__":
    user_input = input("Do you want to prepare the full dataset? (y/n): ")
    if user_input.lower() == 'y':
        prepare_full_dataset()
    
    split_dataset_by_gender()
        
    
