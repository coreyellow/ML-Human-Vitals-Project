import pandas as pd
import vitaldb

save_dir = 'Datasets'

if __name__ == "__main__":
    track_names = [
        'Solar8000/ART_DBP',
        'Solar8000/ART_SBP',
        'Solar8000/BT',
        'Solar8000/HR',
        'Solar8000/PLETH_SPO2',
        'Solar8000/RR' 
    ]
    interval = 2
    
    case_ids = vitaldb.find_cases(track_names=track_names)
    print(f"Found {len(case_ids)} cases with the specified tracks.")
    
    for case_id in case_ids:
        print(f"Processing case ID: {case_id}")
        
        vf = vitaldb.VitalFile(case_id, track_names=track_names, interval=interval)
        vf.to_csv(f'{save_dir}/lab_data/{case_id}.csv', track_names=track_names, interval=interval)