import os
import pandas as pd
import glob

hunts_dir = r"C:\veil_finder_project\hunts"
csv_files = glob.glob(os.path.join(hunts_dir, "*", "scans_export", "fog_master.csv"))

for csv_path in csv_files:
    print(f"Found {csv_path}")
    parquet_path = csv_path.replace(".csv", ".parquet")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        print(f"Converted to {parquet_path}. Size: {os.path.getsize(parquet_path) / (1024*1024):.2f} MB")
        
        # Remove the CSV so it doesn't get tracked
        os.remove(csv_path)
        print(f"Deleted original CSV: {csv_path}")
