import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def pull_both_sensors(date: str, destination: str, label: str = "", adb_path: str = r"C:\Watch_data\platform-tools\adb.exe"):
    """
    Pulls all Accel and Gyro data for a given date from an Android watch, cleans and merges them,
    and saves as separate labeled CSVs.

    Parameters:
        date (str): e.g., '2025-07-05-14-36'
        destination (str): Local folder to store CSVs
        label (str): Trial label like 'walking_trial' (optional)
        adb_path (str): Path to adb executable

    Returns:
        Tuple of (accel_csv_path, gyro_csv_path)
    """
    def pull_and_merge(label_type):
        dfs = []
        i = 1
        while True:
            filename = f"{date}-{i}-{label_type}.txt"
            remote_folder = f"sdcard/Download/{date}"
            remote_path = f"{remote_folder}/{filename}"
            local_path = Path(destination) / filename
            result = subprocess.run(
                [adb_path, 'pull', remote_path, str(local_path)],
                capture_output=True, text=True
            )

            if "No such file" in result.stderr or "remote object" in result.stderr:
                break
            elif result.returncode != 0:
                print(f"ADB error while pulling {remote_path}: {result.stderr.strip()}")
                break

            try:
                df = pd.read_csv(local_path, header=None,
                                 names=['x', 'y', 'z', 'timestamp_ns', 'unused'],
                                 on_bad_lines='skip')
                df.drop(columns=['unused'], inplace=True)

                # Timestamp correction
                true_start_time = datetime.strptime(date, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
                actual_start_ns = int(df['timestamp_ns'].iloc[0])
                desired_start_ns = int(true_start_time.timestamp() * 1e9)
                offset_ns = desired_start_ns - actual_start_ns
                df['timestamp_ns'] += offset_ns

                df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns', utc=True)
                df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
                df['time_sec'] = df['time'].dt.floor('s')

                dfs.append(df)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
            i += 1

        if not dfs:
            print(f"No valid files found for {label_type}.")
            return None

        combined = pd.concat(dfs, ignore_index=True)
        suffix = f"{label_type}"
        if label:
            suffix += f"-{label}"
        output_path = Path(destination) / f"{suffix}.csv"
        combined.to_csv(output_path, index=False)
        print(f"Saved merged CSV: {output_path}")
        return output_path

    # Ensure destination exists
    Path(destination).mkdir(parents=True, exist_ok=True)

    # Pull both types
    accel_csv = pull_and_merge("Accel")
    gyro_csv = pull_and_merge("Gyro")

    return accel_csv, gyro_csv

pull_both_sensors(
    date="2025-07-08-12-11",
    destination=r"C:\Watch_data\data_store",
    label="2025-07-08-12-11"
)
