from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData
import matplotlib.pyplot as plt

import re
import pandas as pd
from pathlib import Path

def concatenate_chunks_by_prefix(folder, base_filename_prefix, output_file=None):
    """
    Concatenate all chunked CSVs like 2025-07-05-14-36-1-Accel.csv under the same timestamp prefix.
    
    Args:
        folder (str or Path): Directory containing the CSV chunks.
        base_filename_prefix (str): The timestamp prefix, e.g., '2025-07-05-14-36'.
        output_file (str or Path, optional): If provided, writes the combined file to disk.
        
    Returns:
        pd.DataFrame: Combined DataFrame sorted in time.
    """
    folder = Path(folder)
    pattern = re.compile(rf'{re.escape(base_filename_prefix)}-\d+-.*\.csv')
    
    files = sorted([f for f in folder.glob("*.csv") if pattern.match(f.name)],
                   key=lambda f: int(re.search(r'-(\d+)-', f.name).group(1)))

    if not files:
        raise FileNotFoundError(f"No matching CSVs for prefix {base_filename_prefix}")

    print(f"📦 Found {len(files)} files to merge for prefix '{base_filename_prefix}':")
    for f in files:
        print(f"   • {f.name}")

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True).sort_values("time")
    combined.reset_index(drop=True, inplace=True)

    if output_file:
        combined.to_csv(output_file, index=False)
        print(f"✅ Saved combined file: {output_file}")

    return combined

local_dir = r'C:\Watch_data\data_store'
accel_data = RawAccelerometerXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-07-13-25-1-Accel.txt',
    save_as_csv=True
)
gyro_data = RawGyroscopeXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-07-13-25-1-Gyro.txt',
    save_as_csv=True
)

gyro_data.compute_orientation(other=accel_data)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(gyro_data.df['time'], gyro_data.df['roll'], label='Roll')
ax.scatter(gyro_data.df['time'], gyro_data.df['pitch'], label='Pitch')
ax.scatter(gyro_data.df['time'], gyro_data.df['yaw'], label='Yaw')
ax.set_xlabel('Time')
ax.set_title('Walking 20 steps')
ax.set_ylabel('Orientation (radians)')
ax.legend()
plt.show()