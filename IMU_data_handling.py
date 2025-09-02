import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import subprocess


class GyroscopeData:
    def __init__(self, filepath, fallback_offset_ms=-4 * 3600 * 1000):
        self.filepath = filepath
        self.raw_df = None       # Full-resolution data
        self.df = None           # Second-averaged data (used by all filters)
        self.offset_ms = fallback_offset_ms
        self.load()

    def load(self):
        df_raw = pd.read_csv(self.filepath, header=0, names=['Timestamp', 'Epoch', 'Magnitude', 'raw_datetime'], dtype=str)

        df_raw['Timestamp'] = pd.to_numeric(df_raw['Timestamp'], errors='coerce')
        df_raw['Epoch'] = pd.to_numeric(df_raw['Epoch'], errors='coerce')
        df_raw['Magnitude'] = pd.to_numeric(df_raw['Magnitude'], errors='coerce')
        df = df_raw.dropna(subset=['Epoch', 'Magnitude'])

        if 'raw_datetime' in df.columns and df['raw_datetime'].notna().any():
            df['true_time'] = pd.to_datetime(df['raw_datetime'], errors='coerce')
            df['epoch_time'] = pd.to_datetime(df['Epoch'], unit='ms')
            df['offset_ms'] = (df['true_time'] - df['epoch_time']).dt.total_seconds() * 1000
            self.offset_ms = df['offset_ms'].median()

        df['correct_epoch'] = df['Epoch'] + self.offset_ms
        df['time'] = pd.to_datetime(df['correct_epoch'], unit='ms')
        df['time_sec'] = df['time'].dt.floor('s')

        self.raw_df = df  # Save original data
        # Compute second averages here
        self.df = df.groupby('time_sec')['Magnitude'].mean().reset_index()
        self.df.rename(columns={'Magnitude': 'Magnitude_avg', 'time_sec': 'time'}, inplace=True)

    def apply_savgol_filter(self, window_length=11, polyorder=2):
        self.df['savgol'] = savgol_filter(self.df['Magnitude_avg'], window_length, polyorder)

    def apply_average_filter(self, window_size=5):
        self.df['average'] = self.df['Magnitude_avg'].rolling(window_size, center=True).mean()

    def apply_kalman_filter(self, R=0.5, Q=0.001): # change these values depending
        from pykalman import KalmanFilter
        kf = KalmanFilter(initial_state_mean=self.df['Magnitude_avg'].iloc[0], observation_covariance=R, transition_covariance=Q)
        state_means, _ = kf.filter(self.df['Magnitude_avg'].values)
        self.df['kf_filtered'] = state_means
    def apply_kalman_smoothing(self, R=5, Q=0.1):
        from pykalman import KalmanFilter
        kf = KalmanFilter(initial_state_mean=self.df['Magnitude_avg'].iloc[0], observation_covariance=R, transition_covariance=Q)
        state_means, _ = kf.smooth(self.df['Magnitude_avg'].values)
        self.df['kf_smoothed'] = state_means
    def compute_derivative(self, column='Magnitude_avg'):
        self.df['dt'] = self.df['time'].diff().dt.total_seconds()
        self.df[f'{column}_derivative'] = self.df[column].diff() / self.df['dt']
class RawGyroscopeXYZData:
    def __init__(self, folder, date, suffix='_Accel.txt', i=None, save_as_csv=True):
        """
        Load and optionally merge chunked sensor files.

        Args:
           folder (str): Path to folder containing data files.
           date (str): The datetime prefix, e.g. '2025-07-15-12-27'.
           suffix (str): File suffix, e.g. '_Accel.txt' or '_Gyro.txt'.
          i (list): List of chunk indices to include, e.g. [1,2,3]. If None, loads only one file.
           save_as_csv (bool): Whether to save merged and timestamp-aligned CSV.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import re

        self.save_as_csv = save_as_csv
        self.folder = Path(folder)
        self.df = None
        self.filepath = None

        # Format filenames
        if i is None:
            i = [1]
        files = [self.folder / f"{date}-{ix}{suffix}" for ix in i]
    
        # Load and concatenate
        df_list = []
        for f in files:
            df = pd.read_csv(f, header=None, names=['x', 'y', 'z', 'timestamp_ns', 'unused'])
            df.drop(columns=['unused'], inplace=True)
            df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
            df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
            df['time_sec'] = df['time'].dt.floor('s')
            df_list.append(df)

        df_all = pd.concat(df_list, ignore_index=True).sort_values('time')

        # Align start time from filename
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', date)
        if not match:
            raise ValueError("Date must be in the form YYYY-MM-DD-HH-MM")
        year, month, day, hour, minute = match.groups()
        aligned_start = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
        offset = aligned_start - df_all['time'].iloc[0]
        df_all['time'] += offset
        df_all['time_sec'] = df_all['time'].dt.floor('s')

        self.df = df_all

        if save_as_csv:
            out_path = self.folder / f"{date}_combined{suffix.replace('.txt', '.csv')}"
            df_all.to_csv(out_path, index=False)
            print(f"Saved combined and aligned CSV to: {out_path}")

    def _load(self):
        import pandas as pd
        import numpy as np
        import re
        if self.filepath.suffix == '.csv':
            self.df = pd.read_csv(self.filepath, parse_dates=['time', 'time_sec'])
            print(f"Loaded cleaned CSV: {self.filepath.name}")
            return

        # Load raw .txt format
        df = pd.read_csv(self.filepath, header=None, names=['x', 'y', 'z', 'timestamp_ns', 'unused'])
        df.drop(columns=['unused'], inplace=True)

        df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df['time_sec'] = df['time'].dt.floor('s')

    # --- Align timestamp using filename ---
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', self.filepath.name)
        print(match)
        if match:
            year, month, day, hour, minute = match.groups()
            target_start = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
            current_start = df['time'].iloc[0]
            offset = target_start - current_start
            df['time'] += offset
            df['time_sec'] = df['time'].dt.floor('s')
            print(f"Shifted timestamps by {offset} to align with {target_start}")
        else:
            print("Could not extract timestamp from filename — no alignment applied.")

        self.df = df

        if self.save_as_csv:
            csv_path = self.filepath.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved cleaned CSV: {csv_path}")
    def resample_and_interpolate(self, interval='50ms'):
        """
        Resamples using 'timestamp_ns' converted to datetime, not relying on pre-parsed 'time' column.
        """
        df = self.df.copy()

        # Use timestamp_ns directly
        if 'timestamp_ns' not in df.columns:
            raise ValueError("timestamp_ns column is required to resample.")

        df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns', errors='coerce')
        df = df.dropna(subset=['time'])

        df.set_index('time', inplace=True)

        

        df_resampled = df.resample(interval).mean()
        df_interpolated = df_resampled.interpolate(method='linear')

        df_interpolated.reset_index(inplace=True)
        df_interpolated['magnitude'] = np.sqrt(
            df_interpolated['x']**2 + df_interpolated['y']**2 + df_interpolated['z']**2
        )
        df_interpolated['time_sec'] = df_interpolated['time'].dt.floor('s')

        self.df = df_interpolated
        print(f"Resampled to {interval}, new length: {len(self.df)}")
    def compute_derivative(self, column):
        self.df['dt'] = self.df['time'].diff().dt.total_seconds()
        self.df[f'{column}_derivative'] = self.df[column].diff() / self.df['dt']
        print('Derivative computed')
    def mark_flat_step_candidates(self, value_column='magnitude', derivative_column='magnitude_derivative', epsilon=1e-3):
        """
        Mark rows where the derivative is near zero and value is non-zero.
        """
        df = self.df
        near_zero_deriv = df[derivative_column].abs() < epsilon
        nonzero_value = df[value_column].abs() >= epsilon
        df['flat_step_candidate'] = near_zero_deriv & nonzero_value
        print(f"🪵 Marked {df['flat_step_candidate'].sum()} flat-step candidates.")

    def mark_sharp_zero_crossings(self, value_column='magnitude', derivative_column='magnitude_derivative', epsilon=1e-3):
        """
        Mark rows where the value is near zero and derivative is non-zero.
        """
        df = self.df
        near_zero_value = df[value_column].abs() < epsilon
        nonzero_deriv = df[derivative_column].abs() >= epsilon
        df['sharp_zero_crossing'] = near_zero_value & nonzero_deriv
        print(f"Marked {df['sharp_zero_crossing'].sum()} sharp-zero crossings.")

    def compute_orientation(self, other):
        from ahrs.filters import Madgwick
        from ahrs.common.orientation import q2euler
        import numpy as np
        import pandas as pd

        df_gyr = self.df.copy()
        df_acc = other.df.copy()
        self.df['time_change']=self.df['timestamp_ns'].diff().fillna(0) / 1e9
        time_changes = self.df['time_change'].values
        # Align both datasets to the same time base
        df = pd.merge_asof(df_acc.sort_values('time'), df_gyr.sort_values('time'),
                           on='time', direction='nearest', suffixes=('_acc', '_gyr'))

        # Ensure units
        acc = df[['x_acc', 'y_acc', 'z_acc']].to_numpy()
        gyr = df[['x_gyr', 'y_gyr', 'z_gyr']].to_numpy()
        madgwick = Madgwick()
        Q = np.tile([1.,0.,0.,0.],(len(acc), 1))
        min_len = min(len(acc), len(gyr), len(time_changes))
        for t in range(1, min_len):
            Q[t] = madgwick.updateIMU(Q[t-1], gyr[t], acc[t], sampleperiod=time_changes[t])
        eulers = np.array([q2euler(q) for q in Q])
        df['w'], df['x'], df['y'], df['z'] = Q.T
        df['roll'], df['pitch'], df['yaw'] = eulers.T
        self.df = df
        print(f"Orientation computed using Madgwick filter over {len(df)} samples")
class RawAccelerometerXYZData:
    def __init__(self, folder, date, suffix='_Accel.txt', i=None, save_as_csv=True):
        """
        Load and optionally merge chunked sensor files.

        Args:
           folder (str): Path to folder containing data files.
           date (str): The datetime prefix, e.g. '2025-07-15-12-27'.
           suffix (str): File suffix, e.g. '_Accel.txt' or '_Gyro.txt'.
          i (list): List of chunk indices to include, e.g. [1,2,3]. If None, loads only one file.
           save_as_csv (bool): Whether to save merged and timestamp-aligned CSV.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import re

        self.save_as_csv = save_as_csv
        self.folder = Path(folder)
        self.df = None
        self.filepath = None

        # Format filenames
        if i is None:
            i = [1]
        files = [self.folder / f"{date}-{ix}{suffix}" for ix in i]
    
        # Load and concatenate
        df_list = []
        for f in files:
            df = pd.read_csv(f, header=None, names=['x', 'y', 'z', 'timestamp_ns', 'unused'])
            df.drop(columns=['unused'], inplace=True)
            df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
            df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
            df['time_sec'] = df['time'].dt.floor('s')
            df_list.append(df)

        df_all = pd.concat(df_list, ignore_index=True).sort_values('time')

        # Align start time from filename
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', date)
        if not match:
            raise ValueError("Date must be in the form YYYY-MM-DD-HH-MM")
        year, month, day, hour, minute = match.groups()
        aligned_start = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
        offset = aligned_start - df_all['time'].iloc[0]
        df_all['time'] += offset
        df_all['time_sec'] = df_all['time'].dt.floor('s')

        self.df = df_all

        if save_as_csv:
            out_path = self.folder / f"{date}_combined{suffix.replace('.txt', '.csv')}"
            df_all.to_csv(out_path, index=False)
            print(f"Saved combined and aligned CSV to: {out_path}")

    def _load(self):
        import pandas as pd
        import numpy as np
        import re

        if self.filepath.suffix == '.csv':
            self.df = pd.read_csv(self.filepath, parse_dates=['time', 'time_sec'])
            print(f"Loaded cleaned CSV: {self.filepath.name}")
            return

        # Load raw .txt format
        df = pd.read_csv(self.filepath, header=None, names=['x', 'y', 'z', 'timestamp_ns', 'unused'])
        df.drop(columns=['unused'], inplace=True)

        df['time'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df['time_sec'] = df['time'].dt.floor('s')

    # --- Align timestamp using filename ---
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', self.filepath.name)
        if match:
            year, month, day, hour, minute = match.groups()
            target_start = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
            current_start = df['time'].iloc[0]
            offset = target_start - current_start
            df['time'] += offset
            df['time_sec'] = df['time'].dt.floor('s')
            print(f"Shifted timestamps by {offset} to align with {target_start}")
        else:
            print("Could not extract timestamp from filename — no alignment applied.")

        self.df = df

        if self.save_as_csv:
            csv_path = self.filepath.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved cleaned CSV: {csv_path}")
    def compute_orientation(self, other, frequency=50):
        from ahrs.filters import Madgwick
        from ahrs.common.orientation import q2euler
        import numpy as np
        import pandas as pd

        df_acc = self.df.copy()
        df_gyr = other.df.copy()

        # Align both datasets to the same time base
        df = pd.merge_asof(df_acc.sort_values('time'), df_gyr.sort_values('time'),
                           on='time', direction='nearest', suffixes=('_acc', '_gyr'))

        # Ensure units
        acc = df[['x_acc', 'y_acc', 'z_acc']].to_numpy()
        gyr = df[['x_gyr', 'y_gyr', 'z_gyr']].to_numpy()

        madgwick = Madgwick(frequency=frequency)
        quats = np.zeros((len(acc), 4))
        for t in range(1, len(acc)):
            quats[t] = madgwick.updateIMU(quats[t-1], gyr[t], acc[t])
        eulers = np.array([q2euler(q) for q in quats])

        df['roll'], df['pitch'], df['yaw'] = eulers.T
        self.df = df
        print(f"Orientation computed using Madgwick filter over {len(df)} samples")
