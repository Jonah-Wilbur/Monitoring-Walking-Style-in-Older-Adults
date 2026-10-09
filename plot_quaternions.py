import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData
from scipy.signal import butter, filtfilt
def plot_quaternion_components(df, time_column='time', q_columns=['w', 'x', 'y', 'z'], gimbal_events=None):
    """
    Plot quaternion components over time.
    
    Args:
        df (pd.DataFrame): DataFrame containing quaternion columns and time.
        time_column (str): Name of the time column (datetime or float).
        q_columns (list): List of quaternion column names in order [w, x, y, z].
        gimbal_events (list): Optional list of timestamps to mark suspected gimbal lock events.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for i, q in enumerate(q_columns):
        if q in df.columns:
            ax[i].plot(df[time_column], df[q], label=q, linewidth=1.5)
            ax[i].set_ylabel(q)
            ax[i].grid(True)
            ax[i].legend(loc='upper right')
            if gimbal_events:
                for event in gimbal_events:
                    ax[i].axvline(pd.to_datetime(event), color='red', linestyle='--', alpha=0.7)

    ax[-1].set_xlabel('Time')
    plt.suptitle('Quaternion Components Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
gyro_data = RawGyroscopeXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-14-13-56-1-Gyro.csv',
    save_as_csv=False
)
accel_data = RawAccelerometerXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-14-13-56-1-Accel.csv',
    save_as_csv=False
)
gyro_data.compute_orientation(other=accel_data)
plot_quaternion_components(gyro_data.df, time_column='time', q_columns=['w', 'x', 'y', 'z'])