import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from pathlib import Path
from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData
from scipy.signal import butter, filtfilt
def highpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)
def plot_motion_and_fft(df, fs=20, cutoff=0.5):
    """
    Creates a 6x3 grid of high-pass filtered signals organized by axis:
    Column 1: x_acc, x_gyr, pitch and their FFTs
    Column 2: y_acc, y_gyr, roll and their FFTs     
    Column 3: z_acc, z_gyr, yaw and their FFTs
    """
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(18, 14))
    axes = axes.reshape((6, 3))

    labels = ['x', 'y', 'z']
    angles = ['pitch', 'roll', 'yaw']

    for i, (axis, angle) in enumerate(zip(labels, angles)):
        # Acceleration time-series
        acc_col = f'{axis}_acc'
        if acc_col in df.columns:
            acc = highpass(df[acc_col] - df[acc_col].mean(), cutoff, fs)
            axes[0][i].plot(df['time'], acc)
            axes[0][i].set_title(f"{axis.upper()} Acceleration (HP)")
            axes[0][i].set_ylabel("Accel (m/s²)")

            N = len(acc)
            fft = np.abs(np.fft.fft(acc * np.hanning(N)))
            freq = np.fft.fftfreq(N, 1/fs)
            pos = freq > 0
            axes[1][i].plot(freq[pos], fft[pos])
            axes[1][i].set_title(f"{axis.upper()} Accel FFT")
            axes[1][i].set_ylabel("Magnitude")

        # Gyroscope time-series
        gyr_col = f'{axis}_gyr'
        if gyr_col in df.columns:
            gyr = highpass(df[gyr_col] - df[gyr_col].mean(), cutoff, fs)
            axes[2][i].plot(df['time'], gyr)
            axes[2][i].set_title(f"{axis.upper()} Gyroscope (HP)")
            axes[2][i].set_ylabel("Gyro (rad/s)")

            N = len(gyr)
            fft = np.abs(np.fft.fft(gyr * np.hanning(N)))
            freq = np.fft.fftfreq(N, 1/fs)
            pos = freq > 0
            axes[3][i].plot(freq[pos], fft[pos])
            axes[3][i].set_title(f"{axis.upper()} Gyro FFT")
            axes[3][i].set_ylabel("Magnitude")

        # Angle time-series and FFT
        if angle in df.columns:
            ang = highpass(df[angle] - df[angle].mean(), cutoff, fs)
            axes[4][i].plot(df['time'], ang)
            axes[4][i].set_title(f"{angle.capitalize()} (HP)")
            axes[4][i].set_ylabel("Radians")

            N = len(ang)
            fft = np.abs(np.fft.fft(ang * np.hanning(N)))
            freq = np.fft.fftfreq(N, 1/fs)
            pos = freq > 0
            axes[5][i].plot(freq[pos], fft[pos])
            axes[5][i].set_title(f"{angle.capitalize()} FFT")
            axes[5][i].set_ylabel("Magnitude")
            axes[5][i].set_xlabel("Frequency (Hz)")

    for row in range(6):
        for col in range(3):
            axes[row][col].grid(False)
    plt.tight_layout()
    plt.show()
local_dir = r'C:\Watch_data\data_store'


accel_data = RawAccelerometerXYZData(
    folder=r'C:\Watch_data\data_store\Walking_upstairs_with_railing',
    date='2025-07-28-12-16',
    suffix='-Accel.txt',
    save_as_csv=True,
    i=[1]
)
gyro_data = RawGyroscopeXYZData(
    folder=r'C:\Watch_data\data_store\Walking_upstairs_with_railing',
    date='2025-07-28-12-16',
    suffix='-Gyro.txt',
    save_as_csv=True,
    i=[1]
)


gyro_data.compute_orientation(other=accel_data)
print(gyro_data.df.head())
plot_motion_and_fft(gyro_data.df, fs=50)