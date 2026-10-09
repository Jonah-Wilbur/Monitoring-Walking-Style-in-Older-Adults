#fourier transforms 
from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData
import matplotlib.pyplot as plt

import re
import pandas as pd
from pathlib import Path
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt



local_dir = r'C:\Watch_data\data_store'
accel_data = RawAccelerometerXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-07-13-25-1-Accel.csv',
    save_as_csv=False
)
gyro_data = RawGyroscopeXYZData(
    filepath=r'C:\Watch_data\data_store\2025-07-07-13-25-1-Gyro.csv',
    save_as_csv=False
)

gyro_data.compute_orientation(other=accel_data)
start = "2025-07-07 13:25:10"
end = "2025-07-07 13:25:25"
print(gyro_data.df.head())
gyro_data.df = gyro_data.df[(gyro_data.df['time'] >= start) & (gyro_data.df['time'] <= end)]
gyro_data.df.set_index('time', inplace=True)
gyro_data.df = gyro_data.df.resample('20ms').mean().interpolate()  # 50 Hz
signal = gyro_data.df['yaw'].values
from scipy.signal import butter, filtfilt

def highpass(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)

filtered_signal = highpass(signal, cutoff=0.5, fs=50)  # Cutoff = 0.5 Hz
fs = 50  # Sampling frequency
N = len(filtered_signal)  # Number of samples
from scipy.signal import windows

window = windows.hann(len(filtered_signal))
signal_windowed = filtered_signal * window
fft_out = fft(signal_windowed)
yaw_xf = fftfreq(N, 1/fs)
yaw_xf = yaw_xf[:N//2] # Take only the positive frequencies
yaw_yf = np.abs(fft_out[:N//2])
filtered_signal_2 = highpass(gyro_data.df['z_gyr'], cutoff=0.5, fs=50)  # Cutoff = 0.5 Hz
window = windows.hann(len(filtered_signal_2))
signal_windowed_2 = filtered_signal_2 * window
z_yf = fft(signal_windowed_2)
z_xf = fftfreq(N, 1/fs)
z_xf = z_xf[:N//2]  # Take only the positive frequencies
z_yf = np.abs(fft(signal_windowed_2)[:N//2])

print(yaw_yf[np.argmax(yaw_yf)])  # Find the index of the maximum value in the FFT output
print(yaw_xf[np.argmax(yaw_yf)])  # Find the index of the maximum value in the FFT output
print(z_yf[np.argmax(z_yf)])  # Find the index of the maximum value in the FFT output
print(z_xf[np.argmax(z_yf)])  # Find the index of the maximum value in the FFT output

fig, axs = plt.subplots(2, 2, figsize=(12, 6))
axs[0, 0].plot(yaw_xf, yaw_yf, label='FFT')
#ax.scatter(gyro_data.df['time'], gyro_data.df['pitch'], label='Pitch')
#ax.scatter(gyro_data.df['time'], gyro_data.df['yaw'], label='Yaw')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_title('Walking 10 steps - Frequency Domain')
axs[0, 0].set_ylabel('Magnitude')
axs[0, 1].plot(gyro_data.df.index, gyro_data.df['yaw'], label='Yaw', color='orange')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Yaw (radians)')
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].plot(gyro_data.df.index, gyro_data.df['z_gyr'], label='gyroscope', color='green')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Z (radians)')
axs[1, 1].plot(z_xf, z_yf, label='FFT of z', color='red')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_title('Gyroscope - Frequency Domain')
axs[1, 1].set_ylabel('Gyroscope Z Magnitude')
axs[1, 0].legend()
axs[1, 1].legend()
plt.tight_layout()
plt.show()

