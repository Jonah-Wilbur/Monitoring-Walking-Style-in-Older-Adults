import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Gyroscope_data import RawGyroscopeXYZData
from Accelerometer_data import RawAccelerometerXYZData

def compute_energy_entropy(detail_coeffs):
    energy = np.sum(detail_coeffs**2)
    p = detail_coeffs**2 / energy if energy > 0 else np.zeros_like(detail_coeffs)
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return energy, entropy

def sliding_window_features(signal, fs=50, window_s=2.0, step_s=0.5, wavelet='db4'):
    window_len = int(window_s * fs)
    step_len = int(step_s * fs)
    n = len(signal)
    
    energies = []
    entropies = []
    times = []

    for start in range(0, n - window_len + 1, step_len):
        segment = signal[start:start+window_len]
        _, cD = pywt.dwt(segment, wavelet)
        e, h = compute_energy_entropy(cD)
        energies.append(e)
        entropies.append(h)
        times.append(start / fs)  # time in seconds

    return np.array(times), np.array(energies), np.array(entropies)

# Load your data
accel_data = RawAccelerometerXYZData(
    folder=r'C:\Watch_data\data_store',
    date='2025-07-24-16-07',
    suffix='-Accel.txt',
    save_as_csv=True,
    i=[1, 2]
)
gyro_data = RawGyroscopeXYZData(
    folder=r'C:\Watch_data\data_store',
    date='2025-07-24-16-07',
    suffix='-Gyro.txt',
    save_as_csv=True,
    i=[1, 2]
)
gyro_data.compute_orientation(other=accel_data)

# Process each axis
results = {}
for axis in ['x_acc', 'y_acc', 'z_acc']:
    times, energies, entropies = sliding_window_features(gyro_data.df[axis].values)
    results[axis] = {
        'times': times,
        'energy': energies,
        'entropy': entropies
    }

# Plot results
fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
for i, axis in enumerate(['x_acc', 'y_acc', 'z_acc']):
    axs[i, 0].plot(results[axis]['times'], results[axis]['energy'], label=f'{axis} energy')
    axs[i, 0].set_ylabel('Energy')
    axs[i, 1].plot(results[axis]['times'], results[axis]['entropy'], label=f'{axis} entropy', color='orange')
    axs[i, 1].set_ylabel('Entropy')
    axs[i, 0].legend(); axs[i, 1].legend()
axs[-1, 0].set_xlabel('Time (s)')
axs[-1, 1].set_xlabel('Time (s)')
plt.suptitle("DWT Energy and Entropy Over Time (Sliding Windows)")
plt.tight_layout()
plt.show()
