from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import pywt
from scipy.stats import skew
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
# Fit model
from sklearn.ensemble import RandomForestClassifier
from Gyroscope_data import RawGyroscopeXYZData, RawAccelerometerXYZData
from scipy.signal import periodogram
import xgboost as xgb
import re
# --- Feature extraction ---
def dominant_freq(signal: np.ndarray, fs: float) -> float:
    f, Pxx = periodogram(signal, fs)
    return f[np.argmax(Pxx)] if len(f) > 0 else 0.0


import pandas as pd
import numpy as np
def iqr(data):
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25
def resample_df(df, time_col='time', target_freq=20.0):
    df = df.sort_values(time_col)
    df[time_col] = pd.to_datetime(df[time_col], unit='ms')
    df.set_index(time_col, inplace=True)

    resampled = df.resample(f'{int(1000/target_freq)}ms').mean()
    resampled = resampled.interpolate(method='linear')  # or 'spline', 'time'
    
    resampled.reset_index(inplace=True)
    resampled[time_col] = resampled[time_col].astype(np.int64) // 10**6  # back to ms
    return resampled

def extract_features_from_gyro_df(df, label, fs=20.0, window_s=2.0, step_s=0.5):
    window_len = int(fs * window_s)
    step_len = int(fs * step_s)
    n = len(df)

    features = []
    labels = []

    for start in range(0, n - window_len + 1, step_len):
        end = start + window_len
        window = df.iloc[start:end]
        f = []


        # Dominant frequency from accel
        for axis in ['x_acc', 'y_acc', 'z_acc']:
            data = window[axis].values
            f.append(dominant_freq(data, fs=fs))
            f.append(np.sum(data**2))    # energy
            f.append(np.var(data))
            f.append(np.mean(data))
            f.append(np.median(data))
            f.append(iqr(data))    # interquartile range
            f.append(np.max(data))
            f.append(np.min(data))
            f.append(skew(data))    # skewness

       # variance
        # Dominant frequency from gyroscope
        for axis in ['x_gyr', 'y_gyr', 'z_gyr']:
            f.append(dominant_freq(data, fs=fs))
            f.append(np.sum(data**2))    # energy
            f.append(np.var(data))
            f.append(np.mean(data))
            f.append(np.median(data))
            f.append(iqr(data))    # interquartile range
            f.append(np.max(data))
            f.append(np.min(data))
            f.append(skew(data))    # skewness


        features.append(f)
        labels.append(label)

    return np.array(features), np.array(labels)

# --- Data loader ---
def load_data_from_labeled_subfolders(base_path: str,
                                      suffix_gyro='-Gyro.txt',
                                      suffix_accel='-Accel.txt') -> Tuple[np.ndarray, np.ndarray]:
    X_all = []
    y_all = []

    base = Path(base_path)
    for subfolder in base.iterdir():
        if not subfolder.is_dir():
            continue

        label = subfolder.name.lower()
        print(f"📂 Processing label: {label}")

        gyro_files = list(subfolder.glob(f"*{suffix_gyro}"))
        pattern = re.compile(r"(.+)-(\d+)" + re.escape(suffix_gyro))  # Matches "prefix-1-Gyro.txt"
        prefixes = []

        for f in gyro_files:
            match = pattern.match(f.name)
            if match:
                base_prefix, i = match.groups()
                prefixes.append((base_prefix, int(i)))

        for base_prefix, i in sorted(set(prefixes)):
            try:
                print(f"🔎 Loading files: {base_prefix}-{i}{suffix_accel} and {base_prefix}-{i}{suffix_gyro}")

                accel = RawAccelerometerXYZData(folder=subfolder, date=base_prefix, suffix=suffix_accel, i=[i], save_as_csv=False)
                gyro  = RawGyroscopeXYZData(folder=subfolder, date=base_prefix, suffix=suffix_gyro, i=[i], save_as_csv=False)
        
                gyro.compute_orientation(other=accel)
                gyro.df = resample_df(gyro.df, target_freq=20.0)

                print(f"🔍 Extracting features for {label}/{base_prefix}-{i}...")
                X, y = extract_features_from_gyro_df(gyro.df, label=label)
                X_all.append(X)
                y_all.append(y)

            except Exception as e:
                print(f"⚠️ Skipped {label}/{base_prefix}-{i} due to error: {e}")

    if not X_all:
        raise RuntimeError("No valid data was loaded. Check folder structure and filenames.")

    return np.vstack(X_all), np.concatenate(y_all)

# --- Run everything ---

base_dir = "C:/Watch_data/data_store"  # Update this if needed

X, y = load_data_from_labeled_subfolders(base_dir)

print(f"\n✅ Loaded {len(X)} windows from {np.unique(y).size} classes.")
for lbl in np.unique(y):
    print(f"  - {lbl}: {(y == lbl).sum()} windows")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train decision tree


def plot_feature_importance(model, feature_names=None):
    importances = model.feature_importances_
    idxs = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[idxs])
    if feature_names:
        plt.xticks(range(len(importances)), [feature_names[i] for i in idxs], rotation=45, ha='right')
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()


feature_names = ['x_acc_freq', 'x_acc_energy', 'x_acc_var', 'x_acc_mean', 'x_acc_median', 'x_acc_iqr', 'x_acc_max', 'x_acc_min', 'x_acc_skew',
                 'y_acc_freq', 'y_acc_energy', 'y_acc_var', 'y_acc_mean', 'y_acc_median', 'y_acc_iqr', 'y_acc_max', 'y_acc_min', 'y_acc_skew',
                 'z_acc_freq', 'z_acc_energy', 'z_acc_var', 'z_acc_mean', 'z_acc_median', 'z_acc_iqr', 'z_acc_max', 'z_acc_min', 'z_acc_skew',
                 'x_gyr_freq', 'x_gyr_energy', 'x_gyr_var', 'x_gyr_mean', 'x_gyr_median', 'x_gyr_iqr', 'x_gyr_max', 'x_gyr_min', 'x_gyr_skew',
                 'y_gyr_freq', 'y_gyr_energy', 'y_gyr_var', 'y_gyr_mean', 'y_gyr_median', 'y_gyr_iqr', 'y_gyr_max', 'y_gyr_min', 'y_gyr_skew',
                 'z_gyr_freq', 'z_gyr_energy', 'z_gyr_var', 'z_gyr_mean', 'z_gyr_median', 'z_gyr_iqr', 'z_gyr_max', 'z_gyr_min', 'z_gyr_skew']
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Converts strings to integers

# Now split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)




# Create model
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(np.unique(y)),
    random_state=42,
    colsample_bytree=0.8,
    subsample=0.8,
    max_depth=5,
    gamma=0,
    min_child_weight=1,
    learning_rate=0.2,
    n_estimators=100

)

xgb_model.fit(X_train, y_train)


# Evaluate
y_pred = xgb_model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)
print("\n🧠 Classification Report (Best XGBoost):")
print(classification_report(y_test_labels, y_pred_labels))
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict on test set
y_pred = xgb_model.predict(X_test)

# If you used LabelEncoder:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)  # Assuming 'y' contains all the labels

# Compute confusion matrix
cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=le.classes_)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
#new_base_path = "C:/Watch_data/Jeff_test"


#X_new, y_new = load_data_from_labeled_subfolders(new_base_path)
booster = xgb_model.get_booster()
booster.feature_names = feature_names
xgb.plot_importance(booster, importance_type='gain', max_num_features=60, height=0.5)
plt.title("XGBoost Feature Importance (by Gain)")
plt.grid(False)
plt.tight_layout()
plt.show()
#y_pred_new = xgb_model.predict(X_new)
#from sklearn.metrics import classification_report, confusion_matrix
#y_pred_labels = le.inverse_transform(y_pred_new)
#print(classification_report(y_new, y_pred_labels))
test_dir = 'C:/Watch_data/Time_based_test'
accel_test = RawAccelerometerXYZData(folder=test_dir, date='2025-08-07-12-46', suffix='-Accel.txt', i=[1, 2], save_as_csv=False)
gyro_test = RawGyroscopeXYZData(folder=test_dir, date='2025-08-07-12-46', suffix='-Gyro.txt', i=[1], save_as_csv=False)
gyro_test.compute_orientation(other=accel_test)
gyro_test.df = resample_df(gyro_test.df, target_freq=20.0)

print(f"🔍 Extracting features for test data...")
X_test_new, y_test_new = extract_features_from_gyro_df(gyro_test.df, label='test')
y_new_pred = xgb_model.predict(X_test_new)
y_new_pred_labels = le.inverse_transform(y_new_pred)
print(y_new_pred)