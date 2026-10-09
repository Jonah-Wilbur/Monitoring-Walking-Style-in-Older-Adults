from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import pywt
# Fit model
from sklearn.ensemble import RandomForestClassifier
from Gyroscope_data import RawGyroscopeXYZData, RawAccelerometerXYZData
from scipy.signal import periodogram
import xgboost as xgb
# --- Feature extraction ---
def dominant_freq(signal: np.ndarray, fs: float) -> float:
    f, Pxx = periodogram(signal, fs)
    return f[np.argmax(Pxx)] if len(f) > 0 else 0.0

def wavelet_energy_features(signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs[1:]]

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
            f.append(dominant_freq(window[axis].values, fs=fs))
        for axis in ['x_acc', 'y_acc', 'z_acc']:
            data = window[axis].values
            f.append(np.sum(data**2))    # energy
            f.append(np.var(data))       # variance
        # Dominant frequency from gyroscope
        for axis in ['x_gyr', 'y_gyr', 'z_gyr']:
            f.append(dominant_freq(window[axis].values, fs=fs))
        # Energy + variance from gyroscope
        for axis in ['x_gyr', 'y_gyr', 'z_gyr']:
            data = window[axis].values
            f.append(np.sum(data**2))    # energy
            f.append(np.var(data))       # variance

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
        prefixes = sorted(f.name.replace(suffix_gyro, '').rsplit('-', 1)[0] for f in gyro_files)

        for prefix in prefixes:
            try:
                print(f"🔎 Loading files: {prefix}-1-Accel.txt and {prefix}-1-Gyro.txt")

                accel = RawAccelerometerXYZData(folder=subfolder, date=prefix, suffix=suffix_accel, i=[1], save_as_csv=False)
                gyro = RawGyroscopeXYZData(folder=subfolder, date=prefix, suffix=suffix_gyro, i=[1], save_as_csv=False)
                gyro.compute_orientation(other=accel)

                X, y = extract_features_from_gyro_df(gyro.df, label=label)
                X_all.append(X)
                y_all.append(y)

            except Exception as e:
                print(f"⚠️ Skipped {label}/{prefix} due to error: {e}")

    if not X_all:
        raise RuntimeError("No valid data was loaded. Check folder structure and filenames.")

    return np.vstack(X_all), np.concatenate(y_all)

# --- Run everything ---
if __name__ == "__main__":
    base_dir = "C:/Watch_data/data_store"  # Update this if needed

    X, y = load_data_from_labeled_subfolders(base_dir)

    print(f"\n✅ Loaded {len(X)} windows from {np.unique(y).size} classes.")
    for lbl in np.unique(y):
        print(f"  - {lbl}: {(y == lbl).sum()} windows")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train decision tree
    
    clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)


    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n🧠 Classification Report:")
    print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_confusion(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()







y_pred = clf.predict(X_test)
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))
plot_tree(clf.estimators_[0],  # first tree in the forest
          feature_names=[
              'x_acc_freq', 'y_acc_freq', 'z_acc_freq',
                'x_acc_energy', 'y_acc_energy', 'z_acc_energy',
                'x_acc_var', 'y_acc_var', 'z_acc_var',
                'x_gyr_freq', 'y_gyr_freq', 'z_gyr_freq',
                'x_gyr_energy', 'y_gyr_energy', 'z_gyr_energy',
                'x_gyr_var', 'y_gyr_var', 'z_gyr_var'
          ],
          class_names=clf.classes_,
          filled=True)
plt.title("One Tree from Random Forest")
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import shap


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
plot_feature_importance(clf, feature_names=[
    'x_acc_freq', 'y_acc_freq', 'z_acc_freq',
    'x_acc_energy', 'y_acc_energy', 'z_acc_energy',
    'x_acc_var', 'y_acc_var', 'z_acc_var',
    'x_gyr_freq', 'y_gyr_freq', 'z_gyr_freq',
    'x_gyr_energy', 'y_gyr_energy', 'z_gyr_energy',
    'x_gyr_var', 'y_gyr_var', 'z_gyr_var'
])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Converts strings to integers

# Now split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define the parameter grid to search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'min_child_weight': [1, 3]
}

# Create model
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(np.unique(y)),
    random_state=42
)

# Grid search with 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='f1_macro'  # or 'accuracy', 'recall_macro' etc.
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\n✅ Best Parameters:")
print(grid_search.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
print("\n🧠 Classification Report (Best XGBoost):")
print(classification_report(y_test, y_pred))
# Create a SHAP explainer for your random forest
#explainer = shap.Explainer(clf, X_train)
#class_index = list(clf.classes_).index('walking_on_stairs')  # Adjust this to your target class
# Compute SHAP values for the test set
#shap_values = explainer(X_test)
#shap.plots.beeswarm(shap_values[:,:,class_index], max_display=10)
#shap.plots.waterfall(shap_values[0,:,class_index])