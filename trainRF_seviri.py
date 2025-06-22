import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import os
import tarfile
import gzip
import shutil
from datetime import datetime, timedelta
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import  precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib


os.makedirs('results_RF', exist_ok=True)
# Configuration
zarr_path = "trentino_satellite_2014-01.zarr"
days_of_interest = [
    "2014-01-12", "2014-01-13", "2014-01-15", "2014-01-16",
    "2014-01-17", "2014-01-18", "2014-01-19", "2014-01-20", "2014-01-21",
    "2014-01-24", "2014-01-25", "2014-01-26", "2014-01-28", "2014-01-29", "2014-01-30"
]
days_of_interest = pd.to_datetime(days_of_interest)
valid_minutes = ["00", "15", "30", "45"]  # keep only 15 min interval data and remove 5 min

# Loading SEVIRI
ds = xr.open_zarr(zarr_path)
# I have randomely choosen the channels
selected_channels = ["IR_108", "IR_120", "WV_062", "WV_073", "VIS008", "IR_134"]

# Confirm they exist in the dataset
channels = [ch for ch in selected_channels if ch in ds.data_vars and ds[ch].dims == ('time', 'y', 'x')]

print("Using channels:", channels)
n_channels = len(channels)

# Filter time indices to select only th days avaible in the data and also the minutes
#  (because there are a lot of missing times)
time_index = ds.time.to_index()
selected_times = time_index[(time_index.floor('D').isin(days_of_interest)) & 
                            (time_index.strftime('%M').isin(valid_minutes))]

# X shape: (squences (1440 if no time is missing), n_channels, 480, 480)
X_list = []
for ch in channels:
    data = ds[ch].sel(time=selected_times).values  # shape (1440, 480, 480) only with the existing times
    X_list.append(data)

X = np.stack(X_list, axis=1)  # shape: (1440, n_channels, 480, 480)

# Load TAASDAAR Data from .tar
# extract ascii files from tar file
def extract_ascii_from_tar(tar_path):
    ascii_dict = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".ascii.gz"):
                # Extract timestamp from filename
                base = os.path.basename(member.name)  # e.g cmaZ201401111815.ascii.gz
                timestamp_str = base[4:16]  # "201401111815"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                except ValueError:
                    continue  

                f = tar.extractfile(member)
                with gzip.open(f, 'rt') as gz:
                    lines = gz.readlines()[6:]
                    arr = np.array([list(map(float, line.strip().split())) for line in lines])
                    arr[arr == -99.0] = np.nan
                    ascii_dict[timestamp] = arr
    return ascii_dict  # {datetime: array}

# Load all days
taasdaar_folder = "january" 
Y = []
valid_times = []
grouped_by_day = selected_times.to_series().groupby(selected_times.floor('D')) # to use same timestamps found in seviri

for day, times in grouped_by_day:
    tar_file = os.path.join(taasdaar_folder, day.strftime('%Y%m%d') + ".tar")
    if not os.path.exists(tar_file):
        print(f"Missing TAASDAAR for {day.strftime('%Y-%m-%d')}")
        continue

    ascii_data = extract_ascii_from_tar(tar_file)  # now a dict {datetime: array}

    for dt in times:
        if dt in ascii_data:
            Y.append(ascii_data[dt])
            valid_times.append(dt)
        else:
            print(f"Missing frame at {dt} in TAASDAAR file {tar_file}")
# Convert to arrays
Y = np.array(Y)  # shape: (1194, 480, 480) (i got 1194 timestaps)

# Align X as well
X_aligned = ds[channels].sel(time=valid_times).to_array().transpose('time', 'variable', 'y', 'x').values
print("SEVIRI shape (X):", X_aligned.shape)
print("TAASDAAR shape (Y):", Y.shape)


# Flatten for training
n_samples = X_aligned.shape[0]
X_rf = X_aligned.reshape(n_samples, n_channels, -1).transpose(0, 2, 1).reshape(-1, n_channels)  # (1194*480*480, n_channels)
Y_rf = Y.reshape(-1)  # (1194*480*480,)

mask = ~np.isnan(Y_rf)
X_rf = X_rf[mask]
Y_rf = Y_rf[mask]
print("X_rf shape:", X_rf.shape)
print("Y_rf shape:", Y_rf.shape)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_rf, Y_rf, test_size=0.2, random_state=42)

# Train
model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "results_RF/rf_precip_model2.pkl")

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R²: {r2:.2f}")


with open('results_RF/results.txt', 'w') as f:
    f.write(f"Test RMSE: {rmse:.2f}\n")
    f.write(f"Test MAE: {mae:.2f}\n")
    f.write(f"Test R²: {r2:.2f}\n")


# save y_test and y_pred
np.save("results_RF/y_test.npy", y_test)
np.save("results_RF/y_pred.npy", y_pred)

# Visiualization
# Plot true vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3)
plt.plot([0, max(y_test[:1000])], [0, max(y_test[:1000])], '--', color='red')
plt.xlabel("True Precipitation (mm)")
plt.ylabel("Predicted Precipitation (mm)")
plt.title("Random Forest: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("results_RF/scatter_true_vs_pred_test.png")
plt.show()


# Plot Line Chart (True vs Predicted) 
plt.figure(figsize=(12, 5))
plt.plot(y_test[:500], label='True', linewidth=1.5)
plt.plot(y_pred[:500], label='Predicted', linewidth=1.5)
plt.title("Precipitation Prediction (First 500 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Precipitation (mm/h)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results_RF/line_true_vs_pred.png")
plt.show()

# Classification by Rain Intensity 
def classify_rain(value):
    if value <= 0.1:
        return 0
    elif value <= 1:
        return 1
    elif value <= 5:
        return 2
    else:
        return 3

y_test_class = np.vectorize(classify_rain)(y_test)
y_pred_class = np.vectorize(classify_rain)(y_pred)

# Plot the first 100 samples 
plt.figure(figsize=(14, 6))
plt.step(range(100), y_test_class[:100], label="True Class", where='mid', linewidth=2, color='royalblue')
plt.step(range(100), y_pred_class[:100], label="Predicted Class", where='mid', linewidth=2, color='darkorange')
plt.yticks([0, 1, 2, 3], ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain'])
plt.xlabel("Sample Index (time or pixel)", fontsize=12)
plt.ylabel("Rain Intensity Class", fontsize=12)
plt.title("Rain Intensity Classification (True vs Predicted)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("results_RF/classification_rain_intensity.png")
plt.show()

#  Compute and Save Metrics ---
binary_y_test = (y_test > 0.1).astype(int)
binary_y_pred = (y_pred > 0.1).astype(int)

binary_conf_matrix = confusion_matrix(binary_y_test, binary_y_pred)
tn, fp, fn, tp = binary_conf_matrix.ravel()

POD = tp / (tp + fn) if (tp + fn) > 0 else 0
FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
BIAS = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0
precision = precision_score(binary_y_test, binary_y_pred)
recall = recall_score(binary_y_test, binary_y_pred)

metrics = {
    "POD": POD,
    "FAR": FAR,
    "BIAS": BIAS,
    "Precision": precision,
    "Recall": recall,
    "Confusion Matrix": binary_conf_matrix.tolist()
}

with open("results_RF/classification_metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
