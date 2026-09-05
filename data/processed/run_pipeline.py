import harmonica as hm
import numpy as np
import xml.etree.ElementTree as ET
import time
import json

print("=" * 60)
print("STEP 1: Load grid")
print("=" * 60)
grid = hm.load_oasis_montaj_grid("data/raw/261_TMI.grd")
print("Shape:", grid.shape, "dims:", grid.dims)

print("\n" + "=" * 60)
print("STEP 2: Real stats (non-NaN)")
print("=" * 60)
valid = grid.values[~np.isnan(grid.values)]
print(f"Min:    {valid.min():.2f} nT")
print(f"Max:    {valid.max():.2f} nT")
print(f"Mean:   {valid.mean():.2f} nT")
print(f"Median: {np.median(valid):.2f} nT")
print(f"Std:    {valid.std():.2f} nT")

print("\n" + "=" * 60)
print("STEP 3: Boundary check")
print("=" * 60)
print("Easting range:", grid.easting.values.min(), grid.easting.values.max())
print("Northing range:", grid.northing.values.min(), grid.northing.values.max())

print("\n" + "=" * 60)
print("STEP 4: RTP + derivatives")
print("=" * 60)
inclination = -11.3996
declination = -5.0348

nan_mask = np.isnan(grid.values)
grid_filled = grid.fillna(grid.mean())

t0 = time.time()
rtp_grid = hm.reduction_to_pole(grid_filled, inclination=inclination, declination=declination)
rtp_grid.values[nan_mask] = np.nan

dx = hm.derivative_easting(grid_filled)
dy = hm.derivative_northing(grid_filled)
dz = hm.derivative_upward(grid_filled)
for arr in (dx, dy, dz):
    arr.values[nan_mask] = np.nan

analytic_signal = np.sqrt(dx**2 + dy**2 + dz**2)
analytic_signal.values[nan_mask] = np.nan

tilt_derivative = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
tilt_derivative.values[nan_mask] = np.nan

horizontal_gradient = np.sqrt(dx**2 + dy**2)
horizontal_gradient.values[nan_mask] = np.nan

feature_stack = np.stack([
    grid.values, rtp_grid.values, analytic_signal.values,
    tilt_derivative.values, horizontal_gradient.values,
], axis=-1)
print(f"Done in {time.time()-t0:.1f}s. Feature stack shape:", feature_stack.shape)
print("NaN count:", np.isnan(feature_stack).sum())

print("\n" + "=" * 60)
print("STEP 5: Consensus labels")
print("=" * 60)
def threshold_filter(arr, percentile=90):
    v = arr[~np.isnan(arr)]
    thr = np.percentile(v, percentile)
    return arr > thr

asa_flag = threshold_filter(analytic_signal.values)
tdr_flag = threshold_filter(np.abs(tilt_derivative.values))
hgm_flag = threshold_filter(horizontal_gradient.values)
agreement = asa_flag.astype(int) + tdr_flag.astype(int) + hgm_flag.astype(int)
consensus_label = agreement >= 2
print("Labeled as lineament:", consensus_label.sum(), f"({100*consensus_label.mean():.2f}%)")

print("\nSensitivity check:")
for p in (85, 90, 95):
    a = threshold_filter(analytic_signal.values, p)
    t = threshold_filter(np.abs(tilt_derivative.values), p)
    h = threshold_filter(horizontal_gradient.values, p)
    lbl = (a.astype(int) + t.astype(int) + h.astype(int)) >= 2
    print(f"  Percentile {p}: {lbl.sum()} pixels ({100*lbl.mean():.2f}%)")

print("\n" + "=" * 60)
print("STEP 6: Save intermediate arrays")
print("=" * 60)
np.save("data/processed/feature_stack.npy", feature_stack)
np.save("data/processed/consensus_label.npy", consensus_label)
np.save("data/processed/rtp_values.npy", rtp_grid.values)
np.save("data/processed/analytic_signal.npy", analytic_signal.values)
np.save("data/processed/tilt_derivative.npy", tilt_derivative.values)
np.save("data/processed/horizontal_gradient.npy", horizontal_gradient.values)
print("Saved.")

print("\n" + "=" * 60)
print("STEP 7: Clean feature stack (TMI + RTP only) + spatial block split")
print("=" * 60)
feature_stack_clean = np.stack([grid.values, rtp_grid.values], axis=-1)

n_rows, n_cols = consensus_label.shape
row_idx, col_idx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

X_full = feature_stack_clean.reshape(-1, feature_stack_clean.shape[-1])
y_full = consensus_label.flatten()
col_full = col_idx.flatten()

valid_rows = ~np.isnan(X_full).any(axis=1)
X = X_full[valid_rows]
y = y_full[valid_rows]
col_v = col_full[valid_rows]

col_threshold = np.percentile(col_v, 70)
train_mask = col_v < col_threshold
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print("Train pixels:", X_train.shape[0], "Test pixels:", X_test.shape[0])
print("Train lineament frac:", y_train.mean(), "Test lineament frac:", y_test.mean())

print("\n" + "=" * 60)
print("STEP 8: Random Forest (TMI+RTP only, spatial split)")
print("=" * 60)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print(f"RF fit time: {time.time()-t0:.1f}s")

rf_preds = rf.predict(X_test)
print(classification_report(y_test, rf_preds, digits=3))
print(confusion_matrix(y_test, rf_preds))

print("\n" + "=" * 60)
print("STEP 9: SVM (TMI+RTP only, spatial split, subsampled)")
print("=" * 60)
from sklearn.svm import SVC
from sklearn.utils import resample

X_train_svm, y_train_svm = resample(X_train, y_train, n_samples=50000, stratify=y_train, random_state=42)
t0 = time.time()
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train_svm, y_train_svm)
print(f"SVM fit time: {time.time()-t0:.1f}s")

svm_preds = svm.predict(X_test)
print(classification_report(y_test, svm_preds, digits=3))
print(confusion_matrix(y_test, svm_preds))

print("\nALL DONE.")
