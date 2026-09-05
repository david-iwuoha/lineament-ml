# Section 6 Results Summary — Lineament ML Model Training

## Setup
- Study area: Ibadan, Sheet 261, NGSA aeromagnetic TMI grid
- IGRF: already removed by NGSA (confirmed from real stats, mean -0.58 nT, range ~1000 nT)
- CRS: WGS_1984_UTM_Zone_31N, EPSG:32631 (confirmed from grd.xml)
- RTP inclination/declination: -11.3996 / -5.0348 deg (NOAA IGRF calc, DGRF2005, lat 7.25N lon 3.75E, epoch 2009-01-01)
- Consensus labels: agreement >=2 of 3 filters (ASA, TDR, HGM) at 90th percentile threshold
  - Sensitivity check: 85th pct = 12.93% labeled, 90th pct = 7.94%, 95th pct = 3.48%
- Train/test split: spatial block split (eastern 30% of grid held out), NOT random split,
  to avoid spatial autocorrelation leakage between neighboring pixels
- Circularity avoided: training features (TMI, RTP, 2nd vertical derivative, upward
  continuation) are disjoint from the three filters used to build consensus labels
  (ASA, TDR, HGM)

## Results

| Model | Features | Recall (True) | Precision (True) | F1 (True) |
|---|---|---|---|---|
| RF baseline | TMI + RTP only, unbalanced, threshold 0.5 | 0.119 | 0.595 | 0.198 |
| RF improved | TMI, RTP, 2nd vert deriv, upward continuation; class_weight=balanced; tuned threshold 0.72 | 0.385 | 0.306 | 0.341 |
| SVM improved | Same 4 features; class_weight=balanced; tuned threshold (decision function ~1.00) | 0.427 | 0.300 | 0.352 |

RF improved feature importances: TMI 0.40, RTP 0.20, 2nd vertical derivative 0.29, upward continuation 0.11

## Known caveats to state in Methods/Discussion
- A naive random pixel split (not used for final results) gave a fake 100% accuracy
  due to spatial autocorrelation leakage between adjacent training/test pixels — this is
  documented as a demonstration of why the spatial block split was necessary
- Including ASA/TDR/HGM directly as model features (not done in final results) also
  produces a fake ~100% accuracy, since consensus_label is a deterministic threshold
  function of those same three arrays — this is why the "improved" feature set uses
  TMI, RTP, 2nd vertical derivative, and upward continuation instead
- SVM was trained on a 50,000-row stratified subsample (not the full ~2.3M training
  pixels) due to O(n^2)-O(n^3) training complexity of RBF-kernel SVC; evaluated on the
  full test set for the baseline result and a 100,000-row stratified subsample for the
  improved/balanced result (27,000+ support vectors made full-test-set decision_function
  calls slow)
- Recall tops out around 40% for both models — roughly 60% of consensus-labeled
  lineaments are not recovered from TMI/RTP/derivative attributes alone. This is a
  real result to report and discuss, not a failure to fix.

## Files in this folder
- (feature_stack.npy was removed to save space — reconstruct it with:
  `np.stack([tmi, rtp_values, analytic_signal, tilt_derivative, horizontal_gradient], axis=-1)`
  where `tmi` is `grid.values` from reloading the original .grd file)
- consensus_label.npy — boolean lineament labels, shape (1845,1843)
- rtp_values.npy, analytic_signal.npy, tilt_derivative.npy, horizontal_gradient.npy — individual filter outputs
- second_vert_deriv.npy, upward_continuation.npy — the two additional independent features
- X_train.npy/X_test.npy/y_train.npy/y_test.npy — baseline (2-feature) spatial split
- X_train_v2.npy/X_test_v2.npy/y_train_v2.npy/y_test_v2.npy — improved (4-feature) spatial split
- rf_model.pkl — baseline RF model
- rf_v2_incremental.pkl — improved RF model (35 trees, depth 15, class-balanced)
- svm_v2.pkl — improved SVM model (50k subsample, class-balanced)
- run_pipeline.py — full reproducible pipeline script (Sections 1-6)
