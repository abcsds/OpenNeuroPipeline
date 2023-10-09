# 25 Sep
Setup pipeline to run with "ds004324".
Running to evaluate results.

# 26 Sep
Pipeline running. SVM hangs. Optuna doesn't timeout with SVMs somehow.

Possible upgrades:
check for hung processes (multiprosses, or celery)
restart hung processes (celery?)
logging (celery?)
Result and model storage (mlflow?)

# 27 Sep
Added report file: Removes subjects where k<=0, normalizes feature importance by k, and visualizes experiment-wide feature importance.

Posible upgrade:
Feature distance between subjects with k<=0 and k>0

Update pipeline: store features.

# 28 Sep
Pipeline: Fix saving settings by removing the relabeling function. EEG info is loaded from raw if the features have been extracted already.
Merge result files: `results.csv`.
Add feature importance normalization by kappa.

# 29 Sep
Catch optuna internal storage error. Probably from NaNs in LDA solver. Pipeline running...

# 03 Oct
Fix settings not storing feat_names. Add visualizations script.
Update pipeline eeg preprocessing to match `abstract`.
Commiting results before re-running pipeline.
Add pdc.

# 04 Oct
Remove epoch rejection by ptp amplitude.

# 05 Oct
Bug in relabel function. Have to redo all studies.
Add visualization scripts for different relabeling.
Change relabel function in pipeline. Rerun pipeline.

# 06 Oct
Bad channel visual inspection from 5-second epochs.
Create condition classification pipeline.

# 09 Oct
Presented results to supervisor. Conc: Let dataset die.