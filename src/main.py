import os
import openneuro as on
import mne
import optuna
import sklearn
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne_features.feature_extraction import FeatureExtractor, get_bivariate_func_names, get_univariate_func_names
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import shutil
import json

# Define the dataset ID and download path
dataset_id = "ds004324"
download_path = "./eeg_data"

# Re-labeling function: Used to edit the markers in the EEG data
def relabel(events, event_dict):
    n = events.shape[0]
    idx = [i % 5 == 0 for i in range(n)] #
    e = events.copy()
    reverse_event_dict = {v: k for k,v in event_dict.items()}
    labels = set([reverse_event_dict[i] for i in e[idx, 2]])
    l = {l: int(l.split("/")[1] == "1") for l in labels if "/" in l} # Selecting only HAPV vs Rest
    for i in labels:
        if i not in l:
            l[i] = 0
    e[idx,2] = np.array([l[reverse_event_dict[i]] for i in events[idx,2]])
    return e[idx,:], l

# Define EEG data processing settings
settings = {
    "rs": 42,
    "l_freq": 1, 
    "h_freq": 40,
    "notch_filter": 50,
    "CAR": False,
    "bads": [],
    "outf": "./results/",
    "drop_channels": ['EOGR', 'EOGU', 'EOGD', 'EOGL', 'ECG', 'GSR', 'x_dir', 'y_dir', 'z_dir', 'MkIdx'],
    "eog_channels": ["EOGU", "EOGD", "EOGL", "EOGR"],
    "stim_channels": ["MkIdx"],
    "ecg_channels": ["ECG"],
    "misc_channels": ["GSR", "x_dir", "y_dir", "z_dir"],
    "montage": "standard_1020",
    # Window
    "tmin": -0.2,
    "tmax": 5.0,
    "baseline": [-0.2, 0.0],
    # Relabeling
    "relabel_func": relabel,
    # Features
    "selected_feats": [
        "mean",  # chnn,
        "variance",  # chnn,
        "std",  # chnn,
        "ptp_amp",  # chnn,
        "skewness",  # chnn,
        "kurtosis",  # chnn,
        "rms",  # chnn,
        "quantile",  # chnn,
        "hurst_exp",  # chnn,
        "app_entropy",  # chnn,
        "samp_entropy",  # chnn,
        "decorr_time",  # chnn,
        "pow_freq_bands",  # band_ch,
        "hjorth_mobility",  # chnn,
        "hjorth_complexity",  # chnn,
        "higuchi_fd",  # chnn,
        "katz_fd",  # chnn,
        "zero_crossings",  # chnn,
        "line_length",  # chnn,
        "spect_entropy",  # chnn,
        "svd_entropy",  # chnn,
        "svd_fisher_info",  # chnn,
        "energy_freq_bands",  # band_ch,
        "spect_edge_freq",  # chnn,
        # "wavelet_coef_energy",  # band_ch, + chnn
        # "teager_kaiser_energy",  # band_ch * 2 + 2,
        # Bivariate
        "max_cross_corr",  # mv_chs_no_self,
        "phase_lock_val",  # mv_chs_no_self,
        # "nonlin_interdep",  # mv_chs_no_self, # Takes too long
        "time_corr",  # mv_chs,
        "spect_corr",  # mv_chs,
        ],
    }

# Create a directory to save results
os.makedirs(settings["outf"], exist_ok=True)

# Define EEG file processing functions
def process_eeg_file(eeg_file):
    global settings
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    if settings["drop_channels"]:
        raw = raw.drop_channels(settings["drop_channels"])
    if settings["montage"]:
        raw.set_montage(settings["montage"])
    if settings["bads"]:
        raw.info["bads"] = settings["bads"]
        raw.interpolate_bads()
        # raw.interpolate_bads(, exclude=settings["eog_channels"] + settings["stim_channels"] + settings["ecg_channels"] + settings["misc_channels"])
    raw.pick_types(eeg=True, stim=False, exclude='bads')
    ## Filtering
    if settings["notch_filter"]:
        raw.notch_filter(settings["notch_filter"])
    if settings["l_freq"] and settings["h_freq"]:
        raw.filter(l_freq=settings["l_freq"], h_freq=settings["h_freq"])
    elif settings["l_freq"]:
        raw.filter(l_freq=settings["l_freq"])
    elif settings["h_freq"]:
        raw.filter(h_freq=settings["h_freq"])
    ## Rereferencing
    if settings["CAR"]:
        mne.set_eeg_reference(raw, 'average', ch_type="eeg", copy=False)
        # raw.plot()
    ## Epoching
    events, event_dict = mne.events_from_annotations(raw)
    assert len(events) == len(set(i["onset"] for i in raw.annotations)), f"Annotations share onset {eeg_file}"
    assert np.abs(np.diff([i["onset"] for i in raw.annotations])).min() > 0.0, f"Annotations share onset {eeg_file}"

    if settings["relabel_func"]:
        events, event_dict = settings["relabel_func"](events, event_dict)

    epochs = mne.Epochs(raw, events,
        event_id=event_dict,
        tmin=settings["tmin"],
        tmax=settings["tmax"],
        baseline=settings["baseline"],
        preload=True,
    )
    data = epochs.get_data()
    
    settings["sfreq"] = epochs.info["sfreq"]
    settings["n_channels"] = epochs.info["nchan"]
    settings["ch_names"] = epochs.info["ch_names"]
    settings["eeg_ch_names"] = epochs.ch_names
    settings["win_len"] = data.shape[-1]

    # Define EEG feature extraction functions (replace with your desired features)
    if settings["selected_feats"]:
        selected_feats = settings["selected_feats"]
    else:
        selected_feats =  get_univariate_func_names() + get_bivariate_func_names()
    
    # Extract features
    feat_extractor = Pipeline([
        ('fe', FeatureExtractor(sfreq=settings["sfreq"], selected_funcs=selected_feats)),
        ('scaler', StandardScaler()),
    ])

    # Time and extract feature space size
    tic = time.time()
    settings["n_feats"] = feat_extractor.fit_transform(np.random.randn(1, settings["n_channels"], settings["win_len"])).shape[1]
    toc = time.time()
    settings["time_per_sample"] = (toc - tic)

    X = feat_extractor.fit_transform(epochs.get_data())
    y = epochs.events[:, 2]
    y = LabelEncoder().fit_transform(y)
    return X, y

# Define machine learning model training and evaluation functions
def train_model(X, y, model_name, subj):
    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=10000)
    elif model_name == "SVM":
        clf = SVC()
    elif model_name == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif model_name == "RandomForest":
        clf = RandomForestClassifier()
    else:
        raise ValueError("Invalid model name")
    
    # Define Optuna study
    study = optuna.create_study(direction="maximize", study_name=model_name+"_"+subj, storage=f"sqlite:///experiments.db", load_if_exists=True)
    
    # Define hyperparameter search space (customize for each model)
    if model_name == "LogisticRegression":
        param_distributions = {
            "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
        }
    elif model_name == "SVM":
        param_distributions = {
            "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
            "kernel": optuna.distributions.CategoricalDistribution(["linear", "poly", "rbf", "sigmoid"]),
            "degree": optuna.distributions.IntDistribution(1, 5),
            "gamma": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
        }
    elif model_name == "LDA":
        param_distributions = {
            "solver": optuna.distributions.CategoricalDistribution(["svd", "lsqr", "eigen"]),
        }
    elif model_name == "RandomForest":
        param_distributions = {
            "n_estimators": optuna.distributions.IntDistribution(10, 500),
            "criterion": optuna.distributions.CategoricalDistribution(["gini", "entropy"]),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_samples_split": optuna.distributions.IntDistribution(2, 15),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 20),
        }
    else:
        raise ValueError("Invalid model name")
    
    # Use OptunaSearchCV for hyperparameter tuning
    optuna_search = optuna.integration.OptunaSearchCV(
        clf, param_distributions, n_trials=100, verbose=0, study=study, n_jobs=-1
    )
    optuna_search.fit(X, y)
    
    # Get the best estimator and store it in a pickle file
    best_model = optuna_search.best_estimator_
    model_file = os.path.join(settings["outf"], "models", f"{model_name.lower()}.pkl")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)

    return best_model


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    recall = sklearn.metrics.recall_score(y_test, y_pred, average='macro')
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
    cohen_kappa = sklearn.metrics.cohen_kappa_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, cohen_kappa

def get_feature_names(n_feats, selected_feats, chnn):
    # Frequency Bands
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    segs = [0.5, 4., 8., 13., 30., 100.]
    segs = [(segs[i], segs[i+1]) for i in range(len(segs)-1)]  # Segments defining every band
    bands = {bn: b for bn,b in zip(band_names, segs)} # Dictionary containing every band and their respective time limits
    band_ch = np.concatenate([[f"{b}_{i}" for i in chnn] for b in band_names]) # A name for every band and channel, used for pow_freq_bands
    # Combinations of channels provide the lower triangle of the correlation matrix
    from itertools import combinations
    mv_chs = [f"{a}_{b}" for a,b in combinations(chnn, 2)]
    mv_chs_no_self = mv_chs.copy()
    mv_chs += chnn # Optionally the feature extractor returns also the diagonal of the correlation matrix (self-correlation, or lambda values)
    # For coherence in the future
    # band_ch_mv = np.concatenate([[f"{b}_{i}" for i in mv_chs] for b in band_names]) # A name for every band and channel, used for energy_freq_bands
    ch_len = ["mean", "variance", "std", "ptp_amp", "skewness", "kurtosis", "rms", "quantile", "hurst_exp", "app_entropy", "samp_entropy", "decorr_time", "hjorth_mobility", "hjorth_complexity", "higuchi_fd", "katz_fd", "zero_crossings", "line_length", "spect_entropy", "svd_entropy", "svd_fisher_info", "spect_edge_freq"]
    feat_dict = {}
    for feat in selected_feats:
        if feat in ch_len:
            feat_dict[feat] = [f"{feat}_{i}" for i in chnn]
        elif feat in ["pow_freq_bands", "energy_freq_bands"]:
            feat_dict[feat] = [f"{feat}_{i}" for i in band_ch]
        elif feat == "wavelet_coef_energy":
            feat_dict[feat] = [f"{feat}_{i}" for i in band_ch] + [f"{feat}_{i}" for i in chnn]
        elif feat == "teager_kaiser_energy":
            feat_dict[feat] = [f"{feat}_{i}_d0" for i in band_ch] + [f"{feat}_{i}_d1" for i in band_ch] + [f"{feat}_d{i}" for i in range(2)] # 2 for the two directions
        elif feat in ["max_cross_corr", "phase_lock_val", "nonlin_interdep"]:
            feat_dict[feat] = [f"{feat}_{i}" for i in mv_chs_no_self]
        elif feat in ["time_corr", "spect_corr"]:
            feat_dict[feat] = [f"{feat}_{i}" for i in mv_chs]
        else:
            raise ValueError(f"Invalid feature name: {feat}")
    feat_names = np.hstack(feat_dict.values())
    assert len(feat_names) == n_feats
    settings["feat_names"] = feat_names
    return feat_names

# Download participant data
download_path = "./data"
if os.path.exists(download_path):
    shutil.rmtree(download_path, ignore_errors=True)
on.download(dataset=dataset_id, target_dir=download_path, include=["participants.tsv", "participants.json"])

# List all subjects in the dataset
subjects = pd.read_csv(os.path.join(download_path, "participants.tsv"), sep="\t")["participant_id"].values

# Iterate over subjects and download/process EEG data
for subject in subjects:
    eeg_file = f"{subject}/ses-01/eeg/{subject}_ses-01_task-RSVP_run-01_eeg.edf"
    eeg_file_path = os.path.join(download_path, eeg_file)
    
    # Download EEG file
    print(f"Downloading EEG file for subject {subject}...")
    on.download(dataset=dataset_id, target_dir=download_path, include=[f"{subject}/*"]) 

    # Process EEG file
    X, labels = process_eeg_file(eeg_file_path)
    
    # Split data into train and test sets (adjust as needed)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, labels, test_size=0.25, random_state=42)
    
    # Train and evaluate models
    model_names = ["LogisticRegression", "SVM", "LDA", "RandomForest"]
    for model_name in model_names:
        print(f"Training {model_name} for subject {subject}...")
        print(f"======== Chance Level: {sum(y_train)/len(y_train)*100:.4f}")
        clf = train_model(X_train, y_train, model_name, subject)
        accuracy, precision, recall, f1, cohen_kappa = evaluate_model(clf, X_test, y_test)
        print(f"Results for {model_name} on subject {subject}:")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1: {f1:.2f}")
        print(f"  Cohen's Kappa: {cohen_kappa:.2f}")
        # Feature importance for RandomForest
        if model_name == "RandomForest":
            feat_names = get_feature_names(X.shape[1], settings["selected_feats"], settings["eeg_ch_names"])
            # best_features = np.argsort(clf.feature_importances_)[::-1]
            df_feat_importance = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
            df_feat_importance.to_csv(os.path.join(settings["outf"], f"ML_RFC_feat_importance_{subject}.csv"), index=False)

        # Save results
        results_file = os.path.join(settings["outf"], f"results_{subject}.csv")
        res_df = pd.DataFrame({
            "subject": subject, 
            "model": model_name, 
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1, 
            "cohen_kappa": cohen_kappa,
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }, index=[0])
        if os.path.exists(results_file):
            res_df.to_csv(results_file, mode="a", header=False, index=False)
    # Make space for the next subject
    shutil.rmtree(os.path.join(download_path, subject), ignore_errors=True)

with open(os.path.join(settings["outf"], "settings.json"), "w") as f:
    json.dump(settings, f, indent=4)