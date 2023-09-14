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
from mne_features.feature_extraction import FeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define the dataset ID and download path
dataset_id = "ds004324"
download_path = "./eeg_data"

# Create a directory to save results
outf = "./results/"
os.makedirs(outf, exist_ok=True)

# Define EEG file processing functions
def process_eeg_file(eeg_file):
    raw = mne.io.read_raw_edf(eeg_file, preload=True)

    # TODO: epochs = mne.Epochs(raw, events, event_id, tmin, tmax, preload=True)
    
    # Define EEG feature extraction functions (replace with your desired features)
    sfreq = raw.info['sfreq']
    feats = ["mean", "std"]
    
    # Extract features
    feat_extractor = Pipeline([
        ('fe', FeatureExtractor(sfreq=sfreq, selected_funcs=feats)),
        ('scaler', StandardScaler()),
    ])
    X = feat_extractor.fit_transform(raw.get_data())
    
    # Load labels (adjust this part based on your dataset structure)
    subject_id = eeg_file.split("_")[0]
    label_file = f"./labels/{subject_id}_labels.csv"
    labels = pd.read_csv(label_file)['label'].values
    
    return X, labels

# Define machine learning model training and evaluation functions
import optuna.integration

def train_model(X, y, model_name, outf):
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
    study = optuna.create_study(direction="maximize", study_name=model_name, storage=f"sqlite:///experiments.db", load_if_exists=True)
    
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
        clf, param_distributions, n_trials=500, verbose=0, study=study, n_jobs=-1
    )
    optuna_search.fit(X, y)
    
    # Get the best estimator and store it in a pickle file
    best_model = optuna_search.best_estimator_
    model_file = os.path.join(outf, "models", f"{model_name.lower()}.pkl")
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

# Download participant data
download_path = "./data"
if os.path.exists(download_path):
    os.rmdir(download_path)
on.download(dataset=dataset_id, target_dir=download_path, include=["participants.tsv", "participants.json"])

# List all subjects in the dataset
subjects = pd.read_csv(os.path.join(download_path, "participants.tsv"), sep="\t")["participant_id"].values

# Iterate over subjects and download/process EEG data
for subject in subjects:
    # subject = subjects[0]
    os.rmdir(download_path) # Make space for the next subject
    eeg_file = f"{subject}/ses-01/eeg/{subject}_ses-01_task-RSVP_run-01_eeg.edf"
    eeg_file_path = os.path.join(download_path, eeg_file)
    
    # Download EEG file
    print(f"Downloading EEG file for subject {subject_id}...")
    on.download(dataset=dataset_id, target_dir=download_path, include=[f"{subject}/*"]) 

    # Process EEG file
    X, labels = process_eeg_file(eeg_file_path)
    
    # Split data into train and test sets (adjust as needed)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, labels, test_size=0.25, random_state=42)
    
    # Train and evaluate models
    model_names = ["LogisticRegression", "SVM", "LDA", "RandomForest"]
    for model_name in model_names:
        print(f"Training {model_name} for subject {subject_id}...")
        clf = train_model(X_train, y_train, model_name)
        accuracy, precision, recall, f1, cohen_kappa = evaluate_model(clf, X_test, y_test)
        print(f"Results for {model_name} on subject {subject_id}:")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1: {f1:.2f}")
        print(f"  Cohen's Kappa: {cohen_kappa:.2f}")

# Plot feature importances (for RandomForest, adjust as needed)
feat_names = ["feat1", "feat2"]  # Replace with your feature names
best_features = np.argsort(clf.feature_importances_)[::-1]
n_best = 50
fig, ax = plt.subplots(figsize=(16, 9))
sns.barplot(x=clf.feature_importances_[best_features[:n_best]], y=feat_names[best_features[:n_best]], ax=ax, errorbar="sd")
plt.savefig(os.path.join(outf, "plots", "ML_RF_best_features.png"))

df_feat_importance = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
df_feat_importance.to_csv(os.path.join(outf, "ML_RFC_feat_importance.csv"), index=False)

