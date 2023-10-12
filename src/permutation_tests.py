import json
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

results_path = "./results/"
# Load models

models_path = os.path.join(results_path, "models")
model_names = [model.split("_")[0] for model in os.listdir(models_path)]
model_names = list(set(model_names))
models = {}
for model_name in model_names:
    print(f"Loading {model_name}...")
    subjects = [fp.split("_")[1] for fp in os.listdir(models_path) if model_name in fp]
    subjects = [subject.split(".")[0] for subject in subjects] # Remove `.pkl`
    for subject in subjects:
        print(f"    Loading {model_name} for {subject}...")
        model_path = os.path.join(models_path, f"{model_name}_{subject}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            models[f"{model_name}_{subject}"] = model

# Load data
features_path = os.path.join(results_path, "features")
features = {}
labels = {}
for subject in subjects:
    print(f"Loading features and labels for {subject}...")
    features[subject] = np.load(os.path.join(features_path, f"features_{subject}.npy"))
    labels[subject] = np.load(os.path.join(features_path, f"labels_{subject}.npy"))

# Function to evaluate different metrics
metrics = ["accuracy", "precision", "recall", "f1", "cohen_kappa"]
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred, average='macro')
    recall = sklearn.metrics.recall_score(y_test, y_pred, average='macro')
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
    cohen_kappa = sklearn.metrics.cohen_kappa_score(y_test, y_pred)

    return accuracy, precision, recall, f1, cohen_kappa

# Load the random state from the settings
if os.path.exists(os.path.join(results_path, "settings.json")):
    with open(os.path.join(results_path, "settings.json"), "r") as f:
        settings = json.load(f)
    rs = settings["rs"]
else:
    rs = 42

# Obtain distribution of scores for experimental hypothesis and null hypothesis via permutation
scores = {}
scores_null = {}
for subject in subjects:
    X, y = features[subject], labels[subject]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.25, random_state=rs
    )
    for model_name in model_names:
        model = models[f"{model_name}_{subject}"]
        print(f"Evaluating {model_name} for {subject}...")
        scores[f"{model_name}_{subject}"] = evaluate_model(model, X_test, y_test)
        permuted_y = np.random.permutation(y_test)
        scores_null[f"{model_name}_{subject}"] = evaluate_model(model, X_test, permuted_y)

# Convert scores into np array for all subjects
dists = {}
for model_name in model_names:
    model_scores = [np.array(list(score)) for model, score in scores.items() if model_name in model]
    model_scores = np.stack(model_scores)
    model_scores_null = [np.array(list(score)) for model, score in scores_null.items() if model_name in model]
    model_scores_null = np.stack(model_scores_null)
    dists[model_name] = (model_scores, model_scores_null)

# Plot violin polots of scores' distributions
fig, axs = plt.subplots(len(model_names), len(metrics), figsize=(20, 4*len(model_names)))
for model_name, ax in zip(model_names, axs):
    model_scores, model_scores_null = dists[model_name]
    data = np.vstack([model_scores, model_scores_null])
    df = pd.DataFrame(data=data, columns=metrics)
    df["hypothesis"] = ["experimental"]*len(model_scores) + ["null"]*len(model_scores_null)
    for j, metric in enumerate(metrics):
        sns.violinplot(data=df, x="hypothesis", y=metric, ax=ax[j], inner="stick")
        ax[j].set_title(f"{model_name} {metric}")
fig.tight_layout()
# plt.show()
plt.savefig(os.path.join(results_path, "img", "scores_distributions.png"))

# Bootstrap the distributions
n_bootstraps = 1000
bootstrap_scores = {}
bootstrap_scores_null = {}
for model_name in model_names:
    model_scores, model_scores_null = dists[model_name]
    bootstrap_scores[model_name] = []
    bootstrap_scores_null[model_name] = []
    for i in range(n_bootstraps):
        bootstrap_scores[model_name].append(
            np.mean(
                sklearn.utils.resample(model_scores, random_state=i),
                axis=0
            )
        )
        bootstrap_scores_null[model_name].append(
            np.mean(
                sklearn.utils.resample(model_scores_null, random_state=i),
                axis=0
            )
        )
    bootstrap_scores[model_name] = np.stack(bootstrap_scores[model_name])
    bootstrap_scores_null[model_name] = np.stack(bootstrap_scores_null[model_name])

# Plot bootstrap distributions
fig, axs = plt.subplots(len(model_names), len(metrics), figsize=(20, 4*len(model_names)))
for model_name, ax in zip(model_names, axs):
    model_scores, model_scores_null = dists[model_name]
    bootstrap_scores_model, bootstrap_scores_null_model = bootstrap_scores[model_name], bootstrap_scores_null[model_name]
    data = np.vstack([bootstrap_scores_model, bootstrap_scores_null_model])
    df = pd.DataFrame(data=data, columns=metrics)
    df["hypothesis"] = ["experimental"]*len(bootstrap_scores_model) + ["null"]*len(bootstrap_scores_null_model)
    for j, metric in enumerate(metrics):
        sns.violinplot(data=df, x="hypothesis", y=metric, ax=ax[j])
        ax[j].set_title(f"{model_name} {metric}")
    # Store bootstrap scores
    bootstrap_scores_df = pd.DataFrame(data=bootstrap_scores_model, columns=metrics)
    bootstrap_scores_df.mean().to_csv(os.path.join(results_path, f"bootstrap_avg_scores_{model_name}.csv"))
fig.tight_layout()
# plt.show()
plt.savefig(os.path.join(results_path, "img", "bootstrap_distributions.png"))

# Calculate p-values # Everything is significant
# from scipy.stats import mannwhitneyu
# p_values = {}
# for model_name in model_names:
#     model_scores, model_scores_null = dists[model_name]
#     bootstrap_scores_model, bootstrap_scores_null_model = bootstrap_scores[model_name], bootstrap_scores_null[model_name]
#     p_values[model_name] = []
#     for j, metric in enumerate(metrics):
#         p_values[model_name].append(
#             mannwhitneyu(bootstrap_scores_model[:, j], bootstrap_scores_null_model[:, j]).pvalue
#         )
#     p_values[model_name] = np.array(p_values[model_name])

# p_values_df = pd.DataFrame(data=p_values, index=metrics)
# p_values_df.to_csv(os.path.join(results_path, "p_values.csv"))
plt.show()