import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

folder = os.path.dirname(os.path.abspath(__name__))
folder = os.path.join(folder, 'results')
outf = os.path.join(folder, 'img')
if not os.path.exists(outf):
    os.makedirs(outf)

plot_sf = 1.5

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
files = [f for f in files if f.startswith("ML_RFC_feat_importance_") and f.endswith(".csv")]
res_df = pd.read_csv(os.path.join(folder, "results.csv"))

res_df.groupby("model")["cohen_kappa"].describe()

scorers = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']
# for scorer in scorers:
#     res_df.groupby("model")[scorer].describe().to_csv(os.path.join(folder, f"results_summary_{scorer}.csv"), float_format="%.3f")

# Plot distribution of different models
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.violinplot(x="model", y="cohen_kappa", data=res_df, ax=ax, inner="stick")
# plt.show()
plt.savefig(os.path.join(outf, "model_dist.png"), dpi=300)
# fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
# sns.boxplot(x="model", y="accuracy", data=res_df, ax=ax)
# # plt.show()

# fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
# sns.violinplot(x="subject", y="cohen_kappa", hue="model", data=res_df, ax=ax, inner="stick")
# plt.show()

fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.violinplot(x="subject", y="cohen_kappa", hue="model", data=res_df[res_df["model"]=="LogisticRegression"], ax=ax, inner="stick", fill=False)
# plt.show()
plt.savefig(os.path.join(outf, "model_dist_logreg.png"), dpi=300)

dfs = []
s_low = {}
for f in files:
    df = pd.read_csv(os.path.join(folder, f))
    subj = f.split(".")[0].split("_")[-1]
    df["subj"] = subj
    df = df.loc[df["feature"] != 0]
    kdf = res_df.loc[res_df["model"]=="RandomForest"]
    k = kdf[kdf["subject"]==subj]["cohen_kappa"].mean()
    print(k)
    if k <= 0:
        print(f"{subj} has a negative kappa value of {k}, not included in featuere relevance analysis.")
        s_low[subj] = k
    else:
        print(f"{subj} has a kappa value of {k}")
        df["kappa"] = k
        dfs.append(df)
print(f"{len(s_low)}/{len(files)} subjects with negative kappa values: {s_low.values()}")
s_low = pd.DataFrame.from_dict(s_low, orient="index", columns=["kappa"])
s_low.to_csv(os.path.join(folder, "low_kappa.csv"))
df = pd.concat(dfs)

# plot distribution of kappa values
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.distplot(df["kappa"], ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "kappa_dist.png"), dpi=300)

feats = df.drop("subj", axis=1).groupby("feature").sum().sort_values("importance", ascending=False)
feats = feats.loc[feats["importance"] != 0]

# feats["importance"] = feats["importance"] / feats["importance"].sum()
# feats.head(10)
# feats[feats["importance"] > 0.02].plot(kind="bar")
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.barplot(x="feature", y="importance", data=feats.reset_index().head(10), ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "feat_importance_t10.png"), dpi=300)

df = pd.concat(dfs)
df["importance"] = df["importance"] * df["kappa"]

feats = df.drop(["subj", "kappa"], axis=1).groupby("feature").sum().sort_values("importance", ascending=False)
feats = feats.loc[feats["importance"] != 0]

feats = feats.sort_values("importance", ascending=False)
# feats.head(10)
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.barplot(x="feature", y="importance", data=feats.reset_index().head(10), ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "feat_importance_kappa_t10.png"), dpi=300)

feats.to_csv(os.path.join(folder, "ML_RFC_feat_importance_k.csv"))
plt.close("all")