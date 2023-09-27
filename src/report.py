import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# bokeh
# from bokeh.io import output_notebook, show, export_png
# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
# from bokeh.layouts import gridplot
# from bokeh.palettes import Category20
# from bokeh.transform import factor_cmap
# from bokeh.models.widgets import Tabs, Panel
# from bokeh.embed import components


folder = os.path.dirname(os.path.abspath(__name__))
folder = os.path.join(folder, 'results')
outf = os.path.join(folder, 'img')
if not os.path.exists(outf):
    os.makedirs(outf)

plot_sf = 6

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
files = [f for f in files if f.startswith("ML_RFC_feat_importance_") and f.endswith(".csv")]

dfs = []
s_low = []
for f in files:
    df = pd.read_csv(os.path.join(folder, f))
    subj = f.split(".")[0].split("_")[-1]
    df["subj"] = subj
    df = df.loc[df["feature"] != 0]
    kdf = pd.read_csv(os.path.join(folder, "results_{}.csv".format(subj)))
    k = kdf[kdf["subject"]==subj]["cohen_kappa"].mean()
    if k <= 0:
        print(f"{subj} has a negative kappa value of {k}, not included in featuere relevance analysis.")
        s_low.append(subj)
    else:
        print(f"{subj} has a kappa value of {k}")
        df["kappa"] = k
        dfs.append(df)
print(f"{len(s_low)}/{len(files)} subjects with negative kappa values: {s_low}")
df = pd.concat(dfs)
feats = df.drop("subj", axis=1).groupby("feature").sum().sort_values("importance", ascending=False)
feats = feats.loc[feats["importance"] != 0]

# plot distribution of kappa values
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.distplot(df["kappa"], ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "kappa_dist.png"), dpi=300)

# feats["importance"] = feats["importance"] / feats["importance"].sum()
# feats.head(10)
# feats[feats["importance"] > 0.02].plot(kind="bar")
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.barplot(x="feature", y="importance", data=feats.reset_index().head(10), ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "feat_importance_t10.png"), dpi=300)

feats["importance"] = feats["importance"] * feats["kappa"]
feats = feats.sort_values("importance", ascending=False)
# feats.head(10)
fig, ax = plt.subplots(figsize=(16*plot_sf, 9*plot_sf))
sns.barplot(x="feature", y="importance", data=feats.reset_index().head(10), ax=ax)
# plt.show()
plt.savefig(os.path.join(outf, "feat_importance_kappa_t10.png"), dpi=300)

feats.drop("kappa", axis=1).to_csv(os.path.join(folder, "ML_RFC_feat_importance_k.csv"))