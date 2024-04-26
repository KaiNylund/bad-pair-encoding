import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

mpl.rcParams['figure.dpi'] = 300

tok_edit_dist_sims = np.load("./embedding_edit_dist_sims.npy", allow_pickle=True).item()

emb_sims = np.array(tok_edit_dist_sims["emb_sim"])
leven_dists = np.array(tok_edit_dist_sims["leven_dist"])
jw_dists = 1.0 - np.array(tok_edit_dist_sims["jw_dist"])

fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs = axs.flatten()

axs[0].set_facecolor("black")
axs[1].set_facecolor("black")

hb1 = axs[0].hist2d(leven_dists, emb_sims, bins=(15, 100), norm=mpl.colors.LogNorm(), cmap='inferno')
cb1 = fig.colorbar(hb1[3], ax=axs[0], label='counts')

hb2 = axs[1].hist2d(jw_dists, emb_sims, bins=(100, 100), norm=mpl.colors.LogNorm(), cmap='inferno')
cb2 = fig.colorbar(hb2[3], ax=axs[1], label='counts')

axs[0].set_ylabel("Cosine similarity between embeddings")
axs[0].set_xlabel("Levenshtein distance between tokens")
axs[1].set_xlabel("Jaro-Winkler distance between tokens")
plt.suptitle("Edit distance vs. embedding similarity between first 5k mT5-small embeddings")
plt.tight_layout()
plt.savefig("edit_dist_sim_2d_bins")

print(f"Correlation between leven dist and jw dist: {stats.pearsonr(leven_dists, jw_dists)}")
print(f"Correlation between leven dist and emb sim: {stats.pearsonr(leven_dists, emb_sims)}")
print(f"Correlation between jw dist and emb sim: {stats.pearsonr(jw_dists, emb_sims)}")

reg0 = LinearRegression().fit(leven_dists[:,np.newaxis], jw_dists)
print(f"Linear fit for leven dist and jw dist: {reg0.coef_} + {reg0.intercept_} with R^2={reg0.score(leven_dists[:,np.newaxis], jw_dists)}")
reg1 = LinearRegression().fit(leven_dists[:,np.newaxis], emb_sims)
print(f"Linear fit for leven dist and emb sim: {reg1.coef_} + {reg1.intercept_} with R^2={reg1.score(leven_dists[:,np.newaxis], emb_sims)}")
reg2 = LinearRegression().fit(jw_dists[:,np.newaxis], emb_sims)
print(f"Linear fit for jw dist and emb sim: {reg2.coef_} + {reg2.intercept_} with R^2={reg2.score(jw_dists[:,np.newaxis], emb_sims)}")