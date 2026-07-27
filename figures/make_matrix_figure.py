import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/content/drive/MyDrive/CHANGE_ME/finetune_outputs"

MODELS = [
    ("mHuBERT-147",                 "mHuBERT-147\n(HuBERT, 147 lang)"),
    ("wav2vec2-xls-r-300m",         "XLS-R-300m\n(wav2vec2, 128 lang)"),
    ("wav2vec2-large-xlsr-turkish", "XLSR-Turkish\n(wav2vec2, +TR ASR)"),
    ("hubert-base-ls960",           "HuBERT-base\n(HuBERT, English)"),
    ("hubert-large-ll60k",          "HuBERT-large\n(HuBERT, English)"),
]
CONDS = [
    ("metrics_turev-%s.json", "test_turev", "TurEV→TurEV\n(in-domain)"),
    ("metrics_turev-%s.json", "test_ours",  "TurEV→CMD\n(transfer)"),
    ("metrics_ours-%s.json",  "test_ours",  "CMD→CMD\n(in-domain)"),
    ("metrics_ours-%s.json",  "test_turev", "CMD→TurEV\n(transfer)"),
]

def uar(fname, key, model):
    p = os.path.join(BASE, fname % model)
    if not os.path.exists(p):
        return np.nan
    try:
        b = json.load(open(p)).get(key)
        return b["uar"] * 100 if isinstance(b, dict) else np.nan
    except Exception:
        return np.nan

M = np.array([[uar(f, k, m) for f, k, _ in CONDS] for m, _ in MODELS])

fig, ax = plt.subplots(figsize=(7.2, 4.2))
# 50 = chance, 100 = perfect. 
im = ax.imshow(M, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")

ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels([c[2] for c in CONDS], fontsize=9)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([m[1] for m in MODELS], fontsize=9)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="grey")
        else:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11,
                    color="black", fontweight="bold" if v >= 90 else "normal")

# separate the in-domain columns from the transfer columns
for x in (0.5, 2.5):
    ax.axvline(x, color="white", linewidth=3)
ax.axvline(1.5, color="black", linewidth=2.5)

cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cb.set_label("UAR (%)   —   50 = chance", fontsize=9)
ax.set_title("Cross-corpus evaluation: in-domain succeeds, transfer collapses in both directions",
             fontsize=10.5, pad=10)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(BASE, f"matrix_figure.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
