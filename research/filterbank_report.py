"""EEGNet vs CSP+LDA vs FilterBank-CSP+LDA, subject-wise. Run: python research/filterbank_report.py"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).parent
M = {"EEGNet": "loso_eegnet", "CSP+LDA": "loso_csp", "FilterBank-CSP+LDA": "loso_filterbank_csp"}
R = {n: {r["held_out"]: r for r in map(json.loads, open(H / "results" / f"{f}.jsonl"))} for n, f in M.items()}
subs = sorted(R["CSP+LDA"]); rng = np.random.default_rng(0)
print("| model | subjects | mean acc | sd | median | min | max | bootstrap 95% CI | mean balanced acc |\n|---|---|---|---|---|---|---|---|---|")
for n, d in R.items():
    a = np.array([d[s]["acc"] for s in subs]); b = np.mean([d[s]["balanced_acc"] for s in subs])
    lo, hi = np.percentile([rng.choice(a, len(a)).mean() for _ in range(10000)], [2.5, 97.5])
    print(f"| {n} | {len(a)} | {a.mean():.3f} | {a.std(ddof=1):.3f} | {np.median(a):.3f} | {a.min():.3f} | {a.max():.3f} | {lo:.3f}-{hi:.3f} | {b:.3f} |")
print("\nsubject | " + " | ".join(R))
for s in subs:
    print(f"S{s} | " + " | ".join(f"{R[n][s]['acc']:.3f}" for n in R))
for other in ("CSP+LDA", "EEGNet"):
    d = np.array([R["FilterBank-CSP+LDA"][s]["acc"] - R[other][s]["acc"] for s in subs])
    print(f"\nFilterBank minus {other}: mean {d.mean():+.3f}; better {(d>0).sum()}, worse {(d<0).sum()}, tied {(d==0).sum()}; Wilcoxon p = {wilcoxon(d).pvalue:.3f}")
fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(subs)); w = 0.27
for k, n in enumerate(R):
    ax.bar(x + (k - 1) * w, [R[n][s]["acc"] for s in subs], w, label=f"{n} (mean {np.mean([R[n][s]['acc'] for s in subs]):.3f})")
ax.axhline(0.5, color="k", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subs]); ax.set_ylim(0.3, 0.85)
ax.set_xlabel("held-out subject (45 test trials each)"); ax.set_ylabel("accuracy"); ax.set_title("LOSO: EEGNet vs CSP+LDA vs FilterBank-CSP+LDA"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(H / "figures" / "loso_filterbank_comparison.png", dpi=130)
