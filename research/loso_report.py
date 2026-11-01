"""Subject-wise LOSO summary, CSV and figure. Run: python research/loso_report.py"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = {}
for m in ("eegnet", "csp"):
    p = HERE / "results" / f"loso_{m}.jsonl"
    if p.exists():
        R[m] = {r["held_out"]: r for r in map(json.loads, open(p) if p.stat().st_size else [])}
names = {"eegnet": "EEGNet", "csp": "CSP+LDA"}
rng = np.random.default_rng(0)
lines = []
with open(HERE / "results" / "loso_subjectwise.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "held_out_subject", "accuracy", "balanced_accuracy", "n_test_trials", "config"])
    for m, d in R.items():
        for s in sorted(d):
            r = d[s]; w.writerow([names[m], s, f"{r['acc']:.4f}", f"{r['balanced_acc']:.4f}", r["n_test"], json.dumps(r["config"])])
for m, d in R.items():
    a = np.array([d[s]["acc"] for s in sorted(d)])
    boot = [rng.choice(a, len(a)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"| {names[m]} | {len(a)} | {a.mean():.3f} | {a.std(ddof=1):.3f} | {np.median(a):.3f} | {a.min():.3f} | {a.max():.3f} | "
                 f"{lo:.3f}–{hi:.3f} |")
print("| model | subjects | mean | sd | median | min | max | bootstrap-over-subjects 95% CI of mean |\n|---|---|---|---|---|---|---|---|")
print("\n".join(lines))
print("\nsubject-wise accuracy:")
subs = sorted(set.intersection(*[set(d) for d in R.values()])) if R else []
for s in subs:
    print(f"  S{s:<3d}" + "  ".join(f"{names[m]} {R[m][s]['acc']:.3f}" for m in R))
if len(R) == 2 and len(subs) >= 6:
    diff = np.array([R['eegnet'][s]['acc'] - R['csp'][s]['acc'] for s in subs])
    print(f"\npaired EEGNet-CSP over {len(subs)} subjects: mean {diff.mean():+.3f}, EEGNet better in {(diff>0).sum()}, "
          f"worse in {(diff<0).sum()}, tied {(diff==0).sum()}; Wilcoxon signed-rank p = {wilcoxon(diff).pvalue:.3f}")
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(subs)); wd = 0.38
for k, m in enumerate(R):
    ax.bar(x + (k - .5) * wd, [R[m][s]["acc"] for s in subs], wd, label=f"{names[m]} (mean {np.mean([R[m][s]['acc'] for s in subs]):.3f})")
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"S{s}" for s in subs]); ax.set_ylim(0.3, 0.85)
ax.set_xlabel("held-out subject (trained on the other 9; 45 test trials each)"); ax.set_ylabel("accuracy")
ax.set_title("Leave-one-subject-out: fists vs feet, 7-30 Hz, 0-3 s"); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "figures" / "loso_eegnet_vs_csp.png", dpi=130)
