"""
EEG Transformer Sprint — entry point.
Run:  python main.py            (about 15-25 min on CPU: 4 models x 5 seeds)

This script:
  1. builds the leak-free cross-subject split (protocol.py: train 1-6 / val 7-8 / test 9-10),
  2. trains EEGNet (Part 1 config), the vanilla transformer, a patch-only ablation (the
     original starter stub) and YOUR model, each over several seeds on the SAME split,
  3. reports mean / std / best / worst, writes results.jsonl,
  4. saves the Part 2 figure (figure_part2.png).

The starter's run_across_seeds / save_figure (peak-test-accuracy harness, t-SNE) are kept below
for reference but are not used by main(): the peak-test number selects on the test set.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import get_channel_positions  # noqa: F401  (kept from starter)
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed
from protocol import make_split, fit_eval


# =========================================================================== #
#  YOUR MODEL — convolutional stem + learned temporal patch tokens.
#
#  Bias injected: TOKENISATION. Instead of one token per 6.25 ms sample (481 tokens,
#  ~232k attention entries per head), a conv stem filters locally in time while mixing
#  electrodes with learned weights, and a strided conv cuts the result into 200 ms patches
#  (15 tokens + CLS = 16, i.e. 256 attention entries per head: ~900x fewer).
#  Why it should help on EEG: motor rhythms are 7-30 Hz oscillations (period
#  33-140 ms), invisible in a single sample, so per-timestep attention must discover them from
#  pairwise sample relations; a conv stem provides local oscillation-aware filters, the
#  nonlinearity (GELU) rectifies them and the patch conv averages over 200 ms, which is
#  close to a band-power / ERD feature. The first conv sees all 64 electrodes with learned
#  weights, so no left-right spatial assumption is baked in (the task is bilateral).
# =========================================================================== #
class _Block(nn.Module):
    """Pre-LN encoder block with explicit scaled dot-product attention."""

    def __init__(self, d, heads, p, p_attn):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
        self.drop, self.attn_drop = nn.Dropout(p), nn.Dropout(p_attn)
        self.last_attn = None            # filled only when keep=True (for the figure)

    def forward(self, x, keep=False):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        # softmax(Q K^T / sqrt(d_k)) V: dividing by sqrt(d_k) keeps logit variance ~1 so the
        # softmax does not saturate (which would give near-one-hot attention and tiny gradients).
        a = (q @ k.transpose(-2, -1) / math.sqrt(self.dk)).softmax(-1)      # (B, h, N, N)
        if keep:
            self.last_attn = a.detach()
        z = (self.attn_drop(a) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.out(z))
        return x + self.drop(self.ff(self.ln2(x)))


class YourModel(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2, patch=32, d_stem=32, k_stem=25,
                 d_model=64, n_heads=4, depth=2, p=0.3, p_attn=0.1):
        super().__init__()
        # Conv stem: learned spatio-temporal filters (k_stem = 156 ms), BN + GELU.
        self.stem = nn.Sequential(
            nn.Conv1d(n_chan, d_stem, k_stem, padding=k_stem // 2, bias=False),
            nn.BatchNorm1d(d_stem), nn.GELU())
        # Learned temporal patches: stride == kernel, no overlap.
        self.patch = nn.Conv1d(d_stem, d_model, patch, stride=patch)
        self.n_tokens = n_time // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))  # order matters (ERD timing)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(_Block(d_model, n_heads, p, p_attn) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False, keep=False):    # x: (B, C, T)
        tok = self.patch(self.stem(x)).transpose(1, 2)       # (B, n_tokens, d)
        z = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], 1) + self.pos
        for blk in self.blocks:
            z = blk(z, keep=keep)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


class PatchOnlyBaseline(nn.Module):
    """The ORIGINAL starter stub (patch tokens, no conv stem), kept as an ablation row."""

    def __init__(self, n_chan, n_time, n_classes=2, patch=32, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.tokenizer = nn.Conv1d(n_chan, d_model, kernel_size=patch, stride=patch)
        n_tokens = n_time // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2,
                                           dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):
        tok = self.tokenizer(x).transpose(1, 2)
        z = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1) + self.pos
        embed = self.norm(self.encoder(z)[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #
# Starter helpers, kept for reference (peak-test-accuracy harness; NOT used by main()).
def run_across_seeds(build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    accs = []
    for s in seeds:
        set_seed(s)
        best, _ = train_model(build_fn(), Xtr, ytr, Xte, yte, device=device, **kw)
        accs.append(best)
    return np.array(accs)


def save_figure(model, Xte, yte, device, path="figure.png"):
    """Starter t-SNE of CLS embeddings on the test subjects (needs return_embed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model.eval()
    with torch.no_grad():
        emb = model(torch.from_numpy(Xte).to(device), return_embed=True)[1].cpu().numpy()
    e2 = TSNE(n_components=2, perplexity=min(30, max(5, len(emb) // 4)), init="pca",
              random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        plt.scatter(e2[yte == c, 0], e2[yte == c, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=130)


# =========================================================================== #
HERE = Path(__file__).parent


def build_models(C, T):
    # EEGNet with the Part 1 configuration (kern 80, dropout 0.5); models.py is not modified.
    return {
        "EEGNet": lambda: EEGNet(C, T, kern=80, p=0.5),
        "VanillaTransformer": lambda: VanillaEEGTransformer(C, T),
        "PatchOnly(starter stub)": lambda: PatchOnlyBaseline(C, T),
        "YourModel": lambda: YourModel(C, T),
    }


def save_part2_figure(rows, model, Xte, yte, path):
    """(a) per-seed accuracies; (b) CLS->patch attention of the trained YourModel (seed 0);
    (c) attention-matrix size, per-timestep vs patched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["model"] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, n in enumerate(names):
        a = np.array([r["test_acc"] for r in rows if r["model"] == n])
        ax[0].scatter(np.full(len(a), i) + np.linspace(-.12, .12, len(a)), a, s=22, zorder=3)
        ax[0].errorbar(i, a.mean(), a.std(ddof=1), fmt="_", color="k", ms=18, capsize=4, zorder=4)
    ax[0].axhline(0.5, color="grey", ls="--", lw=1); ax[0].axhline(0.6, color="grey", ls=":", lw=1)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.replace("(starter stub)", "\n(stub)").replace("Transformer", "\nTransformer")
                           for n in names], fontsize=8)
    ax[0].set_ylabel("test accuracy (subjects 9-10, n=90)")
    ax[0].set_title("(a) per-seed test acc, mean±sd\ndashed: chance, dotted: p<0.05 threshold")

    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(Xte), keep=True)
    att = model.blocks[-1].last_attn[:, :, 0, 1:].mean(1).numpy()      # CLS -> patches, mean over heads
    t = (np.arange(att.shape[1]) + .5) * model.patch.kernel_size[0] / 160
    for c, name in [(0, "fists"), (1, "feet")]:
        m = att[yte == c]
        ax[1].errorbar(t, m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m)), marker="o", ms=3, label=name, capsize=2)
    ax[1].axhline(1 / (att.shape[1] + 1), color="grey", ls="--", lw=1, label="uniform (1/16)")
    ax[1].set_xlabel("time from cue (s), patch centre"); ax[1].set_ylabel("CLS attention (last layer)")
    ax[1].set_title("(b) learned attention, YourModel seed 0\n(test trials, mean±SEM, heads averaged)")
    ax[1].legend(fontsize=8)

    n_t = Xte.shape[2]
    sizes = {"per-timestep": (n_t + 1) ** 2, f"patched ({att.shape[1] + 1} tokens)": (att.shape[1] + 1) ** 2}
    ax[2].bar(list(sizes), list(sizes.values()), color=["#888", "#1f77b4"]); ax[2].set_yscale("log")
    for i, v in enumerate(sizes.values()):
        ax[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax[2].set_ylabel("attention entries / head / layer"); ax[2].set_title("(c) effect of temporal patching")
    ax[2].set_ylim(top=max(sizes.values()) * 8)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  saved {path}")


def report(rows, Xte, yte):
    print("\n=== summary (test accuracy at validation-selected epoch) ===")
    for name in dict.fromkeys(r["model"] for r in rows):
        sel = [r for r in rows if r["model"] == name]
        a = np.array([r["test_acc"] for r in sel])
        print(f"{name:26s} n={len(a)} mean {a.mean():.3f}  std {a.std(ddof=1):.3f}  best {a.max():.3f}  "
              f"worst {a.min():.3f}  | mean train time {np.mean([r['train_seconds'] for r in sel]):.0f}s")
    w = HERE / ".cache" / "yourmodel_seed0.pt"
    if w.exists():
        m = YourModel(Xte.shape[1], Xte.shape[2]); m.load_state_dict(torch.load(w))
        print("\nsaving figure...")
        save_part2_figure(rows, m, Xte, yte, HERE / "figure_part2.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names")
    ap.add_argument("--out", default=None, help="write results_<out>.jsonl and skip the report "
                    "(for parallel runs; merge later with --report)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="merge results_*.jsonl -> results.jsonl, summarise, plot")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    split = make_split()
    Xtr, ytr, Xva, yva, Xte, yte = split

    if args.report:
        rows = [json.loads(l) for p in sorted(HERE.glob("results_*.jsonl")) for l in open(p) if l.strip()]
        (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return report(rows, Xte, yte)

    device = "cpu"      # all models are small; CPU keeps runs reproducible across machines
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"split: train {Xtr.shape} (subj 1-6) | val {Xva.shape} (7-8) | test {Xte.shape} (9-10); "
          f"epochs={args.epochs}, seeds={args.seeds}")
    rows = []
    out = HERE / (f"results_{args.out}.jsonl" if args.out else "results.jsonl")
    out.write_text("")
    for name, build in build_models(C, T).items():
        if args.models and name not in args.models:
            continue
        print(f"\n=== {name}: params {count_params(build()):,} ===")
        for s in args.seeds:
            torch.manual_seed(s)                       # same seed -> same init stream per model
            model, r = fit_eval(build(), split, s, epochs=args.epochs, device=device)
            r.update(model=name, seed=s, params=count_params(model), split="train1-6/val7-8/test9-10",
                     lr=1e-3, weight_decay=1e-3, batch=64)
            rows.append(r)
            if name == "YourModel" and s == 0:
                (HERE / ".cache").mkdir(exist_ok=True)
                torch.save(model.state_dict(), HERE / ".cache" / "yourmodel_seed0.pt")
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  seed {s}: test {r['test_acc']:.3f} (val {r['val_acc']:.3f}, epoch {r['best_epoch']}, "
                  f"final {r['test_final']:.3f}, peak-test[optimistic] {r['test_peak']:.3f}, {r['train_seconds']}s)")
    if not args.out:
        report(rows, Xte, yte)


if __name__ == "__main__":
    main()
