"""Leak-free cross-subject protocol for Part 2 (starter train.py/data.py/models.py are untouched).

The starter harness reports the PEAK TEST accuracy over epochs, i.e. selects on the test set.
Here: train subjects 1-6, validation subjects 7-8, test subjects 9-10. The epoch is chosen on
validation accuracy and test accuracy is read once at that epoch. The starter-style peak test
number is still logged (`test_peak`) but only as an optimistic reference.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import _load_subjects            # starter loader: 7-30 Hz, 0-3 s, imagery runs 6/10/14
from train import make_loaders, evaluate   # starter helpers (unchanged)

TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS = [1, 2, 3, 4, 5, 6], [7, 8], [9, 10]
CACHE = Path(__file__).parent / ".cache" / "subjects.npz"


def load_by_subject(subjects=range(1, 11)):
    """Per-subject arrays (unnormalised): X (N, C, T), y (0 fists / 1 feet), g (subject id)."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["y"], z["g"]
    Xs, ys, gs = [], [], []
    for s in subjects:
        X, y = _load_subjects([s])
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, X=X, y=y, g=g)
    return X, y, g


def make_split():
    """Return (Xtr, ytr, Xva, yva, Xte, yte); z-score statistics come from TRAIN trials only."""
    X, y, g = load_by_subject()
    idx = {n: np.flatnonzero(np.isin(g, s)) for n, s in
           (("tr", TRAIN_SUBJECTS), ("va", VAL_SUBJECTS), ("te", TEST_SUBJECTS))}
    assert not (set(g[idx["tr"]]) & set(g[idx["va"]]) | set(g[idx["tr"]]) & set(g[idx["te"]])
                | set(g[idx["va"]]) & set(g[idx["te"]])), "subject overlap"
    mu = X[idx["tr"]].mean(axis=(0, 2), keepdims=True)
    sd = X[idx["tr"]].std(axis=(0, 2), keepdims=True) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    return tuple(a for n in ("tr", "va", "te") for a in (Z[idx[n]], y[idx[n]]))


def fit_eval(model, split, seed, epochs=40, lr=1e-3, weight_decay=1e-3, device="cpu"):
    """Train with the starter's optimiser/loop settings; select the epoch on validation only."""
    Xtr, ytr, Xva, yva, Xte, yte = split
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    tr, va = make_loaders(Xtr, ytr, Xva, yva)
    _, te = make_loaders(Xtr, ytr, Xte, yte)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_ep, best_state, peak_test, t0 = -1.0, -1, None, 0.0, time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); loss_fn(model(xb.to(device)), yb.to(device)).backward(); opt.step()
        v = evaluate(model, va, device)
        peak_test = max(peak_test, evaluate(model, te, device))  # reference only, never used to select
        if v > best_val:
            best_val, best_ep = v, ep
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    seconds = time.time() - t0
    final_test = evaluate(model, te, device)
    model.load_state_dict(best_state)
    return model, dict(val_acc=best_val, test_acc=evaluate(model, te, device), test_final=final_test,
                       test_peak=peak_test, best_epoch=best_ep, epochs=epochs,
                       train_seconds=round(seconds, 1), n_test=len(yte))


if __name__ == "__main__":
    Xtr, ytr, Xva, yva, Xte, yte = make_split()
    print("train", Xtr.shape, ytr.mean().round(2), "| val", Xva.shape, yva.mean().round(2),
          "| test", Xte.shape, yte.mean().round(2))
    _, _, g = load_by_subject()
    print("trials/subject:", {int(s): int((g == s).sum()) for s in np.unique(g)})
    print("train-split stats from train only: mean|z| on train =",
          float(np.abs(Xtr.mean())).__round__(4), " on test =", float(np.abs(Xte.mean())).__round__(3))
