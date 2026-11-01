"""Checks that the subject split and normalisation have no leakage. Run: python test_pipeline.py"""
import numpy as np
from train import subject_split, zscore_train_only

rng = np.random.default_rng(0)
groups = np.repeat(np.arange(1, 11), 42)
X = rng.normal(size=(len(groups), 4, 50)).astype(np.float32)

for seed in range(20):
    tr, va, te = subject_split(groups, seed)
    gs = [set(groups[i]) for i in (tr, va, te)]
    assert not (gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2]), "subject overlap"
    assert len(tr) + len(va) + len(te) == len(groups)

# Normalised TRAIN data must not change when val/test data are altered.
tr, va, te = subject_split(groups, 0)
Z1 = zscore_train_only(X, tr)
X2 = X.copy(); X2[np.concatenate([va, te])] *= 1000
Z2 = zscore_train_only(X2, tr)
assert np.array_equal(Z1[tr], Z2[tr]), "normalisation uses non-train data"
assert abs(Z1[tr].mean()) < 1e-3 and abs(Z1[tr].std() - 1) < 1e-2
print("OK: no subject overlap (20 seeds); z-score stats are train-only")
