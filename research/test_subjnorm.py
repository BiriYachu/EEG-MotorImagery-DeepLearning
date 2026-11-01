"""Held-out LABELS must never influence predictions (data may, by design: transductive). Run: python research/test_subjnorm.py"""
import numpy as np
from loso import load, loso_folds
from loso_subjnorm import fit_predict
X, y, g = load(); s, _, tr = next(loso_folds(g, False)); assert s not in tr
p1, _ = fit_predict(X, y, g, s, tr, [None], 6)
y2 = y.copy(); y2[g == s] = 1 - y2[g == s]
p2, _ = fit_predict(X, y2, g, s, tr, [None], 6)
assert np.array_equal(p1, p2), "held-out labels influence predictions"
print("OK: held-out labels do not affect predictions; no subject overlap (loso_folds asserts)")
