"""Leakage test for FilterBank-CSP. Run: python research/test_filterbank.py"""
import numpy as np
from loso import load, loso_folds
from loso_filterbank import fit_predict_fbcsp

X, y, g = load()
s, _, tr = next(loso_folds(g, False))
assert s not in tr
itr = np.isin(g, tr); probe = X[g == s][:10]
p1 = fit_predict_fbcsp(X[itr], y[itr], probe)
X2 = X.copy(); X2[g == s] *= 7.0; y2 = y.copy(); y2[g == s] = 1 - y2[g == s]    # corrupt held-out data and labels
p2 = fit_predict_fbcsp(X2[itr], y2[itr], probe)
assert np.array_equal(p1, p2), "FilterBank-CSP depends on held-out subject data/labels"
print("OK: FilterBank-CSP+LDA fit uses only training subjects (held-out data/labels cannot change predictions)")
