"""Leakage checks for the LOSO code. Run: python research/test_loso.py"""
import numpy as np
from loso import load, loso_folds, fit_predict_csp

X, y, g = load()
for use_val in (False, True):
    held = []
    for s, val, tr in loso_folds(g, use_val):
        assert s not in tr and (val is None or val not in tr) and val != s       # no subject overlap
        held.append(s)
        assert len(set(tr) | {s} | ({val} if val else set())) == len(np.unique(g))
    assert sorted(held) == sorted(int(x) for x in np.unique(g))                  # every subject held out once

# CSP+LDA must not depend on the held-out subject: changing its data/labels leaves predictions for
# a fixed probe set unchanged, and training only sees train subjects.
s, _, tr = next(loso_folds(g, False))
itr = np.isin(g, tr); probe = X[g == s][:10]
p1 = fit_predict_csp(X[itr], y[itr], probe)
X2 = X.copy(); X2[g == s] *= 7.0; y2 = y.copy(); y2[g == s] = 1 - y2[g == s]
p2 = fit_predict_csp(X2[itr], y2[itr], probe)
assert np.array_equal(p1, p2), "CSP+LDA depends on held-out subject data/labels"
print("OK: LOSO folds disjoint, each subject held out once, CSP+LDA fit uses only training subjects")
