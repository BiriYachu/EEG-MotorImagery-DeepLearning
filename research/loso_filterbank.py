"""Filter-Bank CSP + LDA, leave-one-subject-out. Separate experiment; loso.py and its results are untouched.

Bands 8-12, 13-20, 20-30 Hz come from FIXED zero-phase Butterworth filters (no learned parameters) applied to
each 0-3 s trial of the cached 7-30 Hz data. Per band: CSP (4 components, Ledoit-Wolf, log-variance) fitted on
TRAIN subjects only; features of the 3 bands are concatenated; LDA fitted on TRAIN features only.
Limitation: edge effects of filtering 3 s epochs; the 20-30 Hz band touches the 30 Hz edge of the cached data.
Usage: python research/loso_filterbank.py
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score
from loso import load, loso_folds

HERE = Path(__file__).parent
BANDS = [(8, 12), (13, 20), (20, 30)]
FS = 160


def bandpass(X, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos"), X, axis=-1)


def fit_predict_fbcsp(Xtr, ytr, Xte, n_components=4):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    ftr, fte = [], []
    for lo, hi in BANDS:
        csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True)
        ftr.append(csp.fit_transform(bandpass(Xtr, lo, hi) * 1e6, ytr))     # fitted on training subjects only
        fte.append(csp.transform(bandpass(Xte, lo, hi) * 1e6))
    lda = LinearDiscriminantAnalysis().fit(np.hstack(ftr), ytr)             # fitted on training features only
    return lda.predict(np.hstack(fte))


def main():
    import mne; mne.set_log_level("ERROR")
    X, y, g = load()
    out = HERE / "results" / "loso_filterbank_csp.jsonl"
    out.write_text("")
    for s, _, tr in loso_folds(g, False):
        t0 = time.time(); itr, ite = np.isin(g, tr), g == s
        pred = fit_predict_fbcsp(X[itr], y[itr], X[ite])
        row = dict(model="filterbank_csp", held_out=s, val_subject=None, train_subjects=tr,
                   acc=float((pred == y[ite]).mean()), balanced_acc=float(balanced_accuracy_score(y[ite], pred)),
                   n_test=int(ite.sum()), config=dict(bands=BANDS, n_components_per_band=4, reg="ledoit_wolf",
                                                      filter="butter4 zero-phase", classifier="LDA"),
                   y_true=y[ite].tolist(), y_pred=pred.tolist(), seconds=round(time.time() - t0, 1))
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"fbcsp held-out S{s}: acc {row['acc']:.3f} bal {row['balanced_acc']:.3f} n={row['n_test']} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
