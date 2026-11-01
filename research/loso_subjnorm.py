"""CSP+LDA and Filter-Bank CSP+LDA with PER-SUBJECT feature standardisation (transductive, label-free). LOSO.
Usage: python research/loso_subjnorm.py   (separate result files; existing results untouched)"""
import json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from loso import load, loso_folds
from loso_filterbank import bandpass, BANDS

HERE = Path(__file__).parent


def csp_feats(Xtr, ytr, Xte, bands, n_comp):
    from mne.decoding import CSP
    ftr, fte = [], []
    for b in bands:
        f = (lambda X: X) if b is None else (lambda X, b=b: bandpass(X, *b))
        csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True)
        ftr.append(csp.fit_transform(f(Xtr) * 1e6, ytr))       # fitted on training subjects only
        fte.append(csp.transform(f(Xte) * 1e6))
    return np.hstack(ftr), np.hstack(fte)


def per_subject_standardise(F, groups):
    out = np.empty_like(F)
    for s in np.unique(groups):                                  # statistics from each subject's own trials, no labels
        m = groups == s
        out[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-9)
    return out


def fit_predict(X, y, g, s, tr, bands, n_comp):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    itr, ite = np.isin(g, tr), g == s
    ftr, fte = csp_feats(X[itr], y[itr], X[ite], bands, n_comp)
    ftr, fte = per_subject_standardise(ftr, g[itr]), per_subject_standardise(fte, g[ite])
    return LinearDiscriminantAnalysis().fit(ftr, y[itr]).predict(fte), ite


def main():
    import mne; mne.set_log_level("ERROR")
    X, y, g = load()
    for name, bands, n_comp in (("csp_subjnorm", [None], 6), ("filterbank_csp_subjnorm", list(BANDS), 4)):
        out = HERE / "results" / f"loso_{name}.jsonl"; out.write_text("")
        for s, _, tr in loso_folds(g, False):
            t0 = time.time(); pred, ite = fit_predict(X, y, g, s, tr, bands, n_comp)
            row = dict(model=name, held_out=s, train_subjects=tr, acc=float((pred == y[ite]).mean()),
                       balanced_acc=float(balanced_accuracy_score(y[ite], pred)), n_test=int(ite.sum()),
                       config=dict(bands=bands, n_components=n_comp, reg="ledoit_wolf", per_subject_standardisation=True, transductive=True),
                       y_true=y[ite].tolist(), y_pred=pred.tolist(), seconds=round(time.time() - t0, 1))
            open(out, "a").write(json.dumps(row) + "\n")
            print(f"{name} S{s}: acc {row['acc']:.3f} bal {row['balanced_acc']:.3f} frac_feet_pred {np.mean(pred):.2f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
