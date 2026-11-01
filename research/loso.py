"""Leave-One-Subject-Out evaluation (EEGNet, CSP+LDA) on the existing 10-subject cache.

Data: part2_transformer/.cache/subjects.npz (7-30 Hz band-pass, 0-3 s, imagery runs 6/10/14, built by the
unchanged starter loader). Each fold holds out ONE whole subject; nothing is fitted on it.
Usage: python research/loso.py --model csp|eegnet
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).parent
P2 = HERE.parent / "part2_transformer"
sys.path.insert(0, str(P2))


def load():
    z = np.load(P2 / ".cache" / "subjects.npz")
    return z["X"], z["y"], z["g"]


def loso_folds(g, val_for_selection):
    """Yield (test_subject, val_subject|None, train_subjects). Subject sets are always disjoint."""
    subs = [int(s) for s in sorted(np.unique(g))]
    for i, s in enumerate(subs):
        val = subs[(i + 1) % len(subs)] if val_for_selection else None
        tr = [x for x in subs if x != s and x != val]
        assert s not in tr and (val is None or (val not in tr and val != s))
        yield s, val, tr


def fit_predict_csp(Xtr, ytr, Xte, n_components=6):
    """CSP (fit on train subjects only) -> log-variance -> LDA (fit on train features only)."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(CSP(n_components=n_components, reg="ledoit_wolf", log=True),
                         LinearDiscriminantAnalysis())
    pipe.fit(Xtr * 1e6, ytr)                 # volts -> microvolts for numerical conditioning
    return pipe.predict(Xte * 1e6)


def fit_predict_eegnet(Xtr, ytr, Xva, yva, Xte, epochs=20, seed=0):
    import torch, torch.nn as nn
    from models import EEGNet                # unchanged Part 2 EEGNet, Part 1 config below
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True); sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-12
    z = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32))   # train-subject statistics only
    Xtr_t, Xva_t, Xte_t = z(Xtr), z(Xva), z(Xte)
    ytr_t = torch.from_numpy(ytr)
    model = EEGNet(Xtr.shape[1], Xtr.shape[2], kern=80, p=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad(); nn.functional.cross_entropy(model(Xtr_t[b]), ytr_t[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = (model(Xva_t).argmax(1).numpy() == yva).mean()
        if v > best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(1).numpy(), float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["csp", "eegnet"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    X, y, g = load()
    out = HERE / "results" / f"loso_{args.model}.jsonl"
    out.write_text("")
    for s, val, tr in loso_folds(g, args.model == "eegnet"):
        t0 = time.time()
        itr, ite = np.isin(g, tr), g == s
        if args.model == "csp":
            pred = fit_predict_csp(X[itr], y[itr], X[ite]); cfg = dict(n_components=6, reg="ledoit_wolf", band="7-30Hz")
            vacc = None
        else:
            pred, vacc = fit_predict_eegnet(X[itr], y[itr], X[g == val], y[g == val], X[ite], epochs=args.epochs)
            cfg = dict(kern=80, dropout=0.5, epochs=args.epochs, lr=1e-3, weight_decay=1e-3, batch=64, seed=0,
                       selection="best accuracy on val subject (a training subject)")
        row = dict(model=args.model, held_out=s, val_subject=val, train_subjects=tr,
                   acc=float((pred == y[ite]).mean()), balanced_acc=float(balanced_accuracy_score(y[ite], pred)),
                   n_test=int(ite.sum()), val_acc=vacc, config=cfg, y_true=y[ite].tolist(), y_pred=pred.tolist(),
                   seconds=round(time.time() - t0, 1))
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{args.model} held-out S{s}: acc {row['acc']:.3f} bal {row['balanced_acc']:.3f} "
              f"n={row['n_test']} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
