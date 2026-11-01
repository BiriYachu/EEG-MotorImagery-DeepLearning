# Filter-Bank CSP + LDA (leave-one-subject-out)

All numbers come from `research/results/loso_filterbank_csp.jsonl` (new file) and the existing, unchanged `loso_eegnet.jsonl` / `loso_csp.jsonl`. Run under a 10-minute limit; the experiment completed (10 of 10 folds).

## Method
- Same data and LOSO folds as `RESEARCH_EXTENSION.md`: 10 subjects, one whole subject held out per fold, trained on the other nine (`loso_folds`), 45 held-out trials per fold.
- The cache is already 7–30 Hz. Three sub-bands (8–12, 13–20, 20–30 Hz) are produced by **fixed** zero-phase 4th-order Butterworth filters applied to each trial (no learned parameters, so nothing to leak).
- Per band: CSP (4 components, Ledoit–Wolf covariance, log-variance), fitted on the nine training subjects only. The three bands' features (12 in total) are concatenated and an LDA is fitted on the training features only. The held-out subject is only transformed and predicted.
- No tuning and no feature selection of any kind. `research/test_filterbank.py` passes: altering the held-out subject's data and labels does not change its predictions.

## Results (accuracy, chance = 0.5)

| model | subjects | mean | sd | median | min | max | bootstrap 95% CI of mean | mean balanced acc |
|---|---|---|---|---|---|---|---|---|
| EEGNet | 10 | 0.536 | 0.062 | 0.533 | 0.444 | 0.644 | 0.500–0.573 | 0.540 |
| CSP+LDA | 10 | 0.518 | 0.060 | 0.511 | 0.422 | 0.622 | 0.484–0.553 | 0.525 |
| **FilterBank-CSP+LDA** | 10 | 0.529 | 0.072 | 0.511 | 0.422 | 0.689 | 0.489–0.573 | 0.527 |

| subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet | 0.644 | 0.600 | 0.444 | 0.556 | 0.578 | 0.511 | 0.467 | 0.511 | 0.489 | 0.556 |
| CSP+LDA | 0.489 | 0.511 | 0.556 | 0.422 | 0.511 | 0.467 | 0.511 | 0.600 | 0.489 | 0.622 |
| FilterBank | 0.533 | 0.533 | 0.511 | 0.422 | 0.511 | 0.467 | 0.511 | 0.600 | 0.511 | 0.689 |

Paired over the 10 subjects: FilterBank minus CSP+LDA = **+0.011** (better in 4, worse in 1, tied in 5; Wilcoxon p = 0.44). FilterBank minus EEGNet = **−0.007** (better in 5, worse in 5; Wilcoxon p = 0.79). Figure: `research/figures/loso_filterbank_comparison.png`; raw summary: `research/results/filterbank_summary.txt`.

## Interpretation
**Measured:** filter-bank CSP did **not** measurably improve accuracy. The mean rose by 0.011 over plain CSP+LDA, and 5 of 10 per-subject accuracies are identical, so this is not distinguishable from no change (p = 0.44). All three models are within noise of each other and the bootstrap intervals of all three include or touch chance. For subjects S5 and S7, both CSP variants predict a single class for all 45 trials (all "hands" for S5, all "feet" for S7), so their accuracy there is just the class proportion (0.511); for the other subjects the predictions of the two CSP variants differ (agreement 0.29–0.98) and several equal accuracies are coincidences.

**Hypotheses (not tested):** (1) the single-class predictions suggest subject-specific shifts in CSP log-variance features that push the held-out subject entirely to one side of the LDA threshold; per-subject feature standardisation or bias calibration might help. (2) More training subjects and subject-specific tuning would likely matter more than finer frequency bands.

## Limitations
10 subjects, 45 trials per subject (one trial = 2.2 points); bands fixed by me in advance, no search (so none was selected on test data); filtering 3 s epochs has edge effects and the 20–30 Hz band touches the 30 Hz edge of the cached data; single pooled cross-subject CSP; LDA on 12 features without regularisation tuning; EEGNet from the previous experiment used 20 epochs and one seed. No permutation test was run, so no claim of significance above chance is made.

## Reproduce
```bash
python research/loso_filterbank.py     # about 2.5 min
python research/filterbank_report.py   # table, paired tests, figure
python research/test_filterbank.py     # leakage test
```
