# Research extension: leave-one-subject-out EEGNet vs CSP+LDA

Everything below comes from actual runs (`research/results/`, `logs/loso_*.log`). Original Part 1 / Part 2 code and results are unchanged. Work done under a 20-minute limit; scope is deliberately small.

## 1. Motivation
The audit (`RESEARCH_AUDIT.md`) found that the assessment's test sets were two subjects (84–90 trials) with one fixed split, so model differences were smaller than the single-run uncertainty (about ±0.10). Leave-one-subject-out (LOSO) uses every one of the 10 subjects as a test subject once, giving 10 subject-level estimates, and a classical CSP+LDA baseline shows whether near-chance accuracy is specific to the deep models.

## 2. LOSO methodology
- Data: the existing cache of the 10 subjects (`part2_transformer/.cache/subjects.npz`): 7–30 Hz band-pass, 0–3 s after the cue, imagery runs 6/10/14, 45 trials per subject, 450 total, built by the unchanged starter loader. The band-pass was applied per subject to continuous data before epoching, so it uses no information from other subjects' labels.
- 10 folds; each holds out one **entire** subject (45 trials) and trains on the other nine. Disjointness is asserted in `loso_folds` and tested (`research/test_loso.py`).
- **EEGNet** (unchanged `part2_transformer/models.py`, Part 1 configuration: kernel 80, dropout 0.5), AdamW lr 1e-3, weight decay 1e-3, batch 64, **20 epochs** (fewer than the 40 used in Part 2, to fit the time limit; so these numbers are not comparable to the Part 2 table). One seed (0). Model selection: the epoch with the best accuracy on one validation subject, which is one of the *training* subjects (the next subject, cyclically), so training uses 8 subjects per fold. Per-channel z-score statistics come from the 8 training subjects only.
- Recorded per fold: held-out subject, accuracy, balanced accuracy, number of test trials, configuration, predictions and labels.

## 3. CSP+LDA methodology
`mne.decoding.CSP` (6 components, Ledoit–Wolf covariance regularisation, log-variance features) followed by `LinearDiscriminantAnalysis`, as one scikit-learn pipeline fitted on the 9 training subjects only (no validation subject, nothing is selected). The held-out subject is transformed and predicted without any fitting on it; `test_loso.py` checks that altering the held-out subject's data and labels does not change the predictions. Limitations: a single broad 7–30 Hz band (no filter bank), no hyperparameter search, a single pooled cross-subject CSP.

## 4. Results (10 held-out subjects, 45 trials each; accuracy, chance = 0.5)

| model | subjects | mean | sd | median | min | max | bootstrap-over-subjects 95% CI of the mean |
|---|---|---|---|---|---|---|---|
| EEGNet | 10 | 0.536 | 0.062 | 0.533 | 0.444 | 0.644 | 0.500–0.573 |
| CSP+LDA | 10 | 0.518 | 0.060 | 0.511 | 0.422 | 0.622 | 0.484–0.553 |

Mean balanced accuracy: EEGNet 0.540, CSP+LDA 0.525. Subject-wise accuracy (also in `research/results/loso_subjectwise.csv`, figure `research/figures/loso_eegnet_vs_csp.png`):

| subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet | 0.644 | 0.600 | 0.444 | 0.556 | 0.578 | 0.511 | 0.467 | 0.511 | 0.489 | 0.556 |
| CSP+LDA | 0.489 | 0.511 | 0.556 | 0.422 | 0.511 | 0.467 | 0.511 | 0.600 | 0.489 | 0.622 |

## 5. EEGNet comparison
Paired over the 10 subjects, EEGNet minus CSP+LDA = +0.018 (EEGNet better in 5 subjects, worse in 4, tied in 1; Wilcoxon signed-rank p = 0.60).

## 6. Interpretation
**Measured:**
- Both models are close to chance across held-out subjects. The EEGNet interval (0.500–0.573) just touches chance; the CSP+LDA interval (0.484–0.553) includes it.
- The two models are not distinguishable at this sample size (p = 0.60, 10 subjects).
- Per-subject accuracy varies from 0.42 to 0.64 for both models, and the subjects on which each model does well differ (EEGNet best on S1, CSP+LDA best on S10).
- The classical baseline is no better than the CNN, so near-chance accuracy is not specific to the deep models or to the Transformer experiments.

**Hypotheses (not tested here):**
- Cross-subject variability and the small number of training subjects (8–9) dominate performance.
- Subject-specific mu/beta frequency, amplitude and topography, and possible subject-specific noise, limit a pooled cross-subject decoder.
- Per-subject calibration, subject-wise normalisation, or training on many more subjects might help.

## 7. Limitations
- 10 subjects and 45 trials per held-out subject: one trial is 2.2 points, so each subject's accuracy has a wide interval (not computed per subject here), and with 10 subjects the comparison has little power.
- EEGNet: one seed, 20 epochs, no tuning, 8 training subjects per fold (one used for validation); its results could change with more epochs or seeds, and I did not run multiple seeds.
- CSP+LDA: untuned, single band, pooled across subjects.
- The mean of 10 subject-level accuracies is the unit of inference; trial-level binomial p-values were deliberately not used. No permutation test was run in this extension, so no claim of significance above chance is made.
- No handling of the cue/EOG confound (see the audit); LOSO accuracy could be partly driven by non-neural signal, and I did not test that.
- These numbers are not comparable with the Part 1 or Part 2 numbers (different split, epoch count, training set size).

## 8. Future work
Multiple EEGNet seeds and an epoch/regularisation sensitivity check; filter-bank CSP and Riemannian tangent-space baselines; a subject-block permutation test for each model; frontal/central channel and window ablations for the cue confound; more subjects (the audit's Phase 1 plan).

## Reproduce
```bash
python research/loso.py --model csp          # about 30 s
python research/loso.py --model eegnet       # about 8 min on CPU
python research/loso_report.py               # CSV, figure, summary
python research/test_loso.py                 # no-overlap / no-leakage checks
```
Requires the cache `part2_transformer/.cache/subjects.npz` (created by `python part2_transformer/protocol.py`).
