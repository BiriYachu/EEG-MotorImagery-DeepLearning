# Final summary

**Bottom line:** no model was shown to decode both-fists vs both-feet imagery from unseen subjects better than chance in a way that holds up statistically. No improvement over a baseline was demonstrated anywhere. The reliable outputs are the corrected, leak-free pipeline, the quantified effect of the original bugs (original 0.724 mean → 0.543 after fixes), and honest evidence about what limits cross-subject decoding on 10 subjects.

## Consolidated results (different protocols are NOT comparable across blocks)

| Block / protocol | Model | n runs | Mean acc | sd | Chance | Main limitation | Improvement demonstrated? |
|---|---|---|---|---|---|---|---|
| **A. Part 1**: seed-chosen 2 test subjects (6/2/2 subjects), 1–40 Hz, 0–4 s, 30 epochs, n_test = 84 per run | EEGNet (all 64 ch) | 5 seeds | 0.543 | 0.043 | 0.5 (p<0.05 needs ≥ 0.607) | test pair varies with seed; 2 test subjects | n/a (baseline) |
| **B. Part 2**: fixed split train 1–6 / val 7–8 / test 9–10 (n = 90), 7–30 Hz, 0–3 s, 40 epochs | EEGNet | 5 seeds | 0.524 | 0.068 | 0.5 (needs ≥ 0.600) | 2 test subjects, one split | n/a (baseline) |
| B | Vanilla Transformer | 4 seeds | 0.506 | 0.028 | 0.5 | only 4 seeds; ~1,257 s/run (runs overlapped) | No (vs EEGNet) |
| B | PatchOnly (starter stub) | 5 seeds | 0.473 | 0.040 | 0.5 | same | No |
| B | YourModel (conv stem + patch tokens) | 5 seeds | 0.467 | 0.042 | 0.5 | same; attention near-uniform | **No**: nominally lowest; hypothesis not supported |
| **C. LOSO**: 10 held-out subjects, train on other 9 (EEGNet: 8 + 1 val subject), 7–30 Hz, 0–3 s, 45 trials per subject | EEGNet (20 epochs, 1 seed) | 10 subjects | 0.536 | 0.062 | 0.5 | one seed, 20 epochs, untuned | No (CI 0.500–0.573 touches chance) |
| C | CSP + LDA | 10 subjects | 0.518 | 0.060 | 0.5 | untuned, single band | No (CI 0.484–0.553) |
| C | Filter-Bank CSP + LDA (8–12, 13–20, 20–30 Hz) | 10 subjects | 0.529 | 0.072 | 0.5 | untuned; S5, S7 predicted as a single class | **No**: +0.011 vs CSP+LDA, Wilcoxon p = 0.44; −0.007 vs EEGNet, p = 0.79 |

| C (transductive) | CSP+LDA with per-subject feature standardisation | 10 subjects | 0.573 | 0.146 | 0.5 | uses the held-out subject's unlabelled trials; not the same protocol as the rows above | **No**: +0.056 vs plain CSP+LDA, Wilcoxon p = 0.25, CI includes chance (`research/SUBJNORM_EXPERIMENT.md`) |
| C (transductive) | Filter-Bank CSP+LDA with per-subject standardisation | 10 subjects | 0.509 | 0.110 | 0.5 | same | No: −0.020, p = 0.56 |

Do not compare numbers across blocks A, B and C: they differ in filter band, window, epoch count, training-set size, test subjects and unit of analysis (A, B: runs on a shared 2-subject test set; C: 10 independent subject-level accuracies).

## Key findings (measured)
- Five defects in the starter fixed one per commit; the original pipeline's 0.724 (5-seed mean) fell to 0.543 after the fixes (ablation, `logs/ablation.jsonl`).
- Corrected EEGNet is at or near chance cross-subject under every protocol used (A, B, C).
- A classical CSP+LDA and a filter-bank variant do no better; near-chance is therefore not specific to the Transformer experiments.
- Part 2: patch tokens with a conv stem did not beat vanilla attention, patch-only or EEGNet; learned attention was near-uniform in time.
- Frontal-only (8 Fp/AF channels) decoding was not worse than all-channel in Part 1 (0.583 vs 0.543, not distinguishable), consistent with, but not proof of, a non-neural (eye-movement/cue) contribution.

## Hypotheses (not tested)
Cross-subject variability with few training subjects dominates; per-subject calibration or many more subjects might help; the top/bottom cue may contribute non-neural signal.

## Main limitations
10 subjects; 45 trials per subject; at most 5 seeds; no permutation test anywhere; no hyperparameter tuning; no EOG channel and no artifact handling; no claim of significance above chance is made.

## Where things are
`README.md` (setup and commands), `ANSWERS.md` (assessment answers, walkthrough prep), `research/RESEARCH_AUDIT.md`, `research/RESEARCH_EXTENSION.md`, `research/FILTERBANK_EXPERIMENT.md`, figures in `part1_diagnose/erd_mu.png`, `part2_transformer/figure_part2.png`, `research/figures/`.
