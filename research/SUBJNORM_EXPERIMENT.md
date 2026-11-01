# Per-subject feature standardisation for CSP+LDA (pre-registered, one attempt)

**Hypothesis (written before running):** In the Filter-Bank experiment, CSP+LDA predicted a single class for every trial of held-out subjects S5 and S7, which suggests a subject-specific shift in the CSP log-variance features. Standardising the features within each subject (subtract that subject's mean, divide by its std, computed over the subject's own trials, **no labels**) should remove the shift and therefore raise the mean held-out-subject accuracy of CSP+LDA and Filter-Bank CSP+LDA, and remove the single-class predictions.

**Protocol (fixed in advance, no tuning, no variants):** same LOSO folds, same CSP / filter-bank configuration and fitting as before (CSP fitted on training subjects only; LDA fitted on training features only). The only change: features of every subject, training subjects and the held-out subject, are standardised per subject. This is **transductive** (it uses the held-out subject's unlabelled feature statistics) and so is **not the same protocol as the earlier LOSO results**; it assumes many unlabelled trials from the new subject are available at test time and that classes are roughly balanced in them (true here, 45 trials, about 50/50). Held-out labels are never used.

**Success criterion:** mean accuracy above the corresponding earlier result with a paired Wilcoxon test over the 10 subjects reported; anything else is reported as no improvement.

## Results
Run once, as pre-registered (`research/loso_subjnorm.py`, results in `research/results/loso_csp_subjnorm.jsonl` and `loso_filterbank_csp_subjnorm.jsonl`, raw summary `results/subjnorm_summary.txt`). 10 held-out subjects, 45 trials each, chance 0.5.

| model | variant | mean acc | sd | median | min | max | bootstrap 95% CI | mean bal acc |
|---|---|---|---|---|---|---|---|---|
| CSP+LDA | earlier LOSO | 0.518 | 0.060 | 0.511 | 0.422 | 0.622 | 0.484–0.556 | 0.525 |
| CSP+LDA | + per-subject standardisation | 0.573 | 0.146 | 0.578 | 0.311 | 0.822 | 0.489–0.660 | 0.574 |
| FilterBank-CSP+LDA | earlier LOSO | 0.529 | 0.072 | 0.511 | 0.422 | 0.689 | 0.489–0.573 | 0.527 |
| FilterBank-CSP+LDA | + per-subject standardisation | 0.509 | 0.110 | 0.511 | 0.311 | 0.689 | 0.442–0.573 | 0.508 |

Paired over the 10 subjects: CSP+LDA **+0.056** (better in 6, worse in 3, tied in 1; Wilcoxon p = 0.25); Filter-Bank **−0.020** (better in 3, worse in 5, tied in 2; p = 0.56).
Per subject (earlier → standardised), CSP+LDA: S1 0.49→0.51, S2 0.51→0.82, S3 0.56→0.51, S4 0.42→0.31, S5 0.51→0.60, S6 0.47→0.42, S7 0.51→0.71, S8 0.60→0.67, S9 0.49→0.56, S10 0.62→0.62.

## Interpretation
**Measured:** the single-class predictions disappeared (predicted-class fractions are now about 0.44–0.60 for every subject, see `logs/loso_subjnorm.log`). The CSP+LDA mean rose by 0.056, driven mostly by S2 (+0.31) and S7 (+0.20), but this is **not statistically supported** (p = 0.25; bootstrap interval 0.489–0.660 includes chance), the spread across subjects roughly doubled (sd 0.146), and some subjects got worse, notably S4 (0.31, systematically below chance). Filter-Bank did not improve (−0.020). **Improvement is not demonstrated.** The hypothesis is therefore only partly supported: the single-class failure is removed, accuracy is not reliably raised.
**Hypotheses (not tested):** the remaining subject-level variation is partly a sign/orientation problem for some subjects (accuracy well below 0.5 for S4 and, for Filter-Bank, S7); with so few trials per subject the per-subject feature statistics are noisy.

## Limitations
Transductive (needs the held-out subject's unlabelled trials; not comparable with the earlier LOSO protocol); one attempt, no tuning, but two models were tested so the one positive-looking mean is a single outcome among two; 10 subjects, 45 trials each; no permutation test; no claim of significance above chance is made.

## Reproduce
```bash
python research/loso_subjnorm.py     # about 1.5 min
python research/test_subjnorm.py     # held-out labels cannot influence predictions
```
