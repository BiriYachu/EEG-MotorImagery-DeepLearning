# Per-subject feature standardisation for CSP+LDA (pre-registered, one attempt)

**Hypothesis (written before running):** In the Filter-Bank experiment, CSP+LDA predicted a single class for every trial of held-out subjects S5 and S7, which suggests a subject-specific shift in the CSP log-variance features. Standardising the features within each subject (subtract that subject's mean, divide by its std, computed over the subject's own trials, **no labels**) should remove the shift and therefore raise the mean held-out-subject accuracy of CSP+LDA and Filter-Bank CSP+LDA, and remove the single-class predictions.

**Protocol (fixed in advance, no tuning, no variants):** same LOSO folds, same CSP / filter-bank configuration and fitting as before (CSP fitted on training subjects only; LDA fitted on training features only). The only change: features of every subject, training subjects and the held-out subject, are standardised per subject. This is **transductive** (it uses the held-out subject's unlabelled feature statistics) and so is **not the same protocol as the earlier LOSO results**; it assumes many unlabelled trials from the new subject are available at test time and that classes are roughly balanced in them (true here, 45 trials, about 50/50). Held-out labels are never used.

**Success criterion:** mean accuracy above the corresponding earlier result with a paired Wilcoxon test over the 10 subjects reported; anything else is reported as no improvement.

## Results
_Pending: filled in after the run._
