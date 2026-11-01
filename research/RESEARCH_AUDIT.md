# Research audit of the assessment project (Phase 1)

Status: audit and proposed plan only. **No new experiments have been run and no existing file was changed.**
All original results (`part1_diagnose/results.jsonl`, `part2_transformer/results.jsonl`, `ANSWERS.md`, `logs/`) are preserved; new work will live under `research/` only.

## 0. What the existing results are, exactly

| | Part 1 (`part1_diagnose`) | Part 2 (`part2_transformer`) |
|---|---|---|
| Subjects | 1–10 | 1–10 |
| Runs / trials | imagery runs 6, 10, 14; 42 usable trials/subject (3 end-of-run trials dropped) | same runs; 45 trials/subject |
| Filter / window | 1–40 Hz, 0–4 s (641 samples) | 7–30 Hz, 0–3 s (481 samples) |
| Split | 2 test / 2 val / 6 train subjects, **chosen by seed** | fixed: train 1–6, val 7–8, test 9–10 |
| Test size | 84 trials (2 subjects) | 90 trials (2 subjects) |
| Epochs, batch, optimiser | 30, 64, AdamW 1e-3 / wd 1e-3 | 40, 64, AdamW 1e-3 / wd 1e-3 |
| Model selection | best validation epoch | best validation epoch |
| Result | EEGNet 0.543 ± 0.043 (5 seeds, all channels) | EEGNet 0.524 ± 0.068; Vanilla 0.506 ± 0.028 (4 seeds); PatchOnly 0.473 ± 0.040; YourModel 0.467 ± 0.042 |

**The Part 1 number (0.543) and the Part 2 EEGNet number (0.524) are not comparable.** They use different filters, windows, trial counts, epoch budgets and different test subjects, and in Part 1 the test pair changes with the seed. Neither is "the" EEGNet accuracy.

## 1. Existing strengths

- Five defects found and fixed one per commit, with a cumulative multi-seed ablation (original 0.724 → corrected 0.543); history preserved.
- Subject-level splitting, train-only normalisation and validation-based selection, with an executable leakage test (`test_pipeline.py`) and a split-hash check for Part 2.
- Negative results are reported, not hidden (YourModel did not help; all models at or near chance).
- Hypothesis for Part 2 recorded in git before the model code; starter vanilla model and EEGNet left untouched.
- Attention block written explicitly and unit-tested against `F.scaled_dot_product_attention`.

## 2. Scientific limitations

- **No model reliably beats chance.** One run of 19 in Part 2 exceeds the 0.600 threshold (n = 90). Any conclusion about architecture ranking or "what the model learned" therefore rests on noise-level differences.
- The "validated EEGNet baseline" is validated for *leakage-freedom*, not for performance: it is near chance cross-subject.
- Part 2's hypothesis (tokenisation is the bottleneck) was tested in a regime where the dominant failure is cross-subject variability, so the experiment could not discriminate the hypothesis well.
- No classical baseline (CSP+LDA, Riemannian methods) was run. On small motor-imagery datasets these are the standard references; without one, "near chance" cannot be attributed to the deep models versus the data/protocol.

## 3. Data limitations

- Only 10 of the 109 subjects in EEGBCI are used; 42–45 trials per subject; imagery runs 6/10/14 only.
- **No EOG channel**, so ocular contamination cannot be measured directly; only frontal proxies (Fp/AF) exist.
- No artifact rejection: peak-to-peak up to about 450–1,280 µV in some subjects; per-subject std varies about 4× (17.8–77.4 µV). Pooled z-scoring does not equalise subjects.
- Very short recordings (3 runs, about 8 minutes of task per subject): the dominant between-subject differences (rhythm frequency, amplitude, BCI "illiteracy") cannot be averaged out.
- Some EEGBCI subjects are known in the literature to have irregular annotations or recordings; this has not been checked here and should be verified programmatically before scaling up, not assumed.

## 4. Evaluation limitations

- **Test sets are two subjects (84–90 trials).** One trial = 1.1–1.2 points. A single run's exact 95% interval at n = 90 is about ±0.10 (e.g. 0.524 → 0.414–0.629), larger than every difference between models.
- Part 2 uses one fixed split; seeds change only initialisation and batch order, so seed sd does not reflect subject variability. Part 1 varies the test pair by seed, mixing seed and subject variance.
- Model selection uses only two validation subjects (noisy; transformers show validation ≈ 0.61–0.66 versus test ≈ 0.47–0.51).
- No confidence intervals, no subject-wise results, no balanced accuracy/F1/confusion matrix stored (`results.jsonl` holds accuracy only), no permutation tests.
- Seeds are not independent evidence about generalisation: they share the same test subjects. Vanilla has 4 seeds, others 5.
- Binomial tests treat the 90 test trials as independent, but trials of the same subject are correlated, so trial-level p-values are anti-conservative for the claim "generalises to new subjects".
- No hyperparameter tuning for any model; different depths and parameter counts across the transformers.

## 5. Architectural limitations

- YourModel mixes all 64 electrodes in the first conv with no spatial structure, then discards electrode identity; the stem's spatial filters could not be inspected.
- Attention (last layer, CLS) is near-uniform over time (0.058–0.063 versus 1/16 = 0.0625), so the encoder adds little beyond pooling patch features. 200 ms patches may be too coarse or too fine; not varied.
- BatchNorm statistics and a learned stem may encode subject-specific amplitude; no subject-invariance mechanism (per-trial/per-subject normalisation, augmentation, domain adaptation).
- Only one inductive bias was tested; spatial (channel) structure, frequency-band structure and multi-scale temporal structure were not.

## 6. Possible confounds (not yet separated)

1. **Visual cue position** (top = fists, bottom = feet): vertical gaze shifts, blinks, and cue-locked visual evoked responses. The 0–0.5 s window includes the evoked transient seen at about 0.25 s in the ERD plot.
2. **Frontal/ocular leakage into central channels** through volume conduction (strongest at low frequencies, but 7–13 Hz is not immune).
3. **Facial/neck EMG** (beta/gamma), not removed.
4. **Subject identity** and per-subject amplitude/scale (relevant when training with few subjects).
5. **Run/session structure:** runs are concatenated; trial order, run effects and baseline drift were not examined.
6. **Reference choice** (no common average reference tested): spatial patterns near Cz can change with reference.
7. **Class balance** is near 50/50 per subject (checked for Part 2), so no majority-class confound, but only accuracy was recorded.

## 7. Most promising improvement opportunities

1. A **statistically meaningful evaluation** (more subjects, LOSO / grouped CV, subject-level intervals and group-aware permutation tests). Without it nothing downstream can be interpreted.
2. A **confound study** with evidence (frontal-only, central-only, ICA with frontal EOG proxies, window ablation).
3. **Strong classical and CNN baselines** before attention models.
4. **Explainability** only after some model is reliably above chance.

## 8. Proposed plan (awaiting approval; nothing implemented)

Ranked by value to a defensible research claim.

| # | Improvement | Scientific value | Complexity | Compute (this 12-core / 15.7 GB CPU machine) | Expected contribution |
|---|---|---|---|---|---|
| 1 | **Subject-level evaluation framework on more subjects**: LOSO and GroupKFold over all 10 subjects, then 30–109 subjects if the download and per-subject checks pass; per-subject accuracy, balanced accuracy, precision/recall/F1, confusion matrices, bootstrap-over-subjects CIs, and a permutation test that permutes labels *within* subject (and run) blocks and aggregates at subject level. Config-driven runner with fixed seeds, metadata, per-experiment result files in `research/results/`. | Highest: turns "n = 2 subjects" into a real generalisation estimate and makes any claim testable | Medium | Small models: EEGNet about 70 s and YourModel about 33 s per fit (measured on the current split), so 10-fold LOSO × 3 seeds is on the order of 15–35 min each; more subjects scales roughly linearly. The vanilla transformer (about 14 min per fit alone) is too slow for full LOSO; run on a subset or 1 seed and say so. | A trustworthy benchmark, publishable in structure, whatever the accuracies turn out to be |
| 2 | **Strong baselines**: CSP + LDA, filter-bank CSP, Riemannian tangent-space + logistic regression (classical, CPU-light), then tuned EEGNet (dropout, weight decay, temporal kernel, D, F1) with nested subject-wise validation | High: tells us whether "near chance" is the data/protocol or the deep models | Low–medium | Very low for classical; EEGNet grid is the main cost (keep it small, e.g. under 10 configs) | A reference point every later model must beat; likely the most informative single addition |
| 3 | **Preprocessing and confound study**: mu / beta / mu+beta, windows (e.g. 0.5–3.5 s to avoid the cue-evoked transient), common average reference, channel groups (frontal, central, occipital, all), ICA (MNE `find_bads_eog` using Fp1/Fp2/AF as EOG proxies since no EOG channel exists) with before/after decoding | High: directly addresses the cue confound that Part 1 only suggested | Medium | Low–medium (ICA per subject is fast; reruns of small models) | Evidence for or against the cue/EOG explanation. I will report what the data support and will not claim artifact removal without a measurable effect (e.g. proxy-EOG correlation drop). |
| 4 | **One hypothesis-driven model at a time, only if baselines justify it**: (a) multi-scale temporal CNN, (b) channel-attention on top of the best CNN, (c) hybrid conv-stem + small encoder. Each with a written hypothesis before coding, evaluated under the framework from #1. Original models not modified. | Medium; depends on #1–#2 showing headroom | Medium–high | Medium (small models only) | A genuine test of the Part 2 idea at an adequate sample size, which the assessment could not do |
| 5 | **Explainability, gated on a model above chance**: saliency / Integrated Gradients per channel × time, channel ablation, band-importance by band-stop filtering, attention maps (labelled as attention, not explanation), compared with expected sensorimotor topography (C3/C4/Cz); Grad-CAM only for the CNN where a conv feature map exists | Medium–high *if* a model works; explaining chance-level models is not meaningful | Medium | Low | A physiology cross-check, with explicit sanity checks (e.g. explanation of a label-shuffled model should look uninformative) |

**Suggested order and decision points:** #1 first (infrastructure + evaluation), then #2 and #3 in parallel, then stop and review. #4 and #5 only if #1–#3 show a model that is reliably above chance under subject-level evaluation; otherwise the honest outcome is a rigorous negative result plus a characterised confound.

**Constraints I will keep:** new code and results only under `research/`; seeds fixed and logged; identical splits across compared models; filtering, normalisation, channel selection and hyperparameter tuning done inside training folds only; no test subject used for selection; one change at a time; no claims of significance without the appropriate test; no push to GitHub.

**Open decisions for you** (they change cost, not honesty): how many subjects to download (10 only, 30–40, or all 109 after programmatic quality checks), and whether the vanilla transformer should be dropped from LOSO because of its cost.
