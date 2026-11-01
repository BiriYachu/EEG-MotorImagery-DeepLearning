# ANSWERS

Setup: PhysioNet EEGBCI, subjects 1–10, 160 Hz, EEGNet (Part 1 `model.py`), CPU.
Reproduce Part 1: `cd part1_diagnose && python train.py --channels all --seed 0` (seeds 0–4; `--channels frontal`), `python erd.py --band mu`, `python test_pipeline.py`.
All numbers below come from `part1_diagnose/results.jsonl` and `logs/`.

---

# PART 1 — Diagnose & Trust

## Task 1 — Bugs

I found **five** defects that I can support with evidence. I did not find a sixth; the brief says six, and I report what I could verify rather than invent one (candidates I checked and rejected are at the end of this section).
One commit per fix (`git log`).

| # | Where | What was wrong | Effect on reported accuracy | Fix (commit) |
|---|---|---|---|---|
| 1 | `data.py` `RUNS` | `[5,9,13]` are *executed* fists/feet runs, not imagery (imagery is 6/10/14, as the docstring and Part 2 state). | Executed movement gives a larger, EMG/movement-contaminated signal → inflates accuracy (expected; single seed 0.667→0.619, noisy). | `fe3af77` |
| 2 | `erd.py` `BASELINE` | `(0.0, 0.5)` is *after* the cue, so the reference power already contains the evoked transient and early ERD. Before-plot: pre-cue curve sat near −10 %, not 0. | Plot only. Shrinks/distorts ERD; no decoder effect. | `76a8a36` |
| 3 | `train.py` z-scoring | Mean/std computed on all trials, test included, before splitting. | Small optimistic leak. Measured: none visible at seed 0 (0.619→0.619). | `899b0eb` |
| 4 | `train.py` split | `train_test_split` over *trials*: every subject in train and test, `groups` unused although the comment says "subject-independent". Model can learn subject signatures. | Inflates accuracy in principle. Seed 0 went 0.619→0.690 but on a *different* test set (subjects 5,7), so that single comparison is not like-for-like. | `5ea6ca2` |
| 5 | `train.py` checkpoint | Best epoch chosen as the max of 30 *test* evaluations; the validation set was computed and never used. | Selection bias → inflates. Seed 0: 0.690→0.583 (val 0.631). | `52dd6f7` |

Also committed: `feat` frontal channel group (`7ee7867`), `refactor` + `test_pipeline.py` (`95bb0a7`, behaviour unchanged, seed 0 reproduces 0.583), results/logs (`e345601`).

### Before / after
- **Original pipeline** (seed 0, all channels, trial-level split): **0.667** (n_test = 84) — `logs/baseline_orig.log`.
- **Corrected pipeline**, all 64 channels, seeds 0–4: **mean 0.543, sd 0.043, min 0.488, max 0.583** (n_test = 84 per run).
- Evaluation: held-out *subjects* (2 test, 2 val, 6 train; the seed chooses which subjects, so seed variance includes subject choice), 30 epochs, AdamW lr 1e-3, batch 64, checkpoint chosen on validation.
- Chance threshold for n = 84 (one-sided 5 %): ≥ 0.607. **No all-channel run exceeds it.** The corrected baseline is therefore close to chance on this split; it is honest, not impressive.

Per-fix ablation over seeds: see "Ablation" below (filled in only if the job finished; otherwise it is stated as not run).

### Verification
- No subject overlap across train/val/test for 20 seeds, and train-set z-scores unchanged when val/test data are altered ×1000: `python test_pipeline.py` → `OK`.
- Original vs. fixed diff touches only `data.py`, `erd.py`, `train.py`.

### Checked and not counted as bugs
- 3 epochs/subject dropped: the −2…5 s window of the last trial of each run crosses the run boundary (`BAD boundary`/`TOO_SHORT`). Few trials, no label bias.
- No artifact rejection: peak-to-peak up to ≈ 450–1280 µV; per-subject std varies 17.8–77.4 µV. A real limitation (pooled z-scoring does not equalise subjects), but a design choice, not a planted defect; I left it.
- Event→label mapping (45 events/subject, T1→0, T2→1), crop (0–4 s = 641 samples), Hilbert-power ERD maths, µV scaling: correct.

## Task 2 — ERD (`part1_diagnose/erd_mu.png`, mu 8–13 Hz, 10 subjects, imagery runs, baseline −1.5…−0.5 s)

All three sites show a brief positive transient ≈ 0.25 s after the cue (a cue-evoked response, not imagery), followed by a sustained mu decrease of roughly −10 to −25 % from ≈ 0.5 s onward. At C3 and C4 hands are slightly more negative than feet around 0.5–1.5 s, which is the direction expected for hand-area desynchronisation; at Cz feet are not clearly stronger than hands, and the hands/feet curves overlap heavily everywhere. The hand area lies laterally (C3/C4), whereas the foot area is medial, in the interhemispheric fissure under Cz, so feet should desynchronise at Cz more than hands do; I do not see that cleanly. Cz may be weak because the foot representation is small, deep and oriented so that its scalp projection is attenuated, because C3/C4/Cz mix through volume conduction (the bilateral hand ERD also spreads medially), because imagery ERD is weaker than movement ERD, and because the average is over only ~21 trials/class/subject with subject-specific frequencies. The curves are noisy and I did not test the hands–feet difference statistically, so this is a qualitative reading.

## Task 3 — What is the model using?

Corrected pipeline, same split protocol, seeds 0–4: **frontal (8 Fp/AF electrodes) mean 0.583, sd 0.089, range 0.500–0.726; all channels mean 0.543, sd 0.043, range 0.488–0.583.** Frontal is not worse than all channels (and one seed, 0.726, is clearly above chance, p < 0.001 for n = 84), although prefrontal electrodes carry almost no sensorimotor mu/beta. With sd ≈ 0.09 and 5 seeds on different test-subject pairs, the two means are not statistically distinguishable, so I cannot claim frontal > all, only that frontal-only decoding is not at chance in some runs. A decoder that works from Fp/AF channels must be using something other than motor-cortex rhythms; since the target is at the top of the screen for fists and the bottom for feet, the leading candidate is vertical eye movement/gaze shift (EOG) and related frontal artefact. Alternative explanations I cannot exclude: frontal EMG, genuine frontal/SMA activity, or chance on a small test set. **I have not shown eye movement is the cause**: the dataset has no EOG channel and I did not run regression/ICA or a spectral/low-frequency analysis. This matters because a high frontal-only score means the all-channel score could be partly cue-driven rather than neural, so it cannot be read as evidence of brain-based decoding.

## Task 4 — Short answers

1. **45 trials, 58 %?** 58 % ≈ 26/45. Under chance (p = 0.5) a one-sided exact binomial test gives p ≈ 0.19; 95 % CI for the accuracy is 0.42–0.72. You need ≥ 28/45 (62 %) to be above chance at 5 %. So no, not demonstrably above chance. Check with a binomial test or a label-permutation test, and account for class balance.
2. **EEGNet depthwise spatial conv (`Conv2d(F1, F1*D, (n_ch,1), groups=F1)`)**: for each of the F1 temporal (band-pass) filters it learns D spatial filters, i.e. weighted sums across electrodes, comparable to CSP-like spatial filters that pick out a scalp pattern for that frequency band (and suppress noise/volume-conducted sources).
3. **Preventing cue exploitation**: counterbalance or randomise target position across classes (so top/bottom is independent of class) or remove the visual cue difference; analytically, regress out / ICA-remove EOG and frontal channels, band-limit to mu/beta over sensorimotor channels, and only count frontal-only performance as a negative control.

## Ablation (per-fix effect over seeds)

_To be filled from `logs/ablation.jsonl` once the background job finishes._
