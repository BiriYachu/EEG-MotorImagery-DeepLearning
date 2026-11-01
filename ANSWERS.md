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
| 1 | `data.py` `RUNS` | `[5,9,13]` are *executed* fists/feet runs, not imagery (imagery is 6/10/14, as the docstring and Part 2 state). | Executed movement gives a larger, EMG/movement-contaminated signal → inflates accuracy (expected; single seed 0.667→0.619, noisy). | `21b21d2` |
| 2 | `erd.py` `BASELINE` | `(0.0, 0.5)` is *after* the cue, so the reference power already contains the evoked transient and early ERD. Before-plot: pre-cue curve sat near −10 %, not 0. | Plot only. Shrinks/distorts ERD; no decoder effect. | `d0878d6` |
| 3 | `train.py` z-scoring | Mean/std computed on all trials, test included, before splitting. | Small optimistic leak. Measured: none visible at seed 0 (0.619→0.619). | `b3b9f34` |
| 4 | `train.py` split | `train_test_split` over *trials*: every subject in train and test, `groups` unused although the comment says "subject-independent". Model can learn subject signatures. | Inflates accuracy in principle. Seed 0 went 0.619→0.690 but on a *different* test set (subjects 5,7), so that single comparison is not like-for-like. | `82066e6` |
| 5 | `train.py` checkpoint | Best epoch chosen as the max of 30 *test* evaluations; the validation set was computed and never used. | Selection bias → inflates. Seed 0: 0.690→0.583 (val 0.631). | `64c96b2` |

Also committed: `feat` frontal channel group (`57ec89d`), `refactor` + `test_pipeline.py` (`5b2073b`, behaviour unchanged, seed 0 reproduces 0.583), results/logs (`96a6a04`).

### Before / after
- **Original pipeline** (seed 0, all channels, trial-level split): **0.667** (n_test = 84) — `logs/baseline_orig.log`.
- **Corrected pipeline**, all 64 channels, seeds 0–4: **mean 0.543, sd 0.043, min 0.488, max 0.583** (n_test = 84 per run).
- Evaluation: held-out *subjects* (2 test, 2 val, 6 train; the seed chooses which subjects, so seed variance includes subject choice), 30 epochs, AdamW lr 1e-3, batch 64, checkpoint chosen on validation.
- Chance threshold for n = 84 (one-sided 5 %): ≥ 0.607. **No all-channel run exceeds it.** The corrected baseline is therefore close to chance on this split; it is honest, not impressive.

Over seeds 0–4 the original pipeline averages **0.724** (sd 0.046), versus 0.543 after all fixes; per-fix effects are in the "Ablation" section below (all 25 runs completed).

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

Cumulative: each row adds one fix on top of the previous one. Same protocol, all 64 channels, seeds 0–4, n_test = 84 per run. Raw rows: `logs/ablation.jsonl` (25 runs, all completed). Each state was run from a git worktree at that commit.

| state (commit) | mean | sd | min | max | change in mean |
|---|---|---|---|---|---|
| original (`53f41a1`) | 0.724 | 0.046 | 0.667 | 0.786 | – |
| + imagery runs 6/10/14 (`21b21d2`) | 0.645 | 0.026 | 0.619 | 0.679 | −0.079 |
| + train-only z-score (`b3b9f34`) | 0.648 | 0.031 | 0.619 | 0.690 | +0.003 |
| + subject-level split (`82066e6`) | 0.600 | 0.060 | 0.524 | 0.690 | −0.048 |
| + validation-based checkpoint (`64c96b2`) | 0.543 | 0.043 | 0.488 | 0.583 | −0.057 |

Reading: the original 0.724 mean was inflated by about 0.18 in total. The directions match the predictions for fixes 1, 4 and 5; fix 3 is negligible (+0.003, within noise). The earlier single-seed rise for fix 4 (0.619 to 0.690) was noise from a different test set; the 5-seed mean falls by 0.048.

Caveats: 5 seeds per state, sd 0.03–0.06, so differences of about 0.05 or less are not reliable. The effects are order-dependent (cumulative), not isolated one-at-a-time effects. Before fix 4 the test set is a random 20 % of trials; after it, 2 whole subjects, so fix 4's row compares different test populations. The ERD fix (2) does not affect the decoder and is not in the table.

---

# LIVE WALKTHROUGH PREPARATION (Part 1 topics; Part 2 topics are added with Part 2)

Format per item: **Short** (say this first), then **Deeper**, then **Repo** (where it lives in my code).

## 1. EEG signal fundamentals
**Short:** Scalp EEG is microvolt-scale voltage from synchronous postsynaptic currents of cortical pyramidal neurons, summed through skull and scalp, so each electrode mixes many sources.
**Deeper:** Spatial resolution is low (volume conduction). Neural amplitudes are tens of µV, versus eye blinks and EMG at hundreds of µV. Signal is analysed in bands (delta, theta, alpha/mu 8–13 Hz, beta 13–30 Hz).
**Repo:** `data.py` returns µV arrays `(trials, 64, 641)`. I measured peak-to-peak up to about 1.2 mV in some subjects, i.e. unrejected artefacts.

## 2. Sampling frequency and Nyquist
**Short:** 160 Hz sampling gives a Nyquist frequency of 80 Hz; content above 80 Hz would alias.
**Deeper:** Nyquist = fs/2. The 40 Hz low-pass sits well under it, and mu/beta (up to 30 Hz) are safely resolved. A 4 s trial is 641 samples (inclusive endpoints).
**Repo:** `data.py` raises if `sfreq != 160`; `CROP=(0,4)` gives 641 samples.

## 3. EEG filtering
**Short:** A band-pass removes slow drift (below 1 Hz) and high-frequency noise/EMG/line noise. I filter the continuous recording, then epoch.
**Deeper:** Filtering continuous data avoids edge artefacts at every epoch boundary. MNE's default FIR (`firwin`) is zero-phase and non-causal: fine offline, not usable as-is for online BCI. 1–40 Hz is broad, so the network must find mu/beta itself. `erd.py` band-passes 8–13 Hz, takes the Hilbert envelope, and squares it to get power.
**Repo:** `L_FREQ, H_FREQ = 1, 40` in `data.py`; `band_power_epochs` in `erd.py`.

## 4. Motor imagery
**Short:** Imagining a movement without executing it modulates sensorimotor rhythms like real movement does, but more weakly.
**Deeper:** Here: imagine opening/closing both fists (hand areas, lateral, C3/C4) versus moving both feet (foot area, medial, near Cz). Cross-subject MI decoding is hard: effects are small and vary by person. Bug 1 used the *executed* runs, which are easier and contaminated by EMG/movement.
**Repo:** `RUNS = [6, 10, 14]`.

## 5. Mu and beta rhythms
**Short:** Mu (8–13 Hz) and beta (13–30 Hz) are sensorimotor rhythms present at rest over motor cortex and suppressed during movement or imagery.
**Deeper:** Mu overlaps the alpha band but is localised over sensorimotor cortex and reacts to movement rather than to eye opening. Beta shows ERD during movement and a rebound afterwards.
**Repo:** `BANDS` in `erd.py`.

## 6. ERD and ERS
**Short:** ERD is a relative band-power decrease versus a rest baseline (cortical activation); ERS is an increase.
**Deeper:** ERD% = (P(t) − P_base) / P_base × 100, with power averaged over trials. The baseline must be genuine rest. **Bug 2:** the baseline 0–0.5 s was after the cue, so the reference included the evoked response and early imagery; I moved it to −1.5…−0.5 s. In my plot the pre-cue curve now sits near 0, and the early positive peak (about 0.25 s) is cue-evoked, not imagery ERS.
**Repo:** `BASELINE = (-1.5, -0.5)`.

## 7. C3, Cz, C4
**Short:** 10–20 positions over the central scalp: C3 left, C4 right, Cz at the vertex. C3/C4 lie over the hand areas; the foot area is medial, near Cz.
**Deeper:** Hand imagery gives (mostly contralateral) ERD at C3/C4; foot imagery gives it near the vertex. For *bilateral* fists versus feet the contrast is medial versus lateral (anterior-posterior in the task brief), not left versus right. Cz can look weak: the foot source is small and deep, its orientation reduces scalp projection, and neighbouring sources mix in.
**Repo:** `PICKS = ["C3","Cz","C4"]`; my observation in Task 2 (Cz not clearly stronger for feet).

## 8. EOG contamination
**Short:** Eye movements and blinks create large frontal voltages (hundreds of µV) that dominate low frequencies.
**Deeper:** Prefrontal electrodes (Fp/AF) are closest to the eyes. With the target at the top (fists) or bottom (feet) of the screen, gaze direction can correlate with the class, giving a non-neural label signal. To claim it I would need an EOG channel, ICA/regression of EOG, or a spectral analysis. I have none of these, so it is only the leading hypothesis.
**Repo:** `CHANNEL_GROUPS["frontal"]` = Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8.

## 9. Cross-subject generalisation
**Short:** Train on some people and test on unseen people: the realistic setting, and much harder than within-subject.
**Deeper:** Subjects differ in anatomy, impedance, rhythm frequency and amplitude (I measured about a 4× spread in per-subject std). With six training subjects, corrected EEGNet is near chance in my runs.
**Repo:** `subject_split` in `train.py`.

## 10. Data leakage
**Short:** Any information from test data influencing training or model selection.
**Deeper:** Three instances in the starter: (a) trials from the same subject in train and test; (b) normalisation statistics from all data; (c) checkpoint chosen on test accuracy, which is the max of 30 noisy estimates. I fixed each in its own commit.
**Repo:** `subject_split`, `zscore_train_only`, validation-based selection; `test_pipeline.py` asserts the first two.

## 11. Subject-level splitting (why and how)
**Short:** Hold out whole subjects so the test measures generalisation to new people.
**Deeper:** 10 subjects give 2 test, 2 validation and 6 train, chosen by seed. Cost: only 2 test subjects (n = 84 trials), so the number depends heavily on which subjects; a 5-seed mean is a rough estimate. Leave-one-subject-out CV would be stronger.
**Repo:** `train.py`; limitation stated under Task 1.

## 12. EEGNet architecture
**Short:** Temporal conv (frequency filters), then depthwise spatial conv (per-filter electrode weights), then separable conv, then a linear head. It is tiny: 2,898 parameters here.
**Deeper:** Block 1: Conv2d(1, F1=8, (1,80)) learns band-pass filters (80 samples = 0.5 s); BatchNorm; depthwise Conv2d(8, 16, (64,1), groups=8) learns D=2 spatial filters per temporal filter; BN, ELU, AvgPool 4, dropout 0.5. Block 2: depthwise temporal (1,16) plus pointwise 1×1 to mix feature maps; BN, ELU, AvgPool 8, dropout. The priors (temporal then spatial, few parameters) are why it works on small EEG data.
**Repo:** `model.py`; the parameter count is printed by `train.py`.

## 13. Depthwise and separable convolutions
**Short:** Depthwise means one filter per input channel (`groups=in_channels`), with no mixing. Separable means depthwise followed by a 1×1 pointwise conv that mixes channels.
**Deeper:** They cut parameters. In EEGNet the depthwise conv spans all electrodes `(n_ch,1)`, so each temporal filter gets D spatial filters (a CSP-like scalp pattern for that frequency band). The separable block summarises temporal structure cheaply.
**Repo:** `groups=F1` in block 1; `groups=F1*D` in block 2.

## 14. The visual-cue confound
**Short:** Fists were cued at the top of the screen and feet at the bottom, so class is confounded with where the eyes look.
**Deeper:** A classifier could exploit gaze/EOG instead of motor cortex. Test: frontal-only (8 Fp/AF) versus all channels. Result: frontal mean 0.583 versus all 0.543, not distinguishable at 5 seeds, but frontal seed 0 = 0.726 (p < 0.001). Frontal channels should carry almost no motor signal, so above-chance frontal performance is a red flag. It is **not proof** of cheating: frontal EMG, genuine frontal/SMA activity or a lucky test pair are alternatives. Remedies: randomise or counterbalance the cue position, record EOG and regress it out, restrict to sensorimotor channels, and keep frontal-only as a negative control.
**Repo:** `train.py --channels frontal`, `logs/final_frontal_s*.log`.

## Answers to the PDF's Part 1 walkthrough questions

**Q. Walk me through each bug: how did you know it was wrong, and which direction did it move the accuracy?**
- *Short:* Five bugs, each found by reading the code against the stated design and then checking it on data. Directions are in the Task 1 table.
- *Deeper:* (1) wrong runs: the docstring and Part 2 say imagery = 6/10/14, and executed runs are stronger, so this inflates. (2) ERD baseline after the cue: visible as the pre-cue curve sitting at about −10 %; plot only. (3) z-score on all data: leak, tiny here. (4) trial-level split with unused `groups`: leak, inflates in principle. (5) test-selected checkpoint: selection bias, inflates. Single-seed numbers are noisy (one trial = 1.2 points), so I do not claim measured magnitudes from them; see the ablation for multi-seed effects.
- *Repo:* commits `21b21d2`, `d0878d6`, `b3b9f34`, `82066e6`, `64c96b2`.

**Q. Your Task 3 result: convince me the model was or wasn't cheating off the cue.**
- *Short:* I can't convince you either way. Frontal-only is not at chance in some runs, which is suspicious, but I have no EOG evidence.
- *Deeper:* See item 14. What would settle it: EOG regression/ICA, frontal-only with EOG removed, a counterbalanced cue, or frontal low-frequency power versus class.
- *Repo:* `results.jsonl` (10 lines) and Task 3 above.

---

# PART 2 — Build & Defend

## 1. Hypothesis (written before any Part 2 model code; see its commit)

**Hypothesis:** Per-timestep attention has no locality prior and, on 481 tokens (about 232 k attention entries per head per layer), must learn from only six training subjects that neighbouring samples are related and that 7–30 Hz oscillations matter, which EEGNet's temporal conv gets for free. Replacing raw-timestep tokens with a small convolutional stem (local temporal filtering across electrodes) followed by 32-sample (200 ms) learned patch tokens (15 tokens + CLS) should give the attention layers locally pre-processed, oscillation-aware inputs, cut the attention matrix by roughly three orders of magnitude, and so reduce overfitting and seed variance.

**Falsifiable predictions** (stated in advance):
1. YourModel beats the vanilla transformer by a margin larger than seed noise, with lower seed variance.
2. YourModel will **not** necessarily beat the corrected EEGNet: the stem re-implements part of what EEGNet already has, and attention adds capacity that six subjects cannot constrain.
3. If YourModel does not beat the vanilla transformer, the sequence-length / locality story is not the bottleneck at this data scale, and the dominant problem is cross-subject variability, not tokenisation.

**Why this and not another bias:** the task is bilateral (hand area versus medial foot area, so a left–right spatial prior is wrong). The stem's first layer mixes all 64 electrodes with *learned* weights (no hard-coded left–right structure), so it can express medial-versus-lateral patterns. I considered electrode tokens and electrode-geometry biases, but with six subjects the sequence length and missing temporal locality are the most direct, most testable weaknesses. The starter stub already shows patch-only tokenisation, so I keep that as a labelled ablation to show what the stem adds.

**Protocol decisions (deliberate, documented):** the starter harness reports the *peak test accuracy over epochs* (test-set selection, the same defect as Part 1 bug 5). I leave `train.py`/`models.py`/`data.py` untouched and add `protocol.py`: train subjects 1–6, validation subjects 7–8, test subjects 9–10, epoch selected on validation, test evaluated once at that epoch. The starter-style peak test accuracy is logged only as an optimistic reference.

## 2. What I changed and why it targets a real weakness of attention on EEG

**Architecture (`YourModel`, `part2_transformer/main.py`), input (B, 64, 481):**

| stage | operation | output |
|---|---|---|
| stem | Conv1d(64→32, kernel 25 = 156 ms) + BatchNorm + GELU | (B, 32, 481) |
| patch embedding | Conv1d(32→64, kernel = stride = 32), i.e. 200 ms patches | (B, 64, 15) |
| tokens | 15 patch tokens + CLS, learned positional embedding | (B, 16, 64) |
| encoder | 2 pre-LN blocks, 4 heads (d_k = 16), explicit softmax(QKᵀ/√d_k)V, attention dropout 0.1, dropout 0.3 | (B, 16, 64) |
| head | LayerNorm on CLS, Linear → 2 classes | (B, 2) |

185,154 parameters (vanilla 135,746; EEGNet 2,738). CPU only.

**Weakness targeted:** per-timestep self-attention has no locality prior and a long sequence. With 481 samples the score matrix is 482 × 482 = 232,324 entries per head per layer, learned from 270 training trials of 6 subjects. A single sample carries no oscillation information, so attention would have to discover 7–30 Hz structure from pairwise sample relations, which EEGNet's temporal conv provides for free. The stem supplies local, oscillation-aware filters (each output channel is a learned spatial pattern over all 64 electrodes × a 156 ms temporal filter, so no left–right assumption is imposed on this bilateral task). GELU followed by averaging over a 200 ms patch behaves like a rectify-and-average, close to a band-power/ERD feature. Patching then shortens the sequence to 16 tokens (256 score entries, about 900× fewer), so attention only has to model slow, trial-level structure.

**Things I deliberately kept simple:** learned positional embedding (16 positions; timing within the trial matters for ERD); no electrode-geometry prior (not the bet I made); depth 2 to stay compact; the attention block is written out explicitly (instead of `nn.TransformerEncoder`) so that the √d_k scaling is visible and the learned attention can be extracted for the figure. `test_model.py` checks it against `F.scaled_dot_product_attention`. `VanillaEEGTransformer` and `EEGNet` in `models.py` are not edited (`git diff` empty); EEGNet is instantiated with the Part 1 configuration (kernel 80, dropout 0.5).

**Starter stub kept as an ablation:** the original `YourModel` stub (patch tokens only, no stem, 3 layers, `nn.TransformerEncoder`) is preserved as `PatchOnly(starter stub)`, so the effect of the stem is visible separately from patching.

## 3.0 Evaluation protocol (identical for every model)

Subject-level split from `protocol.py`: train subjects 1–6 (270 trials), validation 7–8 (90), test 9–10 (90). The split is deterministic and was checked (`logs/part2_split_check.log`: identical array hashes across builds, disjoint subject sets, z-score statistics reproduced from train trials only). Optimiser and loop follow the starter harness (AdamW, lr 1e-3, weight decay 1e-3, batch 64, 40 epochs); the epoch is selected on validation accuracy; test accuracy is read once at that epoch. The starter harness's "peak test accuracy" is logged as `test_peak` but only as an optimistic reference. Every run's seed, parameters, validation accuracy, selected epoch, final-epoch test accuracy and training time are in `part2_transformer/results.jsonl`. Seeds change the initialisation and batch order only; the split is fixed, so seed variance does not include subject choice (unlike Part 1).

**Chance:** for n = 90, accuracy ≥ 0.600 is needed for p < 0.05 (one-sided binomial). Two test subjects give a noisy estimate; one trial is 1.1 points.

## 3. What happened (all numbers from `part2_transformer/results.jsonl`)

Test accuracy at the validation-selected epoch, subjects 9–10 (n = 90), fixed split, seeds change initialisation and batch order only:

| model | seeds | params | test acc mean | sd | best | worst | test @ final epoch (mean) | val acc (mean) | train s/run |
|---|---|---|---|---|---|---|---|---|---|
| EEGNet | 5 | 2,738 | 0.524 | 0.068 | 0.633 | 0.467 | 0.531 | 0.598 | 70 |
| PatchOnly(starter stub) | 5 | 232,898 | 0.473 | 0.040 | 0.522 | 0.422 | 0.496 | 0.660 | 17 |
| YourModel | 5 | 185,154 | 0.467 | 0.042 | 0.522 | 0.411 | 0.471 | 0.658 | 33 |
| VanillaTransformer | 4 | 135,746 | 0.506 | 0.028 | 0.533 | 0.467 | 0.497 | 0.606 | 1257 |

- EEGNet: s0=0.544, s1=0.500, s2=0.478, s3=0.633, s4=0.467
- PatchOnly(starter stub): s0=0.422, s1=0.489, s2=0.489, s3=0.522, s4=0.444
- YourModel: s0=0.467, s1=0.522, s2=0.411, s3=0.444, s4=0.489
- VanillaTransformer: s0=0.467, s1=0.511, s2=0.511, s3=0.533

chance threshold n=90 (one-sided 5%): 0.600

Starter-style *peak* test accuracy (selects on test; optimistic, shown only to quantify the inflation): EEGNet 0.558, PatchOnly(starter stub) 0.576, YourModel 0.538, VanillaTransformer 0.564.

**Best and worst, no cherry-picking:** EEGNet best 0.633 (seed 3) / worst 0.467 (seed 4); YourModel best 0.522 (seed 1) / worst 0.411 (seed 2); VanillaTransformer best 0.533 (seed 3) / worst 0.467 (seed 0); PatchOnly best 0.522 (seed 3) / worst 0.422 (seed 0).

**Honest reading:**
- **All four models are at or near chance on the two unseen test subjects.** Only one of the 19 runs (EEGNet seed 3, 0.633) clears the 0.600 threshold; no model mean does.
- **YourModel did not improve on the vanilla transformer or EEGNet; it was nominally the lowest.** Mean 0.467 versus vanilla 0.506 (4 seeds) and EEGNet 0.524. YourModel minus vanilla is -0.039 (Welch standard error 0.024, about 1.7 SE): suggestive, not significant with 4–5 seeds. Prediction 1 (YourModel beats vanilla by more than noise, with lower variance) is **not supported**: YourModel sd 0.042 versus vanilla 0.028. Prediction 2 (does not beat EEGNet) held, though nothing is distinguishable from chance. Prediction 3 applies: shrinking the attention matrix did not help, so sequence length and locality were not the binding constraint at this data scale.
- **The stem added nothing measurable:** YourModel 0.467 versus the patch-only stub 0.473.
- **Large validation–test gap:** the transformers reach about 0.61–0.66 validation accuracy (subjects 7–8) but about 0.47–0.51 on test (subjects 9–10). Selecting an epoch on two subjects is noisy, and the gap is what cross-subject variability looks like; it is not evidence of a working decoder. Using the starter's peak-test number would have reported 0.54–0.58, an apparent gain from selection alone.
- **Cost:** the attention matrix drops from 482 × 482 = 232,324 to 16 × 16 = 256 entries per head per layer (about 900×). Mean training time per run: YourModel 33 s, patch-only 17 s, EEGNet 70 s, vanilla about 1,257 s. The vanilla figure is inflated because its runs were executed two at a time on the same CPU, so treat it only as an order of magnitude (minutes versus seconds).

## 4. Figure (`part2_transformer/figure_part2.png`)

(a) Per-seed test accuracy with mean ± sd: all models cluster around chance, with one EEGNet seed above the p < 0.05 line. (b) CLS-token attention of the last layer from the trained YourModel (seed 0) on the test trials, averaged over heads: it is almost uniform (0.058–0.063 versus 1/16 = 0.0625 for uniform) and the fists and feet curves overlap within SEM, so the attention is **not selective in time**; I cannot claim it matches the expected late-window ERD, and any class information in the model must come from the stem/patch features, not from where attention looks. (c) Patching shrinks the attention matrix from 232,324 to 256 entries per head per layer. Attention weights show where information is read from, not that it is used, and temporal attention cannot show spatial focus (C3/C4 versus Cz).

## 5. Why it did not help (hypotheses, not tested)

With 270 training trials from six subjects, the dominant problem is between-subject variability, not sequence length; any of these 135 k–233 k-parameter models can fit the training subjects (validation 0.62–0.66) without transferring. EEGNet's few parameters and strong priors protect it only a little here (0.524). The stem's filters are learned from the same six subjects and can overfit subject-specific spatial patterns. The near-uniform attention suggests the 2-layer encoder adds little beyond averaging the patch features. I did not tune any hyperparameter, so a better-tuned variant could behave differently.

## 6. Limitations and future improvements

- Two test subjects (n = 90; one trial = 1.1 points) and one fixed split: the ranking of the models is not statistically reliable. Leave-one-subject-out CV would be the proper test.
- **Vanilla has 4 seeds** (others 5): seed 4 was not run because of the time limit and the failed parallel attempt.
- Different depths (vanilla 3, YourModel 2, stub 3) and parameter counts; no tuning, no augmentation, 40 epochs, epoch chosen on 2 validation subjects.
- No EOG handling, so the Part 1 cue confound may affect every model here too.
- Future: LOSO CV, subject-wise normalisation, augmentation (noise, time shift, channel dropout), electrode/channel tokens or geometry biases aimed at the medial-vs-lateral contrast, EOG removal, and pre-training on more subjects. With 80 subjects the locality prior would likely matter less (see walkthrough).

## 7. Reproducibility

`cd part2_transformer && python main.py` runs everything sequentially in one process (about 80 minutes; the vanilla transformer dominates) and writes `results.jsonl` and `figure_part2.png`. `python test_model.py` checks the model. Split check: `logs/part2_split_check.log`.
What I actually ran: because of time, the vanilla seeds were run as separate processes (at most two at once) (`--models VanillaTransformer --seeds S --out vanilla_sS`, then `python main.py --report` merges `results_*.jsonl`). An earlier attempt to run five vanilla processes in parallel crashed (native segfaults and allocation errors on a 15.7 GB machine); its logs are in `logs/part2_parallel_attempt_failed/` and produced no results. Vanilla seed 4 was **not run**; vanilla seeds ran two at a time (0 and 1, then 2 and 3). Exact bit-for-bit reproduction across machines is not guaranteed.

---

# LIVE WALKTHROUGH PREPARATION — Part 2 topics

Format per item: **Short**, then **Deeper**, then **Repo**.

## 15. Transformer architecture
**Short:** Tokens are embedded, mixed by self-attention (every token reads every other token), then processed per token by an MLP; residual connections and LayerNorm wrap both. A CLS token is read out for classification.
**Deeper:** Self-attention has no built-in notion of order or locality, so order must be injected (positional embeddings) and any locality must be learned. That is the root of the "no inductive bias" problem.
**Repo:** `_Block` in `part2_transformer/main.py` (pre-LN: LN, attention, residual; LN, MLP, residual); `VanillaEEGTransformer` in `models.py` (per-timestep tokens, untouched).

## 16. Query, Key and Value
**Short:** Each token is projected into a query ("what am I looking for"), a key ("what do I offer") and a value ("what I pass on"). Attention weights come from query–key similarity; the output is the weighted sum of values.
**Deeper:** For tokens X: Q = XW_q, K = XW_k, V = XW_v. Row i of softmax(QKᵀ/√d_k) says how much token i reads from each token j; the output is that mixture of V. With heads, d_model is split into h subspaces (here 64 = 4 × 16), each attending independently.
**Repo:** `self.qkv = nn.Linear(d, 3*d)`, reshaped to `(B, heads, N, d_k)` in `_Block.forward`.

## 17. Scaled dot-product attention, and why divide by √d
**Short:** Attention = softmax(QKᵀ/√d_k)V. The division stops the logits from growing with dimension, which would saturate the softmax.
**Deeper:** If q and k have independent zero-mean unit-variance entries, q·k has variance d_k, so its std is √d_k. For d_k = 16 that is std 4 before scaling; large logits push softmax toward one-hot, so most weights get near-zero gradient and attention is brittle. Dividing by √d_k restores variance ≈ 1. Without scaling it would still train sometimes (LayerNorm helps), but it is less stable, especially with large d_k.
**Repo:** `a = (q @ k.transpose(-2,-1) / math.sqrt(self.dk)).softmax(-1)`; `test_model.py` checks it equals `F.scaled_dot_product_attention`.

## 18. Attention complexity (the PDF asks: sequence length, matrix size, cost)
**Short:** Vanilla: 481 samples + CLS = 482 tokens, so a 482 × 482 matrix per head per layer (232,324 entries). YourModel: 15 patches + CLS = 16 tokens, so 16 × 16 = 256 entries, about 900× fewer.
**Deeper:** Time and memory are O(N² · d) per layer for the score matrix. Vanilla: 4 heads × 3 layers × 232 k per trial. YourModel: 4 heads × 2 layers × 256. This showed up directly in wall-clock time: vanilla needed minutes per training run where YourModel needed seconds (see the results table for measured times).
**Repo:** token counts: `VanillaEEGTransformer.pos` (482) vs `YourModel.n_tokens + 1` (16); the figure's panel (c).

## 19. Temporal patch tokenisation
**Short:** Instead of one token per time sample, a strided conv turns each 200 ms window into one learned token, so the sequence is short and each token already summarises local structure.
**Deeper:** Patch = Conv1d with kernel = stride = 32 (no overlap). It is a learned linear projection of 32 samples × stem channels, so it can implement band-power-like features. Trade-off: temporal resolution is 200 ms, and anything that needs finer timing within a patch must be encoded in the token's features.
**Repo:** `self.patch = nn.Conv1d(d_stem, d_model, patch, stride=patch)`.

## 20. Inductive bias
**Short:** An assumption built into the architecture that lets a model generalise from little data, e.g. convolution assumes local, shift-equivariant structure.
**Deeper:** Bias reduces the hypothesis space, so fewer examples are needed, at the cost of being wrong when the assumption fails. EEGNet bakes in: temporal filters then spatial filters, a few parameters. Vanilla attention bakes in almost nothing. My model adds one: local temporal filtering plus patch tokenisation.
**Repo:** conv stem + patch in `YourModel`.

## 21. EEGNet versus Transformer
**Short:** EEGNet has about 2.7 k parameters and strong EEG priors; the transformers have 135–230 k parameters and weak priors. With six training subjects the small, biased model is expected to be hard to beat.
**Deeper:** Attention is flexible but data-hungry: it must learn locality and frequency structure from data. Cross-subject EEG adds large between-subject shifts, which a flexible model can overfit as subject identity. See the results table for what actually happened here; do not assume the textbook ordering.
**Repo:** `build_models` in `main.py`.

## 22. Why transformers struggle on small datasets
**Short:** More parameters and fewer priors mean they memorise the few training subjects instead of finding shared structure.
**Deeper:** 6 subjects × 45 trials = 270 training examples against 135 k–230 k parameters. In NLP/vision, scale and pre-training substitute for the missing bias. Here there is neither.
**Repo:** `protocol.py` split sizes (270 / 90 / 90).

## 23. Neurophysiological interpretation of the attention figure
**Short:** If attention mattered, one would expect the CLS token to weight the later patches, after the early evoked transient, when sustained mu/beta ERD develops (roughly 0.5–3 s). I report what the figure actually shows and whether it differs between fists and feet.
**Deeper:** Fists vs feet involve lateral hand areas (C3/C4) vs medial foot area (near Cz). Attention in *time* alone cannot show spatial focus; the spatial information lives in the stem's electrode weights, which I did not visualise. Attention weights are also not explanations of the decision by themselves (they show where information is read from, not that it is used). See Part 2 section 4 for the observed pattern (near-uniform attention, no fists/feet difference).
**Repo:** `save_part2_figure`, panel (b).

## Questions from the PDF (Part 2)

**Q. Why does scaled dot-product attention divide by d? What breaks if you don't?** Technically by √d_k: see item 17. Without it, logits have std ≈ √d_k, so softmax saturates, gradients vanish for most pairs and training is less stable.

**Q. What's your transformer's sequence length, and the attention matrix's size and cost?** 16 tokens (15 patches + CLS); 16 × 16 per head per layer; O(N²d) per layer. Vanilla: 482 and 482 × 482. See item 18.

**Q. If I gave you 80 subjects instead of 8, would your Part 2 idea still help? Why or why not?** Probably less. With much more data, attention can learn locality itself, so the prior matters less (it can even hurt by limiting resolution). The remaining benefit would be compute (short sequence), and the conv stem would likely stay useful as an efficient front end. I did not test this; it is an argument from the bias-variance logic, not a result.

**Q. Where is your model attending, and is that neurophysiologically sensible for fists vs feet?** Answer from the figure (Part 2 section 4): attention is almost uniform over time and does not differ between fists and feet, so it is not selectively attending to the late ERD window; the honest caveat is that temporal attention weights do not show spatial focus, and a pattern close to uniform means the model is not using attention selectively.
