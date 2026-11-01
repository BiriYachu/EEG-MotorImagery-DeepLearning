# EEG × Deep Learning — Motor-Imagery Decoding (Brainwave Science exam)

Two-part research exercise on imagined **both-fists vs both-feet** decoding from EEG.

- **Part 1 — Diagnose & Trust** (`part1_diagnose/`): a pipeline that runs but reports a wrong number. I found and fixed the defects (one commit each), analysed ERD, and tested for a visual-cue confound.
- **Part 2 — Build & Defend** (`part2_transformer/`): inject one inductive bias into an attention model and compare it honestly with EEGNet and a vanilla transformer. Result: a conv-stem + patch-token transformer did **not** beat EEGNet or the vanilla transformer; all models were at or near chance on 2 unseen subjects (see ANSWERS.md).

## Dataset

PhysioNet EEG Motor Movement/Imagery (EEGBCI), subjects 1–10, 64 channels, 160 Hz. Part 1 uses the **imagined** both-fists (T1) vs both-feet (T2) runs **6, 10, 14** (≈ 42 usable trials/subject, 420 total, balanced). MNE downloads and caches it (~80 MB) in `~/mne_data`; it is **not** stored in this repo.

## Installation

```bash
conda env create -f environment.yml && conda activate eeg-dl-exam
# or
pip install -r part1_diagnose/requirements.txt
```
CPU is sufficient. Tested with Python 3.11, MNE 1.13, PyTorch 2.14.
On a first run MNE may ask where to store data; set it once with
`python -c "import mne; mne.set_config('MNE_DATASETS_EEGBCI_PATH', r'<path>')"`.

## Directory structure

```
ANSWERS.md              Part 1 answers (+ Part 2 later) and live-walkthrough preparation
part1_diagnose/         data.py, erd.py, model.py (EEGNet), train.py, test_pipeline.py,
                        results.jsonl, erd_mu.png
part2_transformer/      protocol.py (leak-free split), main.py (YourModel, runner, figure), test_model.py, results.jsonl, figure_part2.png
logs/                   raw logs: original buggy run, per-seed runs, ablation, ERD before-fix plot
```

## Part 1 — commands

```bash
cd part1_diagnose
python train.py --channels all     --seed 0     # seeds 0-4 were run; appends to results.jsonl
python train.py --channels frontal --seed 0
python erd.py --band mu                         # writes erd_mu.png
python test_pipeline.py                         # leakage checks (no subject overlap, train-only z-score)
```

## Reproducibility

- Evaluation: whole **subjects** held out — 2 test, 2 validation, 6 train. The seed selects which subjects, so seed variance includes subject choice. 30 epochs, AdamW (lr 1e-3, wd 1e-3), batch 64. Checkpoint chosen on **validation** accuracy; test evaluated once.
- Normalisation statistics come from training trials only.
- Every result line in `part1_diagnose/results.jsonl` records channels, seed, best epoch, val acc, test acc, n_test. Raw logs are in `logs/`.
- Seeds fix PyTorch/NumPy RNGs; CPU runs should reproduce closely but bitwise identity across machines/versions is not guaranteed.

## Findings (Part 1; details and caveats in ANSWERS.md)

Five defects fixed (wrong runs, ERD baseline, normalisation leakage, trial-level split, test-set checkpoint selection); I could not support a sixth.

| pipeline (n_test = 84/run) | mean | std | min | max |
|---|---|---|---|---|
| original (seed 0 only) | 0.667 | – | – | – |
| corrected, 64 channels, 5 seeds | 0.543 | 0.043 | 0.488 | 0.583 |
| corrected, frontal Fp/AF (8 ch), 5 seeds | 0.583 | 0.089 | 0.500 | 0.726 |

The chance threshold at n = 84 is 0.607, so the trustworthy baseline is near chance. Frontal-only decoding is not worse than all-channel, which is consistent with a non-neural contribution (e.g. eye movement toward the top/bottom cue) but does not prove it. Cumulative ablation over 5 seeds: original 0.724 → +imagery runs 0.645 → +train-only z-score 0.648 → +subject split 0.600 → +validation checkpoint 0.543 (`logs/ablation.jsonl`). See ANSWERS.md for the ERD reading and caveats.
