# Run Instructions — Team hyper-optimists (Tabular Regression)

This document describes how to install the environment and reproduce our exam
dataset predictions (`predictions.npy`). It follows the two-command structure
required by the exam: one command that trains the AutoML system and produces a
model, and a second command that generates predictions for `X_test`.

In our pipeline both steps are performed by a single invocation of `run.py`
(training and prediction happen in one run), so the "two commands" are shown as
(A) the full run that trains and writes predictions, and (B) how to regenerate
predictions deterministically from the same configuration.

---

## 1. Installation

Create and activate an environment, then install the package.

**Conda (what we used):**
```bash
conda create -n automl-tabular-env python=3.10
conda activate automl-tabular-env
pip install -e .
```

**Or venv:**
```bash
python3 -m venv automl-tabular-env
source automl-tabular-env/bin/activate      # Windows: automl-tabular-env\Scripts\activate
pip install -e .
```

Verify the install:
```bash
python -c "import automl"
```

All dependencies are pinned in `pyproject.toml` (scikit-learn 1.7.2,
lightgbm 4.6.0, catboost 1.2.10, smac 2.4.0, ConfigSpace 1.2.2, pandas 2.3.3,
numpy 1.24.3, pyarrow 24.0.0, matplotlib 3.10.9, tabpfn 8.1.0, ...).

Download the datasets (creates `./data/...`):
```bash
python download-datasets.py
```

---

## 2. Command A — train the AutoML system and produce predictions

This single command runs the full six-stage pipeline on the exam dataset's
training split, selects and refits the best model, and writes predictions for
`X_test`:

```bash
python run.py --task exam_dataset --seed 42 --n_iters 100 --cv_folds 3 \
              --backend smac --output-path data/exam_dataset/predictions.npy
```

It produces:
- `data/exam_dataset/predictions.npy` — the predictions submitted for grading
- `data/exam_dataset/predictions.config.json` — the resolved configuration,
  honest hold-out R2, inner-CV R2, wall-clock time, and dataset meta-features,
  so the run is fully reproducible.

The pipeline completes well within the 24-hour budget: on our hardware the
exam run took ~47 minutes (~3.3% of the budget).

### What the command does (stages)
1. 80/20 hold-out split (HPO never sees the validation split until scoring)
2. Preprocessing fitted **inside** each CV fold (no leakage)
3. Meta-feature extraction
4. Algorithm screening — quick-fit 9 model families on 30% of the data, keep top 3
5. HPO with SMAC (Bayesian optimisation) on the survivors
6. Pareto selection, then refit on all training data and predict `X_test`

---

## 3. Command B — regenerate predictions from the same configuration

Training is fast (< 1 h) and fully seeded, so re-running Command A with the same
arguments deterministically reproduces the identical `predictions.npy`:

```bash
python run.py --task exam_dataset --seed 42 --n_iters 100 --cv_folds 3 \
              --backend smac --output-path data/exam_dataset/predictions.npy
```

The exact configuration selected (best model family and its hyperparameters) is
recorded in `data/exam_dataset/predictions.config.json` from Command A, which
documents the trained model used to generate the predictions.

---

## 4. Command-line options

| Flag            | Default | Meaning                                                        |
|-----------------|---------|----------------------------------------------------------------|
| `--task`        | —       | dataset name (`exam_dataset`, `wine_quality`, ...)             |
| `--fold`        | 1       | outer fold (exam dataset has only fold 1)                      |
| `--seed`        | 42      | random seed                                                    |
| `--n_iters`     | 50      | HPO budget (number of configurations)                         |
| `--cv_folds`    | 3       | inner cross-validation folds during HPO                       |
| `--top_k`       | 3       | model families kept by screening for HPO                      |
| `--backend`     | builtin | `builtin`, `smac`, or `bohb` (see below)                      |
| `--prefer`      | accuracy| Pareto preference: `accuracy` or `speed`                      |
| `--output-path` | —       | where `predictions.npy` is written                            |

**Backend meanings (kept distinct on purpose):**
- `builtin` — multi-fidelity random search (Hyperband budget + random proposals); *not* Bayesian
- `smac` — Bayesian optimisation (random-forest surrogate + expected improvement); *not* BOHB
- `bohb` — BOHB (Hyperband budget + Bayesian proposals)

We submit with `--backend smac`: it gave the best average rank across the five
practice datasets and the best honest hold-out on the exam dataset.

**TabPFN:** included automatically if the `tabpfn` package is installed, but it
is dropped during screening on CPU for datasets above 1000 rows (its CPU limit),
so it does not participate in the exam run.

---

## 5. Reproducing the poster ablation (optional)

The five-method ablation across the practice datasets that backs the poster
figures:

```bash
python benchmark.py \
  --datasets bike_sharing_demand brazilian_houses wine_quality superconductivity yprop_4_1 \
  --methods defaults random_search pipeline pipeline_smac pipeline_bohb \
  --seeds 1 2 3 --n_iters 100 --cv_folds 3 \
  --out ablation_full.csv
```

This regenerates `ablation_full.csv`, the data used for every result on the poster.

---

## 6. Result

On the exam dataset, `--backend smac` gave an honest hold-out R2 of 0.926 and an
inner-CV R2 of 0.929; the GitHub Classroom autograder reported a test score of
**R2 = 0.9351**, above the reference baseline of 0.9290.

---

## 7. Submitting predictions (per the repository README)

Predictions are evaluated on the **`test` branch** only. Generate the file on
your working branch, then move just `predictions.npy` to `test`:

```bash
# on main (or your working branch):
git add data/exam_dataset/predictions.npy
git commit -m "Generated predictions for the exam dataset"
git checkout test
git checkout main -- data/exam_dataset/predictions.npy
git status                      # confirm predictions.npy is staged
git commit -m "Exam predictions ready for evaluation"
git push
git pull                        # scores appear under data/exam_dataset/test_out/
```

Only `predictions.npy` (and the pushed evaluation results) may live in
`data/exam_dataset/` on the `test` branch — do not commit the `.parquet` data.
