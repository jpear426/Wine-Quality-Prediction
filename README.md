# Predicting Premium Wine from Physicochemical Measurements

CHE1147 Chemical Data Science and Engineering, University of Toronto, Fall 2026. Team 8 final project.

**Team:** Marcus Liu, Xing Qian, Jason Pereira

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<github-user>/che1147-wine-quality/blob/main/notebooks/CHE1147_Team8_Wine_Quality_Final.ipynb)

## Project summary

Wine quality is certified by trained sensory panels, which are slow, expensive and not perfectly consistent. Physicochemical lab tests are cheap and run routinely. This project asks whether eleven routine measurements can flag the wines a panel is most likely to rate as **premium** (quality score of 7 or more), so that panel time goes where it matters most.

**Research question.** Can supervised models using eleven physicochemical measurements identify premium wines, separately for red and white wine? The target is a test PR-AUC of at least 0.60, above the no-skill level and above a logistic regression baseline. We also ask which measurements drive the predictions and whether a model trained on one wine type works on the other.

**Data.** UCI Wine Quality dataset: 1,599 red and 4,898 white Portuguese Vinho Verde wines, each with eleven measurements and a median sensory score from at least three tasters (Cortez et al., 2009). After removing exact duplicates, 5,320 wines remain.

## Main results

Four models (logistic regression, k-NN, random forest and XGBoost) were tuned the same way with Optuna using 5-fold stratified cross-validation on the training set. The model with the best cross-validated PR-AUC was selected for each dataset and scored once on the held-out 20% test set.

| Dataset | Selected model | Test PR-AUC (95% interval) | No skill | Logistic regression | 0.60 target |
|---|---|---|---|---|---|
| Red | Random forest | 0.567 (0.44 to 0.72) | 0.136 | 0.558 | Not met |
| White | Random forest | 0.632 (0.57 to 0.70) | 0.208 | 0.594 | Met |
| Combined | Random forest | 0.644 (0.59 to 0.70) | 0.190 | 0.496 | Met |

* The gain over logistic regression is statistically conclusive (paired bootstrap) only for the combined dataset. For red and white wine the interval of the gain includes zero.
* Alcohol and density are among the three most important measurements for every dataset. Sulphates ranks first for red wine.
* A model trained on one wine type loses about a quarter to a third of its PR-AUC on the other type, so each type needs its own model or a combined model.

Full numbers are in `results/tables`, and the notebook's Section 9 prints the summary.

## Repository structure

```
che1147-wine-quality/
├── README.md                  this file
├── pyproject.toml             environment definition (pip install .)
├── requirements.txt           same dependencies, for pip install -r
├── data/
│   ├── winequality-red.csv    UCI Wine Quality, red wines (semicolon separated)
│   └── winequality-white.csv  UCI Wine Quality, white wines (semicolon separated)
├── notebooks/
│   └── CHE1147_Team8_Wine_Quality_Final.ipynb   full pipeline, run top to bottom
└── results/
    ├── tables/                11 csv tables written by the notebook
    └── figures/               12 figures (png, 300 dpi) written by the notebook
```

## How to run

The whole analysis (data loading, EDA, training, tuning, evaluation and interpretation) is in one notebook. Running it top to bottom regenerates every table and figure in `results/`. A full run takes about 10 minutes, most of it in the hyperparameter optimization of Section 6.

### Option 1: Google Colab

1. Click the **Open in Colab** badge above, or upload `notebooks/CHE1147_Team8_Wine_Quality_Final.ipynb` to Colab.
2. Choose **Runtime > Run all**.

The first code cell installs the two packages Colab does not include (`optuna`, `ucimlrepo`). On Colab the notebook downloads the data from the UCI repository and writes the results to `/content/results`.

### Option 2: Local machine

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<github-user>/che1147-wine-quality.git
cd che1147-wine-quality

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install .                      # or: pip install -r requirements.txt
jupyter notebook notebooks/CHE1147_Team8_Wine_Quality_Final.ipynb
```

Then run all cells. Locally the notebook reads the csv files from `data/` and writes to `results/`. To run it without opening Jupyter:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace CHE1147_Team8_Wine_Quality_Final.ipynb
```

## Pipeline

| Notebook section | Step |
|---|---|
| 0 | Setup, data loading, removal of duplicate wines |
| 1 | Binary target (premium if score is 7 or more), stratified 80/20 train/test split for red, white and combined wine |
| 2 | Exploratory data analysis on the training set: summary statistics, distributions, correlations, outlier check |
| 3 | Log transform of five right-skewed measurements, zero-variance and high-correlation checks |
| 4 | Classification metrics (PR-AUC as the main metric), 5-fold stratified cross-validation, dummy baseline |
| 5 | Four models before tuning: logistic regression, k-NN with forward feature selection, random forest, XGBoost with early stopping |
| 6 | Hyperparameter optimization with Optuna (Bayesian optimization, TPE sampler) on cross-validated PR-AUC |
| 7 | Decision threshold from out-of-fold predictions, one-time test evaluation, bootstrap 95% intervals, ROC and PR curves, confusion matrices |
| 8 | Permutation importance and transfer between wine types |
| 9 | Summary and conclusions |

Design choices that protect the test set:

* The split happens before any analysis, and all EDA uses the training set only.
* Scaling sits inside scikit-learn pipelines, so it is refitted on the training folds during cross-validation.
* Model selection, tuning, feature selection and the decision threshold all use cross-validation on the training set. The test set is scored once at the end.

## Outputs

**Tables** (`results/tables`)

| File | Contents |
|---|---|
| `dataset_overview.csv` | Wines per dataset before and after duplicate removal, premium share |
| `train_test_split.csv` | Size and premium share of the training and test sets |
| `summary_statistics.csv` | Mean, spread, range and skewness of each measurement |
| `skewness_before_after_log.csv` | Skewness of the five transformed measurements |
| `baseline_cv_results.csv` | Cross-validated metrics of every model before tuning |
| `knn_selected_features.csv` | Features chosen for k-NN by forward selection |
| `optuna_tuning_results.csv` | Best settings and cross-validated PR-AUC after tuning |
| `test_results.csv` | Test metrics, 95% intervals and gain over logistic regression |
| `confusion_matrices.csv` | Confusion matrix counts at the chosen thresholds |
| `permutation_importance.csv` | Importance of each measurement for the selected models |
| `transfer.csv` | Test PR-AUC when a model is applied to the other wine type |

**Figures** (`results/figures`)

| File | Contents |
|---|---|
| `eda_quality_distribution.png` | Quality score distribution and the premium cut-off |
| `eda_feature_distributions.png` | Distribution of each measurement, red against white |
| `eda_correlation_heatmaps.png` | Correlation matrices for red and white wine |
| `log_transform_residual_sugar.png` | Effect of the log transform on residual sugar |
| `baseline_diagnostics.png` | k-NN forward selection path and XGBoost learning curves |
| `baseline_cv_pr_auc.png` | Cross-validated PR-AUC of every model before tuning |
| `optuna_history.png` | Optuna optimization history per model and dataset |
| `threshold_selection.png` | Precision, recall and F1 against the decision threshold |
| `roc_and_pr_curves.png` | Test ROC and precision-recall curves |
| `confusion_matrices.png` | Test confusion matrices, baseline against the selected model |
| `permutation_importance.png` | Permutation importance of the measurements |
| `transfer.png` | Performance when a model is applied to the other wine type |

## Reproducibility

Every random step (split, cross-validation folds, model seeds, Optuna sampler, bootstrap) uses `RANDOM_SEED = 42`, so a rerun gives the same numbers on the same package versions. The saved results were produced with:

| Package | Version |
|---|---|
| Python | 3.13 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| optuna | 5.0.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

Other versions, such as the ones preinstalled on Colab, can shift some scores in the third decimal place. The last cell of the notebook prints the versions of the current run.

## Data source and citation

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547 to 553. https://doi.org/10.1016/j.dss.2009.05.016

Dataset: UCI Machine Learning Repository, Wine Quality (id 186), https://doi.org/10.24432/C56S3T, licensed under CC BY 4.0.

## Use of generative AI

Generative AI (Claude, Anthropic) was used to draft the code in this repository. The team reviewed, ran and adapted it to the CHE1147 course material. Details are in the generative AI usage statement submitted with the final paper.
