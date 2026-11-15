# lookout

[readme (1).md](https://github.com/user-attachments/files/34867895/readme.1.md)

# Lookout — BeforeTheNotice

**An early-warning research pipeline for California mass-layoff notices.**

Lookout analyzes historical California WARN notices, employee-review activity, and optional financial signals to rank employers by the likelihood of appearing in a WARN notice over the next six months. It also evaluates whether frontline-worker representation and company characteristics affect model performance.

> **Research prototype:** Risk scores are relative ranking signals, not calibrated probabilities or confirmed predictions. A missing WARN filing does not prove that layoffs did not occur.

## Results

The main model combines **prior WARN filing history** with **recent Glassdoor review volume**. Across seven historical out-of-time test dates (October 2025–April 2026), the committed evaluation reports:

| Metric | Main model |
| --- | ---: |
| ROC-AUC | **0.856** |
| PR-AUC | **0.359** |
| Precision @ 10 | **67.1%** (47 / 70) |
| Recall @ 50 | **44.1%** (134 / 304) |
| Evaluation rows | 7,037 |
| Positive cases | 304 |

These are **historical backtest results**, not verified forward-looking accuracy. The main model outperforms the tested random, largest-employer, and recent-WARN-filer baselines at precision @ 10. Adding review-content and stock-market features did not improve that main metric in the committed results.

## How it works

1. **Ingest data:** Read California EDD WARN notices, Glassdoor reviews, O*NET occupation references, and optional Yahoo Finance price history.
2. **Resolve employer identities:** Normalize company names, fuzzy-match WARN filers to review employers, and apply hand-reviewed corrections.
3. **Classify job titles:** Combine O*NET-based rules, manually labeled examples, and a TF-IDF/logistic-regression classifier to identify frontline roles.
4. **Engineer time-aware features:** Produce employer/month-end snapshots using only information available on or before the forecast date. Eligible employers have at least five California reviews in the preceding six months.
5. **Train and evaluate:** Compare heuristic baselines, logistic regression, and histogram gradient boosting using rolling historical tests with a six-month label-maturation gap.
6. **Generate a watchlist:** Score eligible employers in the available feature snapshot, with ranks, risk bands, and explanatory fields.

The prediction label is whether a *matched* WARN notice occurs within six months after a forecast date. A negative label means no matching notice was found in the available records.

## Data sources

| Source | Purpose |
| --- | --- |
| California Employment Development Department (WARN) | Historical layoff-notice outcomes and company filing history |
| Glassdoor employee reviews | Review volume, ratings, text-derived features, and job-title information |
| O*NET occupation data | Frontline-role classification reference |
| Yahoo Finance historical prices | Experimental stock-related predictors |

Data availability, review-selection bias, entity-matching errors, and source terms can limit downstream use. See the original data providers for provenance, permissions, and licensing.

## Repository structure

```text
lookout/
├── BeforeTheNotice.ipynb       # Analysis, figures, and validation worksheet
├── parse_warn.py              # WARN notice preprocessing
├── build_matches.py           # WARN-to-review employer name matching
├── build_frontline.py         # Frontline job-title identification
├── build_training_table.py    # Time-aware feature and target construction
├── train_model.py             # Baselines, ML models, and evaluation
├── make_watchlist.py          # Latest available employer ranking
├── download_stocks.py         # Historical stock-data download
├── download_requirements.txt  # Package list
├── model_results.csv          # Committed model metrics
├── model_subgroups.csv        # Group-specific model evaluation
├── model_errors.csv           # False positives and missed notices
├── watchlist.csv              # Generated employer watchlist
└── *.csv / *.csv.xz           # Source and intermediate data
```

## Quick start

Requires Python 3.10+ (exact environment and versions have not been pinned).

```bash
git clone https://github.com/Sriram-Karthik17/lookout.git
cd lookout

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r download_requirements.txt
```

To regenerate results using the intermediate datasets included in the repository:

```bash
python build_training_table.py
python train_model.py
python make_watchlist.py
```

These commands may overwrite existing output files. Run them in a fresh clone if you want to preserve the committed snapshots. The complete raw-source pipeline requires additional path/configuration validation; in particular, `build_matches.py` defaults to `matches_fixed.csv` while the training-table pipeline consumes `matches_fix.csv`.

## Website Demo

[Live Website Demo](https://lookout-puce.vercel.app/)


## Google Colab Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sriram-Karthik17/lookout/blob/main/Lookout.ipynb)

### How to Run

1. Click the Open in Colab badge above.
2. Select Runtime → Run all.
3. Explore the model results, visualizations,
   fairness analysis, and predictions.

### Reproducing Results

To regenerate the training data and model results:

1. Set REBUILD = True in the notebook.
2. Set RETRAIN = True in the notebook.
3. Select Runtime → Run all.



The committed notebook defaults to reading generated results. Set `REBUILD = True` or `RETRAIN = True` in the appropriate cells only when you intend to regenerate those files.

To upload your latest Google Drive Colab notebook: in Colab, select **File → Download → Download .ipynb**, then open the repository on GitHub, choose **Add file → Upload files**, upload the notebook as `BeforeTheNotice.ipynb` (replacing the older version), and commit. Alternatively, use Colab's **File → Save a copy in GitHub** if available, selecting this repository and the desired file path.

## Evaluation and limitations

- Seven rolling out-of-time test dates with matured six-month labels help reduce future-label leakage.
- The project records subgroup evaluations across frontline share, filing history, stock-data coverage, and sector.
- Historical WARN records are incomplete proxies for all layoffs; employee reviews are nonrandom; company-name matching can be ambiguous.
- Scores should not be used alone for employment, financial, lending, or hiring decisions.


