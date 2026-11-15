# lookout

Lookout — BeforeTheNotice
An early-warning research system for potential mass layoffs at California employers.

Lookout ranks employers by their estimated likelihood of receiving a California WARN layoff notice in the next six months. It combines historical WARN notices with California Glassdoor employee-review activity. The project also examines frontline-worker representation and differences in performance across employer groups.

Important: A model score is a historical ranking signal, not proof that a company will lay off workers. A company without a matching WARN notice is not necessarily free of layoffs.
Project and data
Repository: https://github.com/Sriram-Karthik17/lookout

Data included in this repository:
Input	File(s)	Purpose
California WARN notices	2023-2026WARN_Data.csv	Historical notices and future-window labels
Employee reviews	glassdoor_reviews.csv.xz	Review activity, ratings, sentiment-related indicators
Occupation and job-title reference data	occupation_data.csv, job_titles.csv, job_zones.csv, sample_of_reported_titles.csv, onet_frontline_reference.csv	Frontline classification
Hand-reviewed labels	frontline_labels_top500.csv, vague_title_company_pairs.csv	Label checks and overrides
Company matches	matches_fix.csv	Links WARN employer names to review-company names
Stock data	company_tickers.csv, stock_prices.csv	Optional additional model features


See the original providers and applicable licenses/terms for reuse restrictions. The supplied repository does not currently document full provenance and licenses for every dataset; add those before final submission.
Prediction task
For each eligible employer and forecast date T, the target y_layoff is 1 if a matched WARN layoff notice appears after T and no later than six months afterward, otherwise 0 within the available records. build_training_table.py creates features using information dated at or before T, including prior WARN history, recent review counts and content indicators, and stock-derived features where available. The model ranks employers rather than asserting a calibrated probability of layoffs.
Models and baselines
train_model.py compares:
- Random ranking, most-recent WARN filer, and largest-company baselines
- Logistic regression using WARN history and review volume (main model)
- Logistic regression with additional review-content and stock features
- Histogram gradient boosting with all features or review-only features
The separate build_frontline.py pipeline trains a TF-IDF plus logistic-regression job-title classifier, using O*NET-derived labels and a held-out occupation-group split. Frontline classification also uses manual labels, lookup rules and keyword overrides.
Held-out evaluation
The main model uses seven historical test dates (October 2025–April 2026). For each test date, training rows are restricted to forecast dates whose six-month outcome window ends by the test date. The reported metrics are averages or aggregates from this rolling evaluation, not prospective performance.

Metrics reported in the committed model_results.csv:
Main model metric	Reported value
ROC-AUC	0.856
PR-AUC	0.359
Top-10 precision	0.671
Top-50 recall	0.441


These are previously generated results; the commands below have not been independently verified in a clean environment as part of this README.
Subgroups, errors, and limitations
train_model.py exports model_subgroups.csv, breaking down performance by frontline share, prior WARN history, stock-data availability and sector. It also exports model_errors.csv with top-ranked false alarms and missed layoffs. In the committed results, the top-50 watchlist misses all 82 observed first-time-filer positive cases; this is a major limitation, not a minor edge case. Other limitations include incomplete WARN coverage, uncertain company matching, selective employee reviews, possible label-date ambiguity, and historical changes in reporting.
Reproduce the reported model evaluation
Prerequisites
- Python 3.10+ is a reasonable starting point; the exact tested Python and dependency versions still need to be recorded.
- Install dependencies in an isolated environment.
- Run commands from the repository root.
git clone https://github.com/Sriram-Karthik17/lookout.git
cd lookout
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Action required: Add requirements.txt to the repository and pin the versions from a successful clean run. The current repository does not contain that file. Core libraries used in the reviewed scripts include numpy, pandas, scikit-learn, rapidfuzz, yfinance, and pdfplumber; the notebook also uses matplotlib.
Rebuild from committed intermediate datasets
The repository currently includes matches_fix.csv, glassdoor_reviews_frontline.csv, stock_prices.csv, and other intermediate inputs consumed by build_training_table.py. With dependencies installed:
python build_training_table.py
python train_model.py
python make_watchlist.py
These commands are the intermediate-data-to-results path, not yet a verified original-raw-data-to-results path.

Expected generated artifacts:
File	Description
training_table.csv	Employer/date features and labels
current_features.csv	Latest available feature snapshot
model_results.csv	Baselines and model performance
model_subgroups.csv	Group-level performance
model_errors.csv	False alarms and missed layoffs
model_predictions.csv	Historical model rankings
watchlist.csv	Latest watchlist


Compare the regenerated model_results.csv with the metrics above. Rebuilding files will overwrite committed outputs; consider using a fresh clone.
Full raw-data pipeline — remaining integration work
The project contains the constituent preprocessing scripts, but the complete execution order and required CLI arguments have not yet been verified together:
1. parse_warn.py — prepare source WARN notices if starting from provider documents.
2. build_matches.py — match employer names across WARN and Glassdoor.
3. build_frontline.py — generate frontline labels and review classifications.
4. download_stocks.py — obtain stock history; note that downloading current data may change results.
5. build_training_table.py — assemble dated features and labels.
6. train_model.py — fit baselines/models and export evaluation.
7. make_watchlist.py — generate the latest ranking.
Known integration issue: build_matches.py defaults to writing matches_fixed.csv, while build_training_table.py expects matches_fix.csv. Make those paths consistent before using the full raw-data path. build_frontline.py requires six positional arguments (ONET_DIR, GLASSDOOR, TOP500, PAIRS, REFERENCE, OUT_DIR). Document their tested values and the source-file requirements before claiming a one-command raw-data rebuild. Do not assume downloaded stock history will reproduce the committed snapshot.
Data decisions and manual validation
- Glassdoor company names and WARN employer names are normalized and matched using exact, fuzzy and manually corrected mappings.
- WARN records are deduplicated; employer/date features are constructed from data available up to each forecast date.
- Job titles are mapped to frontline/non-frontline categories using O*NET occupation information, a trained classifier, and manual/rule-based overrides.
- Reviews are filtered to California locations; the training table requires at least five reviews for an employer in the six-month review window.
- Positive targets use matched future WARN layoff notices; negative targets mean no matching notice in the available records for that window.
Notebook section 3.1 displays an eight-case validation worksheet (four positive and four negative cases) and candidate WARN records. The positive cases have supporting notice records; negative cases are provisional absence-of-record checks. The worksheet is a useful inspection aid, not a substitute for a documented independent human review. In particular, broad name searches can surface unrelated WARN employers; verify legal-entity identity and dates before treating a candidate as evidence.
Notebook
Open BeforeTheNotice.ipynb to review the methodology, charts, subgroup analyses, errors, and validation worksheet. Its setup cell must clone this repository (Sriram-Karthik17/lookout), not a differently named repository. Some notebook cells intentionally load committed results (REBUILD = False, RETRAIN = False); set these flags to True only when the underlying inputs and dependencies are ready.
Advanced Track checklist
Requirement	Evidence
Train a predictive or NLP model	train_model.py, build_frontline.py
Compare with simple baselines	train_model.py, model_results.csv
Held-out evaluation	Rolling time split in train_model.py, reported AUC/PR-AUC/top-k metrics
Error and fairness analysis	model_subgroups.csv, model_errors.csv
Explain data decisions and inspect samples	Preprocessing scripts and notebook section 3.1; independent manual confirmation recommended
Reproducibility	Code and inputs committed; clean-run verification and complete raw-data commands still required


Responsible use
Lookout is a research prototype for workforce-risk awareness. Predictions should not be used as definitive employment, investment, credit, or hiring decisions. Employee reviews and WARN notices are incomplete proxies, and subgroup performance varies materially. Consult official notices and independent evidence before drawing conclusions.
