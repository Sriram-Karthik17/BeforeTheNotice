# Data decisions and run guide

## Setup and sources

From the repository root with Python 3.12 or 3.13:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- Glassdoor: obtain an authorized export and its README. Record the actual
  source columns and map a derived copy to `company_name,review_date,rating`
  plus optional `pros,cons,job_title,location,industry`. The source schema
  and date range are pending verification.
- California WARN: use the [EDD report and archive](https://edd.ca.gov/en/jobs_and_training/Layoff_Services_WARN/).
  Collect files covering 2025 onward and earlier years that overlap reviews
  and support the prior-notice baseline. Save URL, retrieval date, covered
  interval and SHA-256 for every snapshot. For each source link, run
  `python src/prepare_data.py download-warn URL data/raw/SNAPSHOT_FILENAME`.
  Use a unique snapshot filename for each download.
- For archival PDFs or worksheets with incompatible layouts, create a validated
  CSV in `data/processed` with `company_name,notice_date` and optional
  `workers_affected,city,effective_date,county,address,notice_type`. Record the
  conversion and check row totals against the source. Use the EDD received date
  consistently for labels. For workbooks, inspect the sheet name and zero-based
  header row and pass `--warn-sheet` and `--warn-header` as needed. Convert
  workbooks with different layouts separately.

## Feasibility check

Adjust these example filenames to your snapshots.

```sh
python src/prepare_data.py prepare --reviews data/raw/reviews.csv --warn data/processed/warn_archive.csv data/processed/warn_current.csv
python src/match_companies.py
jupyter lab notebooks/exploration.ipynb
```

1. Record review dates, employer counts and reviews per employer. Check the
   WARN coverage dates against the source publication periods and resolve gaps.
2. Open `match_audit_sample.csv`. Hand-check up to 100 exact/fuzzy candidates
   and 50 unmatched names using employer and location evidence. Record correct,
   incorrect and unresolved counts by method. Check whether unmatched names
   have plausible aliases, and review collisions among normalized names.
3. Create `data/processed/reviewed_links.csv` with `warn_key,review_key` for
   accepted links. Resolve aliases or subsidiaries with documented evidence.
   Record the scope of review and unresolved matches.
4. Create `data/processed/companies.csv` with one row per eligible California
   employer and columns `company_key,sector,size_group,workforce_group`.
   Include eligible employers without a linked notice. Suggested sector values
   are office/tech, retail and warehouse, healthcare, other and unknown.
   Record sources and as-of dates for size and workforce group. Use unknown when
   evidence is missing. Workforce group reflects employer classification.
5. Run the notebook's `eligible_matched` function at representative quarter
   cutoffs to count reviewed matches with enough reviews by workforce group.
   Choose and document a minimum for positive employers and held-out events
   before proceeding. Stop if overlap or label support is inadequate.

## Build and evaluate

Set shell variables `WARN_START` and `WARN_THROUGH` to the first and last
dates of verified continuous WARN coverage, inclusive. Set `TEST_FROM` to
the first held-out observation quarter in YYYYQn form.

```sh
python src/build_features.py --companies data/processed/companies.csv --links data/processed/reviewed_links.csv --warn-start "${WARN_START}" --warn-through "${WARN_THROUGH}"
python src/train.py --test-from "${TEST_FROM}"
python src/evaluate.py
streamlit run dashboard/app.py
```

Each observation uses reviews from the ending quarter and its predecessor
with a default minimum of five reviews. The target is a linked California WARN
notice received in the following two quarters. A zero means no observed linked
notice in the supplied coverage. The prior-notice feature uses records before
the forecast date within the verified coverage. Training excludes targets
whose windows extend beyond the first holdout forecast date. Evaluation reports
average precision and ROC AUC where both classes occur, alongside counts and
prevalence. Coverage and matching uncertainty can differ across workforce groups.

## Run log

| Item | Record before reporting results |
| --- | --- |
| Inputs | Source URLs, snapshot dates, SHA-256, mappings and exclusions |
| Review coverage | Date span, employer count and reviews per employer |
| WARN coverage | Verified dates, gaps and archive conversion checks |
| Match review | Cutoff, sample sizes, correct/incorrect/unresolved counts and corrections |
| Feasibility | Eligible matched employers by workforce group, threshold and go/no-go |
| Model run | Commit, Python and package versions, parameters and holdout quarters |
| Evaluation | Metric file, cohort limits and group sample sizes |

No real data run has been recorded.
