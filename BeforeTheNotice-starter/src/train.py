"""Fit a logistic baseline on past quarters and score later quarters."""
import argparse

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import PROCESSED

FEATURES = ["prior_warn", "review_count", "mean_rating"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-from", required=True, help="First held-out quarter, such as YYYYQ1")
    args = parser.parse_args()
    test_quarter = pd.Period(args.test_from, freq="Q")
    cutoff = (test_quarter + 1).start_time
    data = pd.read_csv(PROCESSED / "features.csv", parse_dates=["label_end_exclusive"])
    quarters = pd.PeriodIndex(data.quarter, freq="Q")
    train = data[(quarters < test_quarter) & data.label_end_exclusive.le(cutoff)]
    test = data[quarters >= test_quarter].copy()
    if train.target.nunique() != 2 or test.target.nunique() != 2:
        parser.error("Training and holdout each need positive and negative rows after the split.")
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[FEATURES], train.target)
    test["logistic_score"] = model.predict_proba(test[FEATURES])[:, 1]
    test["baseline_score"] = test.prior_warn
    columns = ["company_key", "quarter", "sector", "size_group", "workforce_group",
               "target", "baseline_score", "logistic_score"]
    test[columns].to_csv(PROCESSED / "heldout_scores.csv", index=False)


if __name__ == "__main__":
    main()
