"""Build company-quarter observations with complete six-month label windows."""
import argparse

import pandas as pd

from prepare_data import PROCESSED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", required=True, help="Eligible CA employers and metadata")
    parser.add_argument("--links", required=True, help="Reviewed WARN-to-review company links")
    parser.add_argument("--warn-start", required=True, type=pd.Timestamp)
    parser.add_argument("--warn-through", required=True, type=pd.Timestamp)
    parser.add_argument("--min-reviews", type=int, default=5)
    args = parser.parse_args()
    if args.min_reviews < 1 or args.warn_start >= args.warn_through:
        parser.error("Provide a positive review minimum and an increasing WARN coverage interval.")

    companies = pd.read_csv(args.companies, dtype=str, keep_default_na=False)
    links = pd.read_csv(args.links, dtype=str, keep_default_na=False)
    if companies.company_key.duplicated().any() or links.warn_key.duplicated().any():
        parser.error("Company keys and WARN link keys must be unique.")
    reviews = pd.read_csv(PROCESSED / "reviews.csv", parse_dates=["review_date"], dtype={"company_key": str})
    warn = pd.read_csv(PROCESSED / "warn.csv", parse_dates=["notice_date"], dtype={"company_key": str})
    events = warn.merge(links, left_on="company_key", right_on="warn_key", validate="many_to_one")
    events = events[events.notice_date.between(args.warn_start, args.warn_through)]
    quarters = pd.period_range(reviews.review_date.min(), reviews.review_date.max(), freq="Q")
    tables = []
    for quarter in quarters:
        cutoff = (quarter + 1).start_time
        end = (quarter + 3).start_time
        if cutoff < args.warn_start or end - pd.Timedelta(days=1) > args.warn_through:
            continue
        start = (quarter - 1).start_time
        recent = reviews[reviews.review_date.between(start, cutoff, inclusive="left")]
        stats = recent.groupby("company_key", as_index=False).agg(
            review_count=("rating", "size"), mean_rating=("rating", "mean"),
        )
        stats = stats[stats.review_count.ge(args.min_reviews)].merge(companies, on="company_key")
        stats["quarter"] = str(quarter)
        stats["label_end_exclusive"] = end.date().isoformat()
        stats["prior_warn"] = stats.company_key.isin(events.loc[events.notice_date.lt(cutoff), "review_key"]).astype(int)
        future = events[events.notice_date.between(cutoff, end, inclusive="left")]
        stats["target"] = stats.company_key.isin(future.review_key).astype(int)
        tables.append(stats)
    if not tables or all(table.empty for table in tables):
        parser.error("No eligible company-quarters. Check review overlap, WARN coverage and cohort.")
    features = pd.concat(tables, ignore_index=True)
    features.to_csv(PROCESSED / "features.csv", index=False)
    print(features.groupby("workforce_group").agg(rows=("target", "size"), positives=("target", "sum")))


if __name__ == "__main__":
    main()
