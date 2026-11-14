"""Generate inspectable employer candidates and a reproducible audit sample."""
import argparse
from difflib import SequenceMatcher

import pandas as pd

from prepare_data import PROCESSED


def candidate(name, keys, cutoff):
    if name in keys:
        return name, 1.0, "exact"
    scores = [(SequenceMatcher(None, name, key, autojunk=False).ratio(), key) for key in keys]
    score, best = max(scores)
    unique = sum(value == score for value, _ in scores) == 1
    return best, score, "fuzzy" if unique and score >= cutoff else "unmatched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.cutoff <= 1:
        parser.error("--cutoff must be in (0, 1]")
    reviews = pd.read_csv(PROCESSED / "reviews.csv", keep_default_na=False)
    warn = pd.read_csv(PROCESSED / "warn.csv", keep_default_na=False)
    keys = set(reviews.company_key)
    rows = [(name, *candidate(name, keys, args.cutoff)) for name in sorted(set(warn.company_key))]
    candidates = pd.DataFrame(rows, columns=["warn_key", "review_key", "similarity", "method"])
    candidates.to_csv(PROCESSED / "match_candidates.csv", index=False)
    matched = candidates[candidates.method.ne("unmatched")]
    unmatched = candidates[candidates.method.eq("unmatched")]
    audit = pd.concat([matched.sample(n=min(100, len(matched)), random_state=42),
                       unmatched.sample(n=min(50, len(unmatched)), random_state=42)])
    audit.assign(decision="", evidence="").to_csv(PROCESSED / "match_audit_sample.csv", index=False)
    print(candidates.method.value_counts().to_string())


if __name__ == "__main__":
    main()
