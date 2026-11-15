"""
make_watchlist.py
-----------------
Trains the main model (same settings as train_model.py: WARN history + review
volume, logistic regression) on every row of training_table.csv, then scores
every company in current_features.csv (today's snapshot) and saves watchlist.csv.

Run after build_training_table.py:
    python build_training_table.py
    python make_watchlist.py

watchlist.csv has one row per company:
    rank, company, risk_level, reason, risk_score, last_notice_date,
    notices_past_year, frontline_share, sector, as_of

Risk levels match what was tested in train_model.py:
    High       = top 10   (about 2 in 3 had a layoff notice within 6 months)
    Elevated   = top 50   (about 4 in 10)
    No signal  = everyone else (NOT the same as safe: first-time layoffs aren't caught)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "training_table.csv")
CURRENT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "current_features.csv")
OUT = os.path.join(HERE, "watchlist.csv")

# Keep these in sync with the MAIN model in train_model.py
LABEL = "y_layoff"
WARN = ["w_prior", "log_w_days_since", "w_n_6", "w_n_12", "w_n_24", "w_days_6", "w_days_12", "w_trend",
        "log_w_workers_12", "log_w_maxworkers_12", "w_sites_12", "w_counties_12", "w_layoffs_12", "w_closures_12",
        "log_w_n_ever"]
FEATURES = WARN + ["log_n6"]
HIGH_TOP, ELEVATED_TOP = 10, 50

train = pd.read_csv(TABLE, parse_dates=["T"])
now = pd.read_csv(CURRENT, parse_dates=["T"])
as_of = now["T"].max().date()

# Only train on dates whose 6-month outcome is fully known by the snapshot date.
train = train[train["T"] + pd.DateOffset(months=6) <= now["T"].max()]
med = train[FEATURES].median()
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
model.fit(train[FEATURES].fillna(med), train[LABEL])
now["risk_score"] = model.predict_proba(now[FEATURES].fillna(med))[:, 1]

now = now.sort_values("risk_score", ascending=False).reset_index(drop=True)
now["rank"] = np.arange(1, len(now) + 1)
now["risk_level"] = np.select([now["rank"] <= HIGH_TOP, now["rank"] <= ELEVATED_TOP], ["High", "Elevated"], "No signal")


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def reason(r):
    if r["w_prior"] == 0:
        return "No WARN layoff notices on record in California since mid-2023"
    days = int(r["w_days_since"])
    when = "today" if days == 0 else f"{plural(days, 'day')} ago"
    n12 = int(r["w_n_12"])
    if n12 == 0:
        return f"Last WARN notice {when}; none in the past year"
    bits = [f"{plural(n12, 'WARN notice')} in the past year", f"most recent {when}"]
    if r["w_workers_12"] > 0:
        bits.append(f"{int(r['w_workers_12']):,} workers affected")
    if r["w_sites_12"] > 1:
        bits.append(f"{int(r['w_sites_12'])} sites")
    return ", ".join(bits)


now["reason"] = now.apply(reason, axis=1)
now["notices_past_year"] = now["w_n_12"].astype(int)
now["frontline_share"] = now["front_share"].round(2)
now["as_of"] = as_of
cols = ["rank", "company", "risk_level", "reason", "risk_score", "last_notice_date", "notices_past_year",
        "frontline_share", "sector", "as_of"]
now["risk_score"] = now["risk_score"].round(3)
now[cols].to_csv(OUT, index=False)

print(f"Trained on {len(train)} rows; scored {len(now)} companies as of {as_of}")
print(now["risk_level"].value_counts().reindex(["High", "Elevated", "No signal"]).to_string())
print("\nTop 20:")
print(now[cols].head(20).drop(columns=["as_of"]).to_string(index=False))
print(f"\nSaved {OUT}")
