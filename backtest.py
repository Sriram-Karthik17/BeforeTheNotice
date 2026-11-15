"""Rolling backtest. For each test quarter t, train only on forecast quarters whose
6-month outcome was fully known by t (t' <= t - 6 months), then rank all employers
and score the top 10 / top 20."""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
P = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else "panel.pkl")

# ---- feature engineering (all from columns known at t) ----
P["recency"] = np.where(P["days_since"].isna(), 0, 1 / (1 + P["days_since"] / 30))  # 1 = filed this month, 0 = never
for c in ["n_3m", "n_6m", "n_12m", "n_prev12m", "n_all", "workers_12m", "max_notice_12m", "closures_12m",
          "counties_12m", "sites_12m", "rev_n3", "rev_vol_share", "n_prior"]:
    P["log_" + c] = np.log1p(P[c])
P["trend"] = P["log_n_12m"] - P["log_n_prev12m"]
P["public"] = P["ret3_rel"].notna().astype(int)
P["ret3_rel_f"] = P["ret3_rel"].fillna(0).clip(-0.6, 0.6)
P["drawdown6_f"] = P["drawdown6"].fillna(0)
P["has_rev"] = P["rev_kw"].notna().astype(int)
P["rev_kw_f"] = P["rev_kw"].fillna(0)
P["rev_neg_f"] = P["rev_neg"].fillna(0)
P["rev_former_f"] = P["rev_former"].fillna(0)
P["rating_drop"] = (P["rating_prior"] - P["rev_rating"]).fillna(0)
P["neg_rise"] = (P["rev_neg"] - P["neg_prior"]).fillna(0)
P["kw_rise"] = (P["rev_kw"] - P["kw_prior"]).fillna(0)

BASE = ["filed_before", "log_n_all", "log_rev_n3"]  # roughly the old baseline: prior WARN + review volume
WARN = ["filed_before", "recency", "log_n_3m", "log_n_6m", "log_n_12m", "log_n_all", "trend", "log_workers_12m",
        "log_max_notice_12m", "log_closures_12m", "closure_share", "log_counties_12m", "log_sites_12m"]
STOCK = ["public", "ret3_rel_f", "drawdown6_f"]
REV = ["log_rev_n3", "log_rev_vol_share", "has_rev", "rev_kw_f", "rev_neg_f", "rev_former_f", "rating_drop",
       "neg_rise", "kw_rise"]

MODELS = {
    "Rule: most recent filers first": None,
    "Old-style baseline (filed before + volume)": BASE,
    "WARN history model": WARN,
    "WARN + stock": WARN + STOCK,
    "WARN + reviews": WARN + REV,
    "WARN + stock + reviews": WARN + STOCK + REV,
}


def rule_score(d):
    # most recent filing first, ties broken by filings in last 12 months; never-filed at bottom
    return d["recency"] * 1000 + d["n_12m"]


def evaluate(test_qs, train_min=None, feats_need=None):
    out = []
    for t in test_qs:
        tr = P[P["t"] <= t - pd.DateOffset(months=6)]
        if train_min is not None:
            tr = tr[tr["t"] >= train_min]
        te = P[P["t"] == t].copy()
        for name, feats in MODELS.items():
            if feats_need and name not in feats_need:
                continue
            if feats is None:
                s = rule_score(te)
            else:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
                mdl.fit(tr[feats], tr["y"])
                s = mdl.predict_proba(te[feats])[:, 1]
            te["s"] = np.asarray(s) + np.random.default_rng(0).uniform(0, 1e-9, len(te))
            r = te.sort_values("s", ascending=False)
            top10, top20 = r.head(10), r.head(20)
            first = te[(te.y == 1) & (te.filed_before == 0)]
            out.append({"t": t.date(), "model": name, "hits10": int(top10.y.sum()), "hits20": int(top20.y.sum()),
                        "pos": int(te.y.sum()), "ap": average_precision_score(te.y, te.s),
                        "first_time_pos": len(first),
                        "first_time_in_top20": int(((top20.y == 1) & (top20.filed_before == 0)).sum())})
    return pd.DataFrame(out)


def summarize(res, label):
    s = res.groupby("model", sort=False).agg(quarters=("t", "nunique"), hits10=("hits10", "sum"),
                                            hits20=("hits20", "sum"), pos=("pos", "sum"), ap=("ap", "mean"),
                                            first_pos=("first_time_pos", "sum"), first_top20=("first_time_in_top20", "sum"))
    s["prec@10"] = (s.hits10 / (10 * s.quarters)).round(3)
    s["prec@20"] = (s.hits20 / (20 * s.quarters)).round(3)
    s["recall@10"] = (s.hits10 / s.pos).round(3)
    s["ap"] = s["ap"].round(3)
    print(f"\n=== {label} ===")
    print(s[["quarters", "hits10", "prec@10", "hits20", "prec@20", "recall@10", "ap", "first_pos", "first_top20"]].to_string())
    print("per-quarter hits@10:")
    print(res.pivot(index="t", columns="model", values="hits10")[list(s.index)].to_string())
    return s


base_rate = P.groupby("t").y.mean()
print("base rate per quarter (share of employers that file within 6 months):", base_rate.round(3).to_dict())

QS = pd.to_datetime(sorted(P["t"].unique()))
# A) WARN-only comparisons: every quarter with at least one fully-labelled training quarter
A = evaluate([t for t in QS if t >= pd.Timestamp("2024-07-01")],
             feats_need=["Rule: most recent filers first", "WARN history model"])
summarize(A, "8 test quarters, 2024Q3-2026Q2: simple rule vs WARN model")
# B) stock: needs 3 months of prices (from 2024-09); train from 2024-10, test 2025-04+
B = evaluate([t for t in QS if t >= pd.Timestamp("2025-04-01")], train_min=pd.Timestamp("2024-10-01"),
             feats_need=["Rule: most recent filers first", "WARN history model", "WARN + stock"])
summarize(B, "5 test quarters, 2025Q2-2026Q2: adding stock")
# C) reviews: data starts 2025-01, 3-month window -> train from 2025-04, test 2025-10+
C = evaluate([t for t in QS if t >= pd.Timestamp("2025-10-01")], train_min=pd.Timestamp("2025-04-01"))
summarize(C, "3 test quarters, 2025Q4-2026Q2: adding reviews")
pd.concat([A.assign(test="A"), B.assign(test="B"), C.assign(test="C")]).to_csv("backtest_results.csv", index=False)
