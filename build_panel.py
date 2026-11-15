"""Build a company x forecast-quarter panel with features known at the forecast
date and a label = files a CA WARN notice in the next 6 months.

Run from anywhere:  python build_panel.py <repo_dir> <glassdoor_csv> <out_parquet>
"""
import re
import sys

import numpy as np
import pandas as pd

REPO, GD_CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
import build_matches as bm  # noqa: E402  (reuse the team's name normalization)

HORIZON = pd.DateOffset(months=6)
FORECASTS = pd.date_range("2024-01-01", "2026-04-01", freq="QS")
MIN_CA_REVIEWS = 10

LAYOFF_RX = re.compile(
    r"layoff|laid off|lay off|lay-off|restructur|reorg|downsiz|hiring freeze|cut hours|hours cut|"
    r"hours were cut|cutting hours|store clos|closing stores|closure|shut down|shutting down|"
    r"job security|budget cut|cost cut|cost-cut|rif\b|reduction in force|furlough|offshor|outsourc",
    re.I,
)

# ---------------- WARN -> Glassdoor company ----------------
w = pd.read_csv(f"{REPO}/2023-2026WARN_Data.csv", parse_dates=["notice_date", "received_date"])
w = w.dropna(subset=["received_date"])
w["warn_norm"] = w["company"].map(bm.norm_warn)
m = pd.read_csv(f"{REPO}/matches_fix.csv")
key = dict(zip(m["warn_norm"], m["company_norm"]))
w["cn"] = w["warn_norm"].map(key)
patterns = [(p, cn) for _, p, cn in bm.MANUAL_ADDS]
patterns += [(fix[2], fix[0]) for fix in bm.MANUAL_FIXES.values() if fix[2]]
for p, cn in patterns:
    hit = w["company"].str.contains(p, case=False, regex=True) & w["cn"].isna()
    w.loc[hit, "cn"] = cn
w = w.dropna(subset=["cn"])
w["closure"] = w["type"].str.contains("Closure", na=False)
w["n_employees"] = w["n_employees"].fillna(0)
print("WARN rows linked to Glassdoor companies:", len(w), "companies:", w["cn"].nunique())

# ---------------- Glassdoor ----------------
cols = ["review_id", "rating_date", "company_name", "employee_location", "rating_overall",
        "flags_business_outlook", "employee_status", "employee_type", "summary", "pros", "cons",
        "review_pros", "review_cons", "advice_to_management", "review_advice"]
g = pd.read_csv(GD_CSV, usecols=cols, low_memory=False).drop_duplicates("review_id")
g["d"] = pd.to_datetime(g["rating_date"], errors="coerce", utc=True).dt.tz_localize(None)
g = g.dropna(subset=["d", "company_name"])
g["cn"] = g["company_name"].map(bm.norm_glassdoor)
g["ca"] = g["employee_location"].map(bm.is_ca)
text = g[["summary", "pros", "cons", "review_pros", "review_cons", "advice_to_management", "review_advice"]] \
    .fillna("").astype(str).agg(" ".join, axis=1)
g["kw"] = text.str.contains(LAYOFF_RX)
g["neg_outlook"] = g["flags_business_outlook"].eq("NEGATIVE")
g["former"] = g["employee_type"].fillna("").str.contains("Former")
g["rating"] = pd.to_numeric(g["rating_overall"], errors="coerce")

ca_n = g[g["ca"]].groupby("cn").size()
universe = sorted(set(ca_n[ca_n >= MIN_CA_REVIEWS].index))
print("universe (>= %d CA reviews):" % MIN_CA_REVIEWS, len(universe),
      "| with any WARN match:", len(set(universe) & set(w["cn"])))

# ---------------- stocks ----------------
tick = pd.read_csv(f"{REPO}/company_tickers.csv")
tick["cn"] = tick["company"].map(bm.norm_glassdoor)
tmap = dict(zip(tick["cn"], tick["ticker"]))
px = pd.read_csv(f"{REPO}/stock_prices.csv", parse_dates=["date"])
px = px.pivot_table(index="date", columns="ticker", values="close").sort_index()


def price_at(t, ticker):
    s = px[ticker].loc[:t - pd.Timedelta(days=1)].dropna()
    return s


rows = []
for t in FORECASTS:
    past = w[w["received_date"] < t]
    fut = w[(w["received_date"] >= t) & (w["received_date"] < t + HORIZON)]
    pos = set(fut["cn"])
    gp3 = g[(g["d"] < t) & (g["d"] >= t - pd.DateOffset(months=3))]
    gprior = g[g["d"] < t - pd.DateOffset(months=3)]
    tot3 = max(len(gp3), 1)
    gp3s = gp3.groupby("cn").agg(n=("review_id", "size"), kw=("kw", "mean"), neg=("neg_outlook", "mean"),
                                 former=("former", "mean"), rating=("rating", "mean"))
    gpr = gprior.groupby("cn").agg(rating_prior=("rating", "mean"), neg_prior=("neg_outlook", "mean"),
                                   kw_prior=("kw", "mean"), n_prior=("review_id", "size"))
    sp = price_at(t, "^GSPC")
    for cn in universe:
        h = past[past["cn"] == cn]
        r = {"t": t, "cn": cn, "y": int(cn in pos)}
        h12 = h[h["received_date"] >= t - pd.DateOffset(months=12)]
        h24 = h[(h["received_date"] < t - pd.DateOffset(months=12)) & (h["received_date"] >= t - pd.DateOffset(months=24))]
        r["filed_before"] = int(len(h) > 0)
        r["days_since"] = (t - h["received_date"].max()).days if len(h) else np.nan
        r["n_3m"] = int((h["received_date"] >= t - pd.DateOffset(months=3)).sum())
        r["n_6m"] = int((h["received_date"] >= t - pd.DateOffset(months=6)).sum())
        r["n_12m"] = len(h12)
        r["n_prev12m"] = len(h24)
        r["n_all"] = len(h)
        r["workers_12m"] = h12["n_employees"].sum()
        r["max_notice_12m"] = h12["n_employees"].max() if len(h12) else 0
        r["closures_12m"] = int(h12["closure"].sum())
        r["closure_share"] = h["closure"].mean() if len(h) else 0
        r["counties_12m"] = h12["county"].nunique()
        r["sites_12m"] = h12["address"].nunique()
        # reviews (only meaningful from 2025-04 on; data starts 2025-01)
        if cn in gp3s.index:
            s = gp3s.loc[cn]
            r.update(rev_n3=s["n"], rev_vol_share=s["n"] / tot3 * 1000, rev_kw=s["kw"], rev_neg=s["neg"],
                     rev_former=s["former"], rev_rating=s["rating"])
        else:
            r.update(rev_n3=0, rev_vol_share=0, rev_kw=np.nan, rev_neg=np.nan, rev_former=np.nan, rev_rating=np.nan)
        if cn in gpr.index:
            p = gpr.loc[cn]
            r.update(rating_prior=p["rating_prior"], neg_prior=p["neg_prior"], kw_prior=p["kw_prior"], n_prior=p["n_prior"])
        else:
            r.update(rating_prior=np.nan, neg_prior=np.nan, kw_prior=np.nan, n_prior=0)
        # stock
        tk = tmap.get(cn)
        r["ticker"] = tk
        r["ret3_rel"] = r["drawdown6"] = np.nan
        if tk in px.columns and len(sp):
            s = price_at(t, tk)
            if len(s) and s.index[0] <= t - pd.DateOffset(months=3):
                a = s.loc[:t - pd.DateOffset(months=3)].iloc[-1]
                b = sp.loc[:t - pd.DateOffset(months=3)].iloc[-1]
                r["ret3_rel"] = (s.iloc[-1] / a - 1) - (sp.iloc[-1] / b - 1)
                win = s.loc[t - pd.DateOffset(months=6):]
                r["drawdown6"] = s.iloc[-1] / win.max() - 1
        rows.append(r)

panel = pd.DataFrame(rows)
panel.to_pickle(OUT)
print(panel.groupby("t").agg(n=("y", "size"), pos=("y", "sum"), base=("y", "mean"),
                              first_time=("filed_before", lambda s: 0)).to_string())
ft = panel[panel.y == 1].groupby("t")["filed_before"].apply(lambda s: (s == 0).sum())
print("first-time filers among positives per quarter:\n", ft.to_string())
print("public companies in universe:", panel.loc[panel.t == FORECASTS[-1], "ret3_rel"].notna().sum())
