import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import build_matches as bm

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "training_table.csv")
MIN_REVIEWS = 5
DATES = pd.date_range("2025-03-31", "2026-04-30", freq="ME")

g = pd.read_csv(f"{REPO}/glassdoor_reviews.csv.xz", low_memory=False).drop_duplicates("review_id")
g = g[g["employee_location"].map(bm.is_ca)].copy()
g["date"] = pd.to_datetime(g["rating_date"], utc=True).dt.tz_localize(None)
g["key"] = g["company_name"].astype(str).map(bm.norm_glassdoor)
g = g[g["key"] != ""]
fl = pd.read_csv(f"{REPO}/glassdoor_reviews_frontline.csv")[["review_id", "frontline", "sector"]]
g = g.merge(fl, on="review_id", how="left")
text = (g["review_pros"].fillna("") + " " + g["review_cons"].fillna("") + " " + g["summary"].fillna("")).str.lower()
g["kw_layoff"] = text.str.contains(r"layoff|laid off|lay off|lay-off|downsiz|restructur|reorg|job cuts|\brif\b|headcount").astype(float)
g["kw_staff"] = text.str.contains(r"short.?staffed|under.?staffed|hours cut|cut hours|cutting hours|hiring freeze|turnover").astype(float)
g["kw_unstable"] = text.str.contains(r"job security|unstable|uncertain|bankrupt|closing|shut down|shutting|outsourc|offshor").astype(float)
g["out_neg"] = (g["flags_business_outlook"] == "NEGATIVE").astype(float).where(g["flags_business_outlook"].notna())
g["rec_neg"] = (g["flags_recommend_frend"] == "NEGATIVE").astype(float).where(g["flags_recommend_frend"].notna())
g["ceo_neg"] = (g["flags_ceo_approval"] == "DISAPPROVE").astype(float).where(g["flags_ceo_approval"].notna())
g["former"] = g["employee_type"].fillna("").str.contains("Former").astype(float)
g["front"] = (g["frontline"] == "yes").astype(float).where(g["frontline"].isin(["yes", "no"]))
g["part_time"] = (g["employee_status"] == "PART_TIME").astype(float)
name_of = g.groupby("key")["company_name"].agg(lambda s: s.value_counts().index[0])

w = pd.read_csv(f"{REPO}/2023-2026WARN_Data.csv", low_memory=False, parse_dates=["notice_date", "received_date"])
w = w.drop_duplicates(subset=[c for c in bm.WARN_DEDUPE_COLS if c in w.columns]).copy()
w["warn_norm"] = w["company"].map(bm.norm_warn)
mf = pd.read_csv(f"{REPO}/matches_fix.csv")
w["key"] = w["warn_norm"].map(dict(zip(mf["warn_norm"], mf["company_norm"])))
for wn, (cn, _, pattern) in bm.MANUAL_FIXES.items():
    if pattern:
        w.loc[w["company"].str.contains(pattern, case=False, regex=True) & w["key"].isna(), "key"] = cn
for _, pattern, cn in bm.MANUAL_ADDS:
    w.loc[w["company"].str.contains(pattern, case=False, regex=True), "key"] = cn
w = w.dropna(subset=["key"])
w["date"] = w["received_date"].fillna(w["notice_date"])
w["is_layoff"] = w["type"].fillna("").str.startswith("Layoff")
w["is_closure"] = w["type"].fillna("").str.startswith("Closure")
w["site"] = w["address"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True)

tk = pd.read_csv(f"{REPO}/company_tickers.csv")
tk["key"] = tk["company"].map(bm.norm_glassdoor)
ticker_of = dict(zip(tk["key"], tk["ticker"]))
px = pd.read_csv(f"{REPO}/stock_prices.csv", parse_dates=["date"]).pivot(index="date", columns="ticker", values="close").sort_index().ffill()
rel = px.div(px["^GSPC"], axis=0)


def months(t, n):
    return t - pd.DateOffset(months=n)


def stock_feats(t):
    h = rel[rel.index <= t]
    raw = px[px.index <= t]
    now = h.iloc[-1]
    out = pd.DataFrame({"st_ret3": now / h[h.index <= months(t, 3)].iloc[-1] - 1, "st_ret6": now / h[h.index <= months(t, 6)].iloc[-1] - 1, "st_draw": raw.iloc[-1] / raw[raw.index > months(t, 12)].max() - 1, "st_vol": raw[raw.index > months(t, 3)].pct_change().std()})
    return out


def warn_feats(t):
    h = w[w["date"] <= t]
    f = pd.DataFrame(index=h["key"].unique())
    f["w_prior"] = 1.0
    f["w_days_since"] = (t - h.groupby("key")["date"].max()).dt.days
    f["w_n_ever"] = h.groupby("key").size()
    for m in (6, 12, 24):
        hm = h[h["date"] > months(t, m)]
        f[f"w_n_{m}"] = hm.groupby("key").size()
        if m != 24:
            f[f"w_days_{m}"] = hm.groupby("key")["date"].nunique()
    h12 = h[h["date"] > months(t, 12)]
    f["w_workers_12"] = h12.groupby("key")["n_employees"].sum()
    f["w_maxworkers_12"] = h12.groupby("key")["n_employees"].max()
    f["w_sites_12"] = h12.groupby("key")["site"].nunique()
    f["w_counties_12"] = h12.groupby("key")["county"].nunique()
    f["w_layoffs_12"] = h12.groupby("key")["is_layoff"].sum()
    f["w_closures_12"] = h12.groupby("key")["is_closure"].sum()
    f = f.fillna(0)
    f["w_trend"] = f["w_days_6"] - (f["w_days_12"] - f["w_days_6"])
    return f


def review_feats(t):
    r6 = g[(g["date"] <= t) & (g["date"] > months(t, 6))]
    r3 = r6[r6["date"] > months(t, 3)]
    p3 = r6[r6["date"] <= months(t, 3)]
    agg = {"n6": ("review_id", "size"), "r_rating": ("rating_overall", "mean"), "r_lead": ("rating_senior_leadership", "mean"), "r_wl": ("rating_work_life", "mean"), "r_culture": ("rating_culture_values", "mean"), "r_career": ("rating_career_opportunities", "mean"), "r_comp": ("rating_compensation_benefits", "mean"), "r_out_neg": ("out_neg", "mean"), "r_rec_neg": ("rec_neg", "mean"), "r_ceo_neg": ("ceo_neg", "mean"), "r_kw_layoff": ("kw_layoff", "mean"), "r_kw_staff": ("kw_staff", "mean"), "r_kw_unstable": ("kw_unstable", "mean"), "r_former": ("former", "mean"), "front_share": ("front", "mean"), "r_part_time": ("part_time", "mean"), "r_low_share": ("rating_overall", lambda s: (s <= 2).mean())}
    f = r6.groupby("key").agg(**agg)
    f["n3"] = r3.groupby("key").size()
    f["np3"] = p3.groupby("key").size()
    f[["n3", "np3"]] = f[["n3", "np3"]].fillna(0)
    f["r_vol_trend"] = np.log((f["n3"] + 1) / (f["np3"] + 1))
    f["r_d_rating"] = r3.groupby("key")["rating_overall"].mean() - p3.groupby("key")["rating_overall"].mean()
    f["r_d_out_neg"] = r3.groupby("key")["out_neg"].mean() - p3.groupby("key")["out_neg"].mean()
    f["r_kw_layoff3"] = r3.groupby("key")["kw_layoff"].mean()
    f["n_to_date"] = g[g["date"] <= t].groupby("key").size()
    f["sector"] = r6[r6["sector"].notna() & (r6["sector"] != "unknown")].groupby("key")["sector"].agg(lambda s: s.value_counts().index[0])
    return f[f["n6"] >= MIN_REVIEWS]


def features_at(t, with_label=True):
    f = review_feats(t)
    f = f.join(warn_feats(t), how="left")
    wcols = [c for c in f.columns if c.startswith("w_")]
    f[wcols] = f[wcols].fillna(0)
    f.loc[f["w_prior"] == 0, "w_days_since"] = 2000
    sf = stock_feats(t)
    tick = pd.Series(f.index.map(ticker_of), index=f.index)
    for c in sf.columns:
        f[c] = tick.map(sf[c])
    f["st_has"] = f["st_ret6"].notna().astype(float)
    if with_label:
        fut = w[(w["date"] > t) & (w["date"] <= t + pd.DateOffset(months=6))]
        f["y_any"] = f.index.isin(fut["key"]).astype(int)
        f["y_layoff"] = f.index.isin(fut.loc[fut["is_layoff"], "key"]).astype(int)
        f["workers_next6"] = fut.groupby("key")["n_employees"].sum()
    f["T"] = t
    return f.reset_index(names="key")


def finish(tab):
    tab["company"] = tab["key"].map(name_of)
    tab["log_n6"] = np.log(tab["n6"])
    tab["log_n_to_date"] = np.log(tab["n_to_date"])
    for c in ("w_workers_12", "w_maxworkers_12", "w_n_ever", "w_days_since"):
        tab["log_" + c] = np.log1p(tab[c])
    return tab


tab = finish(pd.concat([features_at(t) for t in DATES], ignore_index=True))
tab["workers_next6"] = tab["workers_next6"].fillna(0)
tab.to_csv(OUT, index=False)
print("WARN notices linked:", len(w), "| companies:", w["key"].nunique())
print(tab.groupby("T").agg(eligible=("key", "size"), any_notice=("y_any", "sum"), layoff_notice=("y_layoff", "sum"), prior=("w_prior", "sum"), with_stock=("st_has", "sum")).to_string())

# Current snapshot for the app: same inputs as of the latest date covered by both WARN and reviews.
# No answer column, because the next 6 months haven't happened yet.
warn_through = pd.read_csv(f"{REPO}/2023-2026WARN_Data.csv", usecols=["received_date"], parse_dates=["received_date"])["received_date"].max()
NOW = min(warn_through, g["date"].max()).normalize()
now = finish(features_at(NOW, with_label=False))
last_notice = w[w["date"] <= NOW].groupby("key")["date"].max()
now["last_notice_date"] = now["key"].map(last_notice).dt.date
CURRENT_OUT = os.path.join(os.path.dirname(os.path.abspath(OUT)), "current_features.csv")
now.to_csv(CURRENT_OUT, index=False)
print(f"current snapshot as of {NOW.date()}: {len(now)} companies -> {CURRENT_OUT}")
