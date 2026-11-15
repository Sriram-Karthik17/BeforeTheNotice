import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_matches as bm

LABEL = "y_layoff"
MAIN = "model: WARN history + review volume"


def path(name):
    return os.path.join(HERE, name)


watch = pd.read_csv(path("watchlist.csv"), parse_dates=["last_notice_date", "as_of"])
current = pd.read_csv(path("current_features.csv")) if os.path.exists(path("current_features.csv")) else pd.DataFrame({"company": [], "n6": []})
table = pd.read_csv(path("training_table.csv"), parse_dates=["T"])
preds = pd.read_csv(path("model_predictions.csv"), parse_dates=["T"])
results = pd.read_csv(path("model_results.csv"))
groups = pd.read_csv(path("model_subgroups.csv"))
warn = pd.read_csv(path("2023-2026WARN_Data.csv"), low_memory=False, parse_dates=["notice_date", "received_date"])
matches = pd.read_csv(path("matches_fix.csv"))
tickers = pd.read_csv(path("company_tickers.csv"))

as_of = watch["as_of"].iloc[0]
preds = preds.merge(table[["T", "company", "w_days_since", "w_n_12", "n6", "workers_next6"]], on=["T", "company"], how="left")
watch = watch.merge(current[["company", "n6"]], on="company", how="left")

names = sorted(set(preds["company"]) | set(watch["company"]))
cid = {n: i for i, n in enumerate(names)}

info = pd.concat([table.sort_values("T").groupby("company").tail(1)[["company", "sector", "front_share"]], watch[["company", "sector", "frontline_share"]].rename(columns={"frontline_share": "front_share"})]).drop_duplicates("company", keep="last").set_index("company")
ticker_of = dict(zip(tickers["company"], tickers["ticker"]))
industry_of = dict(pd.read_csv(path("company_industry.csv")).values) if os.path.exists(path("company_industry.csv")) else {}
companies = []
for n in names:
    sector = info.at[n, "sector"] if n in info.index else None
    share = info.at[n, "front_share"] if n in info.index else None
    companies.append({"n": n, "s": None if pd.isna(sector) else str(sector).replace("_", " "), "f": None if pd.isna(share) else round(float(share), 2), "t": ticker_of.get(n), "i": industry_of.get(n)})

dates = []
for t, d in preds.groupby("T"):
    rows = []
    for r in d.sort_values("rank").itertuples():
        days = None if r.w_days_since >= 2000 else int(r.w_days_since)
        rows.append([cid[r.company], round(float(r.score), 3), int(getattr(r, LABEL)), days, int(r.w_n_12), int(r.n6), int(r.workers_next6)])
    dates.append({"d": t.strftime("%Y-%m-%d"), "live": False, "rows": rows})
rows = []
for r in watch.sort_values("rank").itertuples():
    days = None if pd.isna(r.last_notice_date) else int((as_of - r.last_notice_date).days)
    rows.append([cid[r.company], round(float(r.risk_score), 3), None, days, int(r.notices_past_year), None if pd.isna(r.n6) else int(r.n6), None, r.reason])
dates.append({"d": as_of.strftime("%Y-%m-%d"), "live": True, "rows": rows})

warn = warn.drop_duplicates(subset=[c for c in bm.WARN_DEDUPE_COLS if c in warn.columns]).copy()
warn["key"] = warn["company"].map(bm.norm_warn).map(dict(zip(matches["warn_norm"], matches["company_norm"])))
for _, (cn, _, pattern) in bm.MANUAL_FIXES.items():
    if pattern:
        warn.loc[warn["company"].str.contains(pattern, case=False, regex=True) & warn["key"].isna(), "key"] = cn
for _, pattern, cn in bm.MANUAL_ADDS:
    warn.loc[warn["company"].str.contains(pattern, case=False, regex=True), "key"] = cn
key_to_name = {bm.norm_glassdoor(n): n for n in names}
warn["name"] = warn["key"].map(key_to_name)
linked = warn.dropna(subset=["name"]).copy()
linked["date"] = linked["received_date"].fillna(linked["notice_date"])
filings = {}
for r in linked.sort_values("date", ascending=False).itertuples():
    place = r.county.replace(" County", "").strip() if isinstance(r.county, str) else ""
    kind = "closure" if str(r.type).startswith("Closure") else "layoff"
    filings.setdefault(cid[r.name], []).append([r.date.strftime("%Y-%m-%d"), int(r.n_employees), str(place).title() if isinstance(place, str) else "", kind])

end = warn["notice_date"].max()
recent = warn[warn["notice_date"] > end - pd.DateOffset(months=12)]
county = recent.groupby(recent["county"].str.replace(" County", "", regex=False).str.strip().str.title()).agg(workers=("n_employees", "sum"), notices=("n_employees", "size")).sort_values("workers", ascending=False)

main = results[results["model"] == MAIN].iloc[0]
data = {
    "meta": {"data_through": end.strftime("%Y-%m-%d"), "as_of": as_of.strftime("%Y-%m-%d"), "notices": int(len(warn)), "reviews": int(len(pd.read_csv(path("glassdoor_reviews_frontline.csv"), usecols=["review_id"]))), "rows": int(main["rows"]), "cases": int(main["layoff_cases"]), "auc": float(main["auc"]), "workers_12m": int(recent["n_employees"].sum()), "notices_12m": int(len(recent)), "main": MAIN},
    "companies": companies,
    "dates": dates,
    "filings": filings,
    "counties": [[k, int(v.workers), int(v.notices)] for k, v in county.head(12).iterrows()],
    "results": results[["model", "top10_right", "top10_precision", "top20_precision", "top50_precision", "top10_recall", "top20_recall", "top50_recall", "auc"]].to_dict("records"),
    "groups": groups.replace({np.nan: None}).to_dict("records"),
}
template = open(path("site_template.html"), encoding="utf-8").read()
html = template.replace("/*__LOOKOUT_DATA__*/null", json.dumps(data, separators=(",", ":")))
open(path("index.html"), "w", encoding="utf-8").write(html)
print("as of", as_of.date(), "| employers on the current watchlist:", len(watch), "| employers in the site:", len(names))
print("past forecast dates:", len(dates) - 1, "| filings linked:", sum(len(v) for v in filings.values()), "| page size (MB):", round(len(html) / 1e6, 2))
