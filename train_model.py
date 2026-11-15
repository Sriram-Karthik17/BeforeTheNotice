import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "training_table.csv")
LABEL = "y_layoff"
TEST_DATES = ["2025-10-31", "2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"]
MAIN = "model: WARN history + review volume"

WARN = ["w_prior", "log_w_days_since", "w_n_6", "w_n_12", "w_n_24", "w_days_6", "w_days_12", "w_trend", "log_w_workers_12", "log_w_maxworkers_12", "w_sites_12", "w_counties_12", "w_layoffs_12", "w_closures_12", "log_w_n_ever"]
REVIEWS = ["r_rating", "r_lead", "r_wl", "r_culture", "r_career", "r_comp", "r_out_neg", "r_rec_neg", "r_ceo_neg", "r_kw_layoff", "r_kw_staff", "r_kw_unstable", "r_former", "front_share", "r_part_time", "r_low_share", "r_vol_trend", "r_d_rating", "r_d_out_neg", "r_kw_layoff3"]
STOCK = ["st_has", "st_ret3", "st_ret6", "st_draw", "st_vol"]

tab = pd.read_csv(TABLE, parse_dates=["T"])


def logistic(cols):
    def fit(tr, te):
        med = tr[cols].median()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
        return m.fit(tr[cols].fillna(med), tr[LABEL]).predict_proba(te[cols].fillna(med))[:, 1]
    return fit


def boosted(cols):
    def fit(tr, te):
        out = []
        for seed in range(5):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
            out.append(m.fit(tr[cols], tr[LABEL]).predict_proba(te[cols])[:, 1])
        return np.mean(out, axis=0)
    return fit


MODELS = {
    "baseline: random guess": lambda tr, te: np.random.RandomState(0).rand(len(te)),
    "baseline: most recent WARN filer first": lambda tr, te: -te["w_days_since"].values + te["w_n_12"].values * 1e-3,
    "baseline: biggest company first": lambda tr, te: te["n_to_date"].values,
    MAIN: logistic(WARN + ["log_n6"]),
    "model: + review content": logistic(WARN + ["log_n6"] + REVIEWS),
    "model: + review content + stock": logistic(WARN + ["log_n6"] + REVIEWS + STOCK),
    "boosted trees: everything": boosted(WARN + ["log_n6", "log_n_to_date"] + REVIEWS + STOCK),
    "boosted trees: reviews only": boosted(REVIEWS),
}

keep = ["key", "company", "T", LABEL, "w_prior", "w_days_since", "w_n_12", "front_share", "st_has", "sector", "n6", "workers_next6"]
preds = []
for t in pd.to_datetime(TEST_DATES):
    te = tab[tab["T"] == t]
    tr = tab[tab["T"] + pd.DateOffset(months=6) <= t]
    print(f"forecast date {t.date()}: trained on {len(tr)} earlier rows, scored {len(te)} companies, {int(te[LABEL].sum())} had a layoff notice")
    for name, fit in MODELS.items():
        d = te[keep].copy()
        d["model"], d["score"] = name, fit(tr, te)
        preds.append(d)
P = pd.concat(preds, ignore_index=True)
P["rank"] = P.groupby(["model", "T"])["score"].rank(ascending=False, method="first").astype(int)


def per_date(d, fn):
    return float(np.mean([fn(x[LABEL], x["score"]) for _, x in d.groupby("T") if x[LABEL].nunique() > 1]))


def summarize(d):
    r = {"rows": len(d), "layoff_cases": int(d[LABEL].sum())}
    for k in (10, 20, 50):
        h = d[d["rank"] <= k]
        r[f"top{k}_right"] = f"{int(h[LABEL].sum())} of {len(h)}"
        r[f"top{k}_precision"] = round(h[LABEL].mean(), 3) if len(h) else np.nan
        r[f"top{k}_recall"] = round(h[LABEL].sum() / max(d[LABEL].sum(), 1), 3)
    r["auc"] = round(per_date(d, roc_auc_score), 3)
    r["pr_auc"] = round(per_date(d, average_precision_score), 3)
    return r


results = pd.DataFrame([{"model": name, **summarize(d)} for name, d in P.groupby("model", sort=False)])
results.to_csv(os.path.join(HERE, "model_results.csv"), index=False)

main = P[P["model"] == MAIN].copy()
main["group_frontline"] = np.where(main["front_share"] >= 0.5, "frontline-heavy", "office-heavy")
main["group_history"] = np.where(main["w_prior"] == 1, "filed before", "first-time")
main["group_stock"] = np.where(main["st_has"] == 1, "public (has stock)", "no stock data")
top_sectors = main["sector"].value_counts().head(6).index
main["group_sector"] = np.where(main["sector"].isin(top_sectors), main["sector"], "other or unknown")
rows = []
for col in ("group_frontline", "group_history", "group_stock", "group_sector"):
    for g, d in main.groupby(col):
        s = summarize(d)
        rows.append({"breakdown": col.replace("group_", ""), "group": g, "rows": s["rows"], "layoff_cases": s["layoff_cases"], "share_with_layoff": round(d[LABEL].mean(), 3), "caught_in_top50": s["top50_recall"], "auc": s["auc"]})
groups = pd.DataFrame(rows)
groups.to_csv(os.path.join(HERE, "model_subgroups.csv"), index=False)

cols = ["T", "company", "rank", "score", LABEL, "w_days_since", "w_n_12", "front_share", "sector", "workers_next6"]
false_alarms = main[(main["rank"] <= 10) & (main[LABEL] == 0)][cols].assign(error="false alarm")
missed = main[(main["rank"] > 50) & (main[LABEL] == 1)].sort_values("workers_next6", ascending=False)[cols].assign(error="missed layoff")
pd.concat([false_alarms, missed]).to_csv(os.path.join(HERE, "model_errors.csv"), index=False)
main[["T", "company", "rank", "score", LABEL, "w_prior", "front_share", "sector"]].sort_values(["T", "rank"]).to_csv(os.path.join(HERE, "model_predictions.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print()
print(results[["model", "top10_right", "top20_right", "top50_right", "top50_recall", "auc", "pr_auc"]].to_string(index=False))
print()
print(groups.to_string(index=False))
print()
print("false alarms in the top 10:", len(false_alarms), "| layoffs ranked below 50:", len(missed))
