import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, make_pipeline

ONET_DIR, GLASSDOOR, TOP500, PAIRS, REFERENCE, OUT_DIR = sys.argv[1:7]

MAJOR = {11: "Management", 13: "Business and Financial", 15: "Computer and Math", 17: "Architecture and Engineering", 19: "Science", 21: "Community and Social Service", 23: "Legal", 25: "Education", 27: "Arts, Media, Sports", 29: "Healthcare Practitioners and Technicians", 31: "Healthcare Support", 33: "Protective Service", 35: "Food Preparation and Serving", 37: "Cleaning and Maintenance", 39: "Personal Care and Service", 41: "Sales", 43: "Office and Administrative Support", 45: "Farming, Fishing, Forestry", 47: "Construction and Extraction", 49: "Installation, Maintenance, Repair", 51: "Production", 53: "Transportation and Material Moving", 55: "Military"}
HANDS_ON = {31, 33, 35, 37, 39, 41, 45, 47, 49, 51, 53}
SKIP_PATTERNS = ("registered nurse", "maintenance (tech")


def clean(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_company(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(inc|llc|corp|corporation|co|company|ltd|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_frontline(soc, zone):
    major = int(soc[:2])
    if zone >= 4 or major == 55:
        return False
    if zone <= 2:
        return major in HANDS_ON or soc.startswith(("29-2", "43-4", "43-5", "43-3071"))
    return major in {31, 35, 39} or soc.startswith("29-2")


ref = pd.read_csv(REFERENCE, low_memory=False)
ref_onet = ref[ref["title_type"].str.startswith("onet")]
sector_by_soc = ref_onet.groupby("onet_soc_code")["sector"].agg(lambda s: s.mode().iat[0]).to_dict()

occ = pd.read_csv(f"{ONET_DIR}/occupation_data.csv").iloc[:, :2]
occ.columns = ["soc", "occupation"]
zones = pd.read_csv(f"{ONET_DIR}/job_zones.csv")[["O*NET-SOC Code", "Job Zone"]]
zones.columns = ["soc", "job_zone"]
occ = occ.merge(zones, on="soc", how="left")
for n in (7, 5, 4, 2):
    fill = occ.groupby(occ["soc"].str[:n])["job_zone"].transform("mean").round()
    occ["job_zone"] = occ["job_zone"].fillna(fill)
occ["job_zone"] = occ["job_zone"].fillna(3).astype(int)
occ["major_group"] = occ["soc"].str[:2].astype(int).map(MAJOR)
occ["sector"] = occ["soc"].map(sector_by_soc).fillna("unknown")
occ["occupation_frontline"] = [is_frontline(s, z) for s, z in zip(occ["soc"], occ["job_zone"])]

jt = pd.read_csv(f"{ONET_DIR}/job_titles.csv")[["O*NET-SOC Code", "Job Title"]]
jt.columns = ["soc", "job_title"]
jt["title_source"] = "onet_job_title"
rt = pd.read_csv(f"{ONET_DIR}/sample_of_reported_titles.csv")[["O*NET-SOC Code", "Reported Job Title"]]
rt.columns = ["soc", "job_title"]
rt["title_source"] = "onet_reported_title"
st = ref[ref["title_type"] == "onet_short_title"][["onet_soc_code", "job_title"]].rename(columns={"onet_soc_code": "soc"})
st["title_source"] = "onet_short_title"
ot = occ[["soc", "occupation"]].rename(columns={"occupation": "job_title"})
ot["title_source"] = "onet_occupation_name"
titles = pd.concat([ot, jt, rt, st], ignore_index=True)
titles["title_key"] = titles["job_title"].map(clean)
titles = titles[(titles["title_key"] != "") & titles["soc"].isin(set(occ["soc"]))].drop_duplicates(["soc", "title_key"])
titles = titles.merge(occ, on="soc", how="left")

share = titles.groupby("title_key")["occupation_frontline"].mean()
best = ref_onet[ref_onet["best_match"] == "yes"].assign(title_key=lambda d: d["job_title"].map(clean)).drop_duplicates("title_key")
best_soc = dict(zip(best["title_key"], best["onet_soc_code"]))
front_by_soc = dict(zip(occ["soc"], occ["occupation_frontline"]))


def title_label(k):
    if share[k] == 1:
        return "yes"
    if share[k] == 0:
        return "no"
    soc = best_soc.get(k)
    if soc in front_by_soc:
        return "yes" if front_by_soc[soc] else "no"
    return "mixed"


rule = {k: title_label(k) for k in share.index}
titles["frontline_onet_rule"] = titles["title_key"].map(rule)

top = pd.read_csv(TOP500)
hand = top[top["label"].isin(["frontline", "not_frontline"])].copy()
hand["title_key"] = hand["title"].map(clean)
hand["hand"] = hand["label"].map({"frontline": "yes", "not_frontline": "no"})
hand_map = dict(zip(hand["title_key"], hand["hand"]))

both = hand[hand["title_key"].isin(rule)].copy()
both["onet_rule"] = both["title_key"].map(rule)
both = both[both["onet_rule"] != "mixed"]
agree = (both["hand"] == both["onet_rule"]).mean()
both[both["hand"] != both["onet_rule"]][["title", "n_reviews", "hand", "onet_rule"]].to_csv(f"{OUT_DIR}/hand_vs_onet_disagreements.csv", index=False)

train_df = titles[titles["frontline_onet_rule"] != "mixed"].drop_duplicates("title_key")
X, y, groups = train_df["title_key"].values, (train_df["frontline_onet_rule"] == "yes").astype(int).values, train_df["soc"].values


def make_model():
    feats = FeatureUnion([("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)), ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True))])
    return make_pipeline(feats, LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))


def report(name, yt, yp):
    return {"test_set": name, "n": len(yt), "share_frontline": round(float(np.mean(yt)), 3), "baseline_accuracy": round(max(np.mean(yt), 1 - np.mean(yt)), 3), "accuracy": round(accuracy_score(yt, yp), 3), "precision": round(precision_score(yt, yp), 3), "recall": round(recall_score(yt, yp), 3), "f1": round(f1_score(yt, yp), 3)}


tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
rows = [report("onet_titles_from_unseen_occupations", y[te], make_model().fit(X[tr], y[tr]).predict(X[te]))]
model = make_model().fit(X, y)
hx, hy = hand["title_key"].values, (hand["hand"] == "yes").astype(int).values
hp = model.predict(hx)
unseen = ~hand["title_key"].isin(set(train_df["title_key"])).values
rows.append(report("hand_labeled_titles_not_in_onet", hy[unseen], hp[unseen]))

patterns = [(re.compile(p), f, s) for p, f, s in ref.loc[ref["title_type"] == "keyword_pattern", ["job_title", "frontline", "sector"]].itertuples(index=False) if not p.startswith(SKIP_PATTERNS)]


def pattern_label(title):
    for p, f, s in patterns:
        if p.search(title):
            return f, s
    return None, None


pp = [pattern_label(t)[0] for t in hand["title"]]
got = np.array([p is not None for p in pp])
rows.append(report("keyword_patterns_on_hand_labeled_titles", hy[got], np.array([1 if p == "yes" else 0 for p in pp])[got]))
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT_DIR}/frontline_classifier_metrics.csv", index=False)

titles["frontline"] = titles["title_key"].map(hand_map).fillna(titles["frontline_onet_rule"])
titles["label_source"] = np.where(titles["title_key"].isin(hand_map), "hand_label", "onet_rule")
extra = hand[~hand["title_key"].isin(set(titles["title_key"]))]
extra_rows = pd.DataFrame({"job_title": extra["title"], "title_key": extra["title_key"], "title_source": "glassdoor_hand_labeled", "frontline": extra["hand"], "label_source": "hand_label", "sector": [pattern_label(t)[1] or "unknown" for t in extra["title"]]})
combined = pd.concat([titles, extra_rows], ignore_index=True).rename(columns={"soc": "onet_soc_code"})
combined = combined[["job_title", "title_key", "frontline", "label_source", "sector", "onet_soc_code", "occupation", "major_group", "job_zone", "occupation_frontline", "frontline_onet_rule", "title_source"]]
combined.to_csv(f"{OUT_DIR}/job_titles_frontline.csv", index=False)

g = pd.read_csv(GLASSDOOR, low_memory=False, usecols=["review_id", "employee_location", "employee_job_title", "company_name"]).drop_duplicates("review_id")
g = g[g["employee_location"].fillna("").str.contains(r", CA$|California", regex=True)].copy()
g["title"] = g["employee_job_title"].fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
g["title_key"] = g["title"].map(clean)
g["company"] = g["company_name"].map(norm_company)

special = dict(zip(top["title"], top["label"]))
my_pairs = pd.read_csv(PAIRS)
pair = {(clean(t), norm_company(c)): "yes" for t, c in ref.loc[ref["title_type"] == "vague_title_at_company", ["job_title", "company_name"]].itertuples(index=False)}
pair_src = {k: "company_pair_from_reviews" for k in pair}
for t, c, l in zip(my_pairs["title"], my_pairs["company"], my_pairs["label"]):
    pair[(clean(t), norm_company(c))] = "yes" if l == "frontline" else "no"
    pair_src[(clean(t), norm_company(c))] = "hand_title_company"

ranked = combined.assign(is_best=[best_soc.get(k) == s for k, s in zip(combined["title_key"], combined["onet_soc_code"])], fits=[(f == "yes") == bool(o) for f, o in zip(combined["frontline"], combined["occupation_frontline"])])
first = ranked.sort_values(["fits", "is_best"], ascending=False, kind="stable").drop_duplicates("title_key").set_index("title_key")
lookup, sector_lookup = first["frontline"].to_dict(), first["sector"].to_dict()
keys = g.loc[g["title_key"] != "", "title_key"].unique()
prob = dict(zip(keys, model.predict_proba(keys)[:, 1]))


def label_review(title, key, company):
    psec = pattern_label(title)[1] if key else None
    sector = sector_lookup.get(key) if sector_lookup.get(key, "unknown") != "unknown" else (psec or "unknown")
    if key == "":
        return "unknown", "no_title", np.nan, "unknown"
    kind = special.get(title)
    if kind in ("exclude", "intern_student"):
        return kind, "hand_label", np.nan, sector
    if (key, company) in pair:
        return pair[(key, company)], pair_src[(key, company)], np.nan, sector
    if kind == "vague":
        return "vague", "hand_label", prob.get(key, np.nan), sector
    hit = lookup.get(key)
    if hit in ("yes", "no"):
        return hit, "hand_label" if key in hand_map else "onet_lookup", prob.get(key, np.nan), sector
    p = prob[key]
    if p >= 0.8:
        return "yes", "classifier", p, sector
    if p <= 0.2:
        return "no", "classifier", p, sector
    pl = pattern_label(title)[0]
    if pl in ("yes", "no"):
        return pl, "keyword_pattern", p, sector
    return "unknown", "classifier_unsure", p, sector


res = [label_review(t, k, c) for t, k, c in zip(g["title"], g["title_key"], g["company"])]
g["frontline"], g["label_source"], g["prob_frontline"], g["sector"] = zip(*res)

clear = g[g["frontline"].isin(["yes", "no"])]
co_share = clear.groupby("company")["frontline"].agg(lambda s: (s == "yes").mean())
co_n = clear.groupby("company").size()
ok = (g["frontline"] == "vague") & (g["company"].map(co_n).fillna(0) >= 5)
g.loc[ok, "frontline"] = np.where(g.loc[ok, "company"].map(co_share) >= 0.5, "yes", "no")
g.loc[ok, "label_source"] = "company_majority"
g.loc[g["frontline"] == "vague", "frontline"] = "unknown"

g[["review_id", "company_name", "employee_job_title", "frontline", "label_source", "sector", "prob_frontline"]].to_csv(f"{OUT_DIR}/glassdoor_reviews_frontline.csv", index=False)

print("combined rows:", len(combined), "unique titles:", combined["title_key"].nunique())
print(combined.drop_duplicates("title_key")["frontline"].value_counts().to_string())
print("frontline occupations:", int(occ["occupation_frontline"].sum()), "of", len(occ))
print("hand vs onet rule agreement:", round(agree, 3), "on", len(both), "titles")
print(metrics.to_string(index=False))
print("glassdoor california reviews:", len(g))
print(g["frontline"].value_counts(normalize=True).round(3).to_string())
print(g["label_source"].value_counts(normalize=True).round(3).to_string())
print(g.loc[g["frontline"] == "yes", "sector"].value_counts().head(12).to_string())
