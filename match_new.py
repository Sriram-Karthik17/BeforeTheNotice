"""Match WARN companies to Glassdoor companies (standalone, no common.py needed).
Usage: python match.py warn_all.csv glassdoor_reviews.csv matches.csv warn_matched.csv
"""
import sys, re, pandas as pd
from rapidfuzz import process, fuzz

# Same name cleaning as normalize_name() in common.py. Keep the two identical:
# if one changes, change the other, or labels and features will stop joining.
_SUFFIXES = r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|lp|llp|holdings|group|the|usa|us|na)\b"
def normalize_name(name) -> str:
    """'Google LLC' -> 'google', 'The Walt Disney Company' -> 'walt disney'."""
    if pd.isna(name):
        return ""
    s = str(name).lower()
    s = s.replace("&", " and ").replace("'", "").replace("\u2019", "")
    s = re.sub(r"\.com\b", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(_SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()
w = pd.read_csv(sys.argv[1], parse_dates=["notice_date"])
g = pd.read_csv(sys.argv[2], usecols=["company_name","employee_location","rating_date"], low_memory=False)
def clean_warn(s):
    s = str(s)
    s = re.split(r"\bd/?b/?a\b|\bdba\b", s, flags=re.I)[-1]          # "X dba Y" -> Y (brand)
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)                              # drop (site info)
    s = re.sub(r"\s[-–]\s.*$|\s-\d+.*$|#\s?\d+.*$|\s\d{3,5}$", " ", s)             # drop " - 1320", "#123"
    return normalize_name(s)
w["company_norm"] = w.company.map(clean_warn)
g["company_norm"] = g.company_name.map(normalize_name)
gc = g.groupby("company_norm").agg(n_reviews=("company_name","size"),
        n_ca=("employee_location", lambda s: s.str.contains(r", CA\b|California", na=False).sum()),
        glassdoor_name=("company_name","first")).reset_index()
gc = gc[gc.company_norm.str.len() > 2]
names = gc.company_norm.tolist(); nameset = set(names)
wc = w.groupby("company_norm").agg(n_notices=("company","size"), warn_example=("company","first")).reset_index()
rows = []
for _, r in wc.iterrows():
    n = r.company_norm
    if not n: continue
    if n in nameset: rows.append((n, n, 100, "exact")); continue
    hit = process.extractOne(n, names, scorer=fuzz.token_sort_ratio, score_cutoff=90)
    if hit and hit[0].split()[0] == n.split()[0]: rows.append((n, hit[0], hit[1], "fuzzy"))
m = pd.DataFrame(rows, columns=["warn_norm","company_norm","score","how"])
m = m.merge(wc.rename(columns={"company_norm":"warn_norm"}), on="warn_norm").merge(gc, on="company_norm")
m.to_csv(sys.argv[3], index=False)
w2 = w.merge(m[["warn_norm","company_norm"]].rename(columns={"company_norm":"gd"}), left_on="company_norm", right_on="warn_norm", how="left")
w2.to_csv(sys.argv[4], index=False)
print("WARN companies:", len(wc), "| matched:", m.warn_norm.nunique(), f"({m.how.value_counts().to_dict()})")
win = w2[(w2.notice_date >= "2025-04-01") & w2.gd.notna()]
for th in [10, 30, 100]:
    ok = gc[gc.n_reviews >= th].company_norm
    print(f"Glassdoor cos w/ >={th} reviews that filed WARN Apr2025-Nov2026: {win[win.gd.isin(ok)].gd.nunique()}  (of {len(ok)} such cos)")
big = win[win.n_employees >= 50]
print("...same, only notices with 50+ employees (>=30 reviews):", big[big.gd.isin(gc[gc.n_reviews>=30].company_norm)].gd.nunique())
