import sys
import pandas as pd
import build_matches as bm

WARN = sys.argv[1] if len(sys.argv) > 1 else "2023-2026WARN_Data.csv"
MATCHES = sys.argv[2] if len(sys.argv) > 2 else "matches_fix.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else WARN
PREFIX_BLOCKLIST = {"block by block"}
MIN_PREFIX_LEN = 5

w = pd.read_csv(WARN, low_memory=False).drop(columns=["glassdoor_company", "link_method"], errors="ignore")
mf = pd.read_csv(MATCHES)
norm = w["company"].map(bm.norm_warn)
key = norm.map(dict(zip(mf["warn_norm"], mf["company_norm"])))
method = key.notna().map({True: "name_match", False: None})

for _, (cn, _, pattern) in bm.MANUAL_FIXES.items():
    if pattern:
        hit = w["company"].str.contains(pattern, case=False, regex=True) & key.isna()
        key[hit], method[hit] = cn, "manual"
for _, pattern, cn in bm.MANUAL_ADDS:
    hit = w["company"].str.contains(pattern, case=False, regex=True)
    key[hit], method[hit] = cn, "manual"

target = {**dict(zip(mf["warn_norm"], mf["company_norm"])), **dict(zip(mf["company_norm"], mf["company_norm"]))}
known = sorted((k for k in target if len(k) >= MIN_PREFIX_LEN), key=len, reverse=True)


def by_prefix(name):
    if any(name.startswith(b) for b in PREFIX_BLOCKLIST):
        return None
    for k in known:
        if name.startswith(k + " "):
            return target[k]
    return None


todo = key.isna()
found = norm[todo].map(by_prefix)
key[todo] = found
method[todo & key.notna()] = "name_prefix"

w["glassdoor_company"] = key.map(dict(zip(mf["company_norm"], mf["glassdoor_name"])))
w["link_method"] = method.where(w["glassdoor_company"].notna())
w.to_csv(OUT, index=False)
print("notices:", len(w), "| linked to a Glassdoor company:", int(w["glassdoor_company"].notna().sum()))
print(w["link_method"].value_counts().to_string())
