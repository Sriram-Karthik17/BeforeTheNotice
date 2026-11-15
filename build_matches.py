"""
build_matches.py
----------------
Builds matches_fixed.csv: links California WARN layoff notices to Glassdoor
employers, then adds layoff-timing stats and frontline-review counts.

Output columns (one row per normalized WARN employer name):
    warn_norm, company_norm, score, how, n_notices, warn_example,
    n_reviews, n_ca, glassdoor_name, fix_note,
    n_notices_all, n_notices_window, workers_window,
    first_notice_window, first_notice_ever, filed_before_2025,
    frontline_reviews, frontline_reviews_ca, frontline_share

Usage:
    python build_matches.py \
        --warn warn_all_1.csv \
        --glassdoor glassdoor_reviews.csv \
        --out matches_fixed.csv

Edit the CONFIG block below if your column names differ.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# WARN file columns (CA EDD WARN report)
WARN_COMPANY_COL = "company"
WARN_DATE_COL = "notice_date"
WARN_WORKERS_COL = "n_employees"
WARN_DEDUPE_COLS = ["company", "notice_date", "effective_date", "county", "n_employees", "address"]

# Glassdoor review columns
GD_FIRM_COL = "company_name"
GD_TITLE_COL = "employee_job_title"
GD_LOCATION_COL = "employee_location"

FUZZY_CUTOFF = 90.0                         # minimum fuzz.ratio for a fuzzy match
WINDOW_START = pd.Timestamp("2025-07-01")   # start of the "recent notices" window
BEFORE_CUTOFF = pd.Timestamp("2025-01-01")  # filed_before_2025 threshold

# Words dropped from names after cleaning (legal suffixes + filler)
STOPWORDS = {
    "the", "inc", "incorporated", "llc", "corp", "corporation", "company", "co",
    "ltd", "lp", "plc", "group", "holdings", "us", "usa",
}

# Job-title keywords that mark a review as "frontline" (whole-word match,
# case-insensitive). Tuned against the original matches_fixed.csv counts.
FRONTLINE_KEYWORDS = [
    "cashier", "sales associate", "store associate", "retail associate",
    "production associate", "warehouse associate", "warehouse", "picker", "packer",
    "clerk", "stocker", "driver", "delivery", "courier", "package handler",
    "material handler", "forklift", "server", "bartender", "barista", "cook",
    "dishwasher", "host", "hostess", "housekeeper", "housekeeping", "janitor",
    "custodian", "security officer", "security guard", "caregiver",
    "nursing assistant", "cna", "medical assistant", "registered nurse", "rn",
    "lvn", "nurse", "receptionist", "front desk", "shift supervisor", "shift lead",
    "crew member", "team member", "maintenance technician", "pharmacy technician",
    "attendant", "customer service", "order selector", "selector",
    "machine operator", "operator", "brand ambassador", "fueler", "merchandiser",
    "production worker", "production builder", "program leader",
    "fulfillment associate", "fulfillment", "assembler", "laborer", "loader",
    "mechanic", "teller", "aide", "associate", "crew", "crew trainer",
    "shift manager", "shift leader", "greeter", "shopper", "personal shopper",
    "guest advocate", "guest service", "guest services", "asset protection",
    "cart pusher", "kitchen", "chef", "burger", "fries", "atendente", "cajero",
    "empleado", "stylist", "beauty consultant", "style consultant", "genius",
    "team leader",
]

# A title containing any of these is never frontline (keeps "Associate Director",
# "Software Engineer", "Research Associate" etc. out).
NOT_FRONTLINE = [
    "director", "scientist", "software", "engineer", "engineering", "attorney",
    "counsel", "analyst", "product manager", "program manager", "project manager",
    "marketing", "finance", "financial", "account executive", "accountant",
    "recruiter", "research associate", "data", "consultant at", "partner",
    "vice president", "intern", "internship", "developer", "designer", "architect",
    "professor", "physician", "nurse practitioner", "case manager",
]

# Manual corrections to bad automatic matches: warn_norm -> (company_norm, note, optional regex)
# The regex (if given) widens which raw WARN company names count toward the
# window/date stats for that row.
MANUAL_FIXES = {
    "cvs": ("cvs health", "FIXED: was 'CVS Group' (UK veterinary company)", None),
    "kaiser": (
        "kaiser permanente",
        "FIXED: was 'KAISER' (2 reviews); WARN 'Kaiser' sites are Kaiser Permanente",
        r"\bkaiser\b",
    ),
    "cisco systems": ("cisco", "FIXED: was 'Cisco Systems' (1 review); main Glassdoor page is 'Cisco'", None),
}

MANUAL_ADDS = [
    ("vons", r"^the vons companies", "vons"),
    ("marshalls", r"^marshalls", "marshalls"),
    ("uber technologies", r"^uber technologies", "uber"),
    ("walt disney parks and resorts", r"^walt disney parks", "walt disney"),
    ("hilton", r"^hilton", "hilton"),
    ("staples fulfillment center", r"^staples fulfillment", "staples"),
    ("six flags entertainment", r"^six flags", "six flags"),
    ("shake shack enterprises", r"^shake shack", "shake shack"),
    ("aramark campus", r"^aramark campus", "aramark"),
    ("marriott san mateo hotel", r"^marriott san mateo", "marriott international"),
    ("smart and final stores", r"^smart & final", "smart and final"),
    ("99 cents only stores", r"^99 cents only", "99 cents only"),
    ("michaels stores", r"^michaels stores procurement", "michaels"),
    ("petco animal supplies stores", r"^petco animal", "petco"),
    ("fedex supply chain", r"^fedex supply chain", "fedex"),
    ("universal protection service", r"^universal protection service", "allied universal"),
    ("meta platforms", r"^meta platforms", "meta"),
    ("oracle america", r"^oracle", "oracle"),
    ("paramount global", r"^paramount (global|skydance)", "paramount"),
    ("herbalife international", r"^herbalife", "herbalife"),
    ("planned parenthood", r"^planned parenthood", "planned parenthood"),
    ("rivian automotive", r"^rivian", "rivian"),
    ("becton dickinson and", r"^becton", "becton dickinson"),
    ("farmers group", r"^farmers (group|insurance)", "farmers insurance"),
    ("infineon technologies americas", r"^infineon", "infineon technologies"),
    ("gxo logistics supply chain", r"^gxo", "gxo logistics"),
    ("maxar space", r"^maxar", "maxar technologies"),
    ("blue shield of ca", r"^blue shield of ca\b", "blue shield of california"),
    ("renesas electronics", r"^renesas", "renesas"),
    ("chick fil a", r"^chick-fil-a", "chick fil a restaurants"),
    ("robinhood markets", r"^robinhood", "robinhood"),
    ("sharp metropolitan medical", r"^sharp metropolitan", "sharp healthcare"),
    ("jones lang lasalle americas", r"^jones lang lasalle", "jones lang lasalle"),
    ("marvell semiconductor", r"^marvell", "marvell technology"),
    ("international business machines", r"^international business machines", "ibm"),
    ("pacific biosciences of california", r"^pacific biosciences", "pacific biosciences"),
    ("laz parking california", r"^laz parking", "laz parking"),
    ("raleys", r"^raley", "raleys bel air nob hill"),
    ("lucid usa", r"^lucid usa", "lucid motors"),
    ("sap america", r"^sap america", "sap"),
    ("vca", r"^vca\b", "vca"),
    ("novartis pharmaceuticals", r"^novartis", "novartis"),
    ("providence health", r"^providence (holy cross|st\b|saint|little company|tarzana|health)", "providence"),
    ("bill operations", r"^bill operations", "bill"),
    ("goodwill", r"^goodwill", "goodwill"),
    ("kbr services", r"^kbr\b", "kbr"),
    ("joann stores", r"^joann", "joann stores"),
    ("red lobster restaurants", r"^red lobster", "red lobster"),
    ("red bull north america", r"^red bull", "red bull"),
    ("ttm technologies", r"^ttm technologies", "ttm technologies"),
    ("levi strauss and", r"^levi strauss", "levi strauss"),
    ("l3harris technologies", r"^l3harris", "l3harris"),
    ("blackline systems", r"^blackline", "blackline"),
    ("unity technologies", r"^unity technologies", "unity"),
    ("panera", r"^panera", "panera bread"),
    ("centene management", r"^centene", "centene"),
    ("catalent", r"^catalent", "catalent pharma"),
    ("quantumscape battery", r"^quantumscape", "quantumscape"),
    ("proterra operating", r"^proterra", "proterra"),
    ("twitch interactive", r"^twitch", "twitch"),
    ("universal city studios", r"^universal city studios", "universal studios"),
    ("valero refining", r"^valero", "valero"),
    ("vans headquarters", r"^van'?s (corporate|hq)", "vans"),
    ("tencent america", r"^tencent", "tencent"),
    ("adidas america", r"^adidas", "adidas"),
    ("williams sonoma direct", r"^williams-sonoma", "williams sonoma"),
    ("verily health", r"^verily", "verily"),
    ("abbott laboratories", r"^abbott laboratories", "abbott"),
    ("rtx building", r"^rtx _", "rtx"),
    ("sambanova systems", r"^sambanova", "sambanova"),
    ("roche molecular systems", r"^roche ", "roche"),
    ("charles schwab and", r"^charles schwab", "charles schwab"),
    ("chime financial", r"^chime financial", "chime"),
    ("buzzfeed media enterprises", r"^buzzfeed", "buzzfeed"),
    ("capital one financial", r"^capital one", "capital one"),
    ("bloomingdales", r"^bloomingdale", "bloomingdales"),
    ("anheuser busch", r"^anheuser", "anheuser busch inbev"),
    ("intuitive surgical", r"^intuitive surgical", "intuitive"),
    ("mettler toledo rainin", r"^mettler", "mettler toledo"),
    ("balfour beatty civils", r"^balfour beatty", "balfour beatty"),
]

ADD_NOTE = "ADDED: company missed by name matching"

# warn_norm values to drop entirely (bad matches you don't want to keep)
# (these are wrong fuzzy matches, e.g. "baxalta" -> "axalta", "newark" -> "newmark")
BLOCKLIST = {
    "california college of asu", "stepping stone",
    "aclu of southern california", "alard engineering", "amq solutions", "baxalta",
    "dji services", "gsc solutions", "host international", "ikea distribution services",
    "levine management", "newark", "noa technologies", "nsi services", "sac health",
    "sap america", "tao hospitality", "volta", "endeavors", "alliance", "arc",
    "coach",   # Coach USA (bus company) != Coach (fashion brand)
    "nestle",  # would fuzzy-match the accent-stripped "nestl"
}

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Shared cleanup: lowercase, &->and, drop apostrophes/accents, punctuation->space."""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"['’`]", "", s)
    s = s.encode("ascii", "ignore").decode()   # drop accents/odd chars (Nestlé -> nestl)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = [w for w in s.split() if w not in STOPWORDS]
    return " ".join(words)


def norm_warn(name: str) -> str:
    """Normalize a raw WARN company name (keeps the DBA name, strips site info)."""
    s = str(name)
    # keep the "doing business as" name if present
    m = re.search(r"\b(?:d/b/a|dba)\b\.?\s*(.*)", s, flags=re.I)
    if m and m.group(1).strip():
        s = m.group(1)
    s = re.sub(r"\(.*?\)", " ", s)        # (site / address / location)
    s = re.sub(r"\(.*$", " ", s)          # unclosed paren
    s = re.sub(r"\s[-–]\s*.*$", " ", s)  # " - Site Name", " -1160"
    s = re.sub(r"#\s*\d+", " ", s)        # store numbers
    s = _clean(s)
    return re.sub(r"(\s\d{3,})+$", "", s)  # trailing site numbers ("... california 3021")


def norm_glassdoor(name: str) -> str:
    """Normalize a Glassdoor firm name."""
    return _clean(name)

# ---------------------------------------------------------------------------
# Fuzzy score (same as rapidfuzz.fuzz.ratio; pure-python fallback)
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz, process

    def ratio(a, b):
        return fuzz.ratio(a, b)

    def best_match(query, choices):
        hit = process.extractOne(query, choices, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
        return (hit[0], hit[1]) if hit else (None, 0.0)

except ImportError:
    def ratio(a, b):
        if not a and not b:
            return 100.0
        prev = [0] * (len(b) + 1)
        for ca in a:
            cur = [0]
            for j, cb in enumerate(b):
                cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
            prev = cur
        return 200.0 * prev[-1] / (len(a) + len(b))

    def best_match(query, choices):
        best, best_score = None, 0.0
        for c in choices:
            # quick length bound: ratio can't beat 200*min/(sum)
            if 200.0 * min(len(query), len(c)) / (len(query) + len(c)) < FUZZY_CUTOFF:
                continue
            sc = ratio(query, c)
            if sc > best_score:
                best, best_score = c, sc
        return (best, best_score) if best_score >= FUZZY_CUTOFF else (None, 0.0)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_any(path, usecols=None):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, usecols=usecols)
    return pd.read_csv(p, usecols=usecols, low_memory=False)


def load_warn(paths):
    df = pd.concat([read_any(p) for p in paths], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df[WARN_COMPANY_COL].notna()].copy()
    df["raw"] = df[WARN_COMPANY_COL].astype(str).str.strip()
    df["warn_norm"] = df["raw"].map(norm_warn)
    df["date"] = pd.to_datetime(df[WARN_DATE_COL], errors="coerce")
    df["workers"] = pd.to_numeric(df[WARN_WORKERS_COL], errors="coerce").fillna(0)
    return df[df["warn_norm"] != ""]


_FRONTLINE_RX = re.compile(r"\b(?:" + "|".join(map(re.escape, FRONTLINE_KEYWORDS)) + r")\b")
_NOT_FRONTLINE_RX = re.compile(r"\b(?:" + "|".join(map(re.escape, NOT_FRONTLINE)) + r")")


def is_frontline(title) -> bool:
    if pd.isna(title):
        return False
    t = str(title).lower()
    return bool(_FRONTLINE_RX.search(t)) and not _NOT_FRONTLINE_RX.search(t)


def is_ca(loc: str) -> bool:
    l = str(loc).strip()
    return l.endswith(", CA") or l == "California"


def load_glassdoor(path):
    gd = read_any(path, usecols=[GD_FIRM_COL, GD_TITLE_COL, GD_LOCATION_COL])
    gd = gd[gd[GD_FIRM_COL].notna()].copy()
    gd["company_norm"] = gd[GD_FIRM_COL].astype(str).map(norm_glassdoor)
    gd = gd[gd["company_norm"] != ""]
    gd["ca"] = gd[GD_LOCATION_COL].map(is_ca)
    gd["frontline"] = gd[GD_TITLE_COL].map(is_frontline)

    # one row per company_norm; display name = most-reviewed raw firm name
    name = (gd.groupby(["company_norm", GD_FIRM_COL]).size()
              .reset_index(name="n").sort_values("n", ascending=False)
              .drop_duplicates("company_norm")
              .set_index("company_norm")[GD_FIRM_COL])
    stats = gd.groupby("company_norm").agg(
        n_reviews=("ca", "size"),
        n_ca=("ca", "sum"),
        frontline_reviews=("frontline", "sum"),
        frontline_reviews_ca=("frontline", lambda s: int((s & gd.loc[s.index, "ca"]).sum())),
    )
    stats["glassdoor_name"] = name
    stats["frontline_share"] = (stats["frontline_reviews"] / stats["n_reviews"]).round(2)
    return stats

# ---------------------------------------------------------------------------
# Notice stats
# ---------------------------------------------------------------------------

def notice_stats(rows: pd.DataFrame) -> dict:
    dated = rows.dropna(subset=["date"])
    dated = dated.drop_duplicates(subset=[c for c in WARN_DEDUPE_COLS if c in dated.columns])
    win = dated[dated["date"] >= WINDOW_START]
    first_ever = dated["date"].min()
    first_win = win["date"].min()
    return {
        "n_notices_all": len(dated),
        "n_notices_window": len(win),
        "workers_window": int(win["workers"].sum()),
        "first_notice_window": first_win.date().isoformat() if pd.notna(first_win) else "",
        "first_notice_ever": first_ever.date().isoformat() if pd.notna(first_ever) else "",
        "filed_before_2025": bool(pd.notna(first_ever) and first_ever < BEFORE_CUTOFF),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(warn_paths, glassdoor_path):
    warn = load_warn(warn_paths)
    gd = load_glassdoor(glassdoor_path)
    gd_names = set(gd.index)
    gd_list = list(gd.index)

    rows = []
    manual_labels = {label for label, _, _ in MANUAL_ADDS}

    # 1) automatic matches (exact, then fuzzy)
    for wn, grp in warn.groupby("warn_norm", sort=True):
        if wn in BLOCKLIST or wn in manual_labels:
            continue
        fix = MANUAL_FIXES.get(wn)
        if fix:
            cn, how, score, note, pattern = fix[0], "manual", 100.0, fix[1], fix[2]
        elif wn in gd_names:
            cn, how, score, note, pattern = wn, "exact", 100.0, "", None
        else:
            cn, score = best_match(wn, gd_list)
            if cn is None:
                continue
            how, note, pattern = "fuzzy", "", None
        if cn not in gd_names:
            continue

        stat_rows = warn[warn["raw"].str.contains(pattern, case=False, regex=True)] if pattern else grp
        rows.append({
            "warn_norm": wn, "company_norm": cn, "score": float(score), "how": how,
            "n_notices": len(grp), "warn_example": grp["raw"].iat[0],
            "n_reviews": int(gd.at[cn, "n_reviews"]), "n_ca": int(gd.at[cn, "n_ca"]),
            "glassdoor_name": gd.at[cn, "glassdoor_name"], "fix_note": note,
            **notice_stats(stat_rows),
            "frontline_reviews": int(gd.at[cn, "frontline_reviews"]),
            "frontline_reviews_ca": int(gd.at[cn, "frontline_reviews_ca"]),
            "frontline_share": float(gd.at[cn, "frontline_share"]),
        })

    # 2) manual additions
    for label, pattern, cn in MANUAL_ADDS:
        if cn not in gd_names:
            print(f"  skip manual add {label!r}: '{cn}' not in Glassdoor data")
            continue
        grp = warn[warn["raw"].str.contains(pattern, case=False, regex=True)]
        if grp.empty:
            print(f"  skip manual add {label!r}: no WARN rows match {pattern!r}")
            continue
        rows.append({
            "warn_norm": label, "company_norm": cn, "score": 100.0, "how": "manual",
            "n_notices": len(grp), "warn_example": grp["raw"].iat[0],
            "n_reviews": int(gd.at[cn, "n_reviews"]), "n_ca": int(gd.at[cn, "n_ca"]),
            "glassdoor_name": gd.at[cn, "glassdoor_name"], "fix_note": ADD_NOTE,
            **notice_stats(grp),
            "frontline_reviews": int(gd.at[cn, "frontline_reviews"]),
            "frontline_reviews_ca": int(gd.at[cn, "frontline_reviews_ca"]),
            "frontline_share": float(gd.at[cn, "frontline_share"]),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warn", nargs="+", required=True, help="WARN notice file(s), .xlsx or .csv")
    ap.add_argument("--glassdoor", required=True, help="Glassdoor reviews file, .csv or .xlsx")
    ap.add_argument("--out", default="matches_fixed.csv")
    args = ap.parse_args()

    out = build(args.warn, args.glassdoor)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    print(out["how"].value_counts().to_string())


if __name__ == "__main__":
    main()
