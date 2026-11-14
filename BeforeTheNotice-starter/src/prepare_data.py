"""Standardize reviews and WARN exports while keeping source files intact."""
import argparse
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROCESSED = Path("data/processed")


def company_key(name):
    name = unicodedata.normalize("NFKC", str(name)).casefold()
    name = re.sub(r"[^\w]+|_", " ", name).split()
    while name and name[-1] in {"inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", "llp"}:
        name.pop()
    return " ".join(name)


def read_table(path, sheet, header):
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name=sheet, header=header, dtype=str)
    else:
        frame = pd.read_csv(path, header=header, dtype=str)
    frame.columns = frame.columns.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return frame


def clean(frame, path, required, optional, date):
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}. Found {frame.columns.tolist()}")
    frame = frame.reindex(columns=[*required, *optional]).copy()
    frame["company_name"] = frame["company_name"].fillna("").str.strip()
    frame["company_key"] = frame["company_name"].map(company_key)
    frame[date] = pd.to_datetime(frame[date], format="mixed", errors="coerce").dt.normalize()
    valid = frame["company_key"].ne("") & frame[date].notna()
    print(f"{path}: kept {valid.sum()} of {len(frame)} rows with valid company and {date}")
    if not valid.any():
        raise ValueError(f"{path}: no usable rows")
    return frame.loc[valid].copy()


def load_warn(path, sheet, header):
    frame = read_table(path, sheet, header).rename(columns={
        "company": "company_name", "received_date": "notice_date",
        "warn_received_date": "notice_date", "total_layoff_employees": "workers_affected",
        "no_of_employees": "workers_affected", "number_of_employees": "workers_affected",
    })
    if "notice_date" in frame and frame.columns.duplicated().any():
        raise ValueError(f"{path}: ambiguous headers, map them to the documented WARN schema")
    frame = clean(frame, path, ["company_name", "notice_date"],
                  ["workers_affected", "city", "effective_date", "county", "address", "notice_type"], "notice_date")
    workers = pd.to_numeric(frame["workers_affected"].astype("string").str.replace(",", ""), errors="coerce")
    frame["workers_affected"] = workers.where(workers.ge(0) & workers.mod(1).eq(0)).astype("Int64")
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], format="mixed", errors="coerce").dt.normalize()
    frame["city"] = frame["city"].astype("string").str.strip()
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-warn")
    download.add_argument("url")
    download.add_argument("destination", type=Path)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--reviews", required=True, type=Path)
    prepare.add_argument("--warn", required=True, nargs="+", type=Path)
    prepare.add_argument("--warn-sheet", default="Detailed WARN Report")
    prepare.add_argument("--warn-header", type=int, default=0)
    args = parser.parse_args()
    if args.command == "download-warn":
        with urlopen(args.url, timeout=60) as response:
            payload = response.read()
        with args.destination.open("xb") as output:
            output.write(payload)
        return

    reviews = clean(read_table(args.reviews, 0, 0), args.reviews,
                    ["company_name", "review_date", "rating"],
                    ["pros", "cons", "job_title", "location", "industry"], "review_date")
    rating = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews["rating"] = rating.where(rating.between(1, 5))
    warn = pd.concat([load_warn(path, args.warn_sheet, args.warn_header if path.suffix.lower() == ".xlsx" else 0)
                      for path in args.warn], ignore_index=True).drop_duplicates()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(PROCESSED / "reviews.csv", index=False)
    warn.to_csv(PROCESSED / "warn.csv", index=False)


if __name__ == "__main__":
    main()
