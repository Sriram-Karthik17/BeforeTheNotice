import sys, re, html
import pdfplumber, pandas as pd
from pathlib import Path
raw = Path(sys.argv[1]); out = Path(sys.argv[2])
STD = {"notice date":"notice_date","received date":"received_date","processed date":"received_date",
       "effective date":"effective_date","company":"company","county":"county","county/parish":"county",
       "no. of employees":"n_employees","layoff/closure type":"type","layoff/ closure":"type","layoff/closure":"type",
       "address":"address","related industry":"industry"}
def norm_hdr(h): return re.sub(r"\s+"," ",str(h or "").replace("\n"," ")).strip().lower()
frames=[]
for pdf in sorted(raw.glob("*.pdf")):
    rows=[]; header=None
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            for t in page.extract_tables():
                for r in t:
                    h=[norm_hdr(c) for c in r]
                    if "company" in h and any("notice" in c for c in h): header=[STD.get(c,c) for c in h]; continue
                    if header and len(r)==len(header) and any(r): rows.append(r)
    df=pd.DataFrame(rows,columns=header); df["source"]=pdf.name; frames.append(df)
    print(pdf.name, len(df), header)
# latest xlsx: Detailed WARN Report sheet, find header row
x=pd.read_excel(raw/"warn_latest.xlsx",sheet_name="Detailed WARN Report ",header=None)
hr=x.index[x.apply(lambda r: r.astype(str).str.contains("Company").any() and r.astype(str).str.contains("Notice").any(),axis=1)][0]
x.columns=[STD.get(norm_hdr(c),norm_hdr(c)) for c in x.iloc[hr]]; x=x.iloc[hr+1:].dropna(subset=["company"])
x["source"]="warn_latest.xlsx"; frames.append(x); print("xlsx",len(x),list(x.columns))
w=pd.concat(frames,ignore_index=True)
for c in ["company","county","type","address"]:
    w[c]=w[c].astype(str).map(lambda s: html.unescape(re.sub(r"\s+"," ",s)).strip())
for c in ["notice_date","received_date","effective_date"]:
    w[c]=pd.to_datetime(w[c],errors="coerce")
w["n_employees"]=pd.to_numeric(w["n_employees"].astype(str).str.replace(",",""),errors="coerce")
w["city"]=w.address.str.extract(r"([A-Za-z .'-]+?)\s+CA\s*\d{5}")[0].str.strip().str.split().str[-3:].str.join(" ")
w["zip"]=w.address.str.extract(r"CA\s*(\d{5})")[0]
w=w.drop_duplicates(subset=["notice_date","company","address","n_employees"])
w.to_csv(out,index=False); print("TOTAL",len(w))
