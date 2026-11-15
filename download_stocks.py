import sys
import pandas as pd
import yfinance as yf

START = "2024-06-01"
tickers = sorted(pd.read_csv("company_tickers.csv")["ticker"].unique().tolist()) + ["^GSPC"]
data = yf.download(tickers, start=START, auto_adjust=True, progress=False, threads=True)["Close"]
long = data.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close").dropna(subset=["close"])
long = long.rename(columns={"Date": "date"}).sort_values(["ticker", "date"])
long.to_csv("stock_prices.csv", index=False)
missing = sorted(set(tickers) - set(long["ticker"]))
print("rows:", len(long), "| tickers with prices:", long["ticker"].nunique(), "of", len(tickers))
print("dates:", long["date"].min(), "to", long["date"].max())
print("no data for:", missing)
