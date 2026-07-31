"""Download and cache the public market and Treasury data.

Market indices are downloaded from Yahoo Finance and the US three-month
Treasury-bill discount yield is downloaded from FRED.

Usage:
    python scripts/download_data.py --start 1970-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

FRED_DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

TICKERS = {
    "SPX": "^GSPC",
    "SPX_TR": "^SP500TR",
    "DAX": "^GDAXI",
    "NIKKEI": "^N225",
}


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one Yahoo Finance series and return it ascending by date, deduplicated."""
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df


def download_all(out_dir: Path, start: str, end: str, force: bool) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, symbol in TICKERS.items():
        path = out_dir / f"{name.lower()}.parquet"
        if path.exists() and not force:
            print(f"skip {name}: {path} already exists (use --force to refresh)")
            written[name] = path
            continue
        print(f"downloading {name} ({symbol}) from Yahoo Finance...")
        df = fetch_one(symbol, start=start, end=end)
        df.to_parquet(path)
        print(f"  wrote {len(df)} rows -> {path}")
        written[name] = path
    return written


def download_risk_free_rate(
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download the US three-month Treasury bill rate from FRED."""

    rates = pd.read_csv(
        FRED_DTB3_URL,
        na_values=".",
    )

    rates = rates.rename(
        columns={
            "observation_date": "date",
            "DTB3": "annual_discount_yield_pct",
        }
    )

    rates["date"] = pd.to_datetime(rates["date"])

    rates = rates.set_index("date").sort_index()
    rates = rates.loc[start:end]

    if rates.empty:
        raise ValueError("No risk-free observations were downloaded")

    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--start", type=str, default="1990-01-01")
    parser.add_argument("--end", type=str, default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    download_all(args.out, start=args.start, end=args.end, force=args.force)

    risk_free = download_risk_free_rate(start=args.start, end=args.end)

    risk_free_path = args.out / "us_tbill_3m.parquet"
    risk_free.to_parquet(risk_free_path)

    print(f"Saved risk-free rates to {risk_free_path}")


if __name__ == "__main__":
    main()
