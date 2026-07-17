"""Download daily SPX, NDX, and DAX price history from Yahoo Finance and cache as parquet.

Usage:
    python scripts/download_data.py [--out data] [--start 1990-01-01] [--force]

Note: this originally targeted Stooq via pandas-datareader (no API key needed),
but Stooq now gates every request -- including the CSV download endpoint --
behind a client-side JS proof-of-work challenge that plain HTTP clients cannot
pass. yfinance (Yahoo Finance) is used instead; it requires no API key either.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "DAX": "^GDAXI",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--start", type=str, default="1990-01-01")
    parser.add_argument("--end", type=str, default=date.today().isoformat())
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    download_all(args.out, start=args.start, end=args.end, force=args.force)


if __name__ == "__main__":
    main()
