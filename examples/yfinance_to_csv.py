#!/usr/bin/env python3
"""
Download equity data via yfinance and convert to SignalForge CSV format.

Output columns: timestamp, ticker, metric, value
  - One row per bar per metric (Open, High, Low, Close, Volume)
  - timestamp is ISO 8601, UTC

Usage
-----
# Last 5 days of GME at 1-minute resolution (default)
uv run python examples/yfinance_to_csv.py

# Specific ticker and interval
uv run python examples/yfinance_to_csv.py --ticker AAPL --interval 1m

# Daily bars for GME going back 2 years (includes the 2021 squeeze)
uv run python examples/yfinance_to_csv.py --ticker GME --interval 1d --period 2y

# Save to a specific file
uv run python examples/yfinance_to_csv.py --ticker GME --interval 1d --period 2y --out data/gme_daily.csv

Notes
-----
- 1-minute data: available for the trailing ~7 days only (yfinance free tier).
  For historical minute data (e.g. GME January 2021), use a paid provider
  such as Polygon.io or Alpaca.
- Daily data: available going back years, sufficient to capture the GME squeeze.
- Requires: pip install yfinance  (or: uv add yfinance)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: uv add yfinance", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {ticker}  interval={interval}  period={period} ...")
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)

    if df.empty:
        print(f"No data returned for {ticker}. Check ticker symbol and date range.", file=sys.stderr)
        sys.exit(1)

    # yfinance MultiIndex columns when auto_adjust=True: (metric, ticker)
    # Flatten to single-level if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def to_signalforge_csv(df: pd.DataFrame, ticker: str, metrics: list[str]) -> pd.DataFrame:
    """Melt OHLCV DataFrame into long-form SignalForge CSV."""
    rows = []
    for ts, row in df.iterrows():
        # Ensure UTC
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        elif hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")

        ts_str = ts.isoformat()

        for metric in metrics:
            if metric in row and pd.notna(row[metric]):
                rows.append({
                    "timestamp": ts_str,
                    "ticker": ticker,
                    "metric": metric,
                    "value": float(row[metric]),
                })

    return pd.DataFrame(rows, columns=["timestamp", "ticker", "metric", "value"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Download equity data to SignalForge CSV format.")
    ap.add_argument("--ticker",   default="GME",  help="Ticker symbol (default: GME)")
    ap.add_argument("--interval", default="1m",   help="Bar interval: 1m, 5m, 1h, 1d (default: 1m)")
    ap.add_argument("--period",   default="5d",   help="Lookback period: 1d, 5d, 1mo, 3mo, 1y, 2y, max (default: 5d)")
    ap.add_argument("--metrics",  default="Close,Volume", help="Comma-separated metrics to include (default: Close,Volume)")
    ap.add_argument("--out",      default=None,   help="Output CSV path (default: <ticker>_<interval>.csv)")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    out_path = args.out or f"{args.ticker.lower()}_{args.interval}.csv"

    df = download(args.ticker, args.interval, args.period)
    out_df = to_signalforge_csv(df, args.ticker, metrics)

    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df):,} rows → {out_path}")
    print(f"  Metrics : {metrics}")
    print(f"  Bars    : {len(df):,}")
    print(f"  Range   : {out_df['timestamp'].iloc[0]}  →  {out_df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
