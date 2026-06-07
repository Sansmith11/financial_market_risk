"""
data.py
Data layer: Yahoo Finance download + local CSV ingestion + validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from config import DataConfig, CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    ticker: Optional[str] = None,
    csv_path: Optional[str] = None,
    period: str = "1y",
    interval: str = "1d",
    cfg: DataConfig = CONFIG.data,
) -> pd.DataFrame:
    """
    Load OHLCV data from local CSV or Yahoo Finance.

    Priority: csv_path > cfg.local_csv_path > Yahoo Finance.

    Returns
    -------
    pd.DataFrame with columns: [Open, High, Low, Close, Volume]
        Index: DatetimeIndex (ascending).
    """
    source = csv_path or cfg.local_csv_path

    if source:
        df = _load_csv(source, cfg)
        logger.info("Loaded %d rows from CSV: %s", len(df), source)
    elif YFINANCE_AVAILABLE:
        sym = ticker or cfg.default_ticker
        df = _load_yfinance(sym, period, interval)
        logger.info("Downloaded %d rows from Yahoo Finance (%s)", len(df), sym)
    else:
        raise RuntimeError(
            "No CSV path provided and yfinance is not installed. "
            "pip install yfinance  or supply csv_path."
        )

    df = _validate(df, cfg)
    return df


def get_multiple_tickers(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download several tickers; silently skip failures."""
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            result[t] = load_data(ticker=t, period=period, interval=interval)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", t, exc)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str, cfg: DataConfig) -> pd.DataFrame:
    """Read NSE-style CSV export (date strings like '04-JUN-2026')."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    col_map = {
        cfg.date_col.strip(): "Date",
        cfg.open_col.strip(): "Open",
        cfg.high_col.strip(): "High",
        cfg.low_col.strip(): "Low",
        cfg.close_col.strip(): "Close",
        cfg.volume_col.strip(): "Volume",
    }
    raw.rename(columns={k.strip(): v for k, v in col_map.items()}, inplace=True)

    # Parse dates
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%b-%Y", errors="coerce")
    raw.dropna(subset=["Date"], inplace=True)
    raw.set_index("Date", inplace=True)
    raw.sort_index(ascending=True, inplace=True)

    # Keep standard OHLCV columns (optional columns may be absent)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[keep].copy()


def _load_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not installed")
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.rename(columns={"Volume": "Volume"}, inplace=True)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].copy()


def _validate(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Validate, coerce numeric types, fill missing values."""
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from data")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    # Fill missing prices
    if cfg.fill_method == "ffill":
        df.ffill(inplace=True)
    elif cfg.fill_method == "bfill":
        df.bfill(inplace=True)

    df.dropna(subset=["Close"], inplace=True)

    n = len(df)
    if n < cfg.min_rows:
        raise ValueError(
            f"Only {n} rows after cleaning — need at least {cfg.min_rows}. "
            "Try a longer period."
        )

    logger.debug("Validated data: %d rows, %s to %s", n,
                 df.index.min().date(), df.index.max().date())
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    """Quick data-layer summary (no feature engineering)."""
    close = df["Close"]
    return {
        "rows": len(df),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "close_min": round(float(close.min()), 2),
        "close_max": round(float(close.max()), 2),
        "close_mean": round(float(close.mean()), 2),
        "missing_pct": round(float(df.isnull().mean().mean()) * 100, 2),
    }
