"""
features.py
Feature engineering: log returns, volatility, drawdowns, tail risk, etc.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import FeatureConfig, CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: FeatureConfig = CONFIG.features) -> pd.DataFrame:
    """
    Compute all features and append them to a copy of *df*.

    Returns a DataFrame with original OHLCV columns plus engineered features.
    """
    out = df.copy()

    # --- core returns ---
    out["log_return"] = _log_returns(out["Close"])
    out["abs_return"] = out["log_return"].abs()
    out["pct_return"] = out["Close"].pct_change()

    # --- rolling features ---
    for w in cfg.rolling_windows:
        out[f"vol_{w}d"] = out["log_return"].rolling(w).std() * np.sqrt(cfg.annualisation_factor)
        out[f"mean_ret_{w}d"] = out["log_return"].rolling(w).mean()
        out[f"skew_{w}d"] = out["log_return"].rolling(w).skew()
        out[f"kurt_{w}d"] = out["log_return"].rolling(w).kurt()

    # --- realised volatility (Parkinson) ---
    if all(c in out.columns for c in ["High", "Low"]):
        out["parkinson_vol"] = _parkinson_vol(out["High"], out["Low"], cfg.annualisation_factor)

    # --- drawdown ---
    out["drawdown"] = _drawdown(out["Close"])
    out["max_drawdown"] = out["drawdown"].expanding().min()

    # --- tail risk ---
    lr = out["log_return"].dropna()
    q = cfg.tail_quantile
    var_val = float(np.quantile(lr, q))
    cvar_val = float(lr[lr <= var_val].mean())
    out["VaR_95"] = var_val                # broadcast scalar
    out["CVaR_95"] = cvar_val

    # --- distribution-specific features ---
    out["squared_return"] = out["log_return"] ** 2
    out["signed_vol"] = out["log_return"].abs().rolling(21).mean()

    out.dropna(subset=["log_return"], inplace=True)
    logger.debug("Features built: %d columns, %d rows", out.shape[1], len(out))
    return out


def get_return_series(df_feat: pd.DataFrame) -> np.ndarray:
    """Return cleaned log-return array for distribution fitting."""
    arr = df_feat["log_return"].dropna().values
    return arr[np.isfinite(arr)]


def feature_summary(df_feat: pd.DataFrame) -> Dict[str, float]:
    """Summary statistics of the return series."""
    lr = df_feat["log_return"].dropna()
    ann = CONFIG.features.annualisation_factor
    return {
        "n": int(len(lr)),
        "mean_daily_return": round(float(lr.mean()), 6),
        "annualised_return": round(float(lr.mean() * ann), 4),
        "daily_volatility": round(float(lr.std()), 6),
        "annualised_volatility": round(float(lr.std() * np.sqrt(ann)), 4),
        "skewness": round(float(lr.skew()), 4),
        "excess_kurtosis": round(float(lr.kurt()), 4),
        "min_return": round(float(lr.min()), 6),
        "max_return": round(float(lr.max()), 6),
        "VaR_5pct": round(float(np.quantile(lr, 0.05)), 6),
        "CVaR_5pct": round(float(lr[lr <= np.quantile(lr, 0.05)].mean()), 6),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _parkinson_vol(high: pd.Series, low: pd.Series, ann: int) -> pd.Series:
    factor = 1.0 / (4.0 * np.log(2))
    daily = np.sqrt(factor * (np.log(high / low) ** 2))
    return daily * np.sqrt(ann)


def _drawdown(close: pd.Series) -> pd.Series:
    rolling_max = close.expanding().max()
    return (close - rolling_max) / rolling_max
