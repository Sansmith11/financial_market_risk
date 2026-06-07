"""
regime.py
Market regime classification engine.
Uses return characteristics, volatility, distribution fit, tail behaviour
and trend indicators to classify the market as one of five regimes.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import RegimeThresholds, CONFIG
from scoring import ScoredDistribution

logger = logging.getLogger(__name__)

REGIMES = ["Strong Bull", "Bull", "Neutral", "Bear", "Strong Bear"]


@dataclass
class RegimeResult:
    regime: str
    regime_index: int           # 0 = Strong Bull, 4 = Strong Bear
    return_signal: str
    volatility_signal: str
    distribution_signal: str
    tail_signal: str
    trend_signal: str
    sub_scores: Dict[str, float]
    raw_score: float            # 0-100, higher = more bullish


def classify_regime(
    df_feat: pd.DataFrame,
    scored: List[ScoredDistribution],
    cfg: RegimeThresholds = CONFIG.regime,
) -> RegimeResult:
    """
    Classify the current market regime.

    Parameters
    ----------
    df_feat : feature-engineered DataFrame
    scored  : ranked distribution list (from scoring.score_distributions)
    """
    lr = df_feat["log_return"].dropna().values
    close = df_feat["Close"].dropna()
    ann = 252

    # 1. Return signal (annualised mean return)
    ann_ret = float(np.mean(lr) * ann)
    return_score = _bucket_return(ann_ret, cfg)
    ret_label = _label_return(ann_ret, cfg)

    # 2. Volatility signal (realised annualised vol)
    ann_vol = float(np.std(lr) * np.sqrt(ann))
    vol_score = _bucket_vol(ann_vol, cfg)
    vol_label = _label_vol(ann_vol, cfg)

    # 3. Distribution signal (tail weight from best-fit distribution)
    best = scored[0]
    dist_score, dist_label = _dist_signal(best, lr)

    # 4. Tail signal (skewness + kurtosis)
    skew = float(pd.Series(lr).skew())
    kurt = float(pd.Series(lr).kurt())
    tail_score_val, tail_label = _tail_signal(skew, kurt)

    # 5. Trend signal (close vs 20-day / 50-day MA)
    trend_score, trend_label = _trend_signal(close)

    # Composite bullish score (0-100)
    raw = (
        0.35 * return_score
        + 0.25 * vol_score
        + 0.15 * dist_score
        + 0.10 * tail_score_val
        + 0.15 * trend_score
    )
    raw = float(np.clip(raw, 0, 100))

    # Map to regime
    if raw >= 75:
        regime, idx = "Strong Bull", 0
    elif raw >= 57:
        regime, idx = "Bull", 1
    elif raw >= 43:
        regime, idx = "Neutral", 2
    elif raw >= 25:
        regime, idx = "Bear", 3
    else:
        regime, idx = "Strong Bear", 4

    logger.info("Regime: %s (score=%.1f)", regime, raw)

    return RegimeResult(
        regime=regime,
        regime_index=idx,
        return_signal=ret_label,
        volatility_signal=vol_label,
        distribution_signal=dist_label,
        tail_signal=tail_label,
        trend_signal=trend_label,
        sub_scores={
            "return": round(return_score, 1),
            "volatility": round(vol_score, 1),
            "distribution": round(dist_score, 1),
            "tail": round(tail_score_val, 1),
            "trend": round(trend_score, 1),
        },
        raw_score=round(raw, 1),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_return(ann_ret: float, cfg: RegimeThresholds) -> float:
    if ann_ret >= cfg.strong_bull_return:
        return 90.0
    elif ann_ret >= cfg.bull_return:
        return 70.0
    elif ann_ret >= cfg.bear_return:
        return 50.0
    elif ann_ret >= cfg.strong_bear_return:
        return 25.0
    return 10.0


def _label_return(ann_ret: float, cfg: RegimeThresholds) -> str:
    if ann_ret >= cfg.strong_bull_return:
        return f"High ({ann_ret*100:.1f}% ann.)"
    elif ann_ret >= cfg.bull_return:
        return f"Moderate ({ann_ret*100:.1f}% ann.)"
    elif ann_ret >= cfg.bear_return:
        return f"Low ({ann_ret*100:.1f}% ann.)"
    return f"Negative ({ann_ret*100:.1f}% ann.)"


def _bucket_vol(ann_vol: float, cfg: RegimeThresholds) -> float:
    """Low volatility in a bull market → bullish signal."""
    if ann_vol < cfg.low_vol:
        return 75.0
    elif ann_vol < cfg.high_vol:
        return 50.0
    return 25.0


def _label_vol(ann_vol: float, cfg: RegimeThresholds) -> str:
    if ann_vol < cfg.low_vol:
        return f"Low ({ann_vol*100:.1f}%)"
    elif ann_vol < cfg.high_vol:
        return f"Moderate ({ann_vol*100:.1f}%)"
    return f"High ({ann_vol*100:.1f}%)"


def _dist_signal(best: ScoredDistribution, lr: np.ndarray) -> tuple[float, str]:
    """Heavier-tailed best-fit → lower bullish score."""
    heavy_tails = {"Cauchy", "Stable", "Davies"}
    if best.name in heavy_tails:
        return 35.0, f"{best.name} (heavy tails)"
    elif best.name in {"StudentT", "Laplace", "GeneralizedNormal"}:
        return 55.0, f"{best.name} (medium tails)"
    return 70.0, f"{best.name} (light tails)"


def _tail_signal(skew: float, kurt: float) -> tuple[float, str]:
    """Negative skew + high kurtosis → bearish tail risk."""
    score = 50.0
    label_parts = []
    if skew < -0.5:
        score -= 15
        label_parts.append(f"neg. skew ({skew:.2f})")
    elif skew > 0.5:
        score += 10
        label_parts.append(f"pos. skew ({skew:.2f})")
    if kurt > 3:
        score -= 15
        label_parts.append(f"leptokurtic ({kurt:.2f})")
    elif kurt < 0:
        score += 10
        label_parts.append(f"platykurtic ({kurt:.2f})")
    label = ", ".join(label_parts) if label_parts else "Normal tails"
    return float(np.clip(score, 0, 100)), label


def _trend_signal(close: pd.Series) -> tuple[float, str]:
    """20-day vs 50-day MA trend."""
    if len(close) < 50:
        return 50.0, "Insufficient data"
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    last = float(close.iloc[-1])
    if last > ma20 > ma50:
        return 80.0, "Uptrend (price > MA20 > MA50)"
    elif last > ma50:
        return 60.0, "Mild uptrend (price > MA50)"
    elif last < ma20 < ma50:
        return 20.0, "Downtrend (price < MA20 < MA50)"
    elif last < ma50:
        return 40.0, "Mild downtrend (price < MA50)"
    return 50.0, "Sideways"
