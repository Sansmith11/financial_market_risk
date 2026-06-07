"""
scoring.py
Unified scoring engine: ranks distributions by a weighted composite score.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import ScoringWeights, CONFIG
from distributions import DistributionResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredDistribution:
    name: str
    rank: int
    final_score: float          # 0-100
    ks_score: float
    ll_score: float
    aic_score: float
    bic_score: float
    stability_score: float
    tail_fit_score: float
    result: DistributionResult


def score_distributions(
    results: Dict[str, DistributionResult],
    weights: ScoringWeights = CONFIG.scoring,
) -> List[ScoredDistribution]:
    """
    Compute normalised component scores and a weighted final score for
    every distribution.  Returns a list sorted best-first.
    """
    names = [n for n, r in results.items() if r.fitted_ok]
    if not names:
        raise RuntimeError("No distributions fitted successfully")

    # --- raw arrays ---
    ks_raw = np.array([results[n].ks_statistic for n in names])
    ll_raw = np.array([results[n].log_likelihood for n in names])
    aic_raw = np.array([results[n].aic for n in names])
    bic_raw = np.array([results[n].bic for n in names])
    gof_raw = np.array([results[n].goodness_of_fit for n in names])
    tail_raw = np.array([results[n].tail_risk_score for n in names])

    # --- normalise (0-100, higher = better) ---
    ks_scores = _norm_lower_better(ks_raw)
    ll_scores = _norm_higher_better(ll_raw)
    aic_scores = _norm_lower_better(aic_raw)
    bic_scores = _norm_lower_better(bic_raw)
    stability_scores = gof_raw                     # already 0-100
    tail_scores = tail_raw * 100.0                 # already 0-1

    final_scores = (
        weights.ks_weight * ks_scores
        + weights.ll_weight * ll_scores
        + weights.aic_weight * aic_scores
        + weights.bic_weight * bic_scores
        + weights.stability_weight * stability_scores
        + weights.tail_weight * tail_scores
    )

    scored = []
    for i, name in enumerate(names):
        scored.append(ScoredDistribution(
            name=name,
            rank=0,                         # assigned below
            final_score=round(float(final_scores[i]), 2),
            ks_score=round(float(ks_scores[i]), 2),
            ll_score=round(float(ll_scores[i]), 2),
            aic_score=round(float(aic_scores[i]), 2),
            bic_score=round(float(bic_scores[i]), 2),
            stability_score=round(float(stability_scores[i]), 2),
            tail_fit_score=round(float(tail_scores[i]), 2),
            result=results[name],
        ))

    scored.sort(key=lambda s: s.final_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1

    logger.info("Best distribution: %s (score=%.1f)", scored[0].name, scored[0].final_score)
    return scored


def ranking_table(scored: List[ScoredDistribution]) -> List[Dict]:
    """Serialisable ranking table."""
    rows = []
    for s in scored:
        rows.append({
            "rank": s.rank,
            "distribution": s.name,
            "final_score": s.final_score,
            "ks_score": s.ks_score,
            "ll_score": s.ll_score,
            "aic_score": s.aic_score,
            "bic_score": s.bic_score,
            "stability_score": s.stability_score,
            "tail_fit_score": s.tail_fit_score,
            "log_likelihood": s.result.log_likelihood,
            "aic": s.result.aic,
            "bic": s.result.bic,
            "ks_stat": s.result.ks_statistic,
            "ks_pvalue": s.result.ks_pvalue,
        })
    return rows


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_lower_better(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 50.0)
    return (hi - arr) / (hi - lo) * 100.0


def _norm_higher_better(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 50.0)
    return (arr - lo) / (hi - lo) * 100.0
