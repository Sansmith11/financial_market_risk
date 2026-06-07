"""
confidence.py
Confidence scoring engine: 0-100% confidence in the analysis.

Factors:
  - Distribution agreement (consensus across top-3 fitters)
  - Statistical significance (KS p-values)
  - Sample size
  - Tail consistency (empirical vs theoretical tails)
  - Regime consistency (strength of regime signal)
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from scoring import ScoredDistribution
from regime import RegimeResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    confidence_pct: float           # 0-100
    distribution_agreement: float
    statistical_significance: float
    sample_size_score: float
    tail_consistency: float
    regime_consistency: float
    interpretation: str


def compute_confidence(
    scored: List[ScoredDistribution],
    regime: RegimeResult,
    n_samples: int,
) -> ConfidenceResult:
    """
    Compute an overall confidence percentage.
    """

    # 1. Distribution agreement — gap between #1 and #2
    if len(scored) >= 2:
        gap = scored[0].final_score - scored[1].final_score
        dist_agree = float(np.clip(gap * 4, 0, 100))
    else:
        dist_agree = 50.0

    # 2. Statistical significance from KS p-values
    top3 = scored[:3]
    avg_p = float(np.mean([s.result.ks_pvalue for s in top3]))
    stat_sig = float(np.clip(avg_p * 500, 0, 100))   # p=0.2 → 100

    # 3. Sample size
    if n_samples >= 252:
        sample_score = 100.0
    elif n_samples >= 63:
        sample_score = 60.0 + 40.0 * (n_samples - 63) / (252 - 63)
    elif n_samples >= 30:
        sample_score = 30.0 + 30.0 * (n_samples - 30) / (63 - 30)
    else:
        sample_score = 10.0

    # 4. Tail consistency — average tail-fit score of top 3
    tail_cons = float(np.mean([s.tail_fit_score for s in top3]))

    # 5. Regime consistency — how decisive the raw score is
    raw = regime.raw_score
    # Score is most decisive near 0 or 100
    decisiveness = abs(raw - 50.0) / 50.0
    regime_cons = float(np.clip(decisiveness * 100, 0, 100))

    # Weighted composite
    confidence = (
        0.30 * dist_agree
        + 0.20 * stat_sig
        + 0.20 * sample_score
        + 0.15 * tail_cons
        + 0.15 * regime_cons
    )
    confidence = float(np.clip(confidence, 0, 100))

    interpretation = _interpret(confidence)

    logger.info("Confidence: %.1f%% (%s)", confidence, interpretation)

    return ConfidenceResult(
        confidence_pct=round(confidence, 1),
        distribution_agreement=round(dist_agree, 1),
        statistical_significance=round(stat_sig, 1),
        sample_size_score=round(sample_score, 1),
        tail_consistency=round(tail_cons, 1),
        regime_consistency=round(regime_cons, 1),
        interpretation=interpretation,
    )


def _interpret(pct: float) -> str:
    if pct >= 80:
        return "Very High — strong statistical evidence"
    elif pct >= 65:
        return "High — reliable signal"
    elif pct >= 50:
        return "Moderate — reasonable confidence"
    elif pct >= 35:
        return "Low — treat with caution"
    return "Very Low — insufficient data or poor model fit"
