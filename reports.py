"""
reports.py
Generates a structured text + dict report from all pipeline components.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import CONFIG
from features import feature_summary
from scoring import ScoredDistribution, ranking_table
from regime import RegimeResult
from confidence import ConfidenceResult

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    df_feat: pd.DataFrame,
    scored: List[ScoredDistribution],
    regime: RegimeResult,
    confidence: ConfidenceResult,
    ticker: str = "NIFTY 500",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Build the master report dict and optionally save to JSON + text.
    """
    feat_summary = feature_summary(df_feat)
    best = scored[0]
    signal = _trading_signal(regime, confidence, best)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "ticker": ticker,
            "data_start": str(df_feat.index.min().date()),
            "data_end": str(df_feat.index.max().date()),
            "pipeline_version": "1.0.0",
        },
        "statistical_summary": feat_summary,
        "best_distribution": {
            "name": best.name,
            "rank": best.rank,
            "final_score": best.final_score,
            "params": best.result.params,
            "log_likelihood": best.result.log_likelihood,
            "aic": best.result.aic,
            "bic": best.result.bic,
            "ks_statistic": best.result.ks_statistic,
            "ks_pvalue": best.result.ks_pvalue,
        },
        "ranking_table": ranking_table(scored),
        "regime": {
            "classification": regime.regime,
            "regime_index": regime.regime_index,
            "raw_score": regime.raw_score,
            "return_signal": regime.return_signal,
            "volatility_signal": regime.volatility_signal,
            "distribution_signal": regime.distribution_signal,
            "tail_signal": regime.tail_signal,
            "trend_signal": regime.trend_signal,
            "sub_scores": regime.sub_scores,
        },
        "confidence": {
            "score_pct": confidence.confidence_pct,
            "interpretation": confidence.interpretation,
            "components": {
                "distribution_agreement": confidence.distribution_agreement,
                "statistical_significance": confidence.statistical_significance,
                "sample_size_score": confidence.sample_size_score,
                "tail_consistency": confidence.tail_consistency,
                "regime_consistency": confidence.regime_consistency,
            },
        },
        "tail_risk": _tail_risk_block(feat_summary, best),
        "trading_signal": signal,
    }

    if output_dir:
        _save_report(report, output_dir, ticker)

    return report


def print_report(report: Dict) -> None:
    """Pretty-print the report to stdout."""
    sep = "=" * 70
    m = report["metadata"]
    s = report["statistical_summary"]
    b = report["best_distribution"]
    reg = report["regime"]
    conf = report["confidence"]
    tr = report["tail_risk"]
    sig = report["trading_signal"]

    print(sep)
    print(f"  NIFTY 500 DISTRIBUTION ANALYSIS PIPELINE")
    print(f"  Ticker: {m['ticker']}  |  {m['data_start']} → {m['data_end']}")
    print(f"  Generated: {m['generated_at']}")
    print(sep)

    print("\n[STATISTICAL SUMMARY]")
    for k, v in s.items():
        print(f"  {k:<30}: {v}")

    print("\n[BEST-FIT DISTRIBUTION]")
    print(f"  Distribution  : {b['name']}")
    print(f"  Final Score   : {b['final_score']}")
    print(f"  Log-Likelihood: {b['log_likelihood']}")
    print(f"  AIC / BIC     : {b['aic']} / {b['bic']}")
    print(f"  KS stat / p   : {b['ks_statistic']} / {b['ks_pvalue']}")
    print(f"  Parameters    : {b['params']}")

    print("\n[DISTRIBUTION RANKING]")
    header = f"  {'Rank':<5}{'Name':<22}{'Score':<10}{'KS':<10}{'AIC':<14}{'BIC'}"
    print(header)
    print("  " + "-" * 65)
    for row in report["ranking_table"]:
        print(f"  {row['rank']:<5}{row['distribution']:<22}{row['final_score']:<10}"
              f"{row['ks_stat']:<10}{row['aic']:<14}{row['bic']}")

    print("\n[MARKET REGIME]")
    print(f"  Classification : {reg['classification']}")
    print(f"  Regime Score   : {reg['raw_score']}/100")
    print(f"  Return Signal  : {reg['return_signal']}")
    print(f"  Volatility     : {reg['volatility_signal']}")
    print(f"  Trend          : {reg['trend_signal']}")
    print(f"  Tail Behaviour : {reg['tail_signal']}")
    print(f"  Distribution   : {reg['distribution_signal']}")

    print("\n[CONFIDENCE]")
    print(f"  Score          : {conf['score_pct']}%")
    print(f"  Interpretation : {conf['interpretation']}")

    print("\n[TAIL RISK]")
    for k, v in tr.items():
        print(f"  {k:<30}: {v}")

    print("\n" + "=" * 70)
    print(f"  *** TRADING SIGNAL: {sig['signal']} ***")
    print(f"  Rationale: {sig['rationale']}")
    print(f"  Risk Note : {sig['risk_note']}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_signal(regime: RegimeResult, conf: ConfidenceResult,
                    best: ScoredDistribution) -> Dict:
    """Derive BUY / SELL / HOLD with rationale."""
    r = regime.regime
    c = conf.confidence_pct

    if r in ("Strong Bull", "Bull") and c >= 50:
        signal = "BUY"
        rationale = (f"{r} regime with {c:.0f}% confidence. "
                     f"Best-fit: {best.name}.")
        risk_note = "Use stop-loss at 2× daily VaR. Regime may shift rapidly."
    elif r in ("Strong Bear", "Bear") and c >= 50:
        signal = "SELL / REDUCE"
        rationale = (f"{r} regime with {c:.0f}% confidence. "
                     f"Heavy-tail distribution ({best.name}) detected.")
        risk_note = "Consider defensive positioning. Tail risk elevated."
    else:
        signal = "HOLD"
        rationale = (f"Neutral or low-confidence signal ({c:.0f}%). "
                     f"Regime: {r}.")
        risk_note = "Wait for clearer directional signal before acting."

    return {"signal": signal, "rationale": rationale, "risk_note": risk_note}


def _tail_risk_block(feat: Dict, best: ScoredDistribution) -> Dict:
    return {
        "VaR_5pct": feat.get("VaR_5pct", "N/A"),
        "CVaR_5pct": feat.get("CVaR_5pct", "N/A"),
        "skewness": feat.get("skewness", "N/A"),
        "excess_kurtosis": feat.get("excess_kurtosis", "N/A"),
        "best_dist_tail_score": best.tail_fit_score,
        "interpretation": (
            "Heavy-tail risk" if best.name in {"Cauchy", "Stable", "Davies"}
            else "Moderate-tail risk" if best.name in {"StudentT", "Laplace"}
            else "Light-tail profile"
        ),
    }


def _save_report(report: Dict, output_dir: str, ticker: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(output_dir) / f"report_{ticker}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved: %s", json_path)
