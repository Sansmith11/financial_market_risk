"""
pipeline.py
End-to-end orchestrator: chains Data → Features → Distributions →
Scoring → Regime → Confidence → Reports → Visualizations.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import PipelineConfig, CONFIG
from data import load_data, summary_stats
from features import build_features, get_return_series, feature_summary
from distributions import fit_all, DistributionResult
from scoring import score_distributions, ScoredDistribution
from regime import classify_regime, RegimeResult
from confidence import compute_confidence, ConfidenceResult
from reports import generate_report, print_report
from visualization import plot_all

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    ticker: str
    df_raw: pd.DataFrame
    df_feat: pd.DataFrame
    dist_results: Dict[str, DistributionResult]
    scored: List[ScoredDistribution]
    regime: RegimeResult
    confidence: ConfidenceResult
    report: Dict
    chart_paths: List[str]
    elapsed_sec: float


def run_pipeline(
    ticker: Optional[str] = None,
    csv_path: Optional[str] = None,
    period: str = "1y",
    interval: str = "1d",
    cfg: PipelineConfig = CONFIG,
    output_dir: Optional[str] = None,
    save_charts: bool = True,
    show_charts: bool = False,
    print_output: bool = True,
) -> PipelineOutput:
    """
    Run the complete NIFTY 500 distribution analysis pipeline.

    Parameters
    ----------
    ticker      : Yahoo Finance ticker symbol (ignored if csv_path given).
    csv_path    : Local OHLCV CSV (NSE-style date format).
    period      : yfinance period string, e.g. "1y", "2y", "6mo".
    interval    : yfinance interval, e.g. "1d", "1wk".
    cfg         : PipelineConfig object (override for custom settings).
    output_dir  : Directory for reports + charts. Defaults to cfg.output_dir.
    save_charts : Whether to save PNG charts.
    show_charts : Whether to call plt.show() (blocks).
    print_output: Whether to print the formatted report to stdout.

    Returns
    -------
    PipelineOutput dataclass with all intermediate and final results.
    """
    t0 = time.perf_counter()
    out_dir = output_dir or cfg.output_dir
    label = ticker or (Path(csv_path).stem if csv_path else cfg.data.default_ticker)

    logger.info("=" * 60)
    logger.info("Starting pipeline for: %s", label)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. DATA LAYER                                                         #
    # ------------------------------------------------------------------ #
    logger.info("[1/7] Loading data ...")
    df_raw = load_data(
        ticker=ticker,
        csv_path=csv_path or cfg.data.local_csv_path,
        period=period,
        interval=interval,
        cfg=cfg.data,
    )
    data_info = summary_stats(df_raw)
    logger.info("Data: %d rows | %s → %s", data_info["rows"],
                data_info["start"], data_info["end"])

    # ------------------------------------------------------------------ #
    # 2. FEATURE ENGINEERING                                               #
    # ------------------------------------------------------------------ #
    logger.info("[2/7] Engineering features ...")
    df_feat = build_features(df_raw, cfg=cfg.features)
    returns = get_return_series(df_feat)
    logger.info("Features: %d columns | %d return observations",
                df_feat.shape[1], len(returns))

    # ------------------------------------------------------------------ #
    # 3. DISTRIBUTION FITTING                                              #
    # ------------------------------------------------------------------ #
    logger.info("[3/7] Fitting distributions ...")
    dist_results = fit_all(returns, names=cfg.distributions)
    n_fitted = sum(1 for r in dist_results.values() if r.fitted_ok)
    logger.info("Fitted: %d/%d distributions successfully", n_fitted, len(dist_results))

    # ------------------------------------------------------------------ #
    # 4. SCORING                                                           #
    # ------------------------------------------------------------------ #
    logger.info("[4/7] Scoring distributions ...")
    scored = score_distributions(dist_results, weights=cfg.scoring)
    logger.info("Best: %s (%.1f)", scored[0].name, scored[0].final_score)

    # ------------------------------------------------------------------ #
    # 5. REGIME CLASSIFICATION                                             #
    # ------------------------------------------------------------------ #
    logger.info("[5/7] Classifying market regime ...")
    regime = classify_regime(df_feat, scored, cfg=cfg.regime)
    logger.info("Regime: %s (score=%.1f)", regime.regime, regime.raw_score)

    # ------------------------------------------------------------------ #
    # 6. CONFIDENCE SCORING                                                #
    # ------------------------------------------------------------------ #
    logger.info("[6/7] Computing confidence ...")
    confidence = compute_confidence(scored, regime, n_samples=len(returns))
    logger.info("Confidence: %.1f%%", confidence.confidence_pct)

    # ------------------------------------------------------------------ #
    # 7. REPORT + VISUALIZATIONS                                           #
    # ------------------------------------------------------------------ #
    logger.info("[7/7] Generating report + charts ...")
    report = generate_report(
        df_feat, scored, regime, confidence,
        ticker=label,
        output_dir=out_dir,
    )

    chart_paths: List[str] = []
    if save_charts:
        chart_paths = plot_all(
            df_feat, scored, regime, confidence,
            output_dir=out_dir,
            ticker=label,
            show=show_charts,
        )

    if print_output:
        print_report(report)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Pipeline complete in %.2fs", elapsed)

    return PipelineOutput(
        ticker=label,
        df_raw=df_raw,
        df_feat=df_feat,
        dist_results=dist_results,
        scored=scored,
        regime=regime,
        confidence=confidence,
        report=report,
        chart_paths=chart_paths,
        elapsed_sec=elapsed,
    )
