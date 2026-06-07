"""
main.py
Entry point for the NIFTY 500 Distribution Analysis Pipeline.

Usage examples
--------------
# Run on uploaded NIFTY 500 CSV:
    python main.py --csv path/to/NIFTY_500.csv

# Run on Yahoo Finance data (NIFTY 500 index):
    python main.py --ticker "^CRSLDX" --period 1y

# Run multiple tickers:
    python main.py --ticker "RELIANCE.NS" --ticker "TCS.NS"

# Custom output directory:
    python main.py --csv data.csv --output results/
"""

import argparse
import logging
import sys
from pathlib import Path

from config import CONFIG, PipelineConfig
from pipeline import run_pipeline
from data import get_multiple_tickers, load_data, summary_stats
from features import build_features, get_return_series
from distributions import fit_all
from scoring import score_distributions
from regime import classify_regime
from confidence import compute_confidence
from reports import generate_report, print_report
from visualization import plot_all


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIFTY 500 Distribution Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to local OHLCV CSV file")
    parser.add_argument("--ticker", type=str, action="append", default=None,
                        help="Yahoo Finance ticker(s) (repeatable)")
    parser.add_argument("--period", type=str, default="1y",
                        help="yfinance period string, e.g. 1y, 2y, 6mo")
    parser.add_argument("--interval", type=str, default="1d",
                        help="yfinance interval, e.g. 1d, 1wk")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for reports and charts")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--show-charts", action="store_true",
                        help="Display charts interactively (blocks until closed)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity")
    parser.add_argument("--distributions", type=str, nargs="+",
                        default=None,
                        help="Subset of distributions to fit (space-separated). "
                             "Choices: Normal StudentT Cauchy Laplace "
                             "GeneralizedNormal Stable Davies")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    cfg = CONFIG
    if args.distributions:
        cfg.distributions = args.distributions

    # Resolve inputs
    csv_path = args.csv
    tickers = args.ticker or []

    if not csv_path and not tickers:
        # Default: use NIFTY 500 index via Yahoo Finance
        tickers = [cfg.data.default_ticker]
        logger.info("No --csv or --ticker specified. Using default: %s", tickers[0])

    Path(args.output).mkdir(parents=True, exist_ok=True)

    results_list = []

    # ---- Single CSV run ----
    if csv_path:
        logger.info("Running pipeline on CSV: %s", csv_path)
        out = run_pipeline(
            csv_path=csv_path,
            cfg=cfg,
            output_dir=args.output,
            save_charts=not args.no_charts,
            show_charts=args.show_charts,
            print_output=True,
        )
        results_list.append(out)

    # ---- Ticker-based runs ----
    for tk in tickers:
        logger.info("Running pipeline on ticker: %s", tk)
        out = run_pipeline(
            ticker=tk,
            period=args.period,
            interval=args.interval,
            cfg=cfg,
            output_dir=args.output,
            save_charts=not args.no_charts,
            show_charts=args.show_charts,
            print_output=True,
        )
        results_list.append(out)

    # ---- Summary ----
    logger.info("\n%s", "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for out in results_list:
        logger.info(
            "  %-30s | Regime: %-12s | Signal: %-6s | Conf: %.0f%% | Time: %.1fs",
            out.ticker,
            out.regime.regime,
            out.report["trading_signal"]["signal"],
            out.confidence.confidence_pct,
            out.elapsed_sec,
        )
    logger.info("Output saved to: %s", args.output)


if __name__ == "__main__":
    main()
