"""
visualization.py
Visualization module: price history, return distributions, Q-Q plots,
distribution comparison, tail risk, regime dashboard.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from scoring import ScoredDistribution
from regime import RegimeResult
from confidence import ConfidenceResult

logger = logging.getLogger(__name__)

# Style
sns.set_theme(style="darkgrid", palette="muted")
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
REGIME_COLORS = {
    "Strong Bull": "#00c851",
    "Bull": "#69c47e",
    "Neutral": "#ffc107",
    "Bear": "#ff6b6b",
    "Strong Bear": "#cc0000",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_all(
    df_feat: pd.DataFrame,
    scored: List[ScoredDistribution],
    regime: RegimeResult,
    confidence: ConfidenceResult,
    output_dir: str = "output",
    ticker: str = "NIFTY 500",
    show: bool = False,
) -> List[str]:
    """
    Generate all charts. Returns list of saved file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lr = df_feat["log_return"].dropna().values
    close = df_feat["Close"].dropna()

    paths = []
    paths.append(_plot_price_history(close, ticker, output_dir, show))
    paths.append(_plot_return_distribution(lr, ticker, output_dir, show))
    paths.append(_plot_hist_vs_pdf(lr, scored, ticker, output_dir, show))
    paths.append(_plot_qq(lr, scored[:3], ticker, output_dir, show))
    paths.append(_plot_distribution_comparison(lr, scored, ticker, output_dir, show))
    paths.append(_plot_tail_risk(lr, scored, ticker, output_dir, show))
    paths.append(_plot_regime_dashboard(df_feat, scored, regime, confidence,
                                        ticker, output_dir, show))
    logger.info("Saved %d charts to %s", len(paths), output_dir)
    return paths


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def _plot_price_history(close: pd.Series, ticker: str,
                        output_dir: str, show: bool) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{ticker} — Price History", fontsize=14, fontweight="bold")

    # Price
    axes[0].plot(close.index, close.values, color="#3a7bd5", lw=1.5)
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        axes[0].plot(close.index, ma20, "--", color="#e74c3c", lw=1, label="MA-20")
    if len(close) >= 50:
        ma50 = close.rolling(50).mean()
        axes[0].plot(close.index, ma50, "--", color="#f39c12", lw=1, label="MA-50")
    axes[0].set_ylabel("Price (₹)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Close Price with Moving Averages")

    # Volume (if available)
    log_ret = np.log(close / close.shift(1)).dropna()
    axes[1].bar(log_ret.index, log_ret.values,
                color=np.where(log_ret.values >= 0, "#27ae60", "#e74c3c"), alpha=0.6, width=1)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Daily Log Return")
    axes[1].set_title("Daily Log Returns")

    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_price_history.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_return_distribution(lr: np.ndarray, ticker: str,
                               output_dir: str, show: bool) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — Return Distribution", fontsize=14, fontweight="bold")

    # Histogram
    axes[0].hist(lr, bins=50, density=True, alpha=0.7, color="#3a7bd5", edgecolor="white")
    mu, sigma = float(np.mean(lr)), float(np.std(lr))
    x = np.linspace(lr.min(), lr.max(), 300)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), "r--", lw=2, label="Normal")
    axes[0].set_xlabel("Log Return")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Return Histogram")
    axes[0].legend()

    # Box plot + violin
    parts = axes[1].violinplot(lr, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#3a7bd5")
        pc.set_alpha(0.7)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Log Returns"])
    axes[1].set_ylabel("Log Return")
    axes[1].set_title("Return Distribution (Violin)")

    # Annotations
    skew = float(pd.Series(lr).skew())
    kurt = float(pd.Series(lr).kurt())
    axes[0].text(0.02, 0.97, f"μ={mu:.4f}\nσ={sigma:.4f}\nSkew={skew:.2f}\nKurt={kurt:.2f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_return_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_hist_vs_pdf(lr: np.ndarray, scored: List[ScoredDistribution],
                      ticker: str, output_dir: str, show: bool) -> str:
    top5 = scored[:5]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(lr, bins=60, density=True, alpha=0.4, color="steelblue",
            edgecolor="none", label="Empirical")

    x = np.linspace(np.quantile(lr, 0.001), np.quantile(lr, 0.999), 400)
    for i, s in enumerate(top5):
        if s.result.fitted_ok and len(s.result.pdf_values) > 0:
            # Recompute on common x grid
            try:
                dist_obj = _get_dist_obj(s.name)
                if dist_obj is not None:
                    pdf = dist_obj._pdf(x, s.result.params)
                    ax.plot(x, pdf, lw=2, color=COLORS[i],
                            label=f"#{s.rank} {s.name} (score={s.final_score:.1f})")
            except Exception:
                pass

    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_title(f"{ticker} — Histogram vs Fitted PDFs")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_hist_vs_pdf.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_qq(lr: np.ndarray, top3: List[ScoredDistribution],
             ticker: str, output_dir: str, show: bool) -> str:
    n = min(len(top3), 3)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{ticker} — Q-Q Plots (Top {n} Distributions)",
                 fontsize=13, fontweight="bold")

    lr_sorted = np.sort(lr)
    empirical_q = np.linspace(0.01, 0.99, len(lr_sorted))

    for i, s in enumerate(top3[:n]):
        ax = axes[i]
        try:
            dist_obj = _get_dist_obj(s.name)
            if dist_obj is None:
                raise ValueError("Unknown dist")
            theo_q = np.array([dist_obj._ppf(p, s.result.params) for p in empirical_q])
            mask = np.isfinite(theo_q)
            ax.scatter(theo_q[mask], lr_sorted[mask], s=8, alpha=0.5, color=COLORS[i])
            lo = min(theo_q[mask].min(), lr_sorted[mask].min())
            hi = max(theo_q[mask].max(), lr_sorted[mask].max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Empirical Quantiles")
            ax.set_title(f"#{s.rank} {s.name}")
            ax.legend(fontsize=8)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Q-Q failed:\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_title(s.name)

    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_qq_plots.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_distribution_comparison(lr: np.ndarray, scored: List[ScoredDistribution],
                                   ticker: str, output_dir: str, show: bool) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{ticker} — Distribution Comparison", fontsize=14, fontweight="bold")

    # Score bar chart
    names = [s.name for s in scored]
    scores = [s.final_score for s in scored]
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    axes[0].barh(names[::-1], scores[::-1], color=bar_colors[::-1], alpha=0.8)
    axes[0].set_xlabel("Final Score (0-100)")
    axes[0].set_title("Distribution Ranking by Score")
    for j, (n, sc) in enumerate(zip(names[::-1], scores[::-1])):
        axes[0].text(sc + 0.5, j, f"{sc:.1f}", va="center", fontsize=9)

    # KS statistic comparison
    ks_vals = [s.result.ks_statistic for s in scored]
    axes[1].barh(names[::-1], [s * 100 for s in ks_vals[::-1]],
                 color=bar_colors[::-1], alpha=0.8)
    axes[1].set_xlabel("KS Statistic × 100 (lower = better)")
    axes[1].set_title("KS Test Statistic Comparison")

    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_distribution_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_tail_risk(lr: np.ndarray, scored: List[ScoredDistribution],
                    ticker: str, output_dir: str, show: bool) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{ticker} — Tail Risk Analysis", fontsize=14, fontweight="bold")

    # Left tail zoom
    var5 = float(np.quantile(lr, 0.05))
    cvar5 = float(lr[lr <= var5].mean())
    left_tail = lr[lr <= np.quantile(lr, 0.15)]

    axes[0].hist(left_tail, bins=30, density=True, color="#e74c3c", alpha=0.7, label="Left tail")
    axes[0].axvline(var5, color="black", lw=2, linestyle="--", label=f"VaR 5% = {var5:.4f}")
    axes[0].axvline(cvar5, color="purple", lw=2, linestyle=":", label=f"CVaR 5% = {cvar5:.4f}")
    axes[0].set_xlabel("Log Return")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Left Tail Distribution")
    axes[0].legend(fontsize=9)

    # Right tail (log scale)
    right_tail = lr[lr >= np.quantile(lr, 0.85)]
    axes[1].hist(right_tail, bins=30, density=True, color="#27ae60", alpha=0.7, label="Right tail")
    axes[1].set_xlabel("Log Return")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Right Tail Distribution")

    # Overlay best dist tail
    best = scored[0]
    try:
        dist_obj = _get_dist_obj(best.name)
        if dist_obj:
            x_l = np.linspace(lr.min(), var5, 200)
            pdf_l = dist_obj._pdf(x_l, best.result.params)
            axes[0].plot(x_l, pdf_l, "b-", lw=2, label=f"{best.name} PDF")
            axes[0].legend(fontsize=9)
    except Exception:
        pass

    plt.tight_layout()
    path = str(Path(output_dir) / f"{ticker}_tail_risk.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


def _plot_regime_dashboard(df_feat: pd.DataFrame, scored: List[ScoredDistribution],
                            regime: RegimeResult, confidence: ConfidenceResult,
                            ticker: str, output_dir: str, show: bool) -> str:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{ticker} — Regime Dashboard", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Regime meter (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    _gauge(ax1, regime.raw_score, regime.regime, REGIME_COLORS.get(regime.regime, "gray"))

    # 2. Sub-score radar (top-middle)
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    _radar(ax2, regime.sub_scores)

    # 3. Confidence gauge (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    _gauge(ax3, confidence.confidence_pct, f"Confidence\n{confidence.confidence_pct:.0f}%", "#3a7bd5")

    # 4. Rolling volatility (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    if "vol_21d" in df_feat.columns:
        vol = df_feat["vol_21d"].dropna()
        ax4.plot(vol.index, vol.values * 100, color="#e74c3c", lw=1.5)
        ax4.set_title("21-day Rolling Volatility (%)")
        ax4.set_ylabel("Ann. Vol %")

    # 5. Drawdown (bottom-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    if "drawdown" in df_feat.columns:
        dd = df_feat["drawdown"].dropna()
        ax5.fill_between(dd.index, dd.values * 100, 0, color="#e74c3c", alpha=0.5)
        ax5.set_title("Drawdown (%)")
        ax5.set_ylabel("Drawdown %")

    # 6. Scoring bar (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    top5 = scored[:5]
    ax6.barh([s.name for s in top5[::-1]],
             [s.final_score for s in top5[::-1]],
             color=[COLORS[i] for i in range(len(top5))][::-1], alpha=0.8)
    ax6.set_title("Top-5 Distribution Scores")
    ax6.set_xlabel("Score")

    path = str(Path(output_dir) / f"{ticker}_regime_dashboard.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _gauge(ax, value: float, label: str, color: str) -> None:
    """Simple semicircle gauge."""
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), "lightgray", lw=10)
    frac = float(np.clip(value / 100, 0, 1))
    theta_v = np.linspace(np.pi, np.pi * (1 - frac), 300)
    ax.plot(np.cos(theta_v), np.sin(theta_v), color=color, lw=10)
    ax.text(0, -0.15, f"{value:.0f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color)
    ax.text(0, -0.45, label, ha="center", va="center", fontsize=10)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")


def _radar(ax, sub_scores: Dict) -> None:
    """Radar / spider chart for sub-scores."""
    categories = list(sub_scores.keys())
    values = [sub_scores[c] for c in categories]
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)] + [0]
    values_plot = values + [values[0]]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in categories], size=8)
    ax.plot(angles, values_plot, "o-", lw=2, color="#3a7bd5")
    ax.fill(angles, values_plot, alpha=0.25, color="#3a7bd5")
    ax.set_ylim(0, 100)
    ax.set_title("Regime Sub-scores", size=10, pad=15)


def _get_dist_obj(name: str):
    """Lazy import to avoid circular import."""
    from distributions import DISTRIBUTION_REGISTRY
    return DISTRIBUTION_REGISTRY.get(name)
