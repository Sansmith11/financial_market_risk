"""
config.py
Central configuration for the NIFTY 500 Distribution Analysis Pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    default_ticker: str = "^CRSLDX"          # NIFTY 500 index
    default_period: str = "1y"
    default_interval: str = "1d"
    local_csv_path: Optional[str] = None      # override with uploaded CSV
    date_col: str = "Date "
    open_col: str = "Open "
    high_col: str = "High "
    low_col: str = "Low "
    close_col: str = "Close "
    volume_col: str = "Shares Traded "
    min_rows: int = 30
    fill_method: str = "ffill"


@dataclass
class FeatureConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    annualisation_factor: int = 252
    tail_quantile: float = 0.05


@dataclass
class ScoringWeights:
    ks_weight: float = 0.25
    ll_weight: float = 0.25
    aic_weight: float = 0.20
    bic_weight: float = 0.15
    stability_weight: float = 0.10
    tail_weight: float = 0.05


@dataclass
class RegimeThresholds:
    strong_bull_return: float = 0.15
    bull_return: float = 0.05
    bear_return: float = -0.05
    strong_bear_return: float = -0.15
    low_vol: float = 0.12
    high_vol: float = 0.25


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    output_dir: str = "output"
    log_level: str = "INFO"
    distributions: List[str] = field(default_factory=lambda: [
        "Normal", "StudentT", "Cauchy", "Laplace",
        "GeneralizedNormal", "Stable", "Davies"
    ])


# Singleton config
CONFIG = PipelineConfig()
