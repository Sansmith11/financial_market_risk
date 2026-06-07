"""
distributions.py
Seven distribution models for financial return analysis.
Each class follows a uniform interface for estimation and diagnostics.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, gammaln

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DistributionResult:
    name: str
    params: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: Optional[float]
    tail_risk_score: float           # 0-1, lower = fatter tail risk
    goodness_of_fit: float           # composite 0-100
    pdf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    cdf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_ok: bool = True
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDistribution(ABC):

    name: str = "Base"
    n_params: int = 2

    def fit_and_evaluate(self, data: np.ndarray) -> DistributionResult:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        n = len(data)

        try:
            params = self._estimate(data)
            ll = self._log_likelihood(data, params)
            aic = 2 * self.n_params - 2 * ll
            bic = self.n_params * np.log(n) - 2 * ll

            x_sorted = np.sort(data)
            pdf_vals = self._pdf(x_sorted, params)
            cdf_vals = self._cdf(x_sorted, params)

            ks_stat, ks_p = stats.kstest(data, lambda x: self._cdf(x, params))

            try:
                ad_stat = self._anderson(data, params)
            except Exception:
                ad_stat = None

            tail_score = self._tail_risk_score(data, params)
            gof = self._goodness_of_fit(ks_stat, ks_p, ll, n)

            return DistributionResult(
                name=self.name,
                params=params,
                log_likelihood=round(ll, 4),
                aic=round(aic, 4),
                bic=round(bic, 4),
                ks_statistic=round(ks_stat, 6),
                ks_pvalue=round(ks_p, 6),
                ad_statistic=round(ad_stat, 4) if ad_stat is not None else None,
                tail_risk_score=round(tail_score, 4),
                goodness_of_fit=round(gof, 2),
                pdf_values=pdf_vals,
                cdf_values=cdf_vals,
            )

        except Exception as exc:
            logger.warning("%s fit failed: %s", self.name, exc)
            return DistributionResult(
                name=self.name,
                params={},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                ks_statistic=1.0,
                ks_pvalue=0.0,
                ad_statistic=None,
                tail_risk_score=1.0,
                goodness_of_fit=0.0,
                fitted_ok=False,
                error_msg=str(exc),
            )

    # ---- subclass interface ----

    @abstractmethod
    def _estimate(self, data: np.ndarray) -> Dict[str, float]: ...

    @abstractmethod
    def _pdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray: ...

    @abstractmethod
    def _cdf(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray: ...

    def _log_likelihood(self, data: np.ndarray, params: Dict[str, float]) -> float:
        pdf = self._pdf(data, params)
        pdf = np.clip(pdf, 1e-300, None)
        return float(np.sum(np.log(pdf)))

    def _anderson(self, data: np.ndarray, params: Dict[str, float]) -> float:
        """Simple Anderson-Darling-style statistic."""
        n = len(data)
        x = np.sort(data)
        cdf = np.clip(self._cdf(x, params), 1e-10, 1 - 1e-10)
        i = np.arange(1, n + 1)
        A2 = -n - np.mean((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1])))
        return float(A2)

    def _tail_risk_score(self, data: np.ndarray, params: Dict[str, float]) -> float:
        """Lower score = heavier tails (more tail risk)."""
        q5 = np.quantile(data, 0.05)
        q95 = np.quantile(data, 0.95)
        theo_q5 = self._ppf(0.05, params)
        theo_q95 = self._ppf(0.95, params)
        if not np.isfinite(theo_q5) or not np.isfinite(theo_q95):
            return 0.5
        err = 0.5 * (abs(q5 - theo_q5) + abs(q95 - theo_q95))
        empirical_range = max(abs(q95 - q5), 1e-10)
        return float(np.clip(1.0 - err / empirical_range, 0.0, 1.0))

    def _ppf(self, p: float, params: Dict[str, float]) -> float:
        return np.nan

    def _goodness_of_fit(self, ks: float, ks_p: float, ll: float, n: int) -> float:
        ks_score = max(0.0, (1.0 - ks) * 100)
        p_score = min(ks_p * 200, 50.0)
        return float(np.clip(ks_score * 0.7 + p_score * 0.3, 0, 100))


# ---------------------------------------------------------------------------
# 1. Normal Distribution
# ---------------------------------------------------------------------------

class NormalDistribution(BaseDistribution):
    name = "Normal"
    n_params = 2

    def _estimate(self, data):
        mu, sigma = stats.norm.fit(data)
        return {"mu": mu, "sigma": sigma}

    def _pdf(self, x, p):
        return stats.norm.pdf(x, loc=p["mu"], scale=p["sigma"])

    def _cdf(self, x, p):
        return stats.norm.cdf(x, loc=p["mu"], scale=p["sigma"])

    def _ppf(self, prob, p):
        return float(stats.norm.ppf(prob, loc=p["mu"], scale=p["sigma"]))


# ---------------------------------------------------------------------------
# 2. Student-t Distribution
# ---------------------------------------------------------------------------

class StudentTDistribution(BaseDistribution):
    name = "StudentT"
    n_params = 3

    def _estimate(self, data):
        df, loc, scale = stats.t.fit(data)
        return {"df": max(df, 2.01), "loc": loc, "scale": scale}

    def _pdf(self, x, p):
        return stats.t.pdf(x, df=p["df"], loc=p["loc"], scale=p["scale"])

    def _cdf(self, x, p):
        return stats.t.cdf(x, df=p["df"], loc=p["loc"], scale=p["scale"])

    def _ppf(self, prob, p):
        return float(stats.t.ppf(prob, df=p["df"], loc=p["loc"], scale=p["scale"]))


# ---------------------------------------------------------------------------
# 3. Cauchy Distribution
# ---------------------------------------------------------------------------

class CauchyDistribution(BaseDistribution):
    name = "Cauchy"
    n_params = 2

    def _estimate(self, data):
        loc, scale = stats.cauchy.fit(data)
        return {"loc": loc, "scale": scale}

    def _pdf(self, x, p):
        return stats.cauchy.pdf(x, loc=p["loc"], scale=p["scale"])

    def _cdf(self, x, p):
        return stats.cauchy.cdf(x, loc=p["loc"], scale=p["scale"])

    def _ppf(self, prob, p):
        return float(stats.cauchy.ppf(prob, loc=p["loc"], scale=p["scale"]))


# ---------------------------------------------------------------------------
# 4. Laplace Distribution
# ---------------------------------------------------------------------------

class LaplaceDistribution(BaseDistribution):
    name = "Laplace"
    n_params = 2

    def _estimate(self, data):
        loc, scale = stats.laplace.fit(data)
        return {"loc": loc, "scale": scale}

    def _pdf(self, x, p):
        return stats.laplace.pdf(x, loc=p["loc"], scale=p["scale"])

    def _cdf(self, x, p):
        return stats.laplace.cdf(x, loc=p["loc"], scale=p["scale"])

    def _ppf(self, prob, p):
        return float(stats.laplace.ppf(prob, loc=p["loc"], scale=p["scale"]))


# ---------------------------------------------------------------------------
# 5. Generalized Normal Distribution
# ---------------------------------------------------------------------------

class GeneralizedNormalDistribution(BaseDistribution):
    name = "GeneralizedNormal"
    n_params = 3

    def _estimate(self, data):
        beta, loc, scale = stats.gennorm.fit(data)
        return {"beta": max(beta, 0.1), "loc": loc, "scale": max(scale, 1e-10)}

    def _pdf(self, x, p):
        return stats.gennorm.pdf(x, beta=p["beta"], loc=p["loc"], scale=p["scale"])

    def _cdf(self, x, p):
        return stats.gennorm.cdf(x, beta=p["beta"], loc=p["loc"], scale=p["scale"])

    def _ppf(self, prob, p):
        return float(stats.gennorm.ppf(prob, beta=p["beta"], loc=p["loc"], scale=p["scale"]))


# ---------------------------------------------------------------------------
# 6. Stable Distribution (Lévy alpha-stable)
# ---------------------------------------------------------------------------

class StableDistribution(BaseDistribution):
    name = "Stable"
    n_params = 4

    def _estimate(self, data):
        # scipy stable: alpha, beta, loc, scale
        alpha, beta, loc, scale = stats.levy_stable.fit(data)
        alpha = np.clip(alpha, 0.5, 2.0)
        beta = np.clip(beta, -1.0, 1.0)
        return {"alpha": alpha, "beta": beta, "loc": loc, "scale": max(scale, 1e-10)}

    def _pdf(self, x, p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats.levy_stable.pdf(
                x, alpha=p["alpha"], beta=p["beta"], loc=p["loc"], scale=p["scale"]
            )

    def _cdf(self, x, p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats.levy_stable.cdf(
                x, alpha=p["alpha"], beta=p["beta"], loc=p["loc"], scale=p["scale"]
            )

    def _ppf(self, prob, p):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(stats.levy_stable.ppf(
                    prob, alpha=p["alpha"], beta=p["beta"], loc=p["loc"], scale=p["scale"]
                ))
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# 7. Davies Distribution (custom implementation)
# ---------------------------------------------------------------------------

class DaviesDistribution(BaseDistribution):
    """
    Davies distribution — a flexible heavy-tailed model.
    PDF: f(x) = C * exp(-|x - mu|^alpha / (2 * sigma^alpha))
                * (1 + kappa * |x - mu| / sigma)^(-1/kappa - 1)
    Approximated via Generalized Hyperbolic / skewed-t hybrid.
    Parameters: mu (location), sigma (scale), alpha (shape), kappa (tail).
    """
    name = "Davies"
    n_params = 4

    def _estimate(self, data: np.ndarray) -> Dict[str, float]:
        mu0 = float(np.median(data))
        sigma0 = float(np.std(data))

        def neg_ll(params):
            mu, log_sigma, log_alpha, log_kappa = params
            sigma = np.exp(log_sigma)
            alpha = np.clip(np.exp(log_alpha), 0.2, 4.0)
            kappa = np.clip(np.exp(log_kappa), 0.01, 5.0)
            p = {"mu": mu, "sigma": sigma, "alpha": alpha, "kappa": kappa}
            pdf = self._pdf(data, p)
            pdf = np.clip(pdf, 1e-300, None)
            return -np.sum(np.log(pdf))

        x0 = [mu0, np.log(sigma0 + 1e-10), np.log(1.5), np.log(0.5)]
        try:
            res = minimize(neg_ll, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
            mu, log_sigma, log_alpha, log_kappa = res.x
            return {
                "mu": float(mu),
                "sigma": float(np.exp(log_sigma)),
                "alpha": float(np.clip(np.exp(log_alpha), 0.2, 4.0)),
                "kappa": float(np.clip(np.exp(log_kappa), 0.01, 5.0)),
            }
        except Exception:
            return {"mu": mu0, "sigma": sigma0, "alpha": 1.5, "kappa": 0.5}

    def _pdf(self, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        mu, sigma, alpha, kappa = p["mu"], p["sigma"], p["alpha"], p["kappa"]
        z = np.abs(x - mu) / max(sigma, 1e-10)
        core = np.exp(-0.5 * z ** alpha)
        tail = (1.0 + kappa * z) ** (-(1.0 / kappa + 1.0))
        unnorm = core * tail
        # Normalise numerically
        x_grid = np.linspace(mu - 20 * sigma, mu + 20 * sigma, 2000)
        z_grid = np.abs(x_grid - mu) / max(sigma, 1e-10)
        norm_grid = np.exp(-0.5 * z_grid ** alpha) * (1.0 + kappa * z_grid) ** (-(1.0 / kappa + 1.0))
        norm_const = float(np.trapezoid(norm_grid, x_grid) if hasattr(np, "trapezoid") else np.trapz(norm_grid, x_grid))
        norm_const = max(norm_const, 1e-10)
        return unnorm / norm_const

    def _cdf(self, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        mu, sigma = p["mu"], p["sigma"]
        x_min = mu - 20 * sigma
        cdf_vals = np.zeros_like(x, dtype=float)
        x_grid = np.linspace(x_min, float(np.max(x)) + 20 * sigma, 3000)
        pdf_grid = self._pdf(x_grid, p)
        dx = x_grid[1] - x_grid[0]
        cum = np.cumsum(pdf_grid) * dx
        cum = cum / max(cum[-1], 1e-10)  # normalise to [0,1]
        cdf_vals = np.interp(x, x_grid, cum)
        return np.clip(cdf_vals, 0.0, 1.0)

    def _ppf(self, prob: float, p: Dict[str, float]) -> float:
        mu, sigma = p["mu"], p["sigma"]
        x_grid = np.linspace(mu - 20 * sigma, mu + 20 * sigma, 3000)
        pdf_grid = self._pdf(x_grid, p)
        cum = np.cumsum(pdf_grid) * (x_grid[1] - x_grid[0])
        cum = cum / max(cum[-1], 1e-10)
        idx = np.searchsorted(cum, prob)
        idx = int(np.clip(idx, 0, len(x_grid) - 1))
        return float(x_grid[idx])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DISTRIBUTION_REGISTRY: Dict[str, BaseDistribution] = {
    "Normal": NormalDistribution(),
    "StudentT": StudentTDistribution(),
    "Cauchy": CauchyDistribution(),
    "Laplace": LaplaceDistribution(),
    "GeneralizedNormal": GeneralizedNormalDistribution(),
    "Stable": StableDistribution(),
    "Davies": DaviesDistribution(),
}


def fit_all(data: np.ndarray, names: Optional[list] = None) -> Dict[str, DistributionResult]:
    """Fit all (or selected) distributions and return results dict."""
    registry = DISTRIBUTION_REGISTRY
    if names:
        registry = {k: v for k, v in registry.items() if k in names}

    results: Dict[str, DistributionResult] = {}
    for name, dist in registry.items():
        logger.info("Fitting %s ...", name)
        results[name] = dist.fit_and_evaluate(data)
    return results
