from __future__ import annotations

import numpy as np
import pandas as pd

from .pricer import price_worst_of_option_mc


def _bump_array(x: np.ndarray, idx: int, h: float) -> np.ndarray:
    y = x.copy()
    y[idx] += h
    return y


def compute_delta_fd(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    relative_eps: float = 1e-3,
    strike: float = 100.0,
    seed: int | None = 42,
) -> pd.Series:
    """Estimate delta for each underlying with central finite differences."""
    s0 = prices_common_scale.iloc[-1].values.astype(float)
    deltas = np.zeros_like(s0)

    for i in range(s0.size):
        h = relative_eps * s0[i]
        df_up = prices_common_scale.copy()
        df_down = prices_common_scale.copy()
        df_up.iloc[-1] = _bump_array(s0, i, +h)
        df_down.iloc[-1] = _bump_array(s0, i, -h)

        c_up, _ = price_worst_of_option_mc(df_up, T, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
        c_down, _ = price_worst_of_option_mc(df_down, T, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
        deltas[i] = (c_up - c_down) / (2 * h)

    return pd.Series(deltas, index=prices_common_scale.columns, name="Delta")


def compute_gamma_fd(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    relative_eps: float = 1e-3,
    strike: float = 100.0,
    seed: int | None = 42,
) -> pd.Series:
    """Estimate gamma for each underlying with central finite differences."""
    s0 = prices_common_scale.iloc[-1].values.astype(float)
    gammas = np.zeros_like(s0)
    c0, _ = price_worst_of_option_mc(prices_common_scale, T, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)

    for i in range(s0.size):
        h = relative_eps * s0[i]
        df_up = prices_common_scale.copy()
        df_down = prices_common_scale.copy()
        df_up.iloc[-1] = _bump_array(s0, i, +h)
        df_down.iloc[-1] = _bump_array(s0, i, -h)

        c_up, _ = price_worst_of_option_mc(df_up, T, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
        c_down, _ = price_worst_of_option_mc(df_down, T, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
        gammas[i] = (c_up - 2 * c0 + c_down) / (h**2)

    return pd.Series(gammas, index=prices_common_scale.columns, name="Gamma")


def compute_vega_fd(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    h: float = 1e-3,
    strike: float = 100.0,
    seed: int | None = 42,
) -> pd.Series:
    """Estimate vega for each underlying by bumping annualized volatility."""
    vegas = np.zeros_like(sigma, dtype=float)

    for i in range(sigma.size):
        sigma_up = _bump_array(sigma, i, +h)
        sigma_down = _bump_array(sigma, i, -h)
        c_up, _ = price_worst_of_option_mc(prices_common_scale, T, r, q, sigma_up, corr, option_type, n_sims, n_steps, strike, seed)
        c_down, _ = price_worst_of_option_mc(prices_common_scale, T, r, q, sigma_down, corr, option_type, n_sims, n_steps, strike, seed)
        vegas[i] = (c_up - c_down) / (2 * h)

    return pd.Series(vegas, index=prices_common_scale.columns, name="Vega")


def compute_rho_fd(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    h: float = 1e-4,
    strike: float = 100.0,
    seed: int | None = 42,
) -> float:
    """Estimate rho with central finite differences."""
    c_up, _ = price_worst_of_option_mc(prices_common_scale, T, r + h, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
    c_down, _ = price_worst_of_option_mc(prices_common_scale, T, r - h, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
    return float((c_up - c_down) / (2 * h))


def compute_theta_fd(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    h: float = 1 / 250,
    strike: float = 100.0,
    seed: int | None = 42,
) -> float:
    """Estimate theta with central finite differences around maturity T."""
    c_up, _ = price_worst_of_option_mc(prices_common_scale, T + h, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
    c_down, _ = price_worst_of_option_mc(prices_common_scale, T - h, r, q, sigma, corr, option_type, n_sims, n_steps, strike, seed)
    return float((c_up - c_down) / (2 * h))
