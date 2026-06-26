from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_log_return_parameters(prices: pd.DataFrame, trading_days: int = 250) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Estimate annualized volatilities and correlation matrix from log returns."""
    if prices.shape[0] < 2:
        raise ValueError("At least two price observations are required.")

    log_returns = np.log(prices).diff().dropna()
    annualized_vol = log_returns.std() * np.sqrt(trading_days)
    corr_matrix = log_returns.corr()

    return annualized_vol, corr_matrix, log_returns


def covariance_from_vol_corr(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Build a covariance matrix from annualized volatilities and correlations."""
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)

    if corr.shape != (sigma.size, sigma.size):
        raise ValueError("corr must be a square matrix with size equal to len(sigma).")

    return np.diag(sigma) @ corr @ np.diag(sigma)
