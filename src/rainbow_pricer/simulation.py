from __future__ import annotations

import numpy as np

from .calibration import covariance_from_vol_corr


def simulate_correlated_gbm_paths(
    s0: np.ndarray,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    n_sims: int = 20_000,
    n_steps: int = 250,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate correlated geometric Brownian motion paths.

    Returns an array with shape ``(n_sims, n_steps + 1, n_assets)``.
    """
    s0 = np.asarray(s0, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)

    n_assets = sigma.size
    if s0.size != n_assets or q.size != n_assets:
        raise ValueError("s0, q and sigma must have the same length.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    cov = covariance_from_vol_corr(sigma, corr)
    # Small diagonal jitter improves numerical stability for near-singular matrices.
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n_assets))

    paths = np.empty((n_sims, n_steps + 1, n_assets), dtype=float)
    paths[:, 0, :] = s0

    drift = (r - q - 0.5 * sigma**2) * dt

    for step in range(1, n_steps + 1):
        z = rng.standard_normal((n_sims, n_assets))
        d_w = z @ L.T * sqrt_dt
        paths[:, step, :] = paths[:, step - 1, :] * np.exp(drift + d_w)

    return paths
