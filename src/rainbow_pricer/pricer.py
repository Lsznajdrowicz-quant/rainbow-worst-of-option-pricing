from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation import simulate_correlated_gbm_paths


def worst_of_payoff(s_terminal: np.ndarray, strike: float, option_type: str = "call") -> np.ndarray:
    """Calculate the payoff of a Rainbow Worst-of call or put option."""
    worst_terminal = s_terminal.min(axis=1)

    option_type = option_type.lower()
    if option_type == "call":
        return np.maximum(worst_terminal - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - worst_terminal, 0.0)

    raise ValueError("option_type must be either 'call' or 'put'.")


def price_worst_of_option_mc(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    n_sims: int = 20_000,
    n_steps: int = 250,
    strike: float = 100.0,
    seed: int | None = 42,
    return_paths: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Price a Rainbow Worst-of option using Monte Carlo simulation.

    The input price frame is expected to be normalized to a common valuation-date
    level, usually 100 for each underlying asset.
    """
    s0 = prices_common_scale.iloc[-1].values.astype(float)
    paths = simulate_correlated_gbm_paths(
        s0=s0,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        corr=corr,
        n_sims=n_sims,
        n_steps=n_steps,
        seed=seed,
    )

    payoffs = worst_of_payoff(paths[:, -1, :], strike=strike, option_type=option_type)
    discount = np.exp(-r * T)
    price = float(discount * payoffs.mean())
    stderr = float(discount * payoffs.std(ddof=1) / np.sqrt(n_sims))

    if return_paths:
        return price, stderr, paths
    return price, stderr


def price_convergence(
    prices_common_scale: pd.DataFrame,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    option_type: str = "call",
    simulation_grid: list[int] | None = None,
    n_steps: int = 250,
    strike: float = 100.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Calculate prices for multiple simulation counts to inspect Monte Carlo convergence."""
    if simulation_grid is None:
        simulation_grid = [1_000, 2_500, 5_000, 10_000, 20_000, 50_000]

    rows = []
    for n_sims in simulation_grid:
        price, stderr = price_worst_of_option_mc(
            prices_common_scale=prices_common_scale,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            corr=corr,
            option_type=option_type,
            n_sims=n_sims,
            n_steps=n_steps,
            strike=strike,
            seed=seed,
        )
        rows.append({"n_sims": n_sims, "price": price, "standard_error": stderr})

    return pd.DataFrame(rows)
