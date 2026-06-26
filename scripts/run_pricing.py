from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rainbow_pricer import (
    download_close_prices,
    normalize_to_common_scale,
    estimate_log_return_parameters,
    price_worst_of_option_mc,
    price_convergence,
    compute_delta_fd,
    compute_gamma_fd,
    compute_vega_fd,
    compute_rho_fd,
    compute_theta_fd,
)
from rainbow_pricer.plots import plot_correlation_matrix, plot_normalized_prices, plot_price_convergence


VALUATION_DATE = "2026-06-25"
START_DATE = "2023-06-25"
BASE_LEVEL = 100.0
T = 1.0
RISK_FREE_RATE = 0.0537
N_SIMS = 20_000
N_STEPS = 250
SEED = 42


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    prices = download_close_prices(start_date=START_DATE, end_date=VALUATION_DATE)
    prices_scaled = normalize_to_common_scale(prices, base_level=BASE_LEVEL)
    sigma, corr, log_returns = estimate_log_return_parameters(prices)

    q = np.zeros(len(sigma))

    call_price, call_se = price_worst_of_option_mc(
        prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values,
        option_type="call", n_sims=N_SIMS, n_steps=N_STEPS, strike=BASE_LEVEL, seed=SEED,
    )
    put_price, put_se = price_worst_of_option_mc(
        prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values,
        option_type="put", n_sims=N_SIMS, n_steps=N_STEPS, strike=BASE_LEVEL, seed=SEED,
    )

    greeks = pd.concat([
        compute_delta_fd(prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values, seed=SEED),
        compute_gamma_fd(prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values, seed=SEED),
        compute_vega_fd(prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values, seed=SEED),
    ], axis=1)

    rho = compute_rho_fd(prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values, seed=SEED)
    theta = compute_theta_fd(prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values, seed=SEED)

    convergence = price_convergence(
        prices_scaled, T, RISK_FREE_RATE, q, sigma.values, corr.values,
        option_type="call", strike=BASE_LEVEL, seed=SEED,
    )

    print("Rainbow Worst-of option pricing on normalized index scale")
    print(f"Call price: {call_price:.4f} | SE: {call_se:.4f}")
    print(f"Put price:  {put_price:.4f} | SE: {put_se:.4f}")
    print("\nGreeks:")
    print(greeks)
    print(f"\nRho:   {rho:.6f}")
    print(f"Theta: {theta:.6f}")

    greeks.to_csv(results_dir / "greeks.csv")
    convergence.to_csv(results_dir / "price_convergence.csv", index=False)

    plot_normalized_prices(prices_scaled, results_dir / "normalized_prices.png")
    plot_correlation_matrix(corr, results_dir / "correlation_matrix.png")
    plot_price_convergence(convergence, results_dir / "price_convergence.png")


if __name__ == "__main__":
    main()
