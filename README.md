# Monte Carlo Pricing of Rainbow Worst-of Option

This repository contains a Monte Carlo pricer for an exotic **Rainbow Worst-of option** written in Python. The option payoff depends on the worst-performing asset from a basket of equity indices. The project was prepared as a quantitative finance portfolio project and demonstrates derivatives pricing, stochastic modelling, Monte Carlo simulation and finite-difference Greeks.

## Project overview

The model prices a European Rainbow Worst-of call/put option based on a basket of three global equity indices:

- S&P 500 (`^GSPC`)
- EuroStoxx 50 (`^STOXX50E`)
- Nikkei 225 (`^N225`)

The option is priced on a normalized index scale. Because the indices have different nominal point levels, each index is scaled to a common level of `100` at the valuation date. This makes the worst-of payoff depend on **relative index performance**, not on raw index levels.

## Financial instrument

For a basket of assets \(S_1, S_2, ..., S_n\), the terminal worst-of value is:

\[
Worst_T = \min(S_1(T), S_2(T), ..., S_n(T))
\]

The payoff is:

\[
Call = \max(Worst_T - K, 0)
\]

\[
Put = \max(K - Worst_T, 0)
\]

In the base setup, the option is at-the-money on the normalized scale:

\[
S_i(0) = 100, \quad K = 100
\]

## Methodology

The project follows the full pricing workflow:

1. Download historical market data with `yfinance`.
2. Calculate log returns.
3. Estimate annualized volatility and empirical correlation matrix.
4. Normalize index levels to a common scale.
5. Simulate correlated GBM paths using Cholesky decomposition.
6. Calculate the Rainbow Worst-of payoff.
7. Discount expected payoff to obtain the option price.
8. Estimate Monte Carlo standard error.
9. Calculate Greeks using finite differences.
10. Check Monte Carlo convergence for different simulation counts.

## Model assumptions

- Underlying assets follow correlated geometric Brownian motion (GBM).
- Volatility and correlation are estimated from historical log returns.
- Risk-free rate is constant.
- Dividends are set to zero in the base version.
- The option is European and can be exercised only at maturity.
- Index levels are normalized to make the worst-of comparison economically meaningful.

## Tech stack

Python | NumPy | pandas | yfinance | Matplotlib | Monte Carlo | GBM | Cholesky decomposition | stochastic processes | Rainbow Worst-of option | finite differences | Greeks | derivatives pricing

## Repository structure

```text
rainbow-worst-of-option-pricing/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── rainbow_worst_of_pricing.ipynb
│   └── original_student_notebook.ipynb
│
├── scripts/
│   └── run_pricing.py
│
├── src/
│   └── rainbow_pricer/
│       ├── __init__.py
│       ├── data.py
│       ├── calibration.py
│       ├── simulation.py
│       ├── pricer.py
│       ├── greeks.py
│       └── plots.py
│
└── results/
    └── .gitkeep
```

## How to run

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/rainbow-worst-of-option-pricing.git
cd rainbow-worst-of-option-pricing
pip install -r requirements.txt
```

Run the end-to-end script:

```bash
PYTHONPATH=src python scripts/run_pricing.py
```

Or open the notebook:

```bash
jupyter notebook notebooks/rainbow_worst_of_pricing.ipynb
```

## Outputs

The script produces:

- estimated worst-of call and put prices,
- Monte Carlo standard errors,
- Delta, Gamma and Vega for each underlying,
- Rho and Theta,
- price convergence table,
- plots saved in the `results/` folder.

## Validation checks

The project includes basic model validation elements:

- Monte Carlo convergence for increasing numbers of simulations,
- standard error reported together with estimated prices,
- empirical correlation matrix used for Cholesky-based path generation,
- common random numbers used in finite-difference Greek estimation to reduce simulation noise.

## Limitations

This project is educational and should not be treated as production-grade pricing software. Main limitations:

- volatility and correlation are historical estimates,
- the model assumes constant volatility and correlation,
- no stochastic volatility or jump component is included,
- dividends are set to zero in the base case,
- liquidity, funding, transaction costs and counterparty risk are not included.

## Author

Lukasz Sznajdrowicz  
Financial Engineering | Quantitative Finance | Machine Learning
