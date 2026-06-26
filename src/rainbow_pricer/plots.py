from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_normalized_prices(prices_scaled: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot normalized index levels."""
    plt.figure(figsize=(10, 5))
    plt.plot(prices_scaled)
    plt.title("Normalized index levels: S_i(0) = 100")
    plt.xlabel("Date")
    plt.ylabel("Normalized level")
    plt.legend(prices_scaled.columns)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(corr: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot empirical correlation matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, interpolation="none", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation matrix of log returns")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_price_convergence(convergence_df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Plot Monte Carlo price convergence."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        convergence_df["n_sims"],
        convergence_df["price"],
        yerr=convergence_df["standard_error"],
        marker="o",
        capsize=4,
    )
    plt.title("Monte Carlo price convergence")
    plt.xlabel("Number of simulations")
    plt.ylabel("Estimated option price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
