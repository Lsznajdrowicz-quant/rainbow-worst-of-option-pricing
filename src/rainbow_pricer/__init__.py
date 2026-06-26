"""Monte Carlo pricer for Rainbow Worst-of options."""

from .data import download_close_prices, normalize_to_common_scale
from .calibration import estimate_log_return_parameters
from .simulation import simulate_correlated_gbm_paths
from .pricer import price_worst_of_option_mc, price_convergence
from .greeks import compute_delta_fd, compute_gamma_fd, compute_vega_fd, compute_rho_fd, compute_theta_fd

__all__ = [
    "download_close_prices",
    "normalize_to_common_scale",
    "estimate_log_return_parameters",
    "simulate_correlated_gbm_paths",
    "price_worst_of_option_mc",
    "price_convergence",
    "compute_delta_fd",
    "compute_gamma_fd",
    "compute_vega_fd",
    "compute_rho_fd",
    "compute_theta_fd",
]
