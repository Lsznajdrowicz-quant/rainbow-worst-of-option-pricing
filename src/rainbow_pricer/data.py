from __future__ import annotations

from datetime import datetime
from typing import Mapping

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = {
    "S&P 500": ["^GSPC", "SPY"],
    "EuroStoxx 50": ["^STOXX50E", "FEZ"],
    "Nikkei 225": ["^N225", "EWJ"],
}


def download_close_prices(
    tickers: dict[str, list[str]] | None = None,
    start_date: str | pd.Timestamp = "2023-01-01",
    end_date: str | pd.Timestamp = "2026-06-25",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download close prices for selected indices or ETF fallbacks.

    Yahoo Finance index tickers such as ^GSPC may sometimes fail.
    Therefore each asset can have a list of fallback tickers.
    """

    tickers = tickers or DEFAULT_TICKERS
    price_series = {}

    for name, symbols in tickers.items():
        if isinstance(symbols, str):
            symbols = [symbols]

        downloaded = None
        used_symbol = None

        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                    end=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=auto_adjust,
                )

                if df.empty:
                    print(f"Warning: no data for {name} ({symbol}). Trying fallback...")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                price_col = "Close" if "Close" in df.columns else "Adj Close"

                if price_col not in df.columns:
                    print(f"Warning: no Close/Adj Close column for {name} ({symbol}).")
                    continue

                downloaded = df[price_col].dropna()
                used_symbol = symbol
                break

            except Exception as exc:
                print(f"Warning: failed to download {name} ({symbol}): {exc}")

        if downloaded is None or downloaded.empty:
            raise ValueError(
                f"No data downloaded for {name}. Tried tickers: {symbols}"
            )

        print(f"Downloaded {name} using ticker: {used_symbol}")
        price_series[name] = downloaded

    prices = pd.concat(price_series, axis=1).dropna()

    if prices.empty:
        raise ValueError("Downloaded data is empty after aligning dates.")

    return prices


def normalize_to_common_scale(prices: pd.DataFrame, base_level: float = 100.0) -> pd.DataFrame:
    """Normalize all assets to the same value at the valuation date.

    For a worst-of payoff, comparing raw index point levels is misleading because
    indices have different nominal scales. This transformation makes the payoff
    depend on relative index performance rather than raw index values.
    """
    if prices.empty:
        raise ValueError("prices cannot be empty.")
    return prices.div(prices.iloc[-1]).mul(base_level)
