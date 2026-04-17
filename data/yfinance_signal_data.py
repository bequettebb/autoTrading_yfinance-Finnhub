"""YFinance-based signal data helpers for model-driven strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import yfinance as yf


class SignalDataError(RuntimeError):
    """Raised when the external signal-data fetch cannot satisfy a request."""


@dataclass(frozen=True)
class SignalDataBundle:
    """Intraday price inputs required by the SOXX directional model."""

    signal_frame: pd.DataFrame
    macro_frames: Mapping[str, pd.Series]


class YFinanceSignalDataClient:
    """Fetch extended-hours 5-minute bars for the model's signal universe."""

    def __init__(self, history_period: str = "10d", interval: str = "5m") -> None:
        """Store the lookback window used for signal generation."""

        self._history_period = history_period
        self._interval = interval

    @staticmethod
    def _normalize_downloaded_frame(
        frame: pd.DataFrame,
        *,
        symbol: str,
        filter_to_extended_session: bool,
    ) -> pd.DataFrame:
        """Normalize yfinance output to lower-case OHLCV columns on ET-naive timestamps."""

        if frame.empty:
            raise SignalDataError(f"{symbol} returned no data from yfinance.")

        normalized = frame.copy()
        normalized.columns = [
            column.lower() if isinstance(column, str) else str(column[0]).lower()
            for column in normalized.columns
        ]

        if normalized.index.tz is None:
            normalized.index = normalized.index.tz_localize("UTC")
        normalized.index = normalized.index.tz_convert("US/Eastern")
        if filter_to_extended_session:
            normalized = normalized.between_time("04:00", "20:00")
        normalized.index = normalized.index.tz_localize(None)
        normalized = normalized.dropna(subset=["close"])
        if normalized.empty:
            raise SignalDataError(f"{symbol} returned no usable bars after normalization.")
        return normalized

    def get_stock_frame(self, symbol: str, period: str | None = None) -> pd.DataFrame:
        """Fetch one stock's 5-minute OHLCV bars including pre/post-market trading."""

        raw = yf.download(
            symbol,
            period=period or self._history_period,
            interval=self._interval,
            prepost=True,
            auto_adjust=True,
            progress=False,
        )
        return self._normalize_downloaded_frame(
            raw,
            symbol=symbol,
            filter_to_extended_session=True,
        )

    def get_macro_close_series(self, symbol: str, period: str | None = None) -> pd.Series:
        """Fetch one macro proxy and return the close series on ET-naive timestamps."""

        raw = yf.download(
            symbol,
            period=period or self._history_period,
            interval=self._interval,
            prepost=True,
            auto_adjust=True,
            progress=False,
        )
        normalized = self._normalize_downloaded_frame(
            raw,
            symbol=symbol,
            filter_to_extended_session=False,
        )
        return normalized["close"]

    def get_signal_bundle(
        self,
        *,
        signal_symbol: str,
        macro_tickers: Mapping[str, str],
        period: str | None = None,
    ) -> SignalDataBundle:
        """Fetch the full model input bundle for the latest inference cycle."""

        signal_frame = self.get_stock_frame(signal_symbol, period=period)
        macro_frames = {
            name: self.get_macro_close_series(ticker, period=period)
            for name, ticker in macro_tickers.items()
        }
        return SignalDataBundle(signal_frame=signal_frame, macro_frames=macro_frames)
