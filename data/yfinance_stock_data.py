"""YFinance stock data wrapper for strategy runtime signals."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


class YFinanceStockDataError(RuntimeError):
    """Raised when yfinance cannot satisfy a stock-data request."""


class YFinanceStockDataClient:
    """Fetch recent OHLCV closes from yfinance for configured runtime symbols."""

    def __init__(self, *, bar_timeframe_minutes: int, lookback: int) -> None:
        if bar_timeframe_minutes <= 0:
            raise ValueError("bar_timeframe_minutes must be positive")
        if lookback <= 1:
            raise ValueError("lookback must be greater than one")
        self._bar_timeframe_minutes = bar_timeframe_minutes
        self._lookback = lookback

    def _interval(self) -> str:
        minute = self._bar_timeframe_minutes
        if minute in {1, 2, 5, 15, 30, 60, 90}:
            return f"{minute}m"
        return "5m"

    def _period(self) -> str:
        # Intraday limits on yfinance require bounded periods for minute intervals.
        need_days = max(2, int(((self._lookback + 24) * self._bar_timeframe_minutes) / (60 * 24)) + 1)
        if need_days <= 7:
            return "7d"
        if need_days <= 30:
            return "1mo"
        return "3mo"

    def _download(self, symbol: str) -> pd.DataFrame:
        raw = yf.download(
            symbol.strip().upper(),
            period=self._period(),
            interval=self._interval(),
            auto_adjust=True,
            prepost=True,
            progress=False,
        )
        if raw is None or raw.empty:
            raise YFinanceStockDataError(f"{symbol} returned no rows from yfinance.")

        frame = raw.copy()
        frame.columns = [
            col.lower() if isinstance(col, str) else str(col[0]).lower()
            for col in frame.columns
        ]
        if "close" not in frame.columns:
            raise YFinanceStockDataError(f"{symbol} missing close column in yfinance payload.")
        frame = frame.dropna(subset=["close"])
        if frame.empty:
            raise YFinanceStockDataError(f"{symbol} returned no usable close rows after cleanup.")
        return frame

    def get_recent_closes(self, symbol: str) -> list[float]:
        frame = self._download(symbol)
        closes = [float(value) for value in frame["close"].tolist() if float(value) > 0]
        if len(closes) < self._lookback:
            raise YFinanceStockDataError(
                f"yfinance returned only {len(closes)} closes for {symbol}; need at least {self._lookback}."
            )
        return closes[-self._lookback :]

    def get_latest_price(self, symbol: str) -> float:
        frame = self._download(symbol)
        last_close = float(frame["close"].iloc[-1])
        if last_close <= 0:
            raise YFinanceStockDataError(f"yfinance returned non-positive last close for {symbol}: {last_close}")
        return last_close

    def market_is_open(self) -> bool:
        # Best-effort approximation for dashboard chip behavior only.
        now_et = datetime.now(ZoneInfo("America/New_York"))
        weekday = now_et.weekday()
        if weekday >= 5:
            return False
        # 09:30-16:00 ET regular session window.
        minutes = now_et.hour * 60 + now_et.minute
        return (9 * 60 + 30) <= minutes <= (16 * 60)
