"""Finnhub market-data wrapper focused on recent stock bars and market status."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FinnhubDataError(RuntimeError):
    """Raised when Finnhub data endpoints cannot satisfy a request."""


@dataclass(frozen=True)
class StockBar:
    """Normalized stock bar returned from Finnhub candle endpoint."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FinnhubStockDataClient:
    """Fetch recent Finnhub bars and quotes while hiding HTTP details from strategies."""

    _BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        *,
        api_key: str,
        bar_timeframe_minutes: int,
        lookback: int,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if bar_timeframe_minutes <= 0:
            raise ValueError("bar_timeframe_minutes must be positive")
        if lookback <= 1:
            raise ValueError("lookback must be greater than one")

        self._api_key = api_key.strip()
        self._bar_timeframe_minutes = bar_timeframe_minutes
        self._lookback = lookback

    def _request_json(self, path: str, params: dict[str, str]) -> dict[str, object]:
        query = urlencode({**params, "token": self._api_key})
        request = Request(
            url=f"{self._BASE_URL}{path}?{query}",
            headers={"Accept": "application/json"},
            method="GET",
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise FinnhubDataError(f"Finnhub request failed with status {exc.code}: {body}") from exc
        except URLError as exc:
            raise FinnhubDataError(f"Failed to reach Finnhub API: {exc.reason}") from exc

        if not isinstance(payload, dict):
            raise FinnhubDataError(f"Unexpected Finnhub payload: {payload!r}")
        return payload

    def _resolution(self) -> str:
        minute = self._bar_timeframe_minutes
        if minute in {1, 5, 15, 30, 60}:
            return str(minute)
        return "5"

    @staticmethod
    def _parse_candle_arrays(payload: dict[str, object], symbol: str) -> list[StockBar]:
        status = str(payload.get("s", "")).lower()
        if status == "no_data":
            return []
        if status != "ok":
            raise FinnhubDataError(f"Unexpected Finnhub candle status for {symbol}: {status!r}")

        ts = payload.get("t")
        opens = payload.get("o")
        highs = payload.get("h")
        lows = payload.get("l")
        closes = payload.get("c")
        volumes = payload.get("v")

        if not all(isinstance(values, list) for values in (ts, opens, highs, lows, closes, volumes)):
            raise FinnhubDataError(f"Unexpected Finnhub candle arrays for {symbol}: {payload!r}")

        length = min(len(ts), len(opens), len(highs), len(lows), len(closes), len(volumes))
        bars: list[StockBar] = []
        for i in range(length):
            try:
                timestamp = datetime.fromtimestamp(int(ts[i]), tz=timezone.utc)
                bar = StockBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i]),
                )
            except (TypeError, ValueError, OSError):
                continue
            bars.append(bar)
        return bars

    def get_bars(self, *, symbol: str, start: datetime, end: datetime) -> list[StockBar]:
        normalized = symbol.strip().upper()
        if not normalized:
            return []
        payload = self._request_json(
            "/stock/candle",
            {
                "symbol": normalized,
                "resolution": self._resolution(),
                "from": str(int(start.astimezone(timezone.utc).timestamp())),
                "to": str(int(end.astimezone(timezone.utc).timestamp())),
            },
        )
        return self._parse_candle_arrays(payload, normalized)

    def get_recent_closes(self, symbol: str) -> list[float]:
        now = datetime.now(timezone.utc)
        span_seconds = (self._lookback + 24) * self._bar_timeframe_minutes * 60
        start = datetime.fromtimestamp(int(now.timestamp()) - span_seconds, tz=timezone.utc)
        bars = self.get_bars(symbol=symbol, start=start, end=now)
        closes = [bar.close for bar in bars if bar.close > 0]
        if len(closes) < self._lookback:
            raise FinnhubDataError(
                f"Finnhub returned only {len(closes)} closes for {symbol}; need at least {self._lookback}."
            )
        return closes[-self._lookback :]

    def get_latest_price(self, symbol: str) -> float:
        normalized = symbol.strip().upper()
        if not normalized:
            raise FinnhubDataError("symbol must be non-empty")
        payload = self._request_json("/quote", {"symbol": normalized})
        price = payload.get("c")
        try:
            value = float(price)
        except (TypeError, ValueError) as exc:
            raise FinnhubDataError(f"Unexpected Finnhub quote payload for {normalized}: {payload!r}") from exc
        if value <= 0:
            raise FinnhubDataError(f"Finnhub quote returned a non-positive price for {normalized}: {value}")
        return value

    def market_is_open(self) -> bool:
        payload = self._request_json("/stock/market-status", {"exchange": "US"})
        return bool(payload.get("isOpen", False))
