"""Load the latest SOXX holdings so news can focus on the fund's largest constituents."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOXX_HOLDINGS_CSV_URL = (
    "https://www.ishares.com/us/products/239705/fund/1467271812596.ajax"
    "?dataType=fund&fileName=SOXX_holdings&fileType=csv"
)
_CACHE_TTL_SECONDS = 60 * 60 * 6


class SoxxHoldingsError(RuntimeError):
    """Raised when the latest SOXX holdings cannot be loaded from iShares."""


@dataclass(frozen=True)
class SoxxHolding:
    """One SOXX constituent used to focus the news overlay."""

    symbol: str
    name: str
    weight_pct: float


DEFAULT_SOXX_TOP_HOLDINGS: tuple[SoxxHolding, ...] = (
    SoxxHolding(symbol="AVGO", name="BROADCOM INC", weight_pct=8.44),
    SoxxHolding(symbol="NVDA", name="NVIDIA CORP", weight_pct=8.11),
    SoxxHolding(symbol="MU", name="MICRON TECHNOLOGY INC", weight_pct=7.39),
    SoxxHolding(symbol="AMD", name="ADVANCED MICRO DEVICES INC", weight_pct=6.66),
    SoxxHolding(symbol="AMAT", name="APPLIED MATERIAL INC", weight_pct=5.73),
    SoxxHolding(symbol="MRVL", name="MARVELL TECHNOLOGY INC", weight_pct=5.40),
    SoxxHolding(symbol="INTC", name="INTEL CORPORATION CORP", weight_pct=4.68),
    SoxxHolding(symbol="MPWR", name="MONOLITHIC POWER SYSTEMS INC", weight_pct=4.21),
    SoxxHolding(symbol="KLAC", name="KLA CORP", weight_pct=4.10),
    SoxxHolding(symbol="TER", name="TERADYNE INC", weight_pct=4.04),
)


def default_soxx_top_holdings(limit: int = 10) -> tuple[SoxxHolding, ...]:
    """Return a stable fallback holdings list if the live iShares source is unavailable."""

    return DEFAULT_SOXX_TOP_HOLDINGS[: max(0, limit)]


class SoxxHoldingsClient:
    """Fetch and cache the latest SOXX constituent weights from iShares."""

    _cache: tuple[float, tuple[SoxxHolding, ...]] | None = None

    def __init__(self, *, holdings_csv_url: str = SOXX_HOLDINGS_CSV_URL) -> None:
        """Store the iShares CSV endpoint so tests can inject a fixture URL."""

        self._holdings_csv_url = holdings_csv_url

    def _download_csv(self) -> str:
        """Load the raw holdings CSV text from iShares."""

        request = Request(
            url=self._holdings_csv_url,
            headers={
                "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.1",
                "User-Agent": "Mozilla/5.0",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8-sig")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SoxxHoldingsError(
                f"SOXX holdings request failed with status {exc.code}: {body}"
            ) from exc
        except URLError as exc:
            raise SoxxHoldingsError(f"Failed to reach iShares holdings source: {exc.reason}") from exc

    @staticmethod
    def _parse_csv_rows(csv_text: str) -> tuple[SoxxHolding, ...]:
        """Parse the holdings CSV and keep only equity constituents sorted by weight."""

        lines = csv_text.splitlines()
        header_index = next(
            (index for index, line in enumerate(lines) if line.startswith("Ticker,Name,Sector,")),
            None,
        )
        if header_index is None:
            raise SoxxHoldingsError("SOXX holdings CSV did not include the expected holdings table header.")

        reader = csv.DictReader(lines[header_index:])
        holdings: list[SoxxHolding] = []
        for row in reader:
            if not isinstance(row, dict):
                continue

            symbol = str(row.get("Ticker", "")).strip().upper()
            name = str(row.get("Name", "")).strip()
            asset_class = str(row.get("Asset Class", "")).strip().lower()
            weight_raw = str(row.get("Weight (%)", "")).strip().replace(",", "")
            if not symbol or asset_class != "equity":
                continue
            try:
                weight_pct = float(weight_raw)
            except ValueError:
                continue

            holdings.append(SoxxHolding(symbol=symbol, name=name, weight_pct=weight_pct))

        if not holdings:
            raise SoxxHoldingsError("SOXX holdings CSV contained no equity constituents.")

        holdings.sort(key=lambda item: item.weight_pct, reverse=True)
        return tuple(holdings)

    def get_top_holdings(self, *, limit: int = 10, force_refresh: bool = False) -> tuple[SoxxHolding, ...]:
        """Return the latest top SOXX holdings while avoiding repeated fetches every cycle."""

        if limit <= 0:
            return ()

        cached = self.__class__._cache
        now = time.monotonic()
        if (
            not force_refresh
            and cached is not None
            and (now - cached[0]) < _CACHE_TTL_SECONDS
            and len(cached[1]) >= limit
        ):
            return cached[1][:limit]

        csv_text = self._download_csv()
        holdings = self._parse_csv_rows(csv_text)
        self.__class__._cache = (now, holdings)
        return holdings[:limit]
