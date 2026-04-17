"""Finnhub company-news wrapper for lower-latency dashboard news updates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from data.news_types import NewsArticle


class FinnhubNewsError(RuntimeError):
    """Raised when the Finnhub news endpoint cannot satisfy a request."""


@dataclass(frozen=True)
class _RawFinnhubArticle:
    """Internal normalized shape for one Finnhub article row."""

    headline: str
    summary: str
    source: str
    url: str | None
    created_at: datetime
    updated_at: datetime


class FinnhubNewsClient:
    """Fetch recent news from Finnhub and normalize it into `NewsArticle` objects."""

    _BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key.strip()

    def _request_json(self, params: dict[str, str]) -> list[object]:
        query = urlencode(params)
        request = Request(
            url=f"{self._BASE_URL}?{query}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise FinnhubNewsError(f"Finnhub request failed with status {exc.code}: {body}") from exc
        except URLError as exc:
            raise FinnhubNewsError(f"Failed to reach Finnhub news API: {exc.reason}") from exc

        if not isinstance(payload, list):
            raise FinnhubNewsError(f"Unexpected Finnhub response payload: {payload!r}")
        return payload

    @staticmethod
    def _as_datetime(epoch_seconds: object) -> datetime | None:
        try:
            epoch = int(float(epoch_seconds))
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    def _normalize_article(
        self,
        raw_article: object,
        *,
        symbol: str,
    ) -> _RawFinnhubArticle | None:
        if not isinstance(raw_article, dict):
            return None
        created_at = self._as_datetime(raw_article.get("datetime"))
        if created_at is None:
            return None

        headline = str(raw_article.get("headline", "")).strip()
        summary = str(raw_article.get("summary", "")).strip()
        source = str(raw_article.get("source", "")).strip() or "Finnhub"
        url = str(raw_article.get("url", "")).strip() or None
        if not headline and not summary:
            return None

        return _RawFinnhubArticle(
            headline=headline or f"{symbol} market update",
            summary=summary,
            source=source,
            url=url,
            created_at=created_at,
            updated_at=created_at,
        )

    @staticmethod
    def _build_article_id(symbol: str, url: str | None, created_at: datetime, headline: str) -> str:
        stable = f"{symbol}|{url or ''}|{created_at.isoformat()}|{headline}"
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        return f"finnhub-{digest[:16]}"

    def get_recent_news(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch latest Finnhub company news rows for the requested symbol list."""

        unique_symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()))
        if not unique_symbols or limit <= 0:
            return []

        # Finnhub expects YYYY-MM-DD for both bounds.
        from_date = start.astimezone(timezone.utc).date().isoformat()
        to_date = end.astimezone(timezone.utc).date().isoformat()

        dedup: dict[str, NewsArticle] = {}
        for symbol in unique_symbols:
            raw_rows = self._request_json(
                {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": self._api_key,
                }
            )

            for raw_row in raw_rows:
                normalized = self._normalize_article(raw_row, symbol=symbol)
                if normalized is None:
                    continue
                if normalized.created_at < start.astimezone(timezone.utc) or normalized.created_at > end.astimezone(
                    timezone.utc
                ):
                    continue

                article_id = self._build_article_id(
                    symbol=symbol,
                    url=normalized.url,
                    created_at=normalized.created_at,
                    headline=normalized.headline,
                )
                existing = dedup.get(article_id)
                if existing is not None and existing.updated_at >= normalized.updated_at:
                    continue
                dedup[article_id] = NewsArticle(
                    article_id=article_id,
                    headline=normalized.headline,
                    summary=normalized.summary,
                    source=normalized.source,
                    url=normalized.url,
                    created_at=normalized.created_at,
                    updated_at=normalized.updated_at,
                    symbols=(symbol,),
                    content=None,
                )

        sorted_articles = sorted(dedup.values(), key=lambda item: item.updated_at, reverse=True)
        return sorted_articles[:limit]
