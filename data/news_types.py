"""Shared news-domain data types used across providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news article consumed by sentiment and explanation code."""

    article_id: str
    headline: str
    summary: str
    source: str
    url: str | None
    created_at: datetime
    updated_at: datetime
    symbols: tuple[str, ...]
    content: str | None
