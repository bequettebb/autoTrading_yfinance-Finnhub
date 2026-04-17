"""Generate dashboard-friendly Korean news overlays from market news and Gemini."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.settings import NewsReasoningSettings
from data.news_types import NewsArticle
from data.finnhub_news import FinnhubNewsClient, FinnhubNewsError
from data.soxx_holdings import (
    SoxxHolding,
    SoxxHoldingsClient,
    SoxxHoldingsError,
    default_soxx_top_holdings,
)

LOGGER = logging.getLogger("auto_trading.news_reasoner")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

_POSITIVE_KEYWORDS = (
    "beat",
    "beats",
    "bullish",
    "demand",
    "expand",
    "gain",
    "growth",
    "guidance",
    "higher",
    "improve",
    "optimistic",
    "outperform",
    "partnership",
    "positive",
    "profit",
    "record",
    "rebound",
    "rally",
    "strong",
    "surge",
    "upgrade",
)
_NEGATIVE_KEYWORDS = (
    "bearish",
    "bankruptcy",
    "cut",
    "cuts",
    "decline",
    "downgrade",
    "drop",
    "fall",
    "investigation",
    "lawsuit",
    "lower",
    "miss",
    "missed",
    "negative",
    "pressure",
    "recall",
    "risk",
    "slump",
    "tariff",
    "warning",
    "weak",
)


@dataclass(frozen=True)
class EvaluationContext:
    """Signal context passed into the news explanation layer."""

    symbol: str
    action: str
    score: float
    reason: str
    signal_symbol: str
    target_symbol: str
    prob_bull: float | None = None
    prob_bear: float | None = None
    prob_neutral: float | None = None


@dataclass(frozen=True)
class SymbolNewsInsight:
    """News-aware explanation for one evaluated symbol."""

    symbol: str
    sentiment_label: str
    sentiment_score: float
    alignment: str
    explanation: str
    related_symbols: tuple[str, ...]
    article_count: int


@dataclass(frozen=True)
class ArticleBrief:
    """Compact Korean summary card for one selected article."""

    article_id: str
    symbol: str
    headline: str
    summary: str
    source: str
    url: str | None
    created_at: datetime
    updated_at: datetime
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class NewsReasoningSnapshot:
    """Serializable news reasoning payload consumed by reports and the dashboard."""

    generated_at: str
    status: str
    provider: str
    model: str | None
    overall_sentiment: str
    confidence: float | None
    summary: str
    leader_symbol: str | None
    leader_explanation: str | None
    article_count: int
    related_symbols: tuple[str, ...]
    focus_holdings: tuple[SoxxHolding, ...]
    symbol_insights: tuple[SymbolNewsInsight, ...]
    article_briefs: tuple[ArticleBrief, ...]
    articles: tuple[NewsArticle, ...]
    warnings: tuple[str, ...] = ()


def _parse_iso_datetime(raw_value: object) -> datetime | None:
    """Parse an ISO datetime string into a timezone-aware value when possible."""

    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_recent_gemini_snapshot(
    snapshot: NewsReasoningSnapshot | None,
    *,
    refresh_interval_minutes: int,
) -> bool:
    """Return whether an existing Gemini snapshot is fresh enough to reuse."""

    if snapshot is None or snapshot.provider != "gemini":
        return False
    generated_at = _parse_iso_datetime(snapshot.generated_at)
    if generated_at is None:
        return False
    age = datetime.now(timezone.utc) - generated_at
    return age < timedelta(minutes=refresh_interval_minutes)


def _article_ids(snapshot: NewsReasoningSnapshot | None) -> tuple[str, ...]:
    """Extract normalized article IDs from a snapshot in stable order."""

    if snapshot is None:
        return ()
    return tuple(article.article_id for article in snapshot.articles if article.article_id)


def _can_reuse_previous_gemini_overlay(
    *,
    base_snapshot: NewsReasoningSnapshot,
    previous_snapshot: NewsReasoningSnapshot | None,
    refresh_interval_minutes: int,
) -> bool:
    """Allow Gemini reuse only when the snapshot is fresh and article inputs are unchanged."""

    if not _is_recent_gemini_snapshot(
        previous_snapshot,
        refresh_interval_minutes=refresh_interval_minutes,
    ):
        return False
    return _article_ids(base_snapshot) == _article_ids(previous_snapshot)


def _snapshot_to_llm_payload(snapshot: NewsReasoningSnapshot) -> dict[str, object]:
    """Project a stored Gemini snapshot back into the overlay payload shape."""

    return {
        "overall_sentiment": snapshot.overall_sentiment,
        "confidence": snapshot.confidence,
        "summary": snapshot.summary,
        "leader_explanation": snapshot.leader_explanation,
        "symbol_insights": [
            {
                "symbol": insight.symbol,
                "sentiment_label": insight.sentiment_label,
                "alignment": insight.alignment,
                "explanation": insight.explanation,
            }
            for insight in snapshot.symbol_insights
        ],
        "article_briefs": [
            {
                "article_id": brief.article_id,
                "headline": brief.headline,
                "summary": brief.summary,
            }
            for brief in snapshot.article_briefs
        ],
    }


def _reuse_previous_gemini_overlay(
    *,
    base_snapshot: NewsReasoningSnapshot,
    previous_snapshot: NewsReasoningSnapshot,
    settings: NewsReasoningSettings,
) -> NewsReasoningSnapshot:
    """Reuse the previous Gemini copy on top of the latest heuristic baseline."""

    reused = _apply_llm_overlay(
        base_snapshot=base_snapshot,
        llm_payload=_snapshot_to_llm_payload(previous_snapshot),
        settings=settings,
    )
    return replace(
        reused,
        generated_at=previous_snapshot.generated_at,
        model=previous_snapshot.model or settings.gemini_model,
        warnings=base_snapshot.warnings,
    )


def _clean_text(text: object, *, max_chars: int) -> str:
    """Normalize whitespace and strip lightweight HTML from external content."""

    cleaned = _HTML_TAG_RE.sub(" ", str(text or ""))
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _normalize_sentiment_label(raw_value: object, *, default: str = "neutral") -> str:
    """Map free-form sentiment labels into the small dashboard taxonomy."""

    value = str(raw_value or "").strip().lower()
    mapping = {
        "positive": "bullish",
        "bull": "bullish",
        "bullish": "bullish",
        "negative": "bearish",
        "bear": "bearish",
        "bearish": "bearish",
        "neutral": "neutral",
        "mixed": "mixed",
        "balanced": "mixed",
        "uncertain": "mixed",
    }
    return mapping.get(value, default)


def _clamp_float(raw_value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Convert an arbitrary numeric-looking value to a bounded float."""

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _sentiment_label_from_score(score: float) -> str:
    """Translate a signed sentiment score into a compact discrete label."""

    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    if abs(score) < 0.08:
        return "neutral"
    return "mixed"


def _sentiment_phrase(label: str) -> str:
    """Translate the internal sentiment label into natural Korean copy."""

    mapping = {
        "bullish": "강세",
        "bearish": "약세",
        "mixed": "혼조",
        "neutral": "중립",
    }
    return mapping.get(label, "중립")


def _alignment_phrase(alignment: str) -> str:
    """Translate the internal alignment label into natural Korean copy."""

    mapping = {
        "supports": "대체로 맞물립니다",
        "conflicts": "다소 어긋납니다",
        "mixed": "뚜렷하게 한쪽으로 기울지 않습니다",
        "insufficient": "판단할 근거가 아직 부족합니다",
    }
    return mapping.get(alignment, "판단할 근거가 아직 부족합니다")


def _action_phrase(action: str) -> str:
    """Translate action labels into short Korean copy."""

    mapping = {
        "buy": "매수",
        "sell": "매도",
        "hold": "관망",
    }
    return mapping.get(action.lower(), action)


def _action_display_phrase(action: str) -> str:
    """Return a natural Korean action label for UI-facing sentences."""

    mapping = {
        "buy": "매수",
        "sell": "매도",
        "hold": "관망",
        "매수": "매수",
        "매도": "매도",
        "관망": "관망",
    }
    return mapping.get(str(action or "").strip().lower(), str(action or "").strip())


def _polish_korean_copy(text: str | None) -> str | None:
    """Smooth out stiff or translated-looking Korean copy before it reaches the dashboard."""

    if text is None:
        return None

    polished = str(text).strip()
    if not polished:
        return polished

    replacements = {
        "SOXX 상위 편입종목": "SOXX 상위 편입 종목",
        "상위 편입종목": "상위 편입 종목",
        "뉴스가 혼재돼 있어": "뉴스 흐름이 엇갈려 있어",
        "뉴스 흐름이 혼재돼 있어": "뉴스 흐름이 엇갈려 있어",
        "강하게 확인하거나 반박하긴 어렵습니다": "한쪽 판단을 강하게 뒷받침한다고 보긴 어렵습니다",
        "강하게 확인하거나 반박하기는 어렵습니다": "한쪽 판단을 강하게 뒷받침한다고 보긴 어렵습니다",
    }
    for source, target in replacements.items():
        polished = polished.replace(source, target)

    polished = re.sub(r"\bhold\b", "관망", polished, flags=re.IGNORECASE)
    polished = re.sub(r"\bbuy\b", "매수", polished, flags=re.IGNORECASE)
    polished = re.sub(r"\bsell\b", "매도", polished, flags=re.IGNORECASE)

    awkward_match = re.search(
        r"(?P<symbol>[A-Z]{1,8})의 현재 (?P<action>관망|매수|매도) 판단을 "
        r"(?:강하게 )?(?:확인하거나 반박하긴 어렵습니다|한쪽 판단을 강하게 뒷받침한다고 보긴 어렵습니다|강하게 뒷받침하진 않습니다)",
        polished,
    )
    if awkward_match and ("엇갈려 있어" in polished or "혼재돼 있어" in polished):
        symbol = awkward_match.group("symbol")
        action = _action_display_phrase(awkward_match.group("action"))
        if action == "관망":
            return "SOXX 상위 편입 종목 뉴스 흐름은 엇갈려 있는 상태입니다."
        return (
            f"SOXX 상위 편입 종목 관련 뉴스 흐름이 엇갈려 있어, "
            f"지금 {symbol}을 {action} 쪽으로 보기엔 확신이 크지 않습니다."
        )

    polished = re.sub(r"\s{2,}", " ", polished).strip()
    return polished


def _keyword_score(text: str) -> float:
    """Score text with a tiny keyword lexicon for the non-LLM fallback path."""

    lowered = text.lower()
    positive_hits = sum(lowered.count(keyword) for keyword in _POSITIVE_KEYWORDS)
    negative_hits = sum(lowered.count(keyword) for keyword in _NEGATIVE_KEYWORDS)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0
    return round((positive_hits - negative_hits) / total_hits, 3)


def _alignment_for_action(action: str, sentiment_score: float, article_count: int) -> str:
    """Estimate whether recent headlines support or conflict with the signal."""

    if article_count == 0:
        return "insufficient"

    action_value = action.lower()
    if action_value == "buy":
        if sentiment_score >= 0.10:
            return "supports"
        if sentiment_score <= -0.10:
            return "conflicts"
        return "mixed"
    if action_value == "sell":
        if sentiment_score <= -0.10:
            return "supports"
        if sentiment_score >= 0.10:
            return "conflicts"
        return "mixed"
    return "mixed"


def _requested_symbols(evaluations: Sequence[EvaluationContext]) -> tuple[str, ...]:
    """Collect the symbols that should drive the Finnhub news query."""

    ordered: dict[str, None] = {}
    for evaluation in evaluations:
        for symbol in (evaluation.symbol, evaluation.signal_symbol, evaluation.target_symbol):
            normalized = symbol.strip().upper()
            if normalized:
                ordered[normalized] = None
    return tuple(ordered.keys())


def _focus_symbols(
    focus_holdings: Sequence[SoxxHolding],
    evaluations: Sequence[EvaluationContext],
) -> tuple[str, ...]:
    """Return the preferred news universe: SOXX top holdings first, then symbol fallbacks."""

    holding_symbols = tuple(holding.symbol for holding in focus_holdings if holding.symbol)
    if holding_symbols:
        return holding_symbols
    return _requested_symbols(evaluations)


def _article_matches(
    article: NewsArticle,
    evaluation: EvaluationContext,
    *,
    focus_symbols: Sequence[str],
) -> bool:
    """Match articles to the model output using the SOXX constituent universe."""

    relevant = {
        evaluation.symbol.upper(),
        evaluation.signal_symbol.upper(),
        evaluation.target_symbol.upper(),
        *(symbol.upper() for symbol in focus_symbols),
    }
    article_symbols = {symbol.upper() for symbol in article.symbols}
    if relevant & article_symbols:
        return True

    title_blob = f"{article.headline} {article.summary}".upper()
    return any(symbol in title_blob for symbol in relevant)


def _serialize_holdings_for_prompt(focus_holdings: Sequence[SoxxHolding]) -> list[dict[str, object]]:
    """Trim holdings data before sending it to Gemini."""

    return [
        {
            "symbol": holding.symbol,
            "name": holding.name,
            "weight_pct": round(holding.weight_pct, 2),
        }
        for holding in focus_holdings
    ]


def _serialize_evaluations_for_prompt(evaluations: Sequence[EvaluationContext]) -> list[dict[str, object]]:
    """Trim live evaluation data down to the fields useful for the LLM prompt."""

    return [
        {
            "symbol": evaluation.symbol,
            "action": evaluation.action,
            "score": round(evaluation.score, 5),
            "reason": evaluation.reason,
            "signal_symbol": evaluation.signal_symbol,
            "target_symbol": evaluation.target_symbol,
            "prob_bull": evaluation.prob_bull,
            "prob_bear": evaluation.prob_bear,
            "prob_neutral": evaluation.prob_neutral,
        }
        for evaluation in evaluations
    ]


def _serialize_articles_for_prompt(articles: Sequence[NewsArticle]) -> list[dict[str, object]]:
    """Trim article payloads before sending them to Gemini."""

    return [
        {
            "article_id": article.article_id,
            "headline": _clean_text(article.headline, max_chars=180),
            "summary": _clean_text(article.summary or article.content or "", max_chars=240),
            "source": article.source,
            "updated_at": article.updated_at.isoformat(),
            "symbols": list(article.symbols),
            "url": article.url,
        }
        for article in articles
    ]


def _gemini_reasoning_budget(reasoning_effort: str) -> int | None:
    """Map a small reasoning label to Gemini 2.5 thinking budgets."""

    effort = reasoning_effort.strip().lower()
    mapping = {
        "none": 0,
        "minimal": 0,
        "low": 0,
        "medium": 4096,
        "high": 16384,
    }
    return mapping.get(effort)


def _gemini_response_schema() -> dict[str, object]:
    """Return the JSON schema expected from the Gemini overlay response."""

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "overall_sentiment": {
                "type": "string",
                "enum": ["bullish", "bearish", "mixed", "neutral"],
            },
            "confidence": {"type": "number"},
            "leader_explanation": {"type": "string"},
            "symbol_insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "sentiment_label": {
                            "type": "string",
                            "enum": ["bullish", "bearish", "mixed", "neutral"],
                        },
                        "sentiment_score": {"type": "number"},
                        "alignment": {
                            "type": "string",
                            "enum": ["supports", "conflicts", "mixed", "insufficient"],
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "symbol",
                        "sentiment_label",
                        "sentiment_score",
                        "alignment",
                        "explanation",
                    ],
                    "additionalProperties": False,
                },
            },
            "article_briefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_id": {"type": "string"},
                        "symbol": {"type": "string"},
                        "headline": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["article_id", "symbol", "headline", "summary"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "summary",
            "overall_sentiment",
            "confidence",
            "leader_explanation",
            "symbol_insights",
            "article_briefs",
        ],
        "additionalProperties": False,
    }


def _primary_symbol_for_article(
    article: NewsArticle,
    holdings_by_symbol: dict[str, SoxxHolding],
) -> str:
    """Pick the most relevant SOXX constituent for one article."""

    weighted_symbols = [
        holdings_by_symbol[symbol.upper()]
        for symbol in article.symbols
        if symbol.upper() in holdings_by_symbol
    ]
    if weighted_symbols:
        weighted_symbols.sort(key=lambda item: item.weight_pct, reverse=True)
        return weighted_symbols[0].symbol
    return next(iter(article.symbols), "")


def _article_priority(
    article: NewsArticle,
    *,
    now: datetime,
    lookback_minutes: int,
    holdings_by_symbol: dict[str, SoxxHolding],
) -> float:
    """Rank articles so the dashboard shows only the highest-signal items."""

    max_weight = max(
        (holdings_by_symbol[symbol.upper()].weight_pct for symbol in article.symbols if symbol.upper() in holdings_by_symbol),
        default=0.0,
    )
    age_minutes = max(0.0, (now - article.updated_at).total_seconds() / 60.0)
    recency_score = max(0.0, 1.0 - (age_minutes / max(1, lookback_minutes)))
    tone_score = abs(_keyword_score(f"{article.headline} {article.summary} {article.content or ''}"))
    return round((max_weight * 10.0) + (recency_score * 3.0) + tone_score, 4)


def _select_article_briefs(
    articles: Sequence[NewsArticle],
    *,
    settings: NewsReasoningSettings,
    focus_holdings: Sequence[SoxxHolding],
) -> tuple[NewsArticle, ...]:
    """Select only a few high-value articles for dashboard display."""

    if not articles:
        return ()

    holdings_by_symbol = {holding.symbol: holding for holding in focus_holdings}
    now = datetime.now(timezone.utc)
    ranked = sorted(
        articles,
        key=lambda article: (
            _article_priority(
                article,
                now=now,
                lookback_minutes=settings.realtime_lookback_minutes,
                holdings_by_symbol=holdings_by_symbol,
            ),
            article.updated_at.timestamp(),
        ),
        reverse=True,
    )
    return tuple(ranked[: settings.article_brief_count])


def _build_heuristic_article_briefs(
    selected_articles: Sequence[NewsArticle],
    *,
    focus_holdings: Sequence[SoxxHolding],
) -> tuple[ArticleBrief, ...]:
    """Build compact article cards even when Gemini is unavailable."""

    holdings_by_symbol = {holding.symbol: holding for holding in focus_holdings}
    briefs: list[ArticleBrief] = []
    for article in selected_articles:
        primary_symbol = _primary_symbol_for_article(article, holdings_by_symbol)
        headline = _clean_text(article.headline or primary_symbol, max_chars=96)
        raw_summary = article.summary or article.content or article.headline
        summary = _clean_text(
            f"{primary_symbol or '관련 종목'} 관련 기사입니다. {raw_summary}",
            max_chars=170,
        )
        briefs.append(
            ArticleBrief(
                article_id=article.article_id,
                symbol=primary_symbol or "SOXX",
                headline=headline,
                summary=summary,
                source=article.source,
                url=article.url,
                created_at=article.created_at,
                updated_at=article.updated_at,
                symbols=article.symbols,
            )
        )
    return tuple(briefs)


def _build_heuristic_explanation(
    evaluation: EvaluationContext,
    *,
    label: str,
    alignment: str,
    related_articles: Sequence[NewsArticle],
    focus_holdings: Sequence[SoxxHolding],
) -> str:
    """Generate a compact Korean explanation from recent constituent headlines."""

    if not related_articles:
        return (
            f"{evaluation.signal_symbol} 상위 편입종목 쪽에서 바로 참고할 만한 기사가 많지 않아, "
            "이번 판단은 모델 신호를 중심으로 보는 편이 좋겠습니다."
        )

    holdings_by_symbol = {holding.symbol: holding for holding in focus_holdings}
    lead_symbols = ", ".join(
        symbol
        for symbol in (
            _primary_symbol_for_article(article, holdings_by_symbol)
            for article in related_articles[:2]
        )
        if symbol
    )
    if alignment == "supports":
        return (
            f"{evaluation.signal_symbol} 상위 편입종목 뉴스 흐름은 {_sentiment_phrase(label)} 쪽이며, "
            f"{evaluation.symbol}의 현재 {_action_phrase(evaluation.action)} 판단과도 대체로 맞물립니다."
            f"{f' 핵심 종목: {lead_symbols}.' if lead_symbols else ''}"
        )
    if alignment == "conflicts":
        return (
            f"{evaluation.signal_symbol} 상위 편입종목 뉴스 흐름은 {_sentiment_phrase(label)} 쪽이지만, "
            f"{evaluation.symbol}의 현재 {_action_phrase(evaluation.action)} 판단과는 다소 어긋납니다."
            f"{f' 핵심 종목: {lead_symbols}.' if lead_symbols else ''}"
        )
    return (
        f"{evaluation.signal_symbol} 상위 편입종목 뉴스 흐름이 엇갈려 있어 "
        f"{evaluation.symbol}의 현재 {_action_phrase(evaluation.action)} 판단을 강하게 뒷받침하진 않습니다."
        f"{f' 관찰된 핵심 종목: {lead_symbols}.' if lead_symbols else ''}"
    )


def _build_heuristic_snapshot(
    *,
    settings: NewsReasoningSettings,
    evaluations: Sequence[EvaluationContext],
    leader_symbol: str | None,
    articles: Sequence[NewsArticle],
    focus_holdings: Sequence[SoxxHolding],
    warnings: Sequence[str] = (),
) -> NewsReasoningSnapshot:
    """Build a dashboard snapshot using lightweight heuristics only."""

    focus_symbols = _focus_symbols(focus_holdings, evaluations)
    symbol_insights: list[SymbolNewsInsight] = []
    weighted_scores: list[tuple[float, int]] = []

    for evaluation in evaluations:
        related_articles = [
            article
            for article in articles
            if _article_matches(article, evaluation, focus_symbols=focus_symbols)
        ]
        article_scores = [
            _keyword_score(f"{article.headline} {article.summary} {article.content or ''}")
            for article in related_articles
        ]
        sentiment_score = round(sum(article_scores) / len(article_scores), 3) if article_scores else 0.0
        sentiment_label = _sentiment_label_from_score(sentiment_score)
        alignment = _alignment_for_action(evaluation.action, sentiment_score, len(related_articles))
        symbol_insights.append(
            SymbolNewsInsight(
                symbol=evaluation.symbol,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                alignment=alignment,
                explanation=_build_heuristic_explanation(
                    evaluation,
                    label=sentiment_label,
                    alignment=alignment,
                    related_articles=related_articles,
                    focus_holdings=focus_holdings,
                ),
                related_symbols=tuple(
                    sorted(
                        {
                            symbol
                            for article in related_articles
                            for symbol in article.symbols
                        }
                    )
                ),
                article_count=len(related_articles),
            )
        )
        if related_articles:
            weighted_scores.append((sentiment_score, len(related_articles)))

    overall_score = 0.0
    if weighted_scores:
        score_total = sum(score * weight for score, weight in weighted_scores)
        weight_total = sum(weight for _, weight in weighted_scores)
        if weight_total:
            overall_score = round(score_total / weight_total, 3)

    overall_sentiment = _sentiment_label_from_score(overall_score)
    leader_insight = next((insight for insight in symbol_insights if insight.symbol == leader_symbol), None)
    selected_articles = _select_article_briefs(
        articles,
        settings=settings,
        focus_holdings=focus_holdings,
    )
    article_briefs = _build_heuristic_article_briefs(
        selected_articles,
        focus_holdings=focus_holdings,
    )

    if not articles:
        summary = (
            f"최근 {settings.realtime_lookback_minutes}분 동안 SOXX 상위 편입종목 관련 기사가 많지 않아, "
            "이번 화면은 모델 판단을 중심으로 보여드립니다."
        )
    elif leader_insight is not None:
        summary = _clean_text(
            f"SOXX 상위 편입종목 기준 최근 뉴스 흐름은 {_sentiment_phrase(overall_sentiment)} 쪽이며, "
            f"현재 선두 종목 {leader_insight.symbol} 판단과는 {_alignment_phrase(leader_insight.alignment)}.",
            max_chars=220,
        )
    else:
        summary = _clean_text(
            f"SOXX 상위 편입종목 관련 기사 {len(articles)}건을 바탕으로 보면, "
            f"최근 뉴스 흐름은 {_sentiment_phrase(overall_sentiment)} 쪽입니다.",
            max_chars=220,
        )

    confidence = round(min(1.0, 0.30 + (len(articles) * 0.08)), 2) if articles else None

    return NewsReasoningSnapshot(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status="ok" if articles else "no_news",
        provider="heuristic",
        model=None,
        overall_sentiment=overall_sentiment,
        confidence=confidence,
        summary=summary,
        leader_symbol=leader_symbol,
        leader_explanation=leader_insight.explanation if leader_insight is not None else None,
        article_count=len(articles),
        related_symbols=tuple(
            sorted(
                {
                    symbol
                    for article in articles
                    for symbol in article.symbols
                }
            )
        ),
        focus_holdings=tuple(focus_holdings),
        symbol_insights=tuple(symbol_insights),
        article_briefs=article_briefs,
        articles=tuple(articles),
        warnings=tuple(warnings),
    )


def _extract_gemini_text(payload: dict[str, object]) -> str:
    """Read plain text from a Gemini `generateContent` response payload."""

    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return ""

    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())

    combined = "".join(chunks).strip()
    if combined.startswith("```"):
        combined = combined.strip("`")
        if combined.lower().startswith("json"):
            combined = combined[4:].strip()
    return combined


def _gemini_finish_reason(payload: dict[str, object]) -> str | None:
    """Extract the primary Gemini finish reason so truncation can be handled explicitly."""

    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    finish_reason = first.get("finishReason")
    return str(finish_reason).strip().upper() or None


def _call_gemini_overlay(
    *,
    settings: NewsReasoningSettings,
    evaluations: Sequence[EvaluationContext],
    leader_symbol: str | None,
    articles: Sequence[NewsArticle],
    focus_holdings: Sequence[SoxxHolding],
) -> dict[str, object]:
    """Call Gemini's `generateContent` endpoint and return the parsed JSON object."""
    evaluation_symbols = [evaluation.symbol for evaluation in evaluations]
    article_variants: tuple[tuple[NewsArticle, ...], ...]
    if len(articles) > 2:
        article_variants = (tuple(articles), tuple(articles[:2]))
    else:
        article_variants = (tuple(articles),)

    last_error: Exception | None = None
    for attempt_index, article_batch in enumerate(article_variants, start=1):
        compact_mode = attempt_index > 1
        prompt = json.dumps(
            {
                "instructions": (
                    "You are a semiconductor trading-news analyst. Read the SOXX top holdings, the machine "
                    "trading signals, and the selected Finnhub articles. Return only a JSON object matching the schema. "
                    "Every human-facing string must be written in natural Korean. Use only the supplied facts. "
                    "Create symbol_insights only for the trade symbols listed in evaluations, not for the SOXX holdings list. "
                    "Return exactly one symbol_insight per evaluated symbol. Keep summary to at most two short sentences, "
                    "leader_explanation to one short sentence, each symbol explanation to one short sentence, and each article "
                    "brief to one short headline plus one short summary. Use overall_sentiment in {bullish,bearish,mixed,neutral}. "
                    "Use confidence from 0 to 1. Use sentiment_label in {bullish,bearish,mixed,neutral}. "
                    "Use alignment in {supports,conflicts,mixed,insufficient}. Keep sentiment_score between -1 and 1. "
                    f"Return at most {min(settings.article_brief_count, len(article_batch))} article_briefs and keep them tightly concise."
                ),
                "leader_symbol": leader_symbol,
                "evaluation_symbols": evaluation_symbols,
                "focus_holdings": _serialize_holdings_for_prompt(focus_holdings),
                "evaluations": _serialize_evaluations_for_prompt(evaluations),
                "articles": _serialize_articles_for_prompt(article_batch),
            },
            ensure_ascii=False,
        )
        generation_config: dict[str, object] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": _gemini_response_schema(),
            "temperature": 0.1,
            "maxOutputTokens": 2200 if compact_mode else 1600,
        }
        thinking_budget = 0 if compact_mode else _gemini_reasoning_budget(settings.gemini_reasoning_effort)
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }

        request = Request(
            url=(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
            ),
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini generateContent request failed with status {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach Gemini API: {exc.reason}") from exc

        finish_reason = _gemini_finish_reason(payload)
        response_text = _extract_gemini_text(payload)
        if not response_text:
            last_error = RuntimeError("Gemini returned no usable text output.")
            continue
        if finish_reason == "MAX_TOKENS":
            last_error = RuntimeError("Gemini response hit MAX_TOKENS before completing the JSON output.")
            continue

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            last_error = RuntimeError(f"Gemini returned invalid JSON: {exc}")
            continue
        if not isinstance(parsed, dict):
            last_error = RuntimeError(f"Expected a JSON object from Gemini, got: {type(parsed)!r}")
            continue
        return parsed

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini overlay failed without a usable response.")


def _apply_llm_overlay(
    *,
    base_snapshot: NewsReasoningSnapshot,
    llm_payload: dict[str, object],
    settings: NewsReasoningSettings,
) -> NewsReasoningSnapshot:
    """Merge Gemini-generated Korean dashboard copy onto the heuristic baseline."""

    base_by_symbol = {insight.symbol: insight for insight in base_snapshot.symbol_insights}
    raw_insights = llm_payload.get("symbol_insights", [])
    llm_by_symbol: dict[str, dict[str, object]] = {}
    if isinstance(raw_insights, list):
        for item in raw_insights:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol:
                llm_by_symbol[symbol] = item

    merged_insights: list[SymbolNewsInsight] = []
    for symbol, base_insight in base_by_symbol.items():
        llm_insight = llm_by_symbol.get(symbol)
        if llm_insight is None:
            merged_insights.append(base_insight)
            continue

        merged_insights.append(
            SymbolNewsInsight(
                symbol=symbol,
                sentiment_label=_normalize_sentiment_label(
                    llm_insight.get("sentiment_label"),
                    default=base_insight.sentiment_label,
                ),
                sentiment_score=_clamp_float(
                    llm_insight.get("sentiment_score"),
                    default=base_insight.sentiment_score,
                    minimum=-1.0,
                    maximum=1.0,
                ),
                alignment=str(llm_insight.get("alignment", base_insight.alignment)).strip().lower() or base_insight.alignment,
                explanation=_clean_text(
                    llm_insight.get("explanation", base_insight.explanation),
                    max_chars=320,
                )
                or base_insight.explanation,
                related_symbols=base_insight.related_symbols,
                article_count=base_insight.article_count,
            )
        )

    base_briefs_by_id = {brief.article_id: brief for brief in base_snapshot.article_briefs}
    raw_briefs = llm_payload.get("article_briefs", [])
    llm_briefs_by_id: dict[str, dict[str, object]] = {}
    if isinstance(raw_briefs, list):
        for item in raw_briefs:
            if not isinstance(item, dict):
                continue
            article_id = str(item.get("article_id", "")).strip()
            if article_id and article_id in base_briefs_by_id:
                llm_briefs_by_id[article_id] = item

    merged_briefs: list[ArticleBrief] = []
    for article_id, base_brief in base_briefs_by_id.items():
        llm_brief = llm_briefs_by_id.get(article_id)
        if llm_brief is None:
            merged_briefs.append(base_brief)
            continue

        merged_briefs.append(
            ArticleBrief(
                article_id=article_id,
                symbol=str(llm_brief.get("symbol", base_brief.symbol)).strip().upper() or base_brief.symbol,
                headline=_clean_text(llm_brief.get("headline", base_brief.headline), max_chars=96) or base_brief.headline,
                summary=_clean_text(llm_brief.get("summary", base_brief.summary), max_chars=170) or base_brief.summary,
                source=base_brief.source,
                url=base_brief.url,
                created_at=base_brief.created_at,
                updated_at=base_brief.updated_at,
                symbols=base_brief.symbols,
            )
        )

    summary = _clean_text(llm_payload.get("summary", base_snapshot.summary), max_chars=220) or base_snapshot.summary
    leader_explanation = _clean_text(
        llm_payload.get("leader_explanation", base_snapshot.leader_explanation or ""),
        max_chars=320,
    ) or base_snapshot.leader_explanation

    return replace(
        base_snapshot,
        provider="gemini",
        model=settings.gemini_model,
        overall_sentiment=_normalize_sentiment_label(
            llm_payload.get("overall_sentiment"),
            default=base_snapshot.overall_sentiment,
        ),
        confidence=_clamp_float(
            llm_payload.get("confidence"),
            default=base_snapshot.confidence if base_snapshot.confidence is not None else 0.5,
            minimum=0.0,
            maximum=1.0,
        ),
        summary=summary,
        leader_explanation=leader_explanation,
        symbol_insights=tuple(merged_insights),
        article_briefs=tuple(merged_briefs),
    )


def build_news_reasoning_from_articles(
    *,
    settings: NewsReasoningSettings,
    evaluations: Sequence[EvaluationContext],
    leader_symbol: str | None,
    articles: Sequence[NewsArticle],
    focus_holdings: Sequence[SoxxHolding] = (),
    warnings: Sequence[str] = (),
    previous_snapshot: NewsReasoningSnapshot | None = None,
) -> NewsReasoningSnapshot:
    """Build the dashboard news overlay from already-fetched Finnhub articles."""

    if not settings.enabled:
        return NewsReasoningSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status="disabled",
            provider="none",
            model=None,
            overall_sentiment="neutral",
            confidence=None,
            summary="현재 설정에서는 뉴스 해설을 사용하지 않습니다.",
            leader_symbol=leader_symbol,
            leader_explanation=None,
            article_count=0,
            related_symbols=(),
            focus_holdings=tuple(focus_holdings),
            symbol_insights=(),
            article_briefs=(),
            articles=(),
            warnings=tuple(warnings),
        )

    base_snapshot = _build_heuristic_snapshot(
        settings=settings,
        evaluations=evaluations,
        leader_symbol=leader_symbol,
        articles=articles,
        focus_holdings=focus_holdings,
        warnings=warnings,
    )
    if base_snapshot.status != "ok":
        return base_snapshot

    if not settings.gemini_api_key:
        extra_warning = ("Gemini 키가 없어, 뉴스는 기본 요약으로 보여드립니다.",)
        return replace(
            base_snapshot,
            warnings=base_snapshot.warnings + extra_warning,
        )

    if _can_reuse_previous_gemini_overlay(
        base_snapshot=base_snapshot,
        previous_snapshot=previous_snapshot,
        refresh_interval_minutes=settings.gemini_refresh_interval_minutes,
    ):
        return _reuse_previous_gemini_overlay(
            base_snapshot=base_snapshot,
            previous_snapshot=previous_snapshot,
            settings=settings,
        )

    llm_articles = _select_article_briefs(
        articles,
        settings=settings,
        focus_holdings=focus_holdings,
    )
    try:
        llm_payload = _call_gemini_overlay(
            settings=settings,
            evaluations=evaluations,
            leader_symbol=leader_symbol,
            articles=llm_articles,
            focus_holdings=focus_holdings,
        )
        return _apply_llm_overlay(
            base_snapshot=base_snapshot,
            llm_payload=llm_payload,
            settings=settings,
        )
    except Exception as exc:
        LOGGER.warning("Falling back to heuristic news reasoning: %s", exc)
        if previous_snapshot is not None and previous_snapshot.provider == "gemini":
            return _reuse_previous_gemini_overlay(
                base_snapshot=base_snapshot,
                previous_snapshot=previous_snapshot,
                settings=settings,
            )
        return replace(
            base_snapshot,
            warnings=base_snapshot.warnings + ("Gemini 요약을 불러오지 못해, 이번에는 기본 요약으로 대신 보여드립니다.",),
        )


def analyze_market_news(
    *,
    settings: NewsReasoningSettings,
    finnhub_api_key: str | None,
    evaluations: Sequence[EvaluationContext],
    leader_symbol: str | None,
    previous_snapshot: NewsReasoningSnapshot | None = None,
) -> NewsReasoningSnapshot:
    """Fetch recent Finnhub news for SOXX-focused symbols and build a dashboard overlay."""

    if not settings.enabled:
        return build_news_reasoning_from_articles(
            settings=settings,
            evaluations=evaluations,
            leader_symbol=leader_symbol,
            articles=(),
        )

    warnings: tuple[str, ...] = ()
    focus_holdings: tuple[SoxxHolding, ...] = ()
    try:
        focus_holdings = SoxxHoldingsClient().get_top_holdings(limit=settings.focus_holdings_count)
    except SoxxHoldingsError as exc:
        LOGGER.warning("Failed to fetch current SOXX holdings: %s", exc)
        focus_holdings = default_soxx_top_holdings(limit=settings.focus_holdings_count)
        if focus_holdings:
            warnings = (
                "실시간 SOXX 편입비중을 불러오지 못해 기본 상위 종목 목록으로 뉴스 범위를 잡았습니다.",
            )

    symbols = _focus_symbols(focus_holdings, evaluations)
    if not symbols:
        return NewsReasoningSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status="no_news",
            provider="none",
            model=None,
            overall_sentiment="neutral",
            confidence=None,
            summary="뉴스 분석에 사용할 종목을 정하지 못했습니다.",
            leader_symbol=leader_symbol,
            leader_explanation=None,
            article_count=0,
            related_symbols=(),
            focus_holdings=focus_holdings,
            symbol_insights=(),
            article_briefs=(),
            articles=(),
            warnings=warnings,
        )

    if not finnhub_api_key:
        return NewsReasoningSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status="error",
            provider="none",
            model=None,
            overall_sentiment="neutral",
            confidence=None,
            summary="FINNHUB_API_KEY가 없어 뉴스 해설을 진행할 수 없습니다.",
            leader_symbol=leader_symbol,
            leader_explanation=None,
            article_count=0,
            related_symbols=symbols,
            focus_holdings=focus_holdings,
            symbol_insights=(),
            article_briefs=(),
            articles=(),
            warnings=warnings + ("뉴스 소스: finnhub",),
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=settings.realtime_lookback_minutes)
    news_client = FinnhubNewsClient(api_key=finnhub_api_key)
    try:
        articles = news_client.get_recent_news(
            symbols=symbols,
            start=start,
            end=end,
            limit=settings.max_articles,
        )
    except FinnhubNewsError as exc:
        LOGGER.warning("Failed to fetch Finnhub news: %s", exc)
        return NewsReasoningSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            status="error",
            provider="none",
            model=None,
            overall_sentiment="neutral",
            confidence=None,
            summary="이번 사이클에서는 Finnhub 뉴스를 불러오지 못해 뉴스 해설을 건너뛰었습니다.",
            leader_symbol=leader_symbol,
            leader_explanation=None,
            article_count=0,
            related_symbols=symbols,
            focus_holdings=focus_holdings,
            symbol_insights=(),
            article_briefs=(),
            articles=(),
            warnings=warnings + (str(exc), "뉴스 소스: finnhub"),
        )

    return build_news_reasoning_from_articles(
        settings=settings,
        evaluations=evaluations,
        leader_symbol=leader_symbol,
        articles=articles,
        focus_holdings=focus_holdings,
        warnings=warnings + ("뉴스 소스: finnhub",),
        previous_snapshot=previous_snapshot,
    )


def deserialize_news_reasoning_snapshot(payload: dict[str, object] | None) -> NewsReasoningSnapshot | None:
    """Convert serialized dashboard data back into a typed news snapshot."""

    if not isinstance(payload, dict):
        return None

    focus_holdings_raw = payload.get("focus_holdings", [])
    symbol_insights_raw = payload.get("symbol_insights", [])
    article_briefs_raw = payload.get("article_briefs", [])
    articles_raw = payload.get("articles", [])
    warnings_raw = payload.get("warnings", [])

    focus_holdings = tuple(
        SoxxHolding(
            symbol=str(item.get("symbol", "")).strip().upper(),
            name=str(item.get("name", "")).strip(),
            weight_pct=_clamp_float(item.get("weight_pct"), default=0.0, minimum=0.0, maximum=100.0),
        )
        for item in focus_holdings_raw
        if isinstance(item, dict) and str(item.get("symbol", "")).strip()
    )
    symbol_insights = tuple(
        SymbolNewsInsight(
            symbol=str(item.get("symbol", "")).strip().upper(),
            sentiment_label=_normalize_sentiment_label(item.get("sentiment_label")),
            sentiment_score=_clamp_float(item.get("sentiment_score"), default=0.0, minimum=-1.0, maximum=1.0),
            alignment=str(item.get("alignment", "mixed")).strip().lower() or "mixed",
            explanation=str(item.get("explanation", "")).strip(),
            related_symbols=tuple(
                symbol.strip().upper()
                for symbol in item.get("related_symbols", [])
                if isinstance(symbol, str) and symbol.strip()
            ),
            article_count=max(0, int(item.get("article_count", 0) or 0)),
        )
        for item in symbol_insights_raw
        if isinstance(item, dict) and str(item.get("symbol", "")).strip()
    )
    article_briefs: list[ArticleBrief] = []
    for item in article_briefs_raw:
        if not isinstance(item, dict):
            continue
        created_at = _parse_iso_datetime(item.get("created_at"))
        updated_at = _parse_iso_datetime(item.get("updated_at"))
        if created_at is None or updated_at is None:
            continue
        article_briefs.append(
            ArticleBrief(
                article_id=str(item.get("article_id", "")).strip(),
                symbol=str(item.get("symbol", "")).strip().upper() or "SOXX",
                headline=str(item.get("headline", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                source=str(item.get("source", "")).strip(),
                url=str(item.get("url")).strip() if item.get("url") else None,
                created_at=created_at,
                updated_at=updated_at,
                symbols=tuple(
                    symbol.strip().upper()
                    for symbol in item.get("symbols", [])
                    if isinstance(symbol, str) and symbol.strip()
                ),
            )
        )
    articles: list[NewsArticle] = []
    for item in articles_raw:
        if not isinstance(item, dict):
            continue
        created_at = _parse_iso_datetime(item.get("created_at"))
        updated_at = _parse_iso_datetime(item.get("updated_at"))
        if created_at is None or updated_at is None:
            continue
        articles.append(
            NewsArticle(
                article_id=str(item.get("article_id", "")).strip(),
                headline=str(item.get("headline", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                source=str(item.get("source", "")).strip(),
                url=str(item.get("url")).strip() if item.get("url") else None,
                created_at=created_at,
                updated_at=updated_at,
                symbols=tuple(
                    symbol.strip().upper()
                    for symbol in item.get("symbols", [])
                    if isinstance(symbol, str) and symbol.strip()
                ),
                content=None,
            )
        )

    return NewsReasoningSnapshot(
        generated_at=str(payload.get("generated_at", datetime.now(timezone.utc).isoformat())),
        status=str(payload.get("status", "ok")),
        provider=str(payload.get("provider", "none")),
        model=str(payload.get("model")).strip() if payload.get("model") else None,
        overall_sentiment=_normalize_sentiment_label(payload.get("overall_sentiment")),
        confidence=(
            _clamp_float(payload.get("confidence"), default=0.0, minimum=0.0, maximum=1.0)
            if payload.get("confidence") is not None
            else None
        ),
        summary=str(payload.get("summary", "")).strip(),
        leader_symbol=str(payload.get("leader_symbol")).strip().upper() if payload.get("leader_symbol") else None,
        leader_explanation=str(payload.get("leader_explanation", "")).strip() or None,
        article_count=max(0, int(payload.get("article_count", 0) or 0)),
        related_symbols=tuple(
            symbol.strip().upper()
            for symbol in payload.get("related_symbols", [])
            if isinstance(symbol, str) and symbol.strip()
        ),
        focus_holdings=focus_holdings,
        symbol_insights=symbol_insights,
        article_briefs=tuple(article_briefs),
        articles=tuple(articles),
        warnings=tuple(
            str(warning).strip()
            for warning in warnings_raw
            if str(warning).strip()
        ),
    )


def serialize_news_reasoning_snapshot(snapshot: NewsReasoningSnapshot | None) -> dict[str, object] | None:
    """Convert the internal news snapshot into JSON-friendly dashboard data."""

    if snapshot is None:
        return None

    return {
        "generated_at": snapshot.generated_at,
        "status": snapshot.status,
        "provider": snapshot.provider,
        "model": snapshot.model,
        "overall_sentiment": snapshot.overall_sentiment,
        "confidence": snapshot.confidence,
        "summary": _polish_korean_copy(snapshot.summary),
        "leader_symbol": snapshot.leader_symbol,
        "leader_explanation": _polish_korean_copy(snapshot.leader_explanation),
        "article_count": snapshot.article_count,
        "related_symbols": list(snapshot.related_symbols),
        "focus_holdings": [
            {
                "symbol": holding.symbol,
                "name": holding.name,
                "weight_pct": holding.weight_pct,
            }
            for holding in snapshot.focus_holdings
        ],
        "warnings": list(snapshot.warnings),
        "symbol_insights": [
            {
                "symbol": insight.symbol,
                "sentiment_label": insight.sentiment_label,
                "sentiment_score": insight.sentiment_score,
                "alignment": insight.alignment,
                "explanation": _polish_korean_copy(insight.explanation),
                "related_symbols": list(insight.related_symbols),
                "article_count": insight.article_count,
            }
            for insight in snapshot.symbol_insights
        ],
        "article_briefs": [
            {
                "article_id": brief.article_id,
                "symbol": brief.symbol,
                "headline": _polish_korean_copy(brief.headline),
                "summary": _polish_korean_copy(brief.summary),
                "source": brief.source,
                "url": brief.url,
                "created_at": brief.created_at.isoformat(),
                "updated_at": brief.updated_at.isoformat(),
                "symbols": list(brief.symbols),
            }
            for brief in snapshot.article_briefs
        ],
        "articles": [
            {
                "article_id": article.article_id,
                "headline": article.headline,
                "summary": article.summary,
                "source": article.source,
                "url": article.url,
                "created_at": article.created_at.isoformat(),
                "updated_at": article.updated_at.isoformat(),
                "symbols": list(article.symbols),
            }
            for article in snapshot.articles
        ],
    }
