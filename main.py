"""Command-line entry point for a Finnhub-only market signal + news dashboard."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config.settings import AppSettings, SettingsError, load_settings
from data.yfinance_stock_data import YFinanceStockDataClient
from reporting.daily_reporter import write_daily_report, write_status_snapshot
from reporting.news_reasoner import (
    EvaluationContext,
    NewsReasoningSnapshot,
    analyze_market_news,
    deserialize_news_reasoning_snapshot,
    serialize_news_reasoning_snapshot,
)
from strategy.breakout_momentum import BreakoutMomentumStrategy
from strategy.leveraged_rotation import LeveragedRotationStrategy, LeveragedSignal

LOGGER = logging.getLogger("auto_trading")


@dataclass(frozen=True)
class TradingRuntime:
    """Long-lived collaborators used across analysis cycles."""

    data_client: YFinanceStockDataClient
    strategy: object


@dataclass(frozen=True)
class EvaluatedSymbol:
    """Evaluation result for one symbol during a single cycle."""

    symbol: str
    closes: list[float]
    signal: LeveragedSignal


def configure_logging(level: str) -> None:
    """Configure process-wide logging with a compact operator-friendly format."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _seconds_until_next_bar(*, now: datetime, timeframe_minutes: int, buffer_seconds: int) -> int:
    """Return the wait time until the next fully closed candle should be available."""

    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive.")
    if buffer_seconds < 0:
        raise ValueError("buffer_seconds must be zero or positive.")

    current_timestamp = now.astimezone(timezone.utc).timestamp()
    timeframe_seconds = timeframe_minutes * 60
    next_boundary = (math.floor(current_timestamp / timeframe_seconds) + 1) * timeframe_seconds
    target_timestamp = next_boundary + buffer_seconds
    return max(1, int(math.ceil(target_timestamp - current_timestamp)))


def _sleep_seconds_for_next_cycle(settings: AppSettings, cycle_started_at: datetime) -> int:
    """Choose the next wake-up time using bar-close alignment or fixed polling."""

    if settings.bot.align_to_bar_close:
        return _seconds_until_next_bar(
            now=cycle_started_at,
            timeframe_minutes=settings.strategy.bar_timeframe_minutes,
            buffer_seconds=settings.bot.bar_close_buffer_seconds,
        )
    return max(1, settings.bot.poll_interval_seconds)


def build_runtime(settings: AppSettings) -> TradingRuntime:
    """Construct Finnhub market-data and strategy components from settings."""

    if settings.strategy.strategy_name == "leveraged_rotation":
        strategy = LeveragedRotationStrategy(
            fast_period=settings.strategy.fast_ma_period,
            slow_period=settings.strategy.slow_ma_period,
            momentum_lookback_bars=settings.strategy.momentum_lookback_bars,
            volatility_lookback_bars=settings.strategy.volatility_lookback_bars,
        )
    elif settings.strategy.strategy_name == "breakout_momentum":
        strategy = BreakoutMomentumStrategy(
            fast_period=settings.strategy.fast_ma_period,
            slow_period=settings.strategy.slow_ma_period,
            breakout_lookback_bars=settings.strategy.breakout_lookback_bars,
            momentum_lookback_bars=settings.strategy.momentum_lookback_bars,
            volatility_lookback_bars=settings.strategy.volatility_lookback_bars,
        )
    else:
        raise SettingsError(
            "STRATEGY_NAME must be leveraged_rotation or breakout_momentum in Finnhub-only mode."
        )

    return TradingRuntime(
        data_client=YFinanceStockDataClient(
            bar_timeframe_minutes=settings.strategy.bar_timeframe_minutes,
            lookback=settings.strategy.bar_lookback,
        ),
        strategy=strategy,
    )


def evaluate_symbols(settings: AppSettings, runtime: TradingRuntime) -> list[EvaluatedSymbol]:
    """Fetch data and compute ranking signals for each configured symbol."""

    evaluations: list[EvaluatedSymbol] = []
    required_len = max(16, settings.strategy.bar_lookback)
    for symbol in settings.strategy.symbols:
        try:
            closes = runtime.data_client.get_recent_closes(symbol)
        except Exception:
            LOGGER.exception("Failed to fetch candles for %s; trying quote fallback", symbol)
            try:
                latest = runtime.data_client.get_latest_price(symbol)
                closes = [latest for _ in range(required_len)]
            except Exception:
                LOGGER.exception("Failed to fetch quote fallback for %s", symbol)
                continue

        try:
            signal = runtime.strategy.evaluate_symbol(closes=closes, has_position=False)
        except Exception:
            LOGGER.exception("Failed to evaluate %s", symbol)
            continue

        evaluations.append(EvaluatedSymbol(symbol=symbol, closes=closes, signal=signal))
        LOGGER.info(
            "%s | action=%s | score=%.5f | momentum=%.4f | fast_ma=%.2f | slow_ma=%.2f | last=%.2f",
            symbol,
            signal.action.value,
            signal.score,
            signal.momentum_return,
            signal.fast_ma,
            signal.slow_ma,
            signal.last_close,
        )

    return evaluations


def select_leader(evaluations: list[EvaluatedSymbol], rotation_buffer: float) -> EvaluatedSymbol | None:
    """Choose the highest-scoring entry candidate above the configured buffer."""

    candidates = [item for item in evaluations if item.signal.entry_candidate and item.signal.score > rotation_buffer]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.signal.score)


def _build_news_context(evaluations: list[EvaluatedSymbol]) -> list[EvaluationContext]:
    """Convert evaluated symbols into the smaller context expected by the news layer."""

    return [
        EvaluationContext(
            symbol=item.symbol,
            action=item.signal.action.value,
            score=item.signal.score,
            reason=item.signal.reason,
            signal_symbol=item.symbol,
            target_symbol=item.symbol,
        )
        for item in evaluations
    ]


def _load_previous_news_analysis_snapshot(report_root: str | Path = "reports") -> NewsReasoningSnapshot | None:
    """Load the previous dashboard news snapshot so Gemini copy can be reused."""

    status_path = Path(report_root) / "status.json"
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Failed to read the previous status snapshot for news reuse.", exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    return deserialize_news_reasoning_snapshot(payload.get("news_analysis"))


def build_news_analysis(
    settings: AppSettings,
    evaluations: list[EvaluatedSymbol],
    leader: EvaluatedSymbol | None,
) -> NewsReasoningSnapshot | None:
    """Build an optional Finnhub-news overlay without interrupting the cycle."""

    contexts = _build_news_context(evaluations)
    if not contexts:
        contexts = [
            EvaluationContext(
                symbol=symbol,
                action="hold",
                score=0.0,
                reason="Price signal unavailable; using symbol-only news context.",
                signal_symbol=symbol,
                target_symbol=symbol,
            )
            for symbol in settings.strategy.symbols
        ]

    previous_snapshot = _load_previous_news_analysis_snapshot()
    try:
        return analyze_market_news(
            settings=settings.news,
            finnhub_api_key=settings.news.finnhub_api_key,
            evaluations=contexts,
            leader_symbol=leader.symbol if leader is not None else None,
            previous_snapshot=previous_snapshot,
        )
    except Exception:
        LOGGER.exception("Failed to build the news explanation overlay.")
        return None


def _serialize_evaluations(
    evaluations: list[EvaluatedSymbol],
    news_analysis: NewsReasoningSnapshot | None = None,
) -> list[dict[str, object]]:
    """Convert evaluated symbol data into report-friendly dictionaries."""

    insight_by_symbol = {}
    if news_analysis is not None:
        insight_by_symbol = {insight.symbol: insight for insight in news_analysis.symbol_insights}

    return [
        {
            "symbol": item.symbol,
            "action": item.signal.action.value,
            "score": item.signal.score,
            "score_label": "시그널 점수",
            "entry_threshold": None,
            "reason": item.signal.reason,
            "momentum_return": item.signal.momentum_return,
            "fast_ma": item.signal.fast_ma,
            "slow_ma": item.signal.slow_ma,
            "last_close": item.signal.last_close,
            "signal_symbol": item.symbol,
            "target_symbol": item.symbol,
            "prob_bull": None,
            "prob_bear": None,
            "prob_neutral": None,
            "metric_primary_label": "Momentum",
            "metric_primary_value": item.signal.momentum_return,
            "metric_secondary_label": "Fast",
            "metric_secondary_value": item.signal.fast_ma,
            "metric_bias_label": "Slow",
            "metric_bias_value": item.signal.slow_ma,
            "news_sentiment_label": (insight_by_symbol[item.symbol].sentiment_label if item.symbol in insight_by_symbol else None),
            "news_sentiment_score": (insight_by_symbol[item.symbol].sentiment_score if item.symbol in insight_by_symbol else None),
            "news_alignment": (insight_by_symbol[item.symbol].alignment if item.symbol in insight_by_symbol else None),
            "llm_reason": (insight_by_symbol[item.symbol].explanation if item.symbol in insight_by_symbol else None),
        }
        for item in evaluations
    ]


def publish_report(
    settings: AppSettings,
    runtime: TradingRuntime,
    market_open: bool,
    evaluations: list[EvaluatedSymbol],
    leader: EvaluatedSymbol | None,
    notes: list[str],
    news_analysis: NewsReasoningSnapshot | None,
) -> None:
    """Write a markdown report summarizing the latest signal state."""

    serialized_evaluations = _serialize_evaluations(evaluations, news_analysis)
    serialized_news_analysis = serialize_news_reasoning_snapshot(news_analysis)

    account = {
        "equity": float(settings.bot.virtual_starting_asset_usd),
        "cash": float(settings.bot.virtual_starting_asset_usd),
        "current_exposure": 0.0,
        "trading_blocked": True,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
    }

    report_path = write_daily_report(
        report_root="reports",
        generated_at=datetime.now().astimezone(),
        trading_mode="analysis",
        strategy_name=settings.strategy.strategy_name,
        market_open=market_open,
        account=account,
        positions=[],
        orders=[],
        evaluations=serialized_evaluations,
        leader_symbol=leader.symbol if leader is not None else None,
        notes=notes,
        news_analysis=serialized_news_analysis,
    )

    write_status_snapshot(
        report_root="reports",
        snapshot={
            "generated_at": datetime.now().astimezone().isoformat(),
            "trading_mode": "analysis",
            "strategy_name": settings.strategy.strategy_name,
            "market_open": market_open,
            "leader_symbol": leader.symbol if leader is not None else None,
            "leader_score": leader.signal.score if leader is not None else None,
            "poll_interval_seconds": settings.bot.poll_interval_seconds,
            "account": account,
            "positions": [],
            "evaluations": serialized_evaluations,
            "orders": [],
            "notes": notes,
            "portfolio_history": [],
            "auto_selection": None,
            "news_analysis": serialized_news_analysis,
        },
    )
    LOGGER.info("Updated daily report: %s", report_path)


def run_cycle(settings: AppSettings) -> None:
    """Run one full analysis cycle and publish dashboard/report snapshots."""

    runtime = build_runtime(settings)
    LOGGER.info(
        "Active config | strategy=%s | symbols=%s",
        settings.strategy.strategy_name,
        ", ".join(settings.strategy.symbols),
    )

    market_open = False
    try:
        market_open = runtime.data_client.market_is_open()
    except Exception as exc:
        LOGGER.warning("Failed to read market status from yfinance helper: %s", exc)

    notes: list[str] = ["Execution/trading is disabled in Finnhub-only mode."]
    evaluations = evaluate_symbols(settings, runtime)

    if not evaluations:
        notes.append("No symbols could be evaluated during this cycle.")
        publish_report(
            settings=settings,
            runtime=runtime,
            market_open=market_open,
            evaluations=[],
            leader=None,
            notes=notes,
            news_analysis=None,
        )
        return

    leader = select_leader(evaluations, settings.strategy.rotation_buffer)
    if leader is not None:
        notes.append(f"Current leader is {leader.symbol} with score {leader.signal.score:.5f}.")
    else:
        notes.append("No current leader met the rotation threshold.")

    news_analysis = build_news_analysis(settings, evaluations, leader)
    publish_report(
        settings=settings,
        runtime=runtime,
        market_open=market_open,
        evaluations=evaluations,
        leader=leader,
        notes=notes,
        news_analysis=news_analysis,
    )


def run_loop(settings: AppSettings) -> None:
    """Keep running analysis cycles at the configured polling interval."""

    while True:
        try:
            run_cycle(settings)
        except Exception as exc:
            LOGGER.exception("Analysis cycle failed")
            try:
                write_status_snapshot(
                    report_root="reports",
                    snapshot={
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "trading_mode": "analysis",
                        "strategy_name": settings.strategy.strategy_name,
                        "market_open": None,
                        "leader_symbol": None,
                        "leader_score": None,
                        "poll_interval_seconds": settings.bot.poll_interval_seconds,
                        "account": {},
                        "positions": [],
                        "evaluations": [],
                        "orders": [],
                        "portfolio_history": [],
                        "auto_selection": None,
                        "news_analysis": None,
                        "notes": [f"Analysis cycle failed: {exc}"],
                    },
                )
            except Exception:
                LOGGER.exception("Failed to write status snapshot after cycle failure")

        cycle_finished_at = datetime.now().astimezone()
        sleep_seconds = _sleep_seconds_for_next_cycle(settings, cycle_finished_at)
        if settings.bot.align_to_bar_close:
            LOGGER.info(
                "Sleeping %s seconds until the next %s-minute bar close (+%ss buffer).",
                sleep_seconds,
                settings.strategy.bar_timeframe_minutes,
                settings.bot.bar_close_buffer_seconds,
            )
        else:
            LOGGER.info("Sleeping %s seconds before the next cycle.", sleep_seconds)
        time.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finnhub-only market signal + news dashboard bot.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true", help="Run one analysis cycle and exit.")
    mode_group.add_argument("--loop", action="store_true", help="Run continuously.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings = load_settings()
    except SettingsError as exc:
        configure_logging("INFO")
        LOGGER.error(str(exc))
        return 2

    configure_logging(settings.bot.log_level)
    LOGGER.info("Initialized in Finnhub-only mode")

    if args.loop:
        run_loop(settings)
        return 0

    try:
        run_cycle(settings)
    except Exception:
        LOGGER.exception("Analysis cycle failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
