"""Application settings loaded from environment variables (Finnhub-only runtime)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv as _external_load_dotenv
except ImportError:
    _external_load_dotenv = None


class SettingsError(ValueError):
    """Raised when the local configuration would make the bot invalid."""


@dataclass(frozen=True)
class FinnhubSettings:
    """Credentials for Finnhub market data and news."""

    api_key: str


@dataclass(frozen=True)
class StrategySettings:
    """Signal strategy configuration kept outside code for quick iteration."""

    strategy_name: str
    symbols: tuple[str, ...]
    bar_timeframe_minutes: int
    bar_lookback: int
    fast_ma_period: int
    slow_ma_period: int
    momentum_lookback_bars: int
    volatility_lookback_bars: int
    breakout_lookback_bars: int
    rotation_buffer: float


@dataclass(frozen=True)
class BotSettings:
    """Runtime settings for polling cadence and logs."""

    poll_interval_seconds: int
    align_to_bar_close: bool
    bar_close_buffer_seconds: int
    log_level: str
    virtual_starting_asset_usd: float


@dataclass(frozen=True)
class NewsReasoningSettings:
    """Settings for the dashboard's news sentiment and LLM explanation overlay."""

    enabled: bool
    provider: str
    finnhub_api_key: str | None
    realtime_lookback_minutes: int
    max_articles: int
    focus_holdings_count: int
    article_brief_count: int
    include_content: bool
    gemini_api_key: str | None
    gemini_model: str
    gemini_reasoning_effort: str
    gemini_timeout_seconds: int
    gemini_refresh_interval_minutes: int


@dataclass(frozen=True)
class AppSettings:
    """Full application settings grouped by concern to keep layers separate."""

    finnhub: FinnhubSettings
    strategy: StrategySettings
    bot: BotSettings
    news: NewsReasoningSettings


def _read_required_str(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SettingsError(f"Required environment variable '{name}' is missing.")
    return value


def _read_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _read_optional_str(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _read_int(name: str, default: int) -> int:
    raw = _read_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"Environment variable '{name}' must be an integer.") from exc


def _read_float(name: str, default: float) -> float:
    raw = _read_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"Environment variable '{name}' must be a float.") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw = _read_str(name, str(default)).lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if raw in truthy:
        return True
    if raw in falsy:
        return False
    raise SettingsError(f"Environment variable '{name}' must be a boolean value.")


def _read_symbols(name: str, default: str) -> tuple[str, ...]:
    raw = _read_str(name, default)
    symbols = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not symbols:
        raise SettingsError(f"Environment variable '{name}' must include at least one symbol.")
    return symbols


def _load_dotenv_fallback(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_dotenv(env_file: str | None = None) -> None:
    if _external_load_dotenv is not None:
        if env_file:
            _external_load_dotenv(dotenv_path=env_file, override=False)
        else:
            _external_load_dotenv(override=False)
        return
    dotenv_path = Path(env_file) if env_file else Path(".env")
    _load_dotenv_fallback(dotenv_path)


def load_settings(env_file: str | None = None) -> AppSettings:
    """Load and validate settings from `.env` for a Finnhub-only runtime."""

    _load_dotenv(env_file)

    finnhub_api_key = _read_required_str("FINNHUB_API_KEY")

    fast_ma_period = _read_int("FAST_MA_PERIOD", 8)
    slow_ma_period = _read_int("SLOW_MA_PERIOD", 18)
    momentum_lookback_bars = _read_int("MOMENTUM_LOOKBACK_BARS", 6)
    volatility_lookback_bars = _read_int("VOLATILITY_LOOKBACK_BARS", 12)
    breakout_lookback_bars = _read_int("BREAKOUT_LOOKBACK_BARS", 10)
    bar_timeframe_minutes = _read_int("BAR_TIMEFRAME_MINUTES", 5)
    bar_lookback = _read_int("BAR_LOOKBACK", 120)
    rotation_buffer = _read_float("ROTATION_BUFFER", 0.0)

    if fast_ma_period <= 0 or slow_ma_period <= 0 or fast_ma_period >= slow_ma_period:
        raise SettingsError("FAST_MA_PERIOD and SLOW_MA_PERIOD are invalid.")
    if momentum_lookback_bars <= 0 or volatility_lookback_bars <= 1 or breakout_lookback_bars <= 1:
        raise SettingsError("Momentum/volatility/breakout lookbacks are invalid.")
    if bar_timeframe_minutes <= 0 or bar_lookback <= 10:
        raise SettingsError("BAR_TIMEFRAME_MINUTES and BAR_LOOKBACK must be positive.")
    if rotation_buffer < 0:
        raise SettingsError("ROTATION_BUFFER must be zero or positive.")

    strategy_name = _read_str("STRATEGY_NAME", "leveraged_rotation").lower()
    if strategy_name not in {"leveraged_rotation", "breakout_momentum"}:
        # Force a Finnhub-only compatible strategy when legacy values (e.g., soxx_model) are set.
        strategy_name = "leveraged_rotation"

    symbols = _read_symbols("SYMBOLS", "SOXL,SOXS")

    poll_interval_seconds = _read_int("POLL_INTERVAL_SECONDS", 300)
    align_to_bar_close = _read_bool("ALIGN_TO_BAR_CLOSE", True)
    bar_close_buffer_seconds = _read_int("BAR_CLOSE_BUFFER_SECONDS", 20)
    log_level = _read_str("LOG_LEVEL", "INFO").upper()
    virtual_starting_asset_usd = _read_float("VIRTUAL_STARTING_ASSET_USD", 10000.0)
    if poll_interval_seconds <= 0:
        raise SettingsError("POLL_INTERVAL_SECONDS must be positive.")
    if bar_close_buffer_seconds < 0:
        raise SettingsError("BAR_CLOSE_BUFFER_SECONDS must be zero or positive.")
    if virtual_starting_asset_usd < 0:
        raise SettingsError("VIRTUAL_STARTING_ASSET_USD must be zero or positive.")

    news_enabled = _read_bool("NEWS_REASONING_ENABLED", True)
    news_provider = _read_str("NEWS_PROVIDER", "finnhub").lower()
    if news_provider != "finnhub":
        news_provider = "finnhub"

    news_realtime_lookback_minutes = _read_int("NEWS_REALTIME_LOOKBACK_MINUTES", 90)
    news_max_articles = _read_int("NEWS_MAX_ARTICLES", 8)
    news_focus_holdings_count = _read_int("NEWS_FOCUS_HOLDINGS_COUNT", 10)
    news_article_brief_count = _read_int("NEWS_ARTICLE_BRIEF_COUNT", 4)
    news_include_content = _read_bool("NEWS_INCLUDE_CONTENT", False)
    if news_realtime_lookback_minutes <= 0:
        raise SettingsError("NEWS_REALTIME_LOOKBACK_MINUTES must be positive.")
    if news_max_articles <= 0 or news_focus_holdings_count <= 0 or news_article_brief_count <= 0:
        raise SettingsError("News sizing settings must be positive.")

    gemini_timeout_seconds = _read_int("GEMINI_TIMEOUT_SECONDS", 20)
    gemini_model = _read_str("GEMINI_MODEL", "gemini-2.5-flash-lite")
    gemini_reasoning_effort = _read_str("GEMINI_REASONING_EFFORT", "low").lower()
    gemini_refresh_interval_minutes = _read_int("GEMINI_REFRESH_INTERVAL_MINUTES", 15)
    if gemini_timeout_seconds <= 0 or gemini_refresh_interval_minutes <= 0:
        raise SettingsError("Gemini timeout/refresh settings must be positive.")

    return AppSettings(
        finnhub=FinnhubSettings(api_key=finnhub_api_key),
        strategy=StrategySettings(
            strategy_name=strategy_name,
            symbols=symbols,
            bar_timeframe_minutes=bar_timeframe_minutes,
            bar_lookback=bar_lookback,
            fast_ma_period=fast_ma_period,
            slow_ma_period=slow_ma_period,
            momentum_lookback_bars=momentum_lookback_bars,
            volatility_lookback_bars=volatility_lookback_bars,
            breakout_lookback_bars=breakout_lookback_bars,
            rotation_buffer=rotation_buffer,
        ),
        bot=BotSettings(
            poll_interval_seconds=poll_interval_seconds,
            align_to_bar_close=align_to_bar_close,
            bar_close_buffer_seconds=bar_close_buffer_seconds,
            log_level=log_level,
            virtual_starting_asset_usd=virtual_starting_asset_usd,
        ),
        news=NewsReasoningSettings(
            enabled=news_enabled,
            provider=news_provider,
            finnhub_api_key=finnhub_api_key,
            realtime_lookback_minutes=news_realtime_lookback_minutes,
            max_articles=news_max_articles,
            focus_holdings_count=news_focus_holdings_count,
            article_brief_count=news_article_brief_count,
            include_content=news_include_content,
            gemini_api_key=_read_optional_str("GEMINI_API_KEY"),
            gemini_model=gemini_model,
            gemini_reasoning_effort=gemini_reasoning_effort,
            gemini_timeout_seconds=gemini_timeout_seconds,
            gemini_refresh_interval_minutes=gemini_refresh_interval_minutes,
        ),
    )
