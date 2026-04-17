"""Short-horizon rotation strategy for high-beta leveraged products."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Sequence

from strategy.sma_crossover import SignalAction


@dataclass(frozen=True)
class LeveragedSignal:
    """Signal details used for logging, ranking, and rotation decisions."""

    action: SignalAction
    reason: str
    fast_ma: float
    slow_ma: float
    momentum_return: float
    volatility: float
    score: float
    last_close: float
    entry_candidate: bool


def _simple_moving_average(values: Sequence[float], period: int) -> float:
    """Return the mean of the latest `period` closes."""

    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        raise ValueError("not enough values to compute moving average")
    return float(fmean(values[-period:]))


def _percentage_return(values: Sequence[float], lookback: int) -> float:
    """Return the simple percentage change over the requested lookback."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if len(values) <= lookback:
        raise ValueError("not enough values to compute percentage return")
    start_price = float(values[-lookback - 1])
    end_price = float(values[-1])
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    return (end_price / start_price) - 1.0


def _realized_volatility(values: Sequence[float], lookback: int) -> float:
    """Estimate recent volatility from close-to-close returns."""

    if lookback <= 1:
        raise ValueError("lookback must be greater than one")
    if len(values) <= lookback:
        raise ValueError("not enough values to compute realized volatility")

    returns = []
    for index in range(len(values) - lookback, len(values)):
        previous_close = float(values[index - 1])
        current_close = float(values[index])
        if previous_close <= 0:
            raise ValueError("close prices must be positive")
        returns.append((current_close / previous_close) - 1.0)

    return float(pstdev(returns)) if len(returns) > 1 else 0.0


class LeveragedRotationStrategy:
    """Rank a small universe of high-beta instruments and rotate into the leader."""

    def __init__(
        self,
        fast_period: int,
        slow_period: int,
        momentum_lookback_bars: int,
        volatility_lookback_bars: int,
    ) -> None:
        """Store signal windows used by both live execution and backtests."""

        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("moving-average periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if momentum_lookback_bars <= 0:
            raise ValueError("momentum_lookback_bars must be positive")
        if volatility_lookback_bars <= 1:
            raise ValueError("volatility_lookback_bars must be greater than one")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.momentum_lookback_bars = momentum_lookback_bars
        self.volatility_lookback_bars = volatility_lookback_bars

    @property
    def required_bars(self) -> int:
        """Return the longest lookback required to evaluate one symbol."""

        return max(
            self.slow_period,
            self.momentum_lookback_bars + 1,
            self.volatility_lookback_bars + 1,
        )

    def evaluate_symbol(self, closes: Sequence[float], has_position: bool) -> LeveragedSignal:
        """Evaluate one symbol for trend strength, momentum, and exit urgency."""

        if len(closes) < self.required_bars:
            raise ValueError(
                f"At least {self.required_bars} closes are required to evaluate the strategy."
            )

        fast_ma = _simple_moving_average(closes, self.fast_period)
        slow_ma = _simple_moving_average(closes, self.slow_period)
        momentum_return = _percentage_return(closes, self.momentum_lookback_bars)
        volatility = _realized_volatility(closes, self.volatility_lookback_bars)
        last_close = float(closes[-1])

        trend_strength = (fast_ma / slow_ma) - 1.0
        # Favor symbols with strong short-term acceleration, but penalize noisy moves.
        score = momentum_return + (0.75 * trend_strength) - (0.35 * volatility)
        entry_candidate = fast_ma > slow_ma and momentum_return > 0

        if has_position and not entry_candidate:
            return LeveragedSignal(
                action=SignalAction.SELL,
                reason="Trend or momentum deteriorated versus the rotation rules.",
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                momentum_return=momentum_return,
                volatility=volatility,
                score=score,
                last_close=last_close,
                entry_candidate=entry_candidate,
            )

        if not has_position and entry_candidate:
            return LeveragedSignal(
                action=SignalAction.BUY,
                reason="Symbol is the kind of trend-positive momentum candidate the strategy can rank.",
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                momentum_return=momentum_return,
                volatility=volatility,
                score=score,
                last_close=last_close,
                entry_candidate=entry_candidate,
            )

        return LeveragedSignal(
            action=SignalAction.HOLD,
            reason="No regime change was triggered for this symbol.",
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            momentum_return=momentum_return,
            volatility=volatility,
            score=score,
            last_close=last_close,
            entry_candidate=entry_candidate,
        )

