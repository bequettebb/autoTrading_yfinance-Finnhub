"""Breakout-style momentum strategy for high-beta leveraged products."""

from __future__ import annotations

from typing import Sequence

from strategy.leveraged_rotation import (
    LeveragedSignal,
    _percentage_return,
    _realized_volatility,
    _simple_moving_average,
)
from strategy.sma_crossover import SignalAction


class BreakoutMomentumStrategy:
    """Enter when price breaks recent highs with momentum and exit on deterioration."""

    def __init__(
        self,
        fast_period: int,
        slow_period: int,
        breakout_lookback_bars: int,
        momentum_lookback_bars: int,
        volatility_lookback_bars: int,
    ) -> None:
        """Store windows used for breakout detection, momentum, and volatility checks."""

        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("moving-average periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if breakout_lookback_bars <= 1:
            raise ValueError("breakout_lookback_bars must be greater than one")
        if momentum_lookback_bars <= 0:
            raise ValueError("momentum_lookback_bars must be positive")
        if volatility_lookback_bars <= 1:
            raise ValueError("volatility_lookback_bars must be greater than one")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.breakout_lookback_bars = breakout_lookback_bars
        self.momentum_lookback_bars = momentum_lookback_bars
        self.volatility_lookback_bars = volatility_lookback_bars

    @property
    def required_bars(self) -> int:
        """Return the largest lookback required to evaluate one symbol."""

        return max(
            self.slow_period,
            self.breakout_lookback_bars + 1,
            self.momentum_lookback_bars + 1,
            self.volatility_lookback_bars + 1,
        )

    def evaluate_symbol(self, closes: Sequence[float], has_position: bool) -> LeveragedSignal:
        """Evaluate breakout strength and decide whether the symbol is actionable."""

        if len(closes) < self.required_bars:
            raise ValueError(
                f"At least {self.required_bars} closes are required to evaluate the strategy."
            )

        fast_ma = _simple_moving_average(closes, self.fast_period)
        slow_ma = _simple_moving_average(closes, self.slow_period)
        momentum_return = _percentage_return(closes, self.momentum_lookback_bars)
        volatility = _realized_volatility(closes, self.volatility_lookback_bars)
        last_close = float(closes[-1])
        breakout_level = max(float(price) for price in closes[-self.breakout_lookback_bars - 1 : -1])
        breakout_distance = (last_close / breakout_level) - 1.0 if breakout_level > 0 else 0.0

        score = breakout_distance + (0.8 * momentum_return) + (0.25 * ((fast_ma / slow_ma) - 1.0)) - (0.30 * volatility)
        entry_candidate = last_close > breakout_level and momentum_return > 0 and fast_ma > slow_ma

        if has_position and (last_close < fast_ma or momentum_return < 0):
            return LeveragedSignal(
                action=SignalAction.SELL,
                reason="Breakout weakened and price lost short-term support.",
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
                reason="Price broke recent highs with positive momentum.",
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
            reason="No actionable breakout signal is active.",
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            momentum_return=momentum_return,
            volatility=volatility,
            score=score,
            last_close=last_close,
            entry_candidate=entry_candidate,
        )

