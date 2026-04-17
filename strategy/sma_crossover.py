"""Simple moving-average strategy primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Sequence


class SignalAction(str, Enum):
    """Normalized actions that the execution layer can translate into orders."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class StrategySignal:
    """Strategy output with enough context to log why the action was chosen."""

    action: SignalAction
    reason: str
    fast_ma: float
    slow_ma: float
    last_close: float


def simple_moving_average(values: Sequence[float], period: int) -> float:
    """Calculate a simple moving average for the most recent `period` values."""

    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        raise ValueError("not enough values to compute the requested moving average")
    return float(fmean(values[-period:]))


class MovingAverageCrossoverStrategy:
    """Long-only starter strategy that enters on trend strength and exits on weakness."""

    def __init__(self, fast_period: int, slow_period: int) -> None:
        """Store the moving-average windows after validating the relationship."""

        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("moving-average periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def required_bars(self) -> int:
        """Return the minimum bar count required to generate a signal."""

        return self.slow_period

    def generate_signal(self, closes: Sequence[float], has_position: bool) -> StrategySignal:
        """Generate a regime-based signal from the latest closes and position state."""

        if len(closes) < self.required_bars:
            raise ValueError(
                f"At least {self.required_bars} closes are required to evaluate the strategy."
            )

        fast_ma = simple_moving_average(closes, self.fast_period)
        slow_ma = simple_moving_average(closes, self.slow_period)
        last_close = float(closes[-1])

        if fast_ma > slow_ma and not has_position:
            return StrategySignal(
                action=SignalAction.BUY,
                reason="Fast MA is above slow MA while no position is open.",
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                last_close=last_close,
            )

        if fast_ma < slow_ma and has_position:
            return StrategySignal(
                action=SignalAction.SELL,
                reason="Fast MA is below slow MA while a long position is open.",
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                last_close=last_close,
            )

        return StrategySignal(
            action=SignalAction.HOLD,
            reason="Trend regime does not require a position change.",
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            last_close=last_close,
        )

