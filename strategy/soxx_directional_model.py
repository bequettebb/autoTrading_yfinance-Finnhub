"""SOXX-based directional model that maps signals into SOXL/SOXS execution."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

from strategy.sma_crossover import SignalAction

LOGGER = logging.getLogger("auto_trading.soxx_model")

TargetSide = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class ModelDrivenSignal:
    """Normalized signal object consumed by the orchestration and dashboard layers."""

    action: SignalAction
    reason: str
    score: float
    score_label: str
    entry_threshold: float
    last_close: float
    entry_candidate: bool
    momentum_return: float
    fast_ma: float
    slow_ma: float
    signal_symbol: str
    target_symbol: str
    prob_bull: float
    prob_bear: float
    prob_neutral: float
    metric_primary_label: str
    metric_primary_value: float
    metric_secondary_label: str
    metric_secondary_value: float
    metric_bias_label: str
    metric_bias_value: float


@dataclass(frozen=True)
class DirectionalPrediction:
    """Three-state directional forecast from the SOXX model."""

    signal_symbol: str
    signal_timestamp: pd.Timestamp
    target_side: TargetSide
    prob_bull: float
    prob_bear: float
    prob_neutral: float
    bias: float
    entry_blocked: bool
    reason: str


@dataclass(frozen=True)
class LoadedArtifacts:
    """Loaded model artifacts plus metadata needed for hot reloading."""

    xgb_model: xgb.XGBClassifier
    cnn_model: nn.Module | None
    seq_len: int | None
    feature_cols: tuple[str, ...]
    label_classes: tuple[int, ...]
    best_params: dict[str, Any]
    ensemble_weight: dict[str, float]
    file_mtimes: dict[str, float]
    loaded_at: float


class CNN1D(nn.Module):
    """1D CNN architecture copied from the training script for runtime inference."""

    def __init__(self, n_features: int, n_classes: int = 3) -> None:
        """Mirror the training network exactly so `cnn_model.pt` can be restored."""

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional stack to a `[batch, seq, feature]` tensor."""

        outputs = inputs.permute(0, 2, 1)
        outputs = self.conv(outputs)
        outputs = outputs.reshape(outputs.size(0), -1)
        return self.fc(outputs)


class SoxxDirectionalModelStrategy:
    """Generate SOXX long/short/flat signals and map them to SOXL or SOXS."""

    FEATURE_EXCLUDE = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "label",
        "future_return_pct",
        "vwap",
    }

    def __init__(
        self,
        *,
        artifact_dir: str | Path,
        signal_symbol: str,
        long_symbol: str,
        short_symbol: str,
        allow_cnn: bool = True,
        artifact_reload_seconds: int = 30,
    ) -> None:
        """Store artifact paths and signal-routing settings for live inference."""

        self.artifact_dir = Path(artifact_dir)
        self.signal_symbol = signal_symbol.upper()
        self.long_symbol = long_symbol.upper()
        self.short_symbol = short_symbol.upper()
        self.allow_cnn = allow_cnn
        self.artifact_reload_seconds = max(1, artifact_reload_seconds)
        self._artifacts: LoadedArtifacts | None = None
        self._last_checked_at = 0.0

    @property
    def required_bars(self) -> int:
        """Return the minimum execution-price lookback needed for stops and reporting."""

        seq_len = self._artifacts.seq_len if self._artifacts is not None else 20
        return max(120, 60 + (seq_len or 20))

    @property
    def execution_symbols(self) -> tuple[str, str]:
        """Return the tradable symbols driven by the model's long/short forecast."""

        return (self.long_symbol, self.short_symbol)

    @property
    def model_period(self) -> str:
        """Return the training-period hint stored in the runtime artifacts."""

        self._ensure_artifacts_loaded()
        return str(self._artifacts.best_params.get("period", "60d"))

    @property
    def macro_tickers(self) -> dict[str, str]:
        """Return the macro series referenced by the model's feature pipeline."""

        self._ensure_artifacts_loaded()
        raw = self._artifacts.best_params.get("macro_tickers", {})
        return {
            str(name): str(ticker)
            for name, ticker in raw.items()
        }

    def _artifact_paths(self) -> dict[str, Path]:
        """Return the file paths that define one version of the deployed model."""

        return {
            "xgb": self.artifact_dir / "soxx_model.json",
            "labels": self.artifact_dir / "label_classes.json",
            "params": self.artifact_dir / "best_params.json",
            "features": self.artifact_dir / "feature_list.txt",
            "cnn": self.artifact_dir / "cnn_model.pt",
        }

    def _read_file_mtimes(self) -> dict[str, float]:
        """Read artifact modification times so the bot can hot-reload model updates."""

        mtimes: dict[str, float] = {}
        for name, path in self._artifact_paths().items():
            if path.exists():
                mtimes[name] = path.stat().st_mtime
        return mtimes

    def _ensure_artifacts_loaded(self) -> None:
        """Load the model the first time and reload it when the artifact files change."""

        now = time.monotonic()
        if self._artifacts is None:
            self._artifacts = self._load_artifacts()
            self._last_checked_at = now
            return

        if now - self._last_checked_at < self.artifact_reload_seconds:
            return

        self._last_checked_at = now
        current_mtimes = self._read_file_mtimes()
        if current_mtimes != self._artifacts.file_mtimes:
            LOGGER.info("Detected updated SOXX model artifacts. Reloading from %s", self.artifact_dir)
            self._artifacts = self._load_artifacts()

    def _load_artifacts(self) -> LoadedArtifacts:
        """Load XGBoost, CNN, label metadata, and thresholds from disk."""

        paths = self._artifact_paths()
        missing = [str(path) for path in paths.values() if not path.exists() and path.name != "cnn_model.pt"]
        if missing:
            raise FileNotFoundError(
                f"Missing required SOXX model artifact(s): {', '.join(missing)}"
            )

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(paths["xgb"])

        label_classes = tuple(
            int(value)
            for value in json.loads(paths["labels"].read_text(encoding="utf-8"))
        )
        feature_cols = tuple(
            line.strip()
            for line in paths["features"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        best_params = json.loads(paths["params"].read_text(encoding="utf-8"))

        cnn_model: nn.Module | None = None
        seq_len: int | None = None
        cnn_enabled = bool(best_params.get("cnn_enabled", False)) and self.allow_cnn and paths["cnn"].exists()
        if cnn_enabled:
            checkpoint = torch.load(paths["cnn"], map_location="cpu", weights_only=True)
            seq_len = int(checkpoint["seq_len"])
            n_features = int(checkpoint["n_features"])
            cnn_model = CNN1D(n_features=n_features, n_classes=len(label_classes))
            cnn_model.load_state_dict(checkpoint["state_dict"])
            cnn_model.eval()

        ensemble_weight = {
            "xgb": float(best_params.get("ensemble_weight", {}).get("xgb", 1.0)),
            "cnn": float(best_params.get("ensemble_weight", {}).get("cnn", 0.0)),
        }
        LOGGER.info(
            "Loaded SOXX model artifacts | cnn=%s | labels=%s | features=%s | dir=%s",
            cnn_model is not None,
            label_classes,
            len(feature_cols),
            self.artifact_dir,
        )
        return LoadedArtifacts(
            xgb_model=xgb_model,
            cnn_model=cnn_model,
            seq_len=seq_len,
            feature_cols=feature_cols,
            label_classes=label_classes,
            best_params=best_params,
            ensemble_weight=ensemble_weight,
            file_mtimes=self._read_file_mtimes(),
            loaded_at=time.time(),
        )

    @staticmethod
    def _merge_macro(
        signal_frame: pd.DataFrame,
        macro_frames: Mapping[str, pd.Series],
    ) -> pd.DataFrame:
        """Align macro proxies to the SOXX bar index and derive macro ratios."""

        merged = signal_frame.copy()
        for name, series in macro_frames.items():
            aligned = series.reindex(merged.index, method="ffill")
            ma20 = aligned.rolling(20, min_periods=1).mean()
            merged[f"{name}_ratio"] = (aligned / ma20.replace(0, np.nan)) - 1.0
            merged[f"{name}_ret3"] = aligned.pct_change(3)
        return merged

    @staticmethod
    def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Recreate the live feature pipeline from the training script."""

        df = frame.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        for window in (5, 10, 20, 60):
            moving_average = close.rolling(window, min_periods=1).mean()
            df[f"ma{window}_ratio"] = (close / moving_average.replace(0, np.nan)) - 1.0

        hour_et = df.index.hour
        minute_et = df.index.minute
        total_minutes = hour_et * 60 + minute_et
        pre_start = 4 * 60
        regular_start = 9 * 60 + 30
        regular_end = 16 * 60
        after_end = 20 * 60

        df["is_premarket"] = (
            (total_minutes >= pre_start) & (total_minutes < regular_start)
        ).astype(int)
        df["is_regular"] = (
            (total_minutes >= regular_start) & (total_minutes <= regular_end)
        ).astype(int)
        df["is_aftermarket"] = (
            (total_minutes > regular_end) & (total_minutes <= after_end)
        ).astype(int)

        pre_length = regular_start - pre_start
        regular_length = regular_end - regular_start
        after_length = after_end - regular_end
        df["session_progress"] = np.where(
            df["is_premarket"] == 1,
            (total_minutes - pre_start) / pre_length,
            np.where(
                df["is_regular"] == 1,
                (total_minutes - regular_start) / regular_length,
                np.where(
                    df["is_aftermarket"] == 1,
                    (total_minutes - regular_end) / after_length,
                    np.nan,
                ),
            ),
        )
        df["session_progress"] = df["session_progress"].clip(0.0, 1.0)

        for lag in (1, 3, 6, 12):
            df[f"ret_{lag}"] = close.pct_change(lag)

        df["gap_ratio"] = (df["open"] / close.shift(1).replace(0, np.nan)) - 1.0

        volume_ma20 = volume.rolling(20, min_periods=1).mean()
        df["vol_ratio"] = volume / volume_ma20.replace(0, np.nan)

        obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        obv_ma20 = obv.rolling(20, min_periods=1).mean()
        df["obv_ratio"] = (obv / obv_ma20.replace(0, np.nan)) - 1.0

        high_low = high - low
        high_prev_close = (high - close.shift(1)).abs()
        low_prev_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=1).mean()
        df["atr_ratio"] = atr / close.replace(0, np.nan)
        df["hl_ratio"] = high_low / close.replace(0, np.nan)

        rolling_std = close.rolling(20, min_periods=1).std().fillna(0)
        rolling_mean = close.rolling(20, min_periods=1).mean()
        df["bb_width_ratio"] = (2 * rolling_std) / rolling_mean.replace(0, np.nan)
        df["bb_position"] = (close - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-9)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist_ratio"] = (macd_line - macd_signal) / close.replace(0, np.nan)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep parity with the training script's placeholder derived-feature stage."""

        df = frame.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def _build_cnn_input(frame: pd.DataFrame, feature_cols: Sequence[str], seq_len: int) -> pd.DataFrame:
        """Build the rolling-z-scored CNN tensor inputs expected by `cnn_model.pt`."""

        raw_cols = ["open", "high", "low", "close", "volume"]
        all_cols = raw_cols + [column for column in feature_cols if column not in raw_cols]
        missing = [column for column in all_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"CNN input is missing required columns: {missing}")

        raw_inputs = frame[all_cols].copy().astype(np.float32)
        rolling_mean = raw_inputs.rolling(seq_len, min_periods=seq_len).mean()
        rolling_std = raw_inputs.rolling(seq_len, min_periods=seq_len).std().replace(0, np.nan)
        return ((raw_inputs - rolling_mean) / rolling_std).fillna(0.0)

    def _build_feature_frame(
        self,
        signal_frame: pd.DataFrame,
        macro_frames: Mapping[str, pd.Series],
    ) -> pd.DataFrame:
        """Combine raw bars and macro series into the exact feature frame used at inference."""

        merged = self._merge_macro(signal_frame, macro_frames)
        featured = self._add_features(merged)
        featured = self._add_derived_features(featured)
        featured = featured.sort_index()
        featured = featured.loc[~featured.index.duplicated(keep="last")]
        return featured

    def _xgb_probabilities(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Run the XGBoost model on the latest bar using artifact-ordered features."""

        self._ensure_artifacts_loaded()
        artifacts = self._artifacts
        missing = [column for column in artifacts.feature_cols if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns for SOXX inference: {missing}")

        feature_matrix = feature_frame.loc[:, artifacts.feature_cols].copy()
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        live_medians = feature_matrix.median(numeric_only=True).fillna(0.0)
        feature_matrix.fillna(live_medians, inplace=True)
        return artifacts.xgb_model.predict_proba(feature_matrix.iloc[[-1]].values)[0]

    def _cnn_probabilities(self, feature_frame: pd.DataFrame) -> np.ndarray | None:
        """Run the CNN branch when the artifact set includes one."""

        self._ensure_artifacts_loaded()
        artifacts = self._artifacts
        if artifacts.cnn_model is None or artifacts.seq_len is None:
            return None
        if len(feature_frame) < artifacts.seq_len:
            return None

        cnn_inputs = self._build_cnn_input(feature_frame, artifacts.feature_cols, artifacts.seq_len)
        seq = np.nan_to_num(cnn_inputs.values[-artifacts.seq_len :]).astype(np.float32)
        with torch.no_grad():
            logits = artifacts.cnn_model(torch.tensor(seq[None, :, :], dtype=torch.float32))
            return torch.softmax(logits, dim=1).cpu().numpy()[0]

    def _combine_probabilities(self, xgb_probs: np.ndarray, cnn_probs: np.ndarray | None) -> np.ndarray:
        """Blend XGBoost and CNN outputs using the saved ensemble weights."""

        self._ensure_artifacts_loaded()
        artifacts = self._artifacts
        if cnn_probs is None:
            return xgb_probs

        xgb_weight = artifacts.ensemble_weight.get("xgb", 1.0)
        cnn_weight = artifacts.ensemble_weight.get("cnn", 0.0)
        return (xgb_probs * xgb_weight) + (cnn_probs * cnn_weight)

    def _extract_directional_probabilities(self, probabilities: np.ndarray) -> tuple[float, float, float]:
        """Map model output indices back to raw label classes."""

        self._ensure_artifacts_loaded()
        mapping = dict(enumerate(self._artifacts.label_classes))
        prob_by_label = {int(raw_label): float(probabilities[index]) for index, raw_label in mapping.items()}
        return (
            prob_by_label.get(-1, 0.0),
            prob_by_label.get(0, 0.0),
            prob_by_label.get(1, 0.0),
        )

    def predict(
        self,
        *,
        signal_frame: pd.DataFrame,
        macro_frames: Mapping[str, pd.Series],
    ) -> DirectionalPrediction:
        """Run the deployed model on the latest SOXX feature row."""

        self._ensure_artifacts_loaded()
        feature_frame = self._build_feature_frame(signal_frame, macro_frames)
        xgb_probs = self._xgb_probabilities(feature_frame)
        cnn_probs = self._cnn_probabilities(feature_frame)
        combined_probs = self._combine_probabilities(xgb_probs, cnn_probs)
        prob_bear, prob_neutral, prob_bull = self._extract_directional_probabilities(combined_probs)
        bias = prob_bull - prob_bear

        entry_bull = float(self._artifacts.best_params.get("entry_bull", 0.50))
        entry_bear = float(self._artifacts.best_params.get("entry_bear", 0.50))
        no_entry_hours = {
            int(hour)
            for hour in self._artifacts.best_params.get("no_entry_hours", [])
        }
        signal_timestamp = pd.Timestamp(feature_frame.index[-1])
        entry_blocked = signal_timestamp.hour in no_entry_hours

        if prob_bull >= entry_bull and prob_bull >= prob_bear:
            target_side: TargetSide = "long"
            reason = (
                f"SOXX bullish probability {prob_bull:.2%} cleared the long threshold "
                f"{entry_bull:.0%}."
            )
        elif prob_bear >= entry_bear and prob_bear > prob_bull:
            target_side = "short"
            reason = (
                f"SOXX bearish probability {prob_bear:.2%} cleared the short threshold "
                f"{entry_bear:.0%}."
            )
        else:
            target_side = "flat"
            reason = "Neither bullish nor bearish probability cleared the entry thresholds."

        if entry_blocked and target_side != "flat":
            reason = f"{reason} New entries are blocked during hour {signal_timestamp.hour:02d}:00 ET."

        return DirectionalPrediction(
            signal_symbol=self.signal_symbol,
            signal_timestamp=signal_timestamp,
            target_side=target_side,
            prob_bull=prob_bull,
            prob_bear=prob_bear,
            prob_neutral=prob_neutral,
            bias=bias,
            entry_blocked=entry_blocked,
            reason=reason,
        )

    def _signal_for_symbol(
        self,
        *,
        symbol: str,
        execution_closes: Sequence[float],
        prediction: DirectionalPrediction,
        has_position: bool,
    ) -> ModelDrivenSignal:
        """Map the shared directional prediction to one tradable execution symbol."""

        last_close = float(execution_closes[-1])
        entry_bull = float(self._artifacts.best_params.get("entry_bull", 0.50))
        entry_bear = float(self._artifacts.best_params.get("entry_bear", 0.50))
        exit_threshold = float(self._artifacts.best_params.get("exit_threshold", 0.40))

        bullish_entry = prediction.prob_bull >= entry_bull
        bearish_entry = (not bullish_entry) and prediction.prob_bear >= entry_bear

        if symbol == self.long_symbol:
            if has_position and prediction.prob_bear >= entry_bear:
                action = SignalAction.SELL
                entry_candidate = False
                reason = f"Bearish reversal detected: bear {prediction.prob_bear:.2%} crossed the short entry threshold."
            elif has_position and prediction.prob_bull < exit_threshold:
                action = SignalAction.SELL
                entry_candidate = False
                reason = f"Bullish confidence faded below the exit threshold: {prediction.prob_bull:.2%} < {exit_threshold:.0%}."
            elif has_position:
                action = SignalAction.HOLD
                entry_candidate = True
                reason = f"Model still supports holding {symbol} from the {prediction.signal_symbol} signal."
            elif bullish_entry and not prediction.entry_blocked:
                action = SignalAction.BUY
                entry_candidate = True
                reason = prediction.reason
            else:
                action = SignalAction.HOLD
                entry_candidate = False
                reason = prediction.reason if bullish_entry else f"{symbol} is not the active side of the current {prediction.signal_symbol} signal."
        else:
            if has_position and prediction.prob_bull >= entry_bull:
                action = SignalAction.SELL
                entry_candidate = False
                reason = f"Bullish reversal detected: bull {prediction.prob_bull:.2%} crossed the long entry threshold."
            elif has_position and prediction.prob_bear < exit_threshold:
                action = SignalAction.SELL
                entry_candidate = False
                reason = f"Bearish confidence faded below the exit threshold: {prediction.prob_bear:.2%} < {exit_threshold:.0%}."
            elif has_position:
                action = SignalAction.HOLD
                entry_candidate = True
                reason = f"Model still supports holding {symbol} from the {prediction.signal_symbol} signal."
            elif bearish_entry and not prediction.entry_blocked:
                action = SignalAction.BUY
                entry_candidate = True
                reason = prediction.reason
            else:
                action = SignalAction.HOLD
                entry_candidate = False
                reason = prediction.reason if bearish_entry else f"{symbol} is not the active side of the current {prediction.signal_symbol} signal."

        score = prediction.prob_bull if symbol == self.long_symbol else prediction.prob_bear
        return ModelDrivenSignal(
            action=action,
            reason=reason,
            score=score,
            score_label="상승 확률" if symbol == self.long_symbol else "하락 확률",
            entry_threshold=entry_bull if symbol == self.long_symbol else entry_bear,
            last_close=last_close,
            entry_candidate=entry_candidate,
            momentum_return=prediction.bias,
            fast_ma=prediction.prob_bull,
            slow_ma=prediction.prob_bear,
            signal_symbol=prediction.signal_symbol,
            target_symbol=symbol,
            prob_bull=prediction.prob_bull,
            prob_bear=prediction.prob_bear,
            prob_neutral=prediction.prob_neutral,
            metric_primary_label="P bull",
            metric_primary_value=prediction.prob_bull,
            metric_secondary_label="P bear",
            metric_secondary_value=prediction.prob_bear,
            metric_bias_label="Bias",
            metric_bias_value=prediction.bias,
        )

    def build_execution_signals(
        self,
        *,
        prediction: DirectionalPrediction,
        execution_closes_by_symbol: Mapping[str, Sequence[float]],
        positions_by_symbol: Mapping[str, Any],
    ) -> list[tuple[str, Sequence[float], ModelDrivenSignal]]:
        """Build SOXL/SOXS execution signals from the shared SOXX model output."""

        evaluations: list[tuple[str, Sequence[float], ModelDrivenSignal]] = []
        for symbol in self.execution_symbols:
            closes = execution_closes_by_symbol[symbol]
            signal = self._signal_for_symbol(
                symbol=symbol,
                execution_closes=closes,
                prediction=prediction,
                has_position=symbol in positions_by_symbol,
            )
            evaluations.append((symbol, closes, signal))
        return evaluations
