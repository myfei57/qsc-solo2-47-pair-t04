"""Self-taught normal levels for slowly drifting process signals."""

from dataclasses import dataclass
from typing import Any

from .threshold import Band


@dataclass(frozen=True)
class BaselineSnapshot:
    """The normal level a reading was compared against, frozen at decision time."""

    level: float
    spread: float
    low: float
    high: float
    samples: int
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "spread": self.spread,
            "low": self.low,
            "high": self.high,
            "samples": self.samples,
            "ready": self.ready,
        }


class AdaptiveBaseline:
    """Learns the recent normal level and only learns from readings inside it.

    The band half width is the larger of margin times the learned spread and a
    fixed floor, so a quiet signal gets a tight band and a noisy one a loose
    band. Readings outside the band never move the level, which keeps a real
    departure from dragging the remembered normal along with it.
    """

    def __init__(
        self,
        margin: float,
        minimum_samples: int,
        learn_rate: float = 0.2,
        min_margin: float = 0.0,
    ) -> None:
        if margin <= 0:
            raise ValueError("margin must be positive")
        if minimum_samples < 1:
            raise ValueError("minimum samples must be at least one")
        if not 0.0 < learn_rate <= 1.0:
            raise ValueError("learn rate must be in (0, 1]")
        if min_margin < 0:
            raise ValueError("min margin must not be negative")
        self._margin = float(margin)
        self._minimum = int(minimum_samples)
        self._learn_rate = float(learn_rate)
        self._min_margin = float(min_margin)
        self._level = 0.0
        self._spread = 0.0
        self._samples = 0

    @property
    def level(self) -> float:
        return self._level

    @property
    def spread(self) -> float:
        return self._spread

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def ready(self) -> bool:
        return self._samples >= self._minimum

    def half_width(self) -> float:
        return max(self._margin * self._spread, self._min_margin)

    def band(self) -> Band:
        half = self.half_width()
        return Band(self._level - half, self._level + half)

    def observe(self, reading: float) -> BaselineSnapshot:
        """Compare a reading against the current level, then learn if it belongs."""
        value = float(reading)
        half = self.half_width()
        snapshot = BaselineSnapshot(
            level=self._level,
            spread=self._spread,
            low=self._level - half,
            high=self._level + half,
            samples=self._samples,
            ready=self.ready,
        )
        if not self.ready or snapshot.low <= value <= snapshot.high:
            self._learn(value)
        return snapshot

    def reset(self, level: float | None = None) -> None:
        """Forget the learned level, e.g. when a new batch brings a new normal."""
        self._level = 0.0 if level is None else float(level)
        self._spread = 0.0
        self._samples = 0

    def _learn(self, reading: float) -> None:
        if self._samples == 0:
            self._level = reading
        else:
            deviation = reading - self._level
            self._level += self._learn_rate * deviation
            self._spread += self._learn_rate * (abs(deviation) - self._spread)
        self._samples += 1
