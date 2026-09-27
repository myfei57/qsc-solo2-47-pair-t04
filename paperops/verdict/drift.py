"""Sustained departure verdicts built on an adaptive baseline."""

from dataclasses import dataclass
from typing import Any

from .baseline import AdaptiveBaseline, BaselineSnapshot
from .window import ABOVE_MODE, BELOW_MODE, Window, compare

LEARNING = "learning"
STEADY = "steady"
UNSETTLED = "unsettled"
DRIFT_HIGH = "drift_high"
DRIFT_LOW = "drift_low"


@dataclass(frozen=True)
class DriftVerdict:
    """Outcome of one reading, together with the level it was compared against."""

    verdict: str
    reading: float
    deviation: float
    snapshot: BaselineSnapshot
    window_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reading": self.reading,
            "deviation": self.deviation,
            "baseline": self.snapshot.to_dict(),
            "window_samples": self.window_samples,
        }


class DriftEvaluator:
    """Alarms only when a whole window of readings sits past the baseline band.

    Each ready reading enters the window as a direction: above the band, below
    it or inside. A drift verdict needs a full window on one side, so a lone
    spike reads as unsettled while a departure that holds turns into an alarm.
    """

    def __init__(self, baseline: AdaptiveBaseline, window: Window) -> None:
        self._baseline = baseline
        self._window = window

    @property
    def baseline(self) -> AdaptiveBaseline:
        return self._baseline

    @property
    def window(self) -> Window:
        return self._window

    def push(self, reading: float) -> DriftVerdict:
        value = float(reading)
        snapshot = self._baseline.observe(value)
        if snapshot.ready:
            self._window.push(self._direction(value, snapshot))
        return DriftVerdict(
            verdict=self._judge(snapshot),
            reading=value,
            deviation=value - snapshot.level,
            snapshot=snapshot,
            window_samples=len(self._window),
        )

    def reset(self, level: float | None = None) -> None:
        self._baseline.reset(level)
        self._window.clear()

    @staticmethod
    def _direction(reading: float, snapshot: BaselineSnapshot) -> float:
        if reading > snapshot.high:
            return 1.0
        if reading < snapshot.low:
            return -1.0
        return 0.0

    def _judge(self, snapshot: BaselineSnapshot) -> str:
        if not snapshot.ready or not self._window.is_full():
            return LEARNING
        values = self._window.values()
        if compare(values, 0.0, ABOVE_MODE):
            return DRIFT_HIGH
        if compare(values, 0.0, BELOW_MODE):
            return DRIFT_LOW
        if all(value == 0.0 for value in values):
            return STEADY
        return UNSETTLED
