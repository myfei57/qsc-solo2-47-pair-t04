"""Self-tuning baselines and the sustained drift rule.

A fixed limit band has to be retuned by hand whenever the furnish changes, and a
reading that only visits the band edge for a scan or two is noise, not a fault.
The adaptive baseline learns the recent normal level of one signal from the
readings that classify as normal, and refuses to learn from readings that sit
outside the band, so a genuine drift can never pull the baseline towards itself
and hide. A drift is only called once the readings leave the band on one side
and stay there for a run of consecutive samples. Every judgement carries the
level and band it was compared against, so a historical verdict can always be
traced back to the normal that was in force at the time.
"""

from dataclasses import dataclass
from typing import Any

LEARNING = "learning"
TRACKING = "tracking"
WATCH = "watch"
DRIFT_HIGH = "drift_high"
DRIFT_LOW = "drift_low"


@dataclass(frozen=True)
class BaselineVerdict:
    """One reading judged against the baseline in force at that moment."""

    verdict: str
    reading: float
    level: float
    scatter: float
    low: float
    high: float
    residual: float
    run: int
    hold: int
    samples: int
    generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reading": self.reading,
            "level": self.level,
            "scatter": self.scatter,
            "low": self.low,
            "high": self.high,
            "residual": self.residual,
            "run": self.run,
            "hold": self.hold,
            "samples": self.samples,
            "generation": self.generation,
        }

    def describe(self) -> str:
        return (
            f"reading={self.reading} level={self.level:.3f} "
            f"band=[{self.low:.3f},{self.high:.3f}] gen={self.generation}"
        )


class AdaptiveBaseline:
    """Learns the recent normal level of one signal and judges drift away from it."""

    def __init__(
        self,
        alpha: float,
        width_k: float,
        floor: float,
        warmup: int,
        hold: int,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if float(width_k) <= 0.0:
            raise ValueError("width_k must be positive")
        if float(floor) < 0.0:
            raise ValueError("floor must not be negative")
        if int(warmup) < 1:
            raise ValueError("warmup must be at least one")
        if int(hold) < 1:
            raise ValueError("hold must be at least one")
        self._alpha = float(alpha)
        self._width_k = float(width_k)
        self._floor = float(floor)
        self._warmup = int(warmup)
        self._hold = int(hold)
        self._level = 0.0
        self._scatter = 0.0
        self._samples = 0
        self._run = 0
        self._generation = 1

    # ----------------------------------------------------------------- reading

    @property
    def level(self) -> float:
        return self._level

    @property
    def scatter(self) -> float:
        return self._scatter

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def run(self) -> int:
        return self._run

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def half_width(self) -> float:
        return max(self._width_k * self._scatter, self._floor)

    def band(self) -> tuple[float, float]:
        half = self.half_width
        return (self._level - half, self._level + half)

    def snapshot(self) -> dict[str, Any]:
        """The learned state, for durable storage and later restore."""
        return {
            "level": self._level,
            "scatter": self._scatter,
            "samples": self._samples,
            "run": self._run,
            "generation": self._generation,
        }

    # ------------------------------------------------------------------ writes

    def observe(self, reading: float) -> BaselineVerdict:
        """Judge a reading against the learned normal, then learn if it is normal."""
        value = float(reading)
        residual = value - self._level
        if self._samples < self._warmup:
            verdict = self._judge(LEARNING, value, residual)
            self._absorb(value, 1.0 / (self._samples + 1))
            return verdict
        if abs(residual) <= self.half_width:
            verdict = self._judge(TRACKING, value, residual)
            self._run = 0
            self._absorb(value, self._alpha)
            return verdict
        # Out of band: the baseline stays put so a drift cannot teach it.
        side = 1 if residual > 0 else -1
        self._run = self._run + side if self._run * side > 0 else side
        name = WATCH
        if abs(self._run) >= self._hold:
            name = DRIFT_HIGH if side > 0 else DRIFT_LOW
        return self._judge(name, value, residual)

    def reset(self, hint: float | None = None) -> int:
        """Forget the learned level and re-learn, as when the furnish changes.

        A hint seeds the level with the target of the new grade and counts as
        the first sample, so the re-learn converges one reading sooner.
        """
        self._level = 0.0 if hint is None else float(hint)
        self._scatter = 0.0
        self._samples = 0 if hint is None else 1
        self._run = 0
        self._generation += 1
        return self._generation

    def restore(
        self,
        level: float,
        scatter: float,
        samples: int,
        generation: int,
        run: int,
    ) -> None:
        """Adopt a previously stored snapshot, as after a restart."""
        self._level = float(level)
        self._scatter = float(scatter)
        self._samples = int(samples)
        self._generation = int(generation)
        self._run = int(run)

    # ---------------------------------------------------------------- internals

    def _absorb(self, value: float, weight: float) -> None:
        if self._samples == 0:
            self._level = value
            self._scatter = 0.0
        else:
            residual = value - self._level
            self._level += weight * residual
            self._scatter += weight * (abs(residual) - self._scatter)
        self._samples += 1

    def _judge(self, name: str, value: float, residual: float) -> BaselineVerdict:
        low, high = self.band()
        return BaselineVerdict(
            verdict=name,
            reading=value,
            level=self._level,
            scatter=self._scatter,
            low=low,
            high=high,
            residual=residual,
            run=self._run,
            hold=self._hold,
            samples=self._samples,
            generation=self._generation,
        )
