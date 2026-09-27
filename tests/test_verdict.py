"""Bands, windows, histories, record filters and process batches."""

import pytest

from paperops.errors import DuplicateBatchError, OutOfRangeError, UnknownBatchError
from paperops.verdict import (
    ABOVE,
    ABOVE_MODE,
    BELOW,
    BELOW_MODE,
    DRIFT_HIGH,
    DRIFT_LOW,
    INSUFFICIENT,
    LEARNING,
    STEADY,
    SUSTAINED,
    TRANSIENT,
    UNSETTLED,
    WITHIN,
    AdaptiveBaseline,
    Band,
    BatchRegistry,
    DriftEvaluator,
    HysteresisBand,
    RecordFilter,
    VerdictHistory,
    Window,
    WindowEvaluator,
    compare,
    parse_filter,
)


def test_band_classifies_and_requires():
    band = Band(4.0, 14.0)
    assert band.contains(4.0) is True
    assert band.classify(2.0) == BELOW
    assert band.classify(6.0) == WITHIN
    assert band.classify(20.0) == ABOVE
    with pytest.raises(OutOfRangeError):
        band.require(20.0, "glue concentration")
    assert band.low == 4.0 and band.high == 14.0
    with pytest.raises(ValueError):
        Band(5.0, 1.0)


def test_hysteresis_band_holds_the_previous_verdict():
    band = HysteresisBand(100.0, 800.0, 10.0)
    assert band.classify(900.0) == ABOVE
    assert band.classify(805.0) == ABOVE
    assert band.classify(400.0) == ABOVE
    assert band.classify(80.0) == BELOW
    assert band.last == BELOW


def test_window_keeps_the_most_recent_samples():
    window = Window(3)
    assert window.is_full() is False
    for value in (1.0, 2.0, 3.0, 4.0):
        window.push(value)
    assert window.values() == (2.0, 3.0, 4.0)
    assert window.average() == 3.0
    assert window.span() == 2.0
    assert window.is_full() is True
    assert len(window) == 3
    window.clear()
    assert window.values() == ()
    assert window.average() == 0.0
    assert window.span() == 0.0
    with pytest.raises(ValueError):
        Window(0)


def test_window_evaluator_needs_a_full_window():
    evaluator = WindowEvaluator(Window(3), 12.0, ABOVE_MODE, 3)
    assert evaluator.push(20.0).verdict == INSUFFICIENT
    assert evaluator.push(20.0).verdict == INSUFFICIENT
    result = evaluator.push(20.0)
    assert result.verdict == SUSTAINED
    assert result.samples == 3 and result.required == 3
    assert evaluator.push(3.0).verdict == TRANSIENT
    assert evaluator.window.values() == (20.0, 20.0, 3.0)
    assert evaluator.threshold == 12.0
    assert evaluator.mode == ABOVE_MODE


def test_compare_guards_the_mode():
    assert compare([5.0, 6.0], 4.0, ABOVE_MODE) is True
    assert compare([5.0, 3.0], 4.0, ABOVE_MODE) is False
    assert compare([1.0, 2.0], 4.0, BELOW_MODE) is True
    assert compare([], 4.0, ABOVE_MODE) is False
    with pytest.raises(ValueError):
        compare([1.0], 2.0, "sideways")
    with pytest.raises(ValueError):
        WindowEvaluator(Window(2), 1.0, "sideways", 1)


def test_verdict_history_tracks_watermarks():
    history = VerdictHistory("water.moisture")
    assert history.current() == "unknown"
    history.record("in_band", 1, 1, "reading=6")
    history.record("high", 4, 3, "reading=20")
    assert history.current() == "high"
    assert history.at(0) == "unknown"
    assert history.at(2) == "in_band"
    assert history.changed_since(2) is True
    assert history.subject == "water.moisture"
    assert len(history.points()) == 2
    assert history.to_dict()["current"] == "high"
    history.clear()
    assert history.current() == "unknown"


def test_verdict_history_explains_the_compared_level():
    history = VerdictHistory("water.moisture.drift")
    history.record("steady", 1, 1, "reading=6.0", context={"level": 6.0, "low": 5.7, "high": 6.3})
    history.record("drift_high", 4, 3, "reading=9.8", context={"level": 6.1, "low": 5.8, "high": 6.4})
    point = history.explain_at(4)
    assert point is not None
    assert point.verdict == "drift_high"
    assert point.context["high"] == 6.4
    assert point.to_dict()["context"]["level"] == 6.1
    earlier = history.explain_at(2)
    assert earlier is not None and earlier.verdict == "steady"
    assert history.explain_at(0) is None
    plain = VerdictHistory("steam.pressure")
    plain.record("in_band", 1, 1)
    assert plain.explain_at(1).context == {}


def test_adaptive_baseline_learns_the_normal_level():
    baseline = AdaptiveBaseline(margin=3.0, minimum_samples=3, learn_rate=0.5, min_margin=0.5)
    assert baseline.ready is False
    first = baseline.observe(10.0)
    assert first.ready is False and first.samples == 0
    baseline.observe(10.2)
    snapshot = baseline.observe(9.8)
    assert snapshot.ready is False
    snapshot = baseline.observe(10.1)
    assert snapshot.ready is True
    assert snapshot.low <= 10.1 <= snapshot.high
    assert baseline.level == pytest.approx(10.0, abs=0.2)
    band = baseline.band()
    assert band.low == pytest.approx(baseline.level - baseline.half_width())


def test_adaptive_baseline_never_learns_from_out_of_band_readings():
    baseline = AdaptiveBaseline(margin=3.0, minimum_samples=3, learn_rate=0.5, min_margin=0.5)
    for reading in (10.0, 10.0, 10.0):
        baseline.observe(reading)
    level = baseline.level
    samples = baseline.samples
    snapshot = baseline.observe(20.0)
    assert 20.0 > snapshot.high
    assert baseline.level == level
    assert baseline.samples == samples
    baseline.observe(10.1)
    assert baseline.samples == samples + 1


def test_adaptive_baseline_reset_forgets_the_level():
    baseline = AdaptiveBaseline(margin=3.0, minimum_samples=2, min_margin=0.5)
    baseline.observe(10.0)
    baseline.observe(10.0)
    assert baseline.ready is True
    baseline.reset()
    assert baseline.ready is False
    assert baseline.samples == 0
    with pytest.raises(ValueError):
        AdaptiveBaseline(margin=0.0, minimum_samples=2)
    with pytest.raises(ValueError):
        AdaptiveBaseline(margin=3.0, minimum_samples=0)
    with pytest.raises(ValueError):
        AdaptiveBaseline(margin=3.0, minimum_samples=2, learn_rate=0.0)
    with pytest.raises(ValueError):
        AdaptiveBaseline(margin=3.0, minimum_samples=2, min_margin=-1.0)


def drift_evaluator(window_size=3):
    return DriftEvaluator(
        AdaptiveBaseline(margin=3.0, minimum_samples=3, learn_rate=0.2, min_margin=0.5),
        Window(window_size),
    )


def test_drift_evaluator_waits_for_a_sustained_departure():
    evaluator = drift_evaluator()
    for _ in range(3):
        assert evaluator.push(10.0).verdict == LEARNING
    assert evaluator.push(10.0).verdict == LEARNING
    assert evaluator.push(10.0).verdict == LEARNING
    assert evaluator.push(10.0).verdict == STEADY
    assert evaluator.push(20.0).verdict == UNSETTLED
    assert evaluator.push(20.0).verdict == UNSETTLED
    verdict = evaluator.push(20.0)
    assert verdict.verdict == DRIFT_HIGH
    assert verdict.snapshot.level == pytest.approx(10.0)
    assert verdict.reading == 20.0
    assert verdict.deviation == pytest.approx(10.0)
    assert verdict.window_samples == 3
    body = verdict.to_dict()
    assert body["baseline"]["high"] == pytest.approx(10.5)


def test_drift_evaluator_reports_low_and_recovers():
    evaluator = drift_evaluator()
    for _ in range(6):
        evaluator.push(10.0)
    for _ in range(3):
        verdict = evaluator.push(1.0)
    assert verdict.verdict == DRIFT_LOW
    for _ in range(3):
        verdict = evaluator.push(10.0)
    assert verdict.verdict == STEADY


def test_drift_evaluator_relearns_after_a_reset():
    evaluator = drift_evaluator()
    for _ in range(6):
        evaluator.push(10.0)
    evaluator.reset()
    verdicts = [evaluator.push(6.0).verdict for _ in range(6)]
    assert verdicts[-1] == STEADY
    assert DRIFT_LOW not in verdicts and DRIFT_HIGH not in verdicts


def test_record_filter_matches_and_describes():
    record_filter = RecordFilter(
        topic="steam.valve.changed",
        key_prefix="steam/",
        min_generation=1,
        kinds=("event",),
    )
    row = {
        "topic": "steam.valve.changed",
        "key": "steam/valve",
        "generation": 2,
        "seq": 5,
        "kind": "event",
    }
    assert record_filter.matches(row) is True
    assert record_filter.matches({**row, "generation": 0}) is False
    assert record_filter.matches({**row, "key": "dry/speed"}) is False
    assert record_filter.apply([row]) == [row]
    described = record_filter.describe()
    assert described["kinds"] == ["event"]


def test_parse_filter_reads_query_parameters():
    parsed = parse_filter(
        {"topic": "broke.handled", "min_seq": "3", "max_seq": "", "kind": "event,tombstone"}
    )
    assert parsed.topic == "broke.handled"
    assert parsed.min_seq == 3
    assert parsed.max_seq is None
    assert parsed.kinds == ("event", "tombstone")


def test_batch_registry_reserves_each_batch_once(journal):
    batches = BatchRegistry(journal)
    batches.reserve("B-1", {"volume": 12.0})
    assert batches.is_reserved("B-1") is True
    assert batches.payload("B-1")["volume"] == 12.0
    with pytest.raises(DuplicateBatchError):
        batches.reserve("B-1", {"volume": 1.0})
    assert batches.reserved_ids() == ("B-1",)
    assert len(batches.entries()) == 1
    batches.release("B-1")
    assert batches.is_reserved("B-1") is False
    with pytest.raises(UnknownBatchError):
        batches.release("B-2")
