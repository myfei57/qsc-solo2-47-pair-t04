"""Adaptive baselines, the sustained drift rule and self-explaining judgements."""

import pytest

from paperops.verdict import (
    DRIFT_HIGH,
    DRIFT_LOW,
    LEARNING,
    TRACKING,
    WATCH,
    AdaptiveBaseline,
)

from support import build


def learn(baseline, readings):
    return [baseline.observe(reading) for reading in readings]


def test_baseline_learns_the_recent_normal():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    for verdict in learn(baseline, (6.0, 6.2, 5.8, 6.1, 5.9)):
        assert verdict.verdict == LEARNING
    assert baseline.level == pytest.approx(6.0)
    verdict = baseline.observe(6.05)
    assert verdict.verdict == TRACKING
    assert verdict.low < 6.05 < verdict.high


def test_a_spike_is_a_watch_and_never_teaches_the_baseline():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.0, 6.0, 6.0, 6.0))
    level = baseline.level
    verdict = baseline.observe(9.0)
    assert verdict.verdict == WATCH
    assert verdict.run == 1
    assert baseline.level == level
    assert baseline.observe(6.0).verdict == TRACKING
    assert baseline.run == 0


def test_a_drift_is_called_only_once_the_readings_stay_out():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.0, 6.0, 6.0, 6.0))
    assert baseline.observe(9.0).verdict == WATCH
    assert baseline.observe(9.0).verdict == WATCH
    verdict = baseline.observe(9.0)
    assert verdict.verdict == DRIFT_HIGH
    assert verdict.run == 3
    assert baseline.observe(9.0).verdict == DRIFT_HIGH


def test_a_drift_low_is_called_on_the_low_side():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.0, 6.0, 6.0, 6.0))
    assert baseline.observe(3.0).verdict == WATCH
    assert baseline.observe(3.0).verdict == WATCH
    assert baseline.observe(3.0).verdict == DRIFT_LOW


def test_the_baseline_never_chases_a_drift():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.2, 5.8, 6.1, 5.9))
    verdicts = learn(baseline, [9.0] * 10)
    assert [verdict.verdict for verdict in verdicts] == [WATCH, WATCH] + [DRIFT_HIGH] * 8
    assert baseline.level == pytest.approx(6.0)


def test_a_direction_flip_restarts_the_run():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.1, warmup=2, hold=2)
    learn(baseline, (5.0, 5.0))
    assert baseline.observe(6.0).verdict == WATCH
    assert baseline.observe(6.0).verdict == DRIFT_HIGH
    verdict = baseline.observe(4.0)
    assert verdict.verdict == WATCH
    assert verdict.run == -1
    assert baseline.observe(4.0).verdict == DRIFT_LOW


def test_reset_relearns_a_new_grade_within_the_warmup():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=3, hold=2)
    learn(baseline, (6.0, 6.0, 6.0))
    assert baseline.observe(6.0).verdict == TRACKING
    assert baseline.reset() == 2
    for verdict in learn(baseline, (8.0, 8.0, 8.0)):
        assert verdict.verdict == LEARNING
        assert verdict.generation == 2
    assert baseline.observe(8.0).verdict == TRACKING
    assert baseline.level == pytest.approx(8.0)


def test_reset_can_seed_the_new_grade_as_a_hint():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=3, hold=2)
    learn(baseline, (6.0, 6.0, 6.0, 6.0))
    baseline.reset(hint=8.0)
    assert baseline.samples == 1
    assert baseline.observe(8.1).verdict == LEARNING
    assert baseline.level == pytest.approx(8.05)


def test_the_verdict_carries_the_baseline_it_was_compared_against():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.0, 6.0, 6.0, 6.0))
    verdict = baseline.observe(6.1)
    assert verdict.verdict == TRACKING
    assert verdict.level == pytest.approx(6.0)
    assert verdict.low == pytest.approx(5.7)
    assert verdict.high == pytest.approx(6.3)
    assert verdict.residual == pytest.approx(0.1)
    assert verdict.generation == 1
    assert verdict.samples == 5
    row = verdict.to_dict()
    assert row["level"] == pytest.approx(6.0)
    assert "level=6.000" in verdict.describe()
    assert "band=[5.700,6.300]" in verdict.describe()


def test_snapshot_and_restore_keep_the_learned_state():
    baseline = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    learn(baseline, (6.0, 6.2, 5.8, 6.1, 5.9, 6.05))
    restored = AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=3)
    restored.restore(**baseline.snapshot())
    assert restored.level == baseline.level
    assert restored.scatter == baseline.scatter
    assert restored.samples == baseline.samples
    assert restored.generation == baseline.generation
    assert restored.observe(6.0).verdict == baseline.observe(6.0).verdict


def test_baseline_parameters_are_checked():
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=0.0, width_k=3.0, floor=0.3, warmup=5, hold=3)
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=1.5, width_k=3.0, floor=0.3, warmup=5, hold=3)
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=0.25, width_k=0.0, floor=0.3, warmup=5, hold=3)
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=-0.1, warmup=5, hold=3)
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=0, hold=3)
    with pytest.raises(ValueError):
        AdaptiveBaseline(alpha=0.25, width_k=3.0, floor=0.3, warmup=5, hold=0)


def test_moisture_drift_is_judged_against_the_learned_level(services):
    water = services.water
    for reading in (6.0, 6.1, 5.9, 6.0, 6.1):
        assert water.judge_moisture(reading).verdict == LEARNING
    assert water.judge_moisture(6.05).verdict == TRACKING
    assert water.judge_moisture(9.5).verdict == WATCH
    assert water.judge_moisture(9.5).verdict == WATCH
    assert water.judge_moisture(9.5).verdict == DRIFT_HIGH
    baseline = water.drift_baseline()
    assert baseline["level"] == pytest.approx(6.0275, abs=0.001)
    assert baseline["samples"] == 6
    assert services.state.verdict_history("water.moisture.drift")["current"] == DRIFT_HIGH


def test_stored_judgements_explain_what_they_compared_against(services):
    water = services.water
    for reading in (6.0, 6.1, 5.9, 6.0, 6.1, 6.05):
        water.judge_moisture(reading)
    judgments = water.drift_judgments()
    assert len(judgments) == 6
    last = judgments[-1]
    assert last["verdict"] == TRACKING
    assert last["low"] <= last["reading"] <= last["high"]
    assert last["level"] == pytest.approx(6.02, abs=0.001)
    assert last["generation"] == 1
    assert "tick" in last


def test_drift_baseline_and_judgements_survive_a_restart(config):
    first = build(config)
    for reading in (6.0, 6.1, 5.9, 6.0, 6.1, 6.05):
        first.water.judge_moisture(reading)
    second = build(config)
    baseline = second.water.drift_baseline()
    assert baseline["samples"] == 6
    assert baseline["level"] == pytest.approx(6.0275, abs=0.001)
    assert len(second.water.drift_judgments()) == 6
    assert second.water.judge_moisture(6.0).verdict == TRACKING


def test_drift_judgements_reach_the_audit_log(services):
    for reading in (6.0, 6.1, 5.9, 6.0, 6.1, 6.05):
        services.water.judge_moisture(reading)
    records = services.audit.by_topic("water.drift.judged")
    assert len(records) == 6
    assert records[0].payload["level"] == pytest.approx(6.02, abs=0.001)
    assert records[0].payload["verdict"] == TRACKING


def test_relearn_forgets_the_grade_but_keeps_the_trail(services):
    water = services.water
    for reading in (6.0, 6.1, 5.9, 6.0, 6.1, 6.05):
        water.judge_moisture(reading)
    assert water.relearn_moisture()["generation"] == 2
    assert water.drift_baseline()["samples"] == 0
    for reading in (8.0, 8.1, 7.9, 8.0, 8.1):
        verdict = water.judge_moisture(reading)
        assert verdict.verdict == LEARNING
        assert verdict.generation == 2
    assert water.judge_moisture(8.05).verdict == TRACKING
    judgments = water.drift_judgments()
    assert len(judgments) == 12
    assert judgments[5]["generation"] == 1
    assert judgments[6]["generation"] == 2


def test_pressure_drift_uses_its_own_learned_level(services):
    steam = services.steam
    for reading in (4.0, 4.1, 3.9, 4.0, 4.1):
        assert steam.judge_pressure(reading).verdict == LEARNING
    assert steam.judge_pressure(4.05).verdict == TRACKING
    assert steam.judge_pressure(2.0).verdict == WATCH
    assert steam.judge_pressure(2.0).verdict == WATCH
    assert steam.judge_pressure(2.0).verdict == DRIFT_LOW
    assert steam.drift_baseline()["level"] == pytest.approx(4.0275, abs=0.001)
    assert steam.relearn_pressure()["generation"] == 2
    assert steam.drift_baseline()["samples"] == 0
    assert steam.judge_pressure(4.0).verdict == LEARNING
    assert services.state.verdict_history("steam.pressure.drift")["current"] == LEARNING
