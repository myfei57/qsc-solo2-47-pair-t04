"""Adaptive drift verdicts: learning the normal, alarming on held departures."""

import pytest

from paperops.verdict import DRIFT_HIGH, DRIFT_LOW, LEARNING, STEADY, UNSETTLED

from support import ok


def push_moisture(water, reading, count):
    return [water.moisture_drift(reading).verdict for _ in range(count)]


def push_pressure(steam, reading, count):
    return [steam.pressure_drift(reading).verdict for _ in range(count)]


def test_moisture_drift_learns_the_normal_then_alarms_on_a_held_shift(services):
    water = services.water
    verdicts = push_moisture(water, 6.0, 9)
    assert verdicts[:5] == [LEARNING] * 5
    assert verdicts[-1] == STEADY
    assert water.moisture_drift(9.8).verdict == UNSETTLED
    verdicts = push_moisture(water, 9.8, 3)
    assert verdicts[:2] == [UNSETTLED, UNSETTLED]
    assert verdicts[-1] == DRIFT_HIGH


def test_moisture_drift_relearns_after_a_batch_change(services):
    water = services.water
    push_moisture(water, 6.0, 9)
    services.batches.reserve("B-100", {"grade": "kraft"})
    verdicts = push_moisture(water, 9.0, 9)
    assert verdicts[-1] == STEADY
    assert DRIFT_HIGH not in verdicts and DRIFT_LOW not in verdicts


def test_steam_drift_learns_without_a_calibration(services):
    steam = services.steam
    verdicts = push_pressure(steam, 6.0, 9)
    assert verdicts[-1] == STEADY
    verdicts = push_pressure(steam, 6.9, 4)
    assert verdicts[-1] == DRIFT_HIGH


def test_drift_verdicts_remember_the_level_they_used(services, console):
    water = services.water
    push_moisture(water, 6.0, 9)
    push_moisture(water, 9.8, 4)
    point = water.drift_history().points()[-1]
    assert point.verdict == DRIFT_HIGH
    assert point.context["ready"] is True
    assert point.context["level"] == pytest.approx(6.0, abs=0.2)
    assert point.context["high"] < 9.8
    body = ok(
        console,
        "GET",
        f"/api/verdict/history?subject=water.moisture.drift&at={point.watermark}",
    )
    assert body["at"]["verdict"] == DRIFT_HIGH
    assert body["at"]["context"]["level"] == point.context["level"]


def test_drift_verdict_is_served_on_the_water_endpoint(console, services):
    for _ in range(9):
        ok(console, "GET", "/api/water/verdict?reading=6.0")
    body = ok(console, "GET", "/api/water/verdict?reading=6.0")
    assert body["drift"]["verdict"] == STEADY
    assert body["drift"]["baseline"]["ready"] is True
