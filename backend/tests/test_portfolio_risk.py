"""Adapter tests: performance series to risk metrics."""

from datetime import date, timedelta

import pytest

from app.services.portfolio_risk import risk_for_cumulative_series


def _series(cumulative_returns):
    start = date(2026, 1, 2)
    return [(start + timedelta(days=index), value) for index, value in enumerate(cumulative_returns)]


def test_a_cumulative_series_becomes_the_index_levels():
    metrics = risk_for_cumulative_series(_series([0.0, 10.0, -1.0, 8.9]))

    assert metrics is not None
    assert metrics.observations == 3
    assert metrics.max_drawdown == pytest.approx(-0.10)
    assert metrics.annualized_volatility is not None
    assert metrics.sharpe is not None


def test_a_single_observation_has_no_risk_metrics():
    assert risk_for_cumulative_series(_series([0.0])) is None
    assert risk_for_cumulative_series([]) is None


def test_a_series_that_is_not_a_valid_index_returns_none():
    assert risk_for_cumulative_series(_series([0.0, -150.0, -20.0])) is None


def test_risk_free_rate_is_passed_through():
    series = _series([0.0, 1.0, -0.5, 1.2, 0.4, 2.0])

    without_rf = risk_for_cumulative_series(series)
    with_rf = risk_for_cumulative_series(series, risk_free_rate=0.10)

    assert without_rf is not None and with_rf is not None
    assert with_rf.sharpe is not None and without_rf.sharpe is not None
    assert with_rf.sharpe < without_rf.sharpe


def test_a_monthly_series_is_not_annualized_as_if_it_were_daily():
    # 13 monthly points over a year: 12 observations, so the effective
    # periods per year is 12, not 252.
    start = date(2025, 11, 20)
    series = [
        (date(2025, 11, 20), 0.0),
        *(
            (start + timedelta(days=30 * month), (1.01**month - 1) * 100)
            for month in range(1, 13)
        ),
    ]

    metrics = risk_for_cumulative_series(series)

    assert metrics is not None
    assert metrics.observations == 12
    assert metrics.annualized_return == pytest.approx(1.01**12 - 1, rel=1e-6)
    # A smooth 1%-per-month line has no dispersion, so the ratios stay undefined.
    assert metrics.annualized_volatility == pytest.approx(0.0)
    assert metrics.sharpe is None
