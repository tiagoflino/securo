"""Risk statistics for the Portfolio Performance series.

The performance service produces ``(date, cumulative return)`` series for the
portfolio and for every benchmark, all rebased to 0. The pure math lives in
``risk_metrics``; this module only converts a cumulative series into the index
levels it expects and keeps a short or broken series from failing the whole
response.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

from app.services.risk_metrics import RiskMetrics, RiskMetricsError, compute_risk_metrics


def risk_for_cumulative_series(
    series: Sequence[tuple[date, float]],
    *,
    risk_free_rate: float = 0.0,
) -> Optional[RiskMetrics]:
    """Risk statistics for one performance series, or None when it cannot be measured.

    A benchmark that failed, or a portfolio with a single observation, yields
    None instead of an error: the Performance tab must keep showing the return
    curve even when a statistic is undefined.
    """

    if len(series) < 2:
        return None
    # The performance series carry percentage points (10.0 is 10%), while the
    # risk module works on index levels.
    values = [1.0 + cumulative / 100.0 for _, cumulative in series]
    try:
        return compute_risk_metrics(values, risk_free_rate=risk_free_rate)
    except RiskMetricsError:
        return None
