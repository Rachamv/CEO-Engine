"""Tests for data.py — OHLCV fetch/clean pipeline."""

import pandas as pd
import pytest

from ceo_engine_mt5.data import _clean, _estimate_tick_size


class TestTickSizePreservation:
    """
    Regression tests for the tick_size bug fixed in v2.1.0: _clean() used
    to unconditionally overwrite tick_size with a price-magnitude estimate,
    even when the fetcher (_fetch_mt5) had already supplied the real value
    from mt5.symbol_info(). See CHANGELOG v2.1.0.
    """

    def _raw_df(self):
        return pd.DataFrame({
            "open":  [1.2345, 1.2346],
            "high":  [1.2350, 1.2351],
            "low":   [1.2340, 1.2341],
            "close": [1.2348, 1.2349],
            "volume": [100, 120],
        }, index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))

    def test_mt5_tick_size_survives_clean(self):
        raw = self._raw_df()
        raw.attrs["tick_size"] = 0.00001234  # broker-specific, not what the estimator would guess
        cleaned = _clean(raw, "EURUSD")
        assert cleaned.attrs["tick_size"] == 0.00001234

    def test_estimator_runs_when_no_tick_size_supplied(self):
        raw = self._raw_df()
        raw.attrs = {}  # simulate yfinance/ccxt/csv — no tick_size from the fetcher
        cleaned = _clean(raw, "EURUSD")
        assert cleaned.attrs["tick_size"] == _estimate_tick_size(raw["close"])

    def test_falsy_tick_size_is_treated_as_unset(self):
        """tick_size=0 (or None) should also trigger the estimator, not be
        treated as a 'real' value worth preserving."""
        raw = self._raw_df()
        raw.attrs["tick_size"] = 0
        cleaned = _clean(raw, "EURUSD")
        assert cleaned.attrs["tick_size"] == _estimate_tick_size(raw["close"])


class TestTickSizeEstimator:
    """_estimate_tick_size is a pure function — direct unit tests."""

    @pytest.mark.parametrize("price,expected", [
        (0.005, 0.0000001),
        (0.5,   0.00001),
        (5.0,   0.0001),
        (50.0,  0.01),
        (500.0, 0.1),
        (5000.0, 1.0),
    ])
    def test_estimate_by_price_magnitude(self, price, expected):
        series = pd.Series([price] * 10)
        assert _estimate_tick_size(series) == expected

    def test_empty_series_falls_back_to_default(self):
        assert _estimate_tick_size(pd.Series([], dtype=float)) == 0.00001
