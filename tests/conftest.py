"""
Shared pytest fixtures for the The CEO Protocol test suite.

`synthetic_ohlcv` gives raw OHLCV data with a fixed seed (deterministic —
every test run produces identical bars, so test assertions are exact, not
"roughly"). `enriched_df` runs that through the full Phase 1-3 pipeline
(indicators → signals → candle patterns → CEO structure → geometric
patterns → session filter), which is what almost every test in this
suite actually needs — most tests shouldn't have to rebuild the pipeline
themselves.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(seed: int, n: int, start: str = "2024-01-01", freq: str = "1h") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    price = 2000 + np.cumsum(rng.randn(n) * 3)
    df = pd.DataFrame({
        "open":   price + rng.randn(n) * 0.5,
        "high":   price + abs(rng.randn(n)) * 3,
        "low":    price - abs(rng.randn(n)) * 3,
        "close":  price + rng.randn(n) * 0.5,
        "volume": rng.randint(100, 1000, n).astype(float),
    }, index=idx)
    # Ensure high/low actually bound open/close (real OHLC invariant)
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.1
    df["low"]  = df[["open", "low", "close"]].min(axis=1) - 0.1
    df.attrs["symbol"]    = "XAUUSD"
    df.attrs["tick_size"] = 0.01
    df.index.name = "datetime"
    return df


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """500 bars of deterministic synthetic OHLCV data (seed=42)."""
    return _make_ohlcv(seed=42, n=500)


@pytest.fixture
def synthetic_ohlcv_large() -> pd.DataFrame:
    """3000 bars — enough for walk-forward windows and stable trade counts."""
    return _make_ohlcv(seed=123, n=3000)


@pytest.fixture
def enriched_df(synthetic_ohlcv) -> pd.DataFrame:
    """500 bars run through the full Phase 1-3 pipeline."""
    from ceo_engine_mt5.indicators import calc_all
    from ceo_engine_mt5.signals import build_all, build_confluence
    from ceo_engine_mt5.candle_patterns import build_candle_patterns
    from ceo_engine_mt5.ceo_structure import build_ceo_structure
    from ceo_engine_mt5.patterns import build_patterns
    from ceo_engine_mt5.session_filter import add_session_columns

    df = calc_all(synthetic_ohlcv)
    df = build_all(df)
    df = build_candle_patterns(df)
    df = build_ceo_structure(df)
    # build_confluence() runs after build_ceo_structure(), not inside
    # build_all() -- its quality gate needs the CEO structural bonus
    # build_ceo_structure() just applied to m00_quality_long/short. See
    # signals.build_all()'s docstring for why.
    df = build_confluence(df)
    df = build_patterns(df)
    df = add_session_columns(df, allowed=["all"])
    return df


@pytest.fixture
def enriched_df_large(synthetic_ohlcv_large) -> pd.DataFrame:
    """3000 bars run through the full Phase 1-3 pipeline."""
    from ceo_engine_mt5.indicators import calc_all
    from ceo_engine_mt5.signals import build_all, build_confluence
    from ceo_engine_mt5.candle_patterns import build_candle_patterns
    from ceo_engine_mt5.ceo_structure import build_ceo_structure
    from ceo_engine_mt5.patterns import build_patterns
    from ceo_engine_mt5.session_filter import add_session_columns

    df = calc_all(synthetic_ohlcv_large)
    df = build_all(df)
    df = build_candle_patterns(df)
    df = build_ceo_structure(df)
    df = build_confluence(df)
    df = build_patterns(df)
    df = add_session_columns(df, allowed=["all"])
    return df
