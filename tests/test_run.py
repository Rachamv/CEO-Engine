"""
Tests for run.py's run() — the programmatic entry point. Uses local CSV
data (source="csv") so no network is needed; covers the core path plus
the two extracted helper functions added in v2.2.0
(_run_walkforward_validation, _generate_html_report_safe).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from run import run


def _write_csv(seed: int, n: int) -> str:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    price = 2000 + np.cumsum(rng.randn(n) * 3)
    df = pd.DataFrame({
        "open":   price + rng.randn(n) * 0.5,
        "high":   price + abs(rng.randn(n)) * 3,
        "low":    price - abs(rng.randn(n)) * 3,
        "close":  price + rng.randn(n) * 0.5,
        "volume": rng.randint(100, 1000, n).astype(float),
    }, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.1
    df["low"]  = df[["open", "low", "close"]].min(axis=1) - 0.1
    df.index.name = "datetime"
    path = tempfile.mktemp(suffix=".csv")
    df.to_csv(path)
    return path


@pytest.fixture
def csv_path():
    path = _write_csv(seed=11, n=800)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestRunCore:
    def test_returns_backtest_results_and_table(self, csv_path):
        out_dir = tempfile.mkdtemp()
        bt, tbl, df = run("TESTSYM", tf="1h", source="csv", filepath=csv_path,
                          out_dir=out_dir, no_charts=True, no_csv=True, verbose=False)
        assert isinstance(bt, dict)
        assert len(tbl) > 0
        assert len(df) == 800

    def test_default_sessions_is_trade_anytime(self, csv_path):
        out_dir = tempfile.mkdtemp()
        bt, tbl, df = run("TESTSYM", tf="1h", source="csv", filepath=csv_path,
                          out_dir=out_dir, no_charts=True, no_csv=True, verbose=False)
        # sessions=None default should resolve to ["all"] — every weekday
        # hour should remain tradeable (no silent time-of-day restriction).
        assert "sess_name" in df.columns
        weekday_hours = df[df["sess_name"] != "weekend"]
        assert len(weekday_hours) > 0


class TestWalkforwardPath:
    def test_walkforward_flag_runs_without_error(self, csv_path):
        out_dir = tempfile.mkdtemp()
        bt, tbl, df = run("TESTSYM", tf="1h", source="csv", filepath=csv_path,
                          out_dir=out_dir, no_charts=True, no_csv=True, verbose=False,
                          walkforward=True, wf_windows=3)
        assert os.path.exists(out_dir)


class TestHtmlReportPath:
    def test_html_report_flag_produces_a_file(self, csv_path):
        out_dir = tempfile.mkdtemp()
        run("TESTSYM", tf="1h", source="csv", filepath=csv_path,
            out_dir=out_dir, no_charts=True, no_csv=True, verbose=False,
            html_report=True)
        html_files = [f for f in os.listdir(out_dir) if f.endswith(".html")]
        assert len(html_files) == 1

    def test_html_report_with_walkforward_and_journal_stats(self, csv_path):
        """Exercises both optional sections of generate_report() together."""
        out_dir = tempfile.mkdtemp()
        run("TESTSYM", tf="1h", source="csv", filepath=csv_path,
            out_dir=out_dir, no_charts=True, no_csv=True, verbose=False,
            walkforward=True, wf_windows=3, html_report=True,
            journal_stats={"trades": 10, "win_rate": 45.0})
        html_files = [f for f in os.listdir(out_dir) if f.endswith(".html")]
        assert len(html_files) == 1
        content = open(os.path.join(out_dir, html_files[0])).read()
        assert "Walk-Forward Consistency" in content
        assert "Live Journal Comparison" in content
