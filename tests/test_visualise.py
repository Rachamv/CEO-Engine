"""
Tests for visualise.py -- previously 8% covered. These are matplotlib
chart generators; the valuable thing to test is that each one runs to
completion and actually writes a real, non-empty PNG for realistic
inputs (a real run_backtest() result over enriched_df_large, which
gives multiple models with enough trades to exercise the "happy path"
branches), plus each function's documented "nothing to plot" edge case.
Pixel-level content isn't asserted -- that's the wrong level for this
kind of regression protection; what matters is that these don't raise
for shapes the model-comparison/report pipeline actually produces.
"""

import os

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.backtest import run_backtest, results_table
from ceo_engine_mt5.visualise import (
    plot_model_comparison, plot_equity_curves, plot_wr_vs_pf,
    plot_drawdown, plot_trade_distribution, plot_summary_dashboard,
    plot_all, _max_drawdown, _bar_color, _apply_theme,
)


@pytest.fixture(scope="module")
def bt_and_tbl(synthetic_ohlcv_large_module):
    bt = run_backtest(synthetic_ohlcv_large_module)
    tbl = results_table(bt)
    return bt, tbl


@pytest.fixture(scope="module")
def synthetic_ohlcv_large_module():
    """Module-scoped copy of the enriched_df_large pipeline, since
    building it is expensive and every test in this file reads the
    same backtest result without mutating it."""
    from tests.conftest import _make_ohlcv
    from ceo_engine_mt5.indicators import calc_all
    from ceo_engine_mt5.signals import build_all, build_confluence
    from ceo_engine_mt5.candle_patterns import build_candle_patterns
    from ceo_engine_mt5.ceo_structure import build_ceo_structure
    from ceo_engine_mt5.patterns import build_patterns
    from ceo_engine_mt5.session_filter import add_session_columns

    raw = _make_ohlcv(seed=123, n=3000)
    df = calc_all(raw)
    df = build_all(df)
    df = build_candle_patterns(df)
    df = build_ceo_structure(df)
    df = build_confluence(df)
    df = build_patterns(df)
    df = add_session_columns(df, allowed=["all"])
    return df


def _assert_valid_png(path):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBarColor:
    def test_positive_is_green(self):
        from ceo_engine_mt5.visualise import THEME
        assert _bar_color(1.5) == THEME["green"]

    def test_negative_is_red(self):
        from ceo_engine_mt5.visualise import THEME
        assert _bar_color(-1.5) == THEME["red"]

    def test_zero_is_muted(self):
        from ceo_engine_mt5.visualise import THEME
        assert _bar_color(0) == THEME["muted"]

    def test_nan_is_muted(self):
        from ceo_engine_mt5.visualise import THEME
        assert _bar_color(np.nan) == THEME["muted"]


class TestMaxDrawdown:
    def test_monotonic_gains_have_zero_drawdown(self):
        trades = pd.DataFrame({"r_result": [1.0, 1.0, 1.0]})
        dd = _max_drawdown(trades)
        assert (dd == 0).all()

    def test_drawdown_after_a_loss_is_negative(self):
        trades = pd.DataFrame({"r_result": [2.0, -1.0]})
        dd = _max_drawdown(trades)
        assert dd.iloc[-1] == -1.0


class TestApplyTheme:
    def test_runs_with_single_axis(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _apply_theme(fig, ax)  # should not raise
        plt.close(fig)

    def test_runs_with_no_axes(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        _apply_theme(fig)  # axes=None path
        plt.close(fig)

    def test_runs_with_list_of_axes(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        _apply_theme(fig, axes)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# plot_model_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotModelComparison:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        _, tbl = bt_and_tbl
        out = str(tmp_path / "comparison.png")
        plot_model_comparison(tbl, save_path=out)
        _assert_valid_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# plot_equity_curves
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotEquityCurves:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        out = str(tmp_path / "equity.png")
        plot_equity_curves(bt, min_trades=1, save_path=out)
        _assert_valid_png(out)

    def test_no_eligible_models_skips_without_error(self, tmp_path, capsys):
        out = str(tmp_path / "equity_empty.png")
        empty_bt = {"LQ": pd.DataFrame({"r_result": []})}
        plot_equity_curves(empty_bt, min_trades=5, save_path=out)
        assert not os.path.exists(out)
        assert "No models with enough trades" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# plot_wr_vs_pf
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotWrVsPf:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        _, tbl = bt_and_tbl
        out = str(tmp_path / "wr_pf.png")
        plot_wr_vs_pf(tbl, min_trades=1, save_path=out)
        _assert_valid_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# plot_drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotDrawdown:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        out = str(tmp_path / "drawdown.png")
        plot_drawdown(bt, min_trades=1, save_path=out)
        _assert_valid_png(out)

    def test_no_eligible_models_skips_without_error(self, tmp_path):
        out = str(tmp_path / "drawdown_empty.png")
        empty_bt = {"LQ": pd.DataFrame({"r_result": []})}
        plot_drawdown(empty_bt, min_trades=5, save_path=out)
        assert not os.path.exists(out)


# ─────────────────────────────────────────────────────────────────────────────
# plot_trade_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotTradeDistribution:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        out = str(tmp_path / "dist.png")
        plot_trade_distribution(bt, save_path=out)
        _assert_valid_png(out)

    def test_single_model_still_works(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        non_empty = [k for k, v in bt.items() if not v.empty]
        if not non_empty:
            pytest.skip("synthetic data produced no trades for any model")
        out = str(tmp_path / "dist_single.png")
        plot_trade_distribution(bt, models=[non_empty[0]], save_path=out)
        _assert_valid_png(out)

    def test_no_models_with_trades_skips_without_error(self, tmp_path, capsys):
        out = str(tmp_path / "dist_empty.png")
        empty_bt = {"LQ": pd.DataFrame({"r_result": []})}
        plot_trade_distribution(empty_bt, save_path=out)
        assert not os.path.exists(out)
        assert "No models with trades" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# plot_summary_dashboard
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotSummaryDashboard:
    def test_generates_a_real_png(self, bt_and_tbl, tmp_path):
        bt, tbl = bt_and_tbl
        out = str(tmp_path / "dashboard.png")
        plot_summary_dashboard(bt, tbl, symbol="XAUUSD", save_path=out)
        _assert_valid_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# plot_all -- the master entry point
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotAll:
    def test_generates_all_six_charts(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        out_dir = str(tmp_path / "charts")
        saved = plot_all(bt, symbol="XAUUSD", output_dir=out_dir, min_trades=1)
        assert len(saved) == 6
        for path in saved:
            assert os.path.exists(path)

    def test_creates_output_dir_if_missing(self, bt_and_tbl, tmp_path):
        bt, _ = bt_and_tbl
        out_dir = str(tmp_path / "nested" / "charts_dir")
        plot_all(bt, output_dir=out_dir, min_trades=1)
        assert os.path.isdir(out_dir)
