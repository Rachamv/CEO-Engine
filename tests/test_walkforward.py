"""Tests for walkforward.py — window splitting, model selection, and
consistency validation. The existing smoke tests verified the basic
shape of the output; these add edge-case and correctness coverage
for the logic that feeds the ModelSelector's min_consistency gate."""

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.walkforward import (
    walk_forward, _build_summary, _find_consistent_winner,
)


# ─────────────────────────────────────────────────────────────────────────────
# Basic structure (existing smoke tests, kept for regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForward:
    def test_produces_requested_number_of_windows(self, enriched_df_large):
        wf = walk_forward(enriched_df_large, n_windows=4, min_trades=1, verbose=False)
        assert "summary" in wf
        assert "windows" in wf
        assert len(wf["windows"]) == 4

    def test_summary_has_one_row_per_model_plus_confluence(self, enriched_df_large):
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        from ceo_engine_mt5.signals import NUM_MODELS
        assert len(wf["summary"]) == NUM_MODELS + 1

    def test_summary_columns_present(self, enriched_df_large):
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        for col in ["Valid Windows", "Positive Windows"]:
            assert col in wf["summary"].columns

    def test_too_few_bars_per_window_raises_clear_error(self, enriched_df):
        """walk_forward() requires >= 200 bars/window — documented behavior,
        not a silent degradation, since a thinner window isn't trustworthy."""
        with pytest.raises(ValueError, match="Too few bars per window"):
            walk_forward(enriched_df, n_windows=10, min_trades=5, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Window splitting edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowSplitting:
    def test_last_window_absorbs_remainder_bars(self, enriched_df_large):
        """With 3001 bars and 3 windows: windows 1-2 get 1000 bars each,
        window 3 gets the remaining 1001. Total must equal original length."""
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        total_bars = sum(w["bars"] for w in wf["windows"])
        assert total_bars == len(enriched_df_large)



    def test_minimum_two_windows(self, enriched_df_large):
        """n_windows=2 is the smallest meaningful walk-forward run."""
        wf = walk_forward(enriched_df_large, n_windows=2, min_trades=1, verbose=False)
        assert len(wf["windows"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _find_consistent_winner unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFindConsistentWinner:
    """_find_consistent_winner is pure logic — test it directly with
    hand-crafted window lists rather than running the full pipeline."""

    def _make_windows(self, best_models):
        return [{"best_model": m, "best_value": 1.0} for m in best_models]

    def test_clear_majority_winner(self):
        windows = self._make_windows(["LQ", "LQ", "LQ + Trend", "LQ"])
        best, consistency = _find_consistent_winner(windows)
        assert best == "LQ"
        assert consistency == pytest.approx(3 / 4)

    def test_tie_returns_one_of_the_tied_models(self):
        windows = self._make_windows(["LQ", "LQ + FVG", "LQ", "LQ + FVG"])
        best, consistency = _find_consistent_winner(windows)
        assert best in ("LQ", "LQ + FVG")
        assert consistency == pytest.approx(0.5)

    def test_all_windows_have_no_winner(self):
        windows = self._make_windows(["—", "—", "—"])
        best, consistency = _find_consistent_winner(windows)
        assert best == "—"
        assert consistency == 0.0

    def test_single_window(self):
        windows = self._make_windows(["LQ + All Filters"])
        best, consistency = _find_consistent_winner(windows)
        assert best == "LQ + All Filters"
        assert consistency == pytest.approx(1.0)

    def test_all_different_models(self):
        windows = self._make_windows(["LQ", "LQ + Trend", "LQ + FVG", "LQ + Volume"])
        best, consistency = _find_consistent_winner(windows)
        # Any model appears exactly once — consistency = 1/4
        assert consistency == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────────────────────
# _build_summary unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSummary:
    """_build_summary aggregates per-window tables into the stability report.
    Test using minimal synthetic tables so we control the math exactly."""

    def _make_table(self, model_name, net_r, win_rate, trades):
        """Build a minimal results_table row for one model."""
        return pd.DataFrame({
            "Trades":     [trades],
            "Win Rate":   [win_rate],
            "Net R":      [net_r],
            "Profit Factor": [1.5],
            "Avg R":      [net_r / max(trades, 1)],
        }, index=[model_name])

    def test_avg_net_r_computed_correctly(self):
        """Two windows with Net R of +2.0 and +4.0 → avg should be +3.0."""
        tbl1 = self._make_table("LQ", net_r=2.0, win_rate=60.0, trades=10)
        tbl2 = self._make_table("LQ", net_r=4.0, win_rate=65.0, trades=12)
        windows = [
            {"window": 1, "best_model": "LQ", "best_value": 2.0,
             "metric": "Net R", "table": tbl1},
            {"window": 2, "best_model": "LQ", "best_value": 4.0,
             "metric": "Net R", "table": tbl2},
        ]
        summary = _build_summary(windows, min_trades=5, metric="Net R")
        assert "LQ" in summary.index
        assert summary.loc["LQ", "Net R (avg)"] == pytest.approx(3.0)

    def test_positive_windows_counts_only_positive_net_r(self):
        """One +2.0 window and one -1.0 window → Positive Windows = 1."""
        tbl1 = self._make_table("LQ", net_r=2.0,  win_rate=60.0, trades=10)
        tbl2 = self._make_table("LQ", net_r=-1.0, win_rate=40.0, trades=10)
        windows = [
            {"window": 1, "best_model": "LQ", "best_value": 2.0,
             "metric": "Net R", "table": tbl1},
            {"window": 2, "best_model": "LQ", "best_value": -1.0,
             "metric": "Net R", "table": tbl2},
        ]
        summary = _build_summary(windows, min_trades=5, metric="Net R")
        assert summary.loc["LQ", "Positive Windows"] == 1
        assert summary.loc["LQ", "Valid Windows"] == 2

    def test_window_below_min_trades_excluded_from_valid_count(self):
        """A window with fewer trades than min_trades should not count as
        'valid' — it gets excluded from Net R (avg) and Valid Windows."""
        tbl1 = self._make_table("LQ", net_r=3.0, win_rate=60.0, trades=10)
        tbl2 = self._make_table("LQ", net_r=1.0, win_rate=55.0, trades=2)   # < min=5
        windows = [
            {"window": 1, "best_model": "LQ", "best_value": 3.0,
             "metric": "Net R", "table": tbl1},
            {"window": 2, "best_model": "LQ", "best_value": 1.0,
             "metric": "Net R", "table": tbl2},
        ]
        summary = _build_summary(windows, min_trades=5, metric="Net R")
        assert summary.loc["LQ", "Valid Windows"] == 1
        # Average should only reflect the one valid window (3.0), not (3+1)/2
        assert summary.loc["LQ", "Net R (avg)"] == pytest.approx(3.0)

    def test_model_absent_from_window_table_is_skipped(self):
        """If a model doesn't appear in a window's results table at all
        (no trades), it should be counted as 0 valid windows, not error."""
        tbl1 = self._make_table("LQ + Trend", net_r=2.0, win_rate=60.0, trades=10)
        windows = [
            {"window": 1, "best_model": "LQ + Trend", "best_value": 2.0,
             "metric": "Net R", "table": tbl1},
        ]
        summary = _build_summary(windows, min_trades=5, metric="Net R")
        # "LQ" should exist in the summary (it's always included) but have 0 valid
        assert "LQ" in summary.index
        assert summary.loc["LQ", "Valid Windows"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# walk_forward — verbose output, HTF resampling, confluence_mode override
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardVerboseAndOptions:
    def test_verbose_true_prints_progress(self, enriched_df_large, capsys):
        walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=True)
        out = capsys.readouterr().out
        assert "Walk-Forward Validation" in out
        assert "Window 1/3" in out

    def test_verbose_false_suppresses_walkforward_banner(self, enriched_df_large, capsys):
        walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        out = capsys.readouterr().out
        # run_backtest() prints its own progress independent of this flag;
        # verbose=False only suppresses walk_forward's own banner/progress lines.
        assert "Walk-Forward Validation" not in out
        assert "Window 1/3" not in out

    def test_htf_tf_resamples_successfully(self, enriched_df_large):
        # "4h" is a supported resample target and enriched_df_large has
        # plenty of bars per window -- this exercises the happy HTF path.
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1,
                           htf_tf="4h", verbose=False)
        assert len(wf["windows"]) == 3

    def test_htf_resample_failure_does_not_crash_window(self, enriched_df_large, monkeypatch, caplog):
        import ceo_engine_mt5.walkforward as wf_mod
        monkeypatch.setattr(wf_mod, "resample_ohlcv",
                             lambda df, tf: (_ for _ in ()).throw(ValueError("boom")))
        with caplog.at_level("WARNING"):
            wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1,
                               htf_tf="4h", verbose=False)
        assert len(wf["windows"]) == 3
        assert "HTF resample failed" in caplog.text

    def test_non_sweep_confluence_mode_is_forced_back_with_warning(self, enriched_df_large, caplog):
        with caplog.at_level("WARNING"):
            wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1,
                               signal_params={"confluence_mode": "full"}, verbose=False)
        assert "confluence_mode='full'" in caplog.text or "requires build_ceo_structure" in caplog.text
        assert len(wf["windows"]) == 3

    def test_sweep_confluence_mode_does_not_warn(self, enriched_df_large, caplog):
        with caplog.at_level("WARNING"):
            walk_forward(enriched_df_large, n_windows=3, min_trades=1,
                          signal_params={"confluence_mode": "sweep"}, verbose=False)
        assert "requires build_ceo_structure" not in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# print_wf_report
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintWfReport:
    def test_prints_report_without_error(self, enriched_df_large, capsys):
        from ceo_engine_mt5.walkforward import print_wf_report
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        print_wf_report(wf)
        out = capsys.readouterr().out
        assert "WALK-FORWARD REPORT" in out
        assert "Best overall" in out
        assert "Per-Model Stability" in out

    def test_handles_all_windows_with_no_valid_model(self, capsys):
        from ceo_engine_mt5.walkforward import print_wf_report, _build_summary
        windows = [{"window": 1, "start": "2024-01-01", "end": "2024-01-02",
                    "bars": 300, "best_model": "—", "best_value": np.nan,
                    "metric": "Net R", "table": pd.DataFrame()}]
        results = {
            "windows": windows,
            "summary": _build_summary(windows, min_trades=5, metric="Net R"),
            "best_overall": "—", "consistency": 0.0,
            "metric": "Net R", "n_windows": 1,
        }
        print_wf_report(results)
        out = capsys.readouterr().out
        assert "WALK-FORWARD REPORT" in out


# ─────────────────────────────────────────────────────────────────────────────
# save_wf_results
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveWfResults:
    def test_writes_both_csv_files(self, enriched_df_large, tmp_path, capsys):
        from ceo_engine_mt5.walkforward import save_wf_results
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        out_dir = tmp_path / "wf_out"
        save_wf_results(wf, out_dir=str(out_dir))
        assert (out_dir / "wf_windows.csv").exists()
        assert (out_dir / "wf_summary.csv").exists()
        out = capsys.readouterr().out
        assert "wf_windows.csv" in out
        assert "wf_summary.csv" in out

    def test_creates_output_directory_if_missing(self, enriched_df_large, tmp_path):
        from ceo_engine_mt5.walkforward import save_wf_results
        wf = walk_forward(enriched_df_large, n_windows=3, min_trades=1, verbose=False)
        out_dir = tmp_path / "does" / "not" / "exist"
        save_wf_results(wf, out_dir=str(out_dir))
        assert out_dir.exists()
