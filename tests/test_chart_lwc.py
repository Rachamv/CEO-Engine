"""
Tests for chart_lwc.py, which replaced chart_html.py's plotly-based
candlestick+overlays renderer with TradingView's Lightweight Charts JS
library -- the same one the live dashboard already loads for
/api/candles, so the whole project now needs zero Python charting
dependencies for anything except the static Telegram-alert PNG
(chart_png.py, matplotlib).

Covers:
  - plot_chart_html()'s contract is unchanged (full HTML doc, literal
    <body>...</body>, same function name/signature) so every caller
    (report.py, dashboard.py, mt5_live_signals.py, run.py) needed zero
    changes
  - the old chart_html.py module is actually gone, not just unused
  - every generated chart's JS is valid (Node syntax check) and behaves
    correctly at runtime (Node execution against a stubbed
    LightweightCharts API)
  - each data-layer builder (candles, EMA, volume, markers, structure
    overlays, pattern trendlines) produces correctly-shaped output
  - report.py's chart embedding still works now that the CDN script
    lives in the report's own <head> instead of the (stripped) embedded
    document's <head>
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5 import chart_lwc
from ceo_engine_mt5.chart_lwc import (
    plot_chart_html, LIGHTWEIGHT_CHARTS_CDN,
    _to_unix_seconds, _build_candle_data, _build_line_data,
    _build_volume_data, _build_pivot_markers, _build_signal_markers,
    _build_structure_overlays, _build_pattern_trendlines,
)
from ceo_engine_mt5.chart import plot_chart_html as chart_module_plot_chart_html


def _node_available() -> bool:
    return shutil.which("node") is not None


# ─────────────────────────────────────────────────────────────────────────────
# chart_html.py is actually gone, plotly is actually gone
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotlyFullyRetired:
    def test_chart_html_module_file_no_longer_exists(self):
        import ceo_engine_mt5
        pkg_dir = os.path.dirname(ceo_engine_mt5.__file__)
        assert not os.path.exists(os.path.join(pkg_dir, "chart_html.py"))

    def test_chart_module_resolves_plot_chart_html_to_lwc_implementation(self):
        assert chart_module_plot_chart_html is plot_chart_html

    def test_chart_lwc_module_has_no_plotly_import(self):
        import inspect
        source = inspect.getsource(chart_lwc)
        code_lines = [l for l in source.splitlines() if not l.strip().startswith(("#", '"', "'"))]
        assert not any(re.match(r"\s*(import plotly|from plotly)", l) for l in code_lines)

    def test_generated_chart_contains_no_plotly_references(self, enriched_df):
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=100)
        assert "plotly" not in html.lower()
        assert "cdn.plot.ly" not in html


# ─────────────────────────────────────────────────────────────────────────────
# Contract preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestContractUnchanged:
    def test_returns_full_html_document(self, enriched_df):
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=100)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")

    def test_contains_literal_body_tags(self, enriched_df):
        """_embed_ceo_chart in report.py string-searches for these exact
        literal tags -- this must never regress."""
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=100)
        assert "<body>" in html
        assert "</body>" in html
        body_start = html.find("<body>")
        body_end = html.find("</body>")
        assert body_end > body_start

    def test_accepts_all_original_keyword_arguments(self, enriched_df):
        # Must not raise for any of the original plotly version's kwargs,
        # including show_sessions which is now a accepted-but-unused
        # simplification (see module docstring).
        html = plot_chart_html(
            enriched_df, symbol="EURUSD", tf="H1", bars=50,
            show_volume=False, show_struct=False, show_patterns=False,
            show_sessions=True, height=500,
        )
        assert "<!DOCTYPE html>" in html

    def test_title_reflects_symbol_and_timeframe(self, enriched_df):
        html = plot_chart_html(enriched_df, symbol="GBPJPY", tf="H4", bars=50)
        assert "GBPJPY" in html
        assert "H4" in html

    def test_cdn_script_present(self, enriched_df):
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=100)
        assert LIGHTWEIGHT_CHARTS_CDN in html


# ─────────────────────────────────────────────────────────────────────────────
# Real JS validation (Node)
# ─────────────────────────────────────────────────────────────────────────────

_STUB_LWC_JS = """
const calls = { seriesCount: 0, dataLens: [], markers: null, priceLines: [] };
class FakeSeries {
  setData(d) { calls.dataLens.push(d.length); }
  setMarkers(m) { calls.markers = m.length; }
  createPriceLine(o) { calls.priceLines.push(o); return {}; }
  priceScale() { return { applyOptions: () => {} }; }
}
class FakeChart {
  addCandlestickSeries(o) { calls.seriesCount++; return new FakeSeries(); }
  addLineSeries(o) { calls.seriesCount++; return new FakeSeries(); }
  addHistogramSeries(o) { calls.seriesCount++; return new FakeSeries(); }
  timeScale() { return { fitContent: () => {} }; }
  applyOptions(o) {}
}
const LightweightCharts = {
  createChart: (el, opts) => new FakeChart(),
  CrosshairMode: { Normal: 0 },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2, LargeDashed: 3, SparseDotted: 4 },
};
const document = { getElementById: (id) => ({ clientWidth: 900 }) };
const window = { addEventListener: () => {} };
"""


class TestGeneratedJsIsValid:
    def test_node_syntax_check(self, enriched_df):
        if not _node_available():
            pytest.skip("node not available in this environment")
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=150)
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        proc = subprocess.run(["node", "--check", "-"], input=script,
                               capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0, proc.stderr

    def test_node_execution_against_stubbed_api(self, enriched_df):
        if not _node_available():
            pytest.skip("node not available in this environment")
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=150)
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        js_source = _STUB_LWC_JS + script + '\nconsole.log(JSON.stringify(calls));\n'
        proc = subprocess.run(["node", "-e", js_source], capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0, proc.stderr
        calls = json.loads(proc.stdout.strip().splitlines()[-1])
        assert calls["seriesCount"] >= 1     # at least the candlestick series
        assert 150 in calls["dataLens"]      # candlestick series got all 150 bars

    def test_empty_overlays_still_produce_valid_js(self, enriched_df):
        """A chart with every optional layer disabled must still be
        syntactically valid -- this is the minimal-payload path."""
        if not _node_available():
            pytest.skip("node not available in this environment")
        html = plot_chart_html(enriched_df, symbol="XAUUSD", tf="M15", bars=50,
                               show_volume=False, show_struct=False, show_patterns=False)
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        proc = subprocess.run(["node", "--check", "-"], input=script,
                               capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0, proc.stderr


# ─────────────────────────────────────────────────────────────────────────────
# Data-layer builders
# ─────────────────────────────────────────────────────────────────────────────

class TestToUnixSeconds:
    def test_converts_naive_datetimes(self):
        times = pd.Series(pd.date_range("2024-01-01", periods=3, freq="1h"))
        out = _to_unix_seconds(times)
        assert len(out) == 3
        assert out[1] - out[0] == 3600

    def test_converts_tz_aware_datetimes_to_utc(self):
        times = pd.Series(pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))
        out = _to_unix_seconds(times)
        assert len(out) == 2
        assert out[1] > out[0]


class TestCandleAndLineData:
    def test_build_candle_data_shape(self):
        times = [1000, 1060, 1120]
        o, h, lo, c = [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5], [1.2, 2.2, 3.2]
        out = _build_candle_data(times, o, h, lo, c)
        assert len(out) == 3
        assert out[0] == {"time": 1000, "open": 1.0, "high": 1.5, "low": 0.5, "close": 1.2}

    def test_build_line_data_skips_nan(self):
        times = [1000, 1060, 1120]
        values = [1.0, np.nan, 3.0]
        out = _build_line_data(times, values)
        assert len(out) == 2
        assert out[0]["time"] == 1000
        assert out[1]["time"] == 1120


class TestVolumeData:
    def test_colors_by_candle_direction(self):
        times = [1000, 1060]
        o = [1.0, 2.0]
        c = [1.5, 1.5]   # first bull (close>open), second bear (close<open)
        vol = [100.0, 200.0]
        out = _build_volume_data(times, o, c, vol)
        assert out[0]["color"] != out[1]["color"]


class TestMarkerBuilders:
    def _sl(self, **cols):
        n = 5
        base = {
            "pivot_high": [np.nan] * n, "pivot_low": [np.nan] * n,
            "base_long": [False] * n, "base_short": [False] * n,
            "cp_bull_confirmation": [False] * n, "cp_bear_confirmation": [False] * n,
            "ceo_long_valid": [False] * n, "ceo_short_valid": [False] * n,
        }
        base.update(cols)
        return pd.DataFrame(base)

    def test_pivot_markers_shape(self):
        sl = self._sl(pivot_high=[np.nan, 105.0, np.nan, np.nan, np.nan],
                      pivot_low=[np.nan, np.nan, np.nan, 95.0, np.nan])
        times = [0, 1, 2, 3, 4]
        atr = np.ones(5)
        markers = _build_pivot_markers(sl, times, atr)
        assert len(markers) == 2
        shapes = {m["shape"] for m in markers}
        assert shapes == {"arrowDown", "arrowUp"}

    def test_signal_markers_sorted_by_time(self):
        sl = self._sl(ceo_long_valid=[False, False, True, False, False],
                      base_short=[False, True, False, False, False])
        times = [10, 20, 30, 40, 50]
        markers = _build_signal_markers(sl, times)
        assert len(markers) == 2
        assert [m["time"] for m in markers] == sorted(m["time"] for m in markers)

    def test_base_signal_suppressed_when_ceo_signal_present_same_bar(self):
        """A CEO-valid entry and its underlying base sweep on the same bar
        should produce one marker, not two overlapping ones."""
        sl = self._sl(ceo_long_valid=[True], base_long=[True],
                      pivot_high=[np.nan], pivot_low=[np.nan],
                      base_short=[False], cp_bull_confirmation=[False],
                      cp_bear_confirmation=[False], ceo_short_valid=[False])
        markers = _build_signal_markers(sl, [100])
        assert len(markers) == 1
        assert markers[0]["text"] == "CEO"


class TestStructureOverlays:
    def test_order_block_zone_produces_two_boundary_lines(self):
        n = 5
        sl = pd.DataFrame({
            "ob_bull_active": [False, True, True, True, True],
            "ob_bull_high":   [np.nan, 110.0, 110.0, 110.0, 110.0],
            "ob_bull_low":    [np.nan, 105.0, 105.0, 105.0, 105.0],
            "ob_bear_active": [False] * n, "ob_bear_high": [np.nan] * n, "ob_bear_low": [np.nan] * n,
            "fib_50": [np.nan] * n,
            "high": [111.0] * n, "low": [104.0] * n,
        })
        times = [0, 1, 2, 3, 4]
        out = _build_structure_overlays(sl, times, sl["high"].values, np.ones(n))
        # One zone (constant high/low across active bars) -> exactly 2 boundary lines
        assert len(out["zone_lines"]) == 2
        prices = {out["zone_lines"][0]["data"][0]["value"], out["zone_lines"][1]["data"][0]["value"]}
        assert prices == {110.0, 105.0}

    def test_fib_50_produces_a_price_line(self):
        n = 3
        sl = pd.DataFrame({
            "ob_bull_active": [False]*n, "ob_bull_high":[np.nan]*n, "ob_bull_low":[np.nan]*n,
            "ob_bear_active": [False]*n, "ob_bear_high":[np.nan]*n, "ob_bear_low":[np.nan]*n,
            "fib_50": [np.nan, 100.5, np.nan],
            "high": [101.0]*n, "low": [99.0]*n,
        })
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        assert any(pl["title"] == "Fib 50%" and pl["price"] == 100.5 for pl in out["price_lines"])

    def test_no_active_zones_returns_empty_lists(self):
        n = 3
        sl = pd.DataFrame({
            "ob_bull_active": [False]*n, "ob_bull_high":[np.nan]*n, "ob_bull_low":[np.nan]*n,
            "ob_bear_active": [False]*n, "ob_bear_high":[np.nan]*n, "ob_bear_low":[np.nan]*n,
            "fib_50": [np.nan]*n,
            "high": [101.0]*n, "low": [99.0]*n,
        })
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        assert out["zone_lines"] == []
        assert out["price_lines"] == []

    def _minimal_sl(self, n, **overrides):
        base = {
            "ob_bull_active": [False]*n, "ob_bull_high":[np.nan]*n, "ob_bull_low":[np.nan]*n,
            "ob_bear_active": [False]*n, "ob_bear_high":[np.nan]*n, "ob_bear_low":[np.nan]*n,
            "fib_50": [np.nan]*n,
            "high": [101.0]*n, "low": [99.0]*n,
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_qm_bull_level_produces_dashed_price_line(self):
        n = 3
        sl = self._minimal_sl(n, qm_bull_active=[False, True, True],
                               qm_bull_level=[np.nan, 105.0, 105.0])
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        qm_lines = [pl for pl in out["price_lines"] if pl["title"] == "QM (bull)"]
        assert len(qm_lines) == 1
        assert qm_lines[0]["price"] == 105.0
        assert qm_lines[0]["dash"] is True

    def test_qm_bear_level_produces_dashed_price_line(self):
        n = 3
        sl = self._minimal_sl(n, qm_bear_active=[False, False, True],
                               qm_bear_level=[np.nan, np.nan, 112.0])
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        qm_lines = [pl for pl in out["price_lines"] if pl["title"] == "QM (bear)"]
        assert len(qm_lines) == 1
        assert qm_lines[0]["price"] == 112.0

    def test_no_qm_active_produces_no_qm_lines(self):
        n = 3
        sl = self._minimal_sl(n)
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        assert not any("QM" in pl["title"] for pl in out["price_lines"])

    def test_bull_fvg_produces_two_boundary_lines(self):
        n = 5
        sl = self._minimal_sl(n,
                               bull_fvg=[False, False, True, False, False],
                               high=[100.0, 100.0, 103.0, 103.0, 103.0],
                               low=[99.0, 99.0, 102.0, 101.0, 101.0])
        out = _build_structure_overlays(sl, [0, 1, 2, 3, 4], sl["high"].values, np.ones(n))
        fvg_lines = [zl for zl in out["zone_lines"]]
        assert len(fvg_lines) == 2
        prices = {fvg_lines[0]["data"][0]["value"], fvg_lines[1]["data"][0]["value"]}
        # bull FVG: top = low[i] (102.0), bottom = high[i-2] (100.0)
        assert prices == {102.0, 100.0}

    def test_bear_fvg_produces_two_boundary_lines(self):
        n = 5
        sl = self._minimal_sl(n,
                               bear_fvg=[False, False, True, False, False],
                               high=[100.0, 100.0, 97.0, 97.0, 97.0],
                               low=[95.0, 95.0, 94.0, 94.0, 94.0])
        out = _build_structure_overlays(sl, [0, 1, 2, 3, 4], sl["high"].values, np.ones(n))
        assert len(out["zone_lines"]) == 2
        prices = {out["zone_lines"][0]["data"][0]["value"], out["zone_lines"][1]["data"][0]["value"]}
        # bear FVG: top = low[i-2] (95.0), bottom = high[i] (97.0)
        assert prices == {95.0, 97.0}

    def test_no_fvg_produces_no_zone_lines(self):
        n = 3
        sl = self._minimal_sl(n)
        out = _build_structure_overlays(sl, [0, 1, 2], sl["high"].values, np.ones(n))
        assert out["zone_lines"] == []


class TestBosMarkers:
    def _sl(self, n, **overrides):
        base = {"bos_long": [False]*n, "bos_short": [False]*n,
                "double_bos_long": [False]*n, "double_bos_short": [False]*n}
        base.update(overrides)
        return pd.DataFrame(base)

    def test_single_bos_long_produces_circle_marker(self):
        sl = self._sl(3, bos_long=[False, True, False])
        markers = chart_lwc._build_bos_markers(sl, [0, 1, 2])
        assert len(markers) == 1
        assert markers[0]["shape"] == "circle"
        assert markers[0]["text"] == "BOS"
        assert markers[0]["position"] == "belowBar"

    def test_double_bos_long_takes_priority_over_single(self):
        sl = self._sl(3, bos_long=[False, True, False],
                       double_bos_long=[False, True, False])
        markers = chart_lwc._build_bos_markers(sl, [0, 1, 2])
        assert len(markers) == 1
        assert markers[0]["shape"] == "arrowUp"
        assert markers[0]["text"] == "BOS2"

    def test_bos_short_marks_above_bar(self):
        sl = self._sl(3, bos_short=[False, False, True])
        markers = chart_lwc._build_bos_markers(sl, [0, 1, 2])
        assert len(markers) == 1
        assert markers[0]["position"] == "aboveBar"

    def test_no_bos_produces_no_markers(self):
        sl = self._sl(3)
        assert chart_lwc._build_bos_markers(sl, [0, 1, 2]) == []


class TestBuildStructurePayload:
    def _df_with_structure(self, n=10):
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        data = {
            "open": np.linspace(100, 110, n), "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n), "close": np.linspace(100.5, 110.5, n),
            "volume": [10]*n, "atr": [0.5]*n,
            "ob_bull_active": [False]*n, "ob_bull_high": [np.nan]*n, "ob_bull_low": [np.nan]*n,
            "ob_bear_active": [False]*n, "ob_bear_high": [np.nan]*n, "ob_bear_low": [np.nan]*n,
            "fib_50": [np.nan]*n,
            "qm_bull_active": [False]*n, "qm_bull_level": [np.nan]*n,
            "qm_bear_active": [False]*n, "qm_bear_level": [np.nan]*n,
            "bull_fvg": [False]*n, "bear_fvg": [False]*n,
            "bos_long": [False]*n, "bos_short": [False]*n,
            "double_bos_long": [False]*n, "double_bos_short": [False]*n,
            "ceo_long_valid": [False]*n, "ceo_short_valid": [False]*n,
            "base_long": [False]*n, "base_short": [False]*n,
            "cp_bull_confirmation": [False]*n, "cp_bear_confirmation": [False]*n,
            "pivot_high": [np.nan]*n, "pivot_low": [np.nan]*n,
            "pat_double_top": [False]*n, "pat_double_bottom": [False]*n,
            "pat_triple_top": [False]*n, "pat_triple_bottom": [False]*n,
        }
        df = pd.DataFrame(data, index=idx)
        df.index.name = "datetime"
        return df.reset_index()

    def test_returns_expected_shape(self):
        df = self._df_with_structure()
        out = chart_lwc.build_structure_payload(df)
        assert set(out.keys()) == {"zoneLines", "priceLines", "markers"}

    def test_empty_dataframe_returns_empty_shape(self):
        out = chart_lwc.build_structure_payload(pd.DataFrame())
        assert out == {"zoneLines": [], "priceLines": [], "markers": []}

    def test_bos_event_appears_in_markers(self):
        df = self._df_with_structure()
        df.loc[df.index[5], "bos_long"] = True
        out = chart_lwc.build_structure_payload(df)
        assert any(m["text"] == "BOS" for m in out["markers"])

    def test_ceo_valid_event_appears_in_markers(self):
        df = self._df_with_structure()
        df.loc[df.index[5], "ceo_long_valid"] = True
        out = chart_lwc.build_structure_payload(df)
        assert any(m["text"] == "CEO" for m in out["markers"])

    def test_order_block_appears_in_zone_lines(self):
        df = self._df_with_structure()
        df.loc[df.index[3]:, "ob_bull_active"] = True
        df.loc[df.index[3]:, "ob_bull_high"] = 105.0
        df.loc[df.index[3]:, "ob_bull_low"] = 103.0
        out = chart_lwc.build_structure_payload(df)
        assert len(out["zoneLines"]) == 2


class TestPatternTrendlines:
    def test_two_pivot_highs_produce_a_trendline(self):
        n = 6
        sl = pd.DataFrame({
            "pivot_high": [np.nan, 110.0, np.nan, 112.0, np.nan, np.nan],
            "pivot_low":  [np.nan]*n,
            "pat_asc_triangle":  [False, False, False, False, True, False],
            "pat_desc_triangle": [False]*n, "pat_sym_triangle": [False]*n,
            "pat_rising_wedge": [False]*n, "pat_falling_wedge": [False]*n,
        })
        times = list(range(n))
        h = np.array([100, 110, 105, 112, 108, 107], dtype=float)
        lo = np.array([95, 100, 98, 102, 99, 100], dtype=float)
        lines = _build_pattern_trendlines(sl, times, h, lo)
        asc_lines = [l for l in lines if "Asc Triangle" in l["name"]]
        assert len(asc_lines) == 1
        assert len(asc_lines[0]["data"]) == 2

    def test_no_pattern_produces_no_lines(self):
        n = 4
        sl = pd.DataFrame({
            "pivot_high": [np.nan]*n, "pivot_low": [np.nan]*n,
            "pat_asc_triangle": [False]*n, "pat_desc_triangle": [False]*n,
            "pat_sym_triangle": [False]*n, "pat_rising_wedge": [False]*n,
            "pat_falling_wedge": [False]*n,
        })
        lines = _build_pattern_trendlines(sl, list(range(n)),
                                          np.zeros(n), np.zeros(n))
        assert lines == []


# ─────────────────────────────────────────────────────────────────────────────
# report.py integration
# ─────────────────────────────────────────────────────────────────────────────

class TestReportIntegration:
    def test_embed_ceo_chart_strips_head_but_report_still_loads_lwc(self, enriched_df):
        from ceo_engine_mt5.backtest import run_backtest, results_table
        from ceo_engine_mt5.report import generate_report

        bt = run_backtest(enriched_df)
        tbl = results_table(bt)
        out_path = tempfile.mktemp(suffix=".html")
        generate_report(enriched_df, bt, tbl, symbol="XAUUSD", tf="M15",
                         out_path=out_path, include_ceo_chart=True)
        content = open(out_path).read()

        assert LIGHTWEIGHT_CHARTS_CDN in content
        assert content.count("<html") == 1     # embedded doc's <html> was stripped, not nested
        assert content.count("<body") == 1
        assert "LightweightCharts.createChart(" in content

    def test_report_generation_survives_chart_embed_failure(self, enriched_df, monkeypatch):
        from ceo_engine_mt5.backtest import run_backtest, results_table
        from ceo_engine_mt5 import report as report_module

        def _boom(*a, **kw):
            raise RuntimeError("simulated chart failure")
        monkeypatch.setattr(report_module, "_embed_ceo_chart", _boom)

        bt = run_backtest(enriched_df)
        tbl = results_table(bt)
        out_path = tempfile.mktemp(suffix=".html")
        # generate_report() guards the _embed_ceo_chart call site too (on
        # top of _embed_ceo_chart's own internal try/except), so even a
        # chart embed that fails outside its own guard must not take down
        # the whole report.
        result_path = report_module.generate_report(
            enriched_df, bt, tbl, symbol="XAUUSD", tf="M15",
            out_path=out_path, include_ceo_chart=True)
        assert result_path == out_path
        assert os.path.exists(out_path)
