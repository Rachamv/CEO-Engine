"""
Tests for report.py's charting layer -- replaced plotly (a heavy,
sometimes-not-installed dependency that made the whole module fail to
import, and therefore the whole report generator fail, if plotly wasn't
present) with hand-built Chart.js configs rendered via a CDN <script>
tag. No Python charting library is required for report.py anymore.

Covers:
  - the module imports and generates a report with zero plotly present
  - every chart embeds a well-formed Chart.js config (validated with a
    real JS parser via Node if available, otherwise a brace-balanced
    Python extractor as a fallback so this doesn't hard-depend on Node)
  - the "no data" placeholder path (empty trades) doesn't emit a broken
    <canvas>/<script> pair
  - numpy/pandas scalar types in the underlying data don't break JSON
    serialization (a common gotcha when charting DataFrame-derived data)
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5 import report as report_module
from ceo_engine_mt5.report import (
    generate_report, _render_chart, _json_default,
    _chart_equity_curve, _chart_model_comparison, _chart_trade_distribution,
    _chart_drawdown, _chart_session_breakdown, _chart_wr_vs_pf,
)
from ceo_engine_mt5.backtest import run_backtest, results_table


# ─────────────────────────────────────────────────────────────────────────────
# No plotly dependency at all
# ─────────────────────────────────────────────────────────────────────────────

class TestNoPlotlyDependency:
    def test_report_module_has_no_plotly_import(self):
        assert "plotly" not in dir(report_module)
        import inspect
        source = inspect.getsource(report_module)
        # Check actual import statements, not prose mentioning "import plotly"
        # in a comment explaining why it was removed.
        code_lines = [line for line in source.splitlines()
                      if not line.strip().startswith(("#", '"', "'"))]
        assert not any(re.match(r"\s*(import plotly|from plotly)", line) for line in code_lines)

    def test_report_module_importable_regardless_of_plotly_availability(self):
        # If this test file imports fine at all, report.py already didn't
        # explode on import -- this just makes the intent explicit.
        import importlib
        importlib.reload(report_module)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract & validate embedded Chart.js configs
# ─────────────────────────────────────────────────────────────────────────────

def _extract_chart_json_blobs(html: str):
    """Pulls the raw JSON/JS text passed to each `new Chart(...)` call,
    using brace-balancing (not a naive non-greedy regex, which breaks on
    the embedded tooltip functions that contain their own `{...}` blocks)."""
    blobs = []
    for m in re.finditer(r"new Chart\(document\.getElementById\(\"[^\"]+\"\), ", html):
        start = m.end()
        depth = 0
        i = start
        in_string = False
        str_char = ""
        while i < len(html):
            ch = html[i]
            if in_string:
                if ch == "\\":
                    i += 2
                    continue
                if ch == str_char:
                    in_string = False
            else:
                if ch in ("'", '"'):
                    in_string = True
                    str_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blobs.append(html[start:i + 1])
                        break
            i += 1
    return blobs


def _node_available() -> bool:
    return shutil.which("node") is not None


class TestChartConfigsAreValidJS:
    """Validates every chart config with Node's actual JS parser when
    available -- the strongest possible check short of a real browser."""

    def test_generated_charts_pass_node_syntax_check(self, enriched_df):
        if not _node_available():
            pytest.skip("node not available in this environment")
        bt = run_backtest(enriched_df)
        tbl = results_table(bt)
        out_path = tempfile.mktemp(suffix=".html")
        generate_report(enriched_df, bt, tbl, symbol="XAUUSD", tf="M15",
                         out_path=out_path, include_ceo_chart=False)
        html = open(out_path).read()
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        assert len(scripts) >= 1

        js_source = "class Chart { constructor(el, cfg) {} }\nconst document = {getElementById:(id)=>({})};\n"
        js_source += "\n".join(scripts)
        proc = subprocess.run(["node", "-e", js_source], capture_output=True, text=True, timeout=10)
        assert proc.returncode == 0, f"node execution failed: {proc.stderr}"

    def test_every_chart_config_has_required_fields(self, enriched_df):
        bt = run_backtest(enriched_df)
        tbl = results_table(bt)
        out_path = tempfile.mktemp(suffix=".html")
        generate_report(enriched_df, bt, tbl, symbol="XAUUSD", tf="M15",
                         out_path=out_path, include_ceo_chart=False)
        html = open(out_path).read()
        blobs = _extract_chart_json_blobs(html)
        assert len(blobs) >= 1
        for blob in blobs:
            # Function literals aren't valid JSON -- stub them out with a
            # brace-balanced replacement before parsing.
            safe = _stub_functions(blob)
            config = json.loads(safe)
            assert "type" in config
            assert "data" in config
            assert config["type"] in ("line", "bar", "bubble")


def _stub_functions(blob: str) -> str:
    """Replaces every `function(...){...}` literal with a JSON string
    placeholder, brace-balanced so nested `{}` inside the function body
    doesn't truncate the match early."""
    out = []
    i = 0
    while i < len(blob):
        if blob[i:i + 8] == "function":
            # find the opening brace of the function body
            brace_start = blob.index("{", i)
            depth = 0
            j = brace_start
            while j < len(blob):
                if blob[j] == "{":
                    depth += 1
                elif blob[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append('"FN"')
            i = j + 1
        else:
            out.append(blob[i])
            i += 1
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# No-data placeholders
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDataPlaceholders:
    def test_equity_curve_with_no_trades_renders_placeholder_not_chart(self):
        html = _chart_equity_curve(pd.DataFrame(), "—")
        assert "<canvas" not in html
        assert "No trades" in html

    def test_trade_distribution_with_no_trades_renders_placeholder(self):
        html = _chart_trade_distribution(pd.DataFrame())
        assert "<canvas" not in html
        assert "No trades" in html

    def test_drawdown_with_no_trades_renders_placeholder(self):
        html = _chart_drawdown(pd.DataFrame())
        assert "<canvas" not in html

    def test_model_comparison_with_no_data_renders_placeholder(self):
        empty_tbl = pd.DataFrame({"Net R": []})
        html = _chart_model_comparison(empty_tbl)
        assert "<canvas" not in html

    def test_wr_vs_pf_with_no_data_renders_placeholder(self):
        empty_tbl = pd.DataFrame({"Win Rate": [], "Profit Factor": [], "Net R": [], "Trades": []})
        html = _chart_wr_vs_pf(empty_tbl)
        assert "<canvas" not in html

    def test_session_breakdown_without_entry_bar_column_renders_placeholder(self):
        trades = pd.DataFrame({"r_result": [1.0, -1.0]})   # no entry_bar column
        html = _chart_session_breakdown(trades)
        assert "<canvas" not in html


# ─────────────────────────────────────────────────────────────────────────────
# JSON-serialization of numpy/pandas scalar types
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyJsonSafety:
    def test_json_default_converts_numpy_scalars(self):
        assert _json_default(np.float64(1.5)) == 1.5
        assert _json_default(np.int64(3)) == 3

    def test_render_chart_handles_numpy_types_in_config(self):
        config = {
            "type": "bar",
            "data": {"labels": ["a"], "datasets": [{"data": [np.float32(2.5)]}]},
        }
        html = _render_chart(config)
        assert "<canvas" in html
        # Must not have raised, and the numpy value must appear as a plain number
        assert "2.5" in html

    def test_full_report_generation_with_real_backtest_data_does_not_raise(self, enriched_df):
        """End-to-end: DataFrame-derived data (numpy int64/float64 dtypes
        throughout) must serialize cleanly across all 6 charts."""
        bt = run_backtest(enriched_df)
        tbl = results_table(bt)
        out_path = tempfile.mktemp(suffix=".html")
        result_path = generate_report(enriched_df, bt, tbl, symbol="XAUUSD", tf="M15",
                                       out_path=out_path, include_ceo_chart=False)
        assert result_path == out_path
        html = open(out_path).read()
        assert "chart.umd" in html   # Chart.js CDN present
        assert html.count("new Chart(") == 6
        assert "plotly" not in html.lower()
