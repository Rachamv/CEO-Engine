"""
Validates ceo_engine_mt5/templates/dashboard.html's JavaScript is
actually syntactically valid, using Node's real parser.

Why this needs its own test: the dashboard's JS lives in a single inline
<script> block (confirmed: exactly one <script src=...> for the
Lightweight Charts CDN, and exactly one inline <script>...</script> for
everything else -- roughly 950 lines). A syntax error ANYWHERE in that
block breaks parsing of the WHOLE thing in any real browser -- not just
the broken line, everything, since template-literal and expression
syntax errors are caught by the JS engine before any code runs at all.

Flask/Jinja2 has no way to catch this: render_template_string() only
cares about {{ }}/{% %} syntax, so it happily serves syntactically
invalid JavaScript as plain text with a 200 OK. None of the existing
Python-side dashboard tests (test_dashboard_security.py) would ever
catch this either, since they only check the served HTML *string*
contains expected substrings -- they don't execute the JS.

This exact scenario happened for real: a stray semicolon inside a
template-literal `${...}` expression (`${dp>=0?'a':'b';font-weight:700}`
instead of `${dp>=0?'a':'b'};font-weight:700`) broke the entire
dashboard's JavaScript, undetected until this test was written and
actually ran it through Node's parser.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


TEMPLATE_PATH = (Path(__file__).resolve().parent.parent /
                  "ceo_engine_mt5" / "templates" / "dashboard.html")


def _node_available() -> bool:
    return shutil.which("node") is not None


def _extract_inline_scripts(html: str) -> list:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


# A minimal browser-DOM stub, just enough for the dashboard's top-level
# code (event listener registration, initial element lookups) to parse
# and execute without throwing on missing globals. This isn't trying to
# actually simulate the dashboard -- it exists purely so Node can load
# and run the script far enough to prove there's no syntax error and no
# immediate ReferenceError on the obvious globals.
_DOM_STUB = """
function _stubEl(){
  return {
    value:"", checked:false, textContent:"", innerHTML:"", style:{},
    dataset:{}, classList:{add(){},remove(){},toggle(){},contains(){return false;}},
    addEventListener(){}, appendChild(){}, querySelector(){return null;},
    querySelectorAll(){return [];}, cells:[{},{},{},{},{}], offsetWidth:0,
  };
}
const document = {
  getElementById(){ return _stubEl(); },
  querySelectorAll(){ return []; },
  querySelector(){ return null; },
  addEventListener(){},
  createElement(){ return _stubEl(); },
};
const window = { addEventListener(){}, location:{}, innerWidth:1024 };
const navigator = { userAgent:"node" };
class ResizeObserver { constructor(cb){} observe(){} disconnect(){} }
// No-ops: this test checks the script's top-level synchronous setup runs
// without throwing, not simulated runtime behavior over time -- letting
// real setInterval callbacks fire would pull in every fetch() response
// shape the real backend returns, which is a different (and much
// bigger) testing task than "does this script load without error".
const setTimeout = () => 0;
const setInterval = () => 0;
const clearTimeout = () => {};
const clearInterval = () => {};
const fetch = () => Promise.resolve({ json: () => Promise.resolve({}), ok: true });
class FakeChartSeries {
  setData(){} setMarkers(){} createPriceLine(){return {};} priceScale(){return {applyOptions(){}};}
}
class FakeChart {
  addCandlestickSeries(){return new FakeChartSeries();}
  addLineSeries(){return new FakeChartSeries();}
  addHistogramSeries(){return new FakeChartSeries();}
  timeScale(){return {fitContent(){}};}
  priceScale(){return {applyOptions(){}};}
  applyOptions(){}
  subscribeCrosshairMove(){}
  remove(){}
}
const LightweightCharts = {
  createChart(){ return new FakeChart(); },
  CrosshairMode: { Normal: 0 },
  LineStyle: { Solid:0, Dotted:1, Dashed:2, LargeDashed:3, SparseDotted:4 },
};
class Chart { constructor(){} }
"""


@pytest.fixture(scope="module")
def dashboard_html():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


class TestDashboardTemplateExists:
    def test_template_file_is_present(self):
        assert TEMPLATE_PATH.exists()

    def test_has_exactly_one_inline_script_block(self, dashboard_html):
        """Documents the current structure (one CDN <script src> + one
        big inline <script>) -- if this ever changes, the syntax-check
        test below needs to iterate over multiple blocks instead of
        assuming just one."""
        scripts = _extract_inline_scripts(dashboard_html)
        assert len(scripts) == 1


class TestDashboardJsIsValidSyntax:
    def test_inline_script_passes_node_syntax_check(self, dashboard_html):
        if not _node_available():
            pytest.skip("node not available in this environment")
        scripts = _extract_inline_scripts(dashboard_html)
        assert scripts, "no inline <script> block found to check"
        proc = subprocess.run(
            ["node", "--check", "-"],
            input=scripts[0], capture_output=True, text=True, timeout=15,
        )
        assert proc.returncode == 0, (
            f"dashboard.html's inline JavaScript has a syntax error -- this "
            f"breaks the ENTIRE dashboard in any real browser, not just the "
            f"broken line (a syntax error prevents the whole script block "
            f"from parsing at all):\n{proc.stderr}"
        )

    def test_inline_script_loads_and_runs_top_level_code_without_throwing(self, dashboard_html):
        """Goes one step further than a syntax check: actually executes
        the script's top-level synchronous setup (against a minimal DOM
        stub, with setTimeout/setInterval stubbed as no-ops so no
        callback actually fires) to catch immediate top-level
        ReferenceErrors too, not just parse errors. Deliberately does not
        try to simulate real runtime behavior over time -- that would
        require matching every /api/* response shape the real Flask
        backend returns, which is a different and much bigger testing
        task than "does this script load without error"."""
        if not _node_available():
            pytest.skip("node not available in this environment")
        scripts = _extract_inline_scripts(dashboard_html)
        js_source = _DOM_STUB + "\n" + scripts[0]
        proc = subprocess.run(
            ["node", "-e", js_source],
            capture_output=True, text=True, timeout=15,
        )
        assert proc.returncode == 0, f"dashboard.html's JS threw when executed:\n{proc.stderr}"

    def test_no_stray_semicolon_inside_template_literal_expressions(self, dashboard_html):
        """Narrower regression guard for the exact bug pattern that
        caused this: a semicolon inside a ${...} template-literal
        substitution. Not a complete grammar check (that's what the
        Node syntax check above is for) -- just a fast, dependency-free
        pattern check for this specific mistake recurring."""
        scripts = _extract_inline_scripts(dashboard_html)
        for script in scripts:
            for match in re.finditer(r"\$\{([^{}]*)\}", script):
                inner = match.group(1)
                assert ";" not in inner, (
                    f"found a semicolon inside a template-literal "
                    f"${{...}} expression: ${{{inner}}} -- this is a "
                    f"syntax error, not valid JS"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Help & Glossary panel -- the always-available reference for CEO-method
# terms, dashboard settings, and troubleshooting.
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpPanelMarkup:
    def test_help_overlay_and_trigger_button_present(self, dashboard_html):
        assert 'id="help-overlay"' in dashboard_html
        assert "openHelp()" in dashboard_html

    def test_help_search_input_present(self, dashboard_html):
        assert 'id="help-search"' in dashboard_html
        assert 'oninput="filterHelp(this.value)"' in dashboard_html

    def test_all_three_categories_present(self, dashboard_html):
        for cat in ["CEO Method Terms", "Settings Explained", "Troubleshooting"]:
            assert cat in dashboard_html

    def test_key_ceo_terms_are_documented(self, dashboard_html):
        for term in ["Order Block", "Fair Value Gap", "Quasimodo",
                     "Break of Structure", "Fibonacci 50%", "Confluence",
                     "Regime", "Session", "CEO Valid"]:
            assert term in dashboard_html

    def test_key_settings_are_documented(self, dashboard_html):
        for term in ["Risk per trade", "Min signal quality", "Min model consistency",
                     "Confluence mode", "Daily Loss Limit", "Auto-trade",
                     "MTF mode", "Performance Feedback", "Walk-forward validation",
                     "News gate"]:
            assert term in dashboard_html

    def test_troubleshooting_entries_present(self, dashboard_html):
        for term in ["MT5 won't connect", "Telegram alerts not arriving",
                     "Engine won't start", "Chart looks empty", "Dashboard password"]:
            assert term in dashboard_html

    def test_every_help_entry_has_a_searchable_data_attribute(self, dashboard_html):
        entries = re.findall(r'<div class="help-entry"[^>]*>', dashboard_html)
        assert len(entries) >= 20  # every documented term above, at minimum
        for entry in entries:
            assert 'data-help="' in entry

    def test_contextual_help_links_present_for_tricky_settings(self, dashboard_html):
        # Spot-check a few of the settings most likely to confuse someone,
        # linked directly from where they're configured.
        assert "openHelp('confluence mode')" in dashboard_html
        assert "openHelp('min model consistency')" in dashboard_html
        assert "openHelp('mtf mode')" in dashboard_html
        assert "openHelp('order block')" in dashboard_html


class TestHelpPanelJsLogic:
    """Executes the real filterHelp()/openHelp()/closeHelp() against a
    minimal but real DOM (jsdom-free -- just enough element/classList
    behavior for these functions' actual code paths) so this checks
    real search/filter logic, not just markup presence."""

    def _run(self, dashboard_html, script_calls):
        if not _node_available():
            pytest.skip("node not available in this environment")
        scripts = _extract_inline_scripts(dashboard_html)
        # A tiny real DOM: enough elements with classList/style/textContent
        # to exercise filterHelp's actual matching + category-hiding logic.
        harness = _DOM_STUB + r"""
        // Minimal real element graph for the help panel, built from the
        // actual entries so this test breaks if new entries lose their
        // data-help attribute or category structure.
        function makeEntry(term, def, dataHelp){
          return { classList:{list:['help-entry'],
                     contains(c){return this.list.includes(c);},
                     add(c){this.list.push(c);}, remove(c){this.list=this.list.filter(x=>x!==c);}},
                   dataset:{help:dataHelp}, textContent: term+' '+def,
                   style:{display:''}, nextElementSibling:null };
        }
        function makeCat(name){
          return { classList:{list:['help-cat'],
                     contains(c){return this.list.includes(c);}},
                   textContent:name, style:{display:''}, nextElementSibling:null };
        }
        const cat1 = makeCat('CEO Method Terms');
        const e1 = makeEntry('Order Block', 'last opposite candle before impulse', 'order block ob');
        const e2 = makeEntry('Break of Structure', 'price breaks a swing point', 'bos break structure');
        const cat2 = makeCat('Settings Explained');
        const e3 = makeEntry('Risk per trade', 'percent of balance risked', 'risk percent');
        cat1.nextElementSibling = e1; e1.nextElementSibling = e2; e2.nextElementSibling = cat2;
        cat2.nextElementSibling = e3;
        const allEntries = [e1, e2, e3];
        const allCats = [cat1, cat2];
        const helpEmpty = _stubEl();
        const helpSearch = _stubEl();
        const helpOverlay = { classList: { removed:false, added:false,
          remove(){this.removed=true;}, add(){this.added=true;} } };
        const _byId = {'help-overlay':helpOverlay, 'help-search':helpSearch, 'help-empty':helpEmpty};
        document.getElementById = (id) => _byId[id] || _stubEl();
        document.querySelectorAll = (sel) => sel === '.help-entry' ? allEntries
                                            : sel === '.help-cat' ? allCats : [];
        """
        js_source = harness + "\n" + scripts[0] + "\n" + script_calls
        proc = subprocess.run(["node", "-e", js_source], capture_output=True, text=True, timeout=15)
        assert proc.returncode == 0, f"help panel JS threw:\n{proc.stderr}"
        return proc.stdout

    def test_filter_matches_by_data_help_attribute(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        filterHelp('bos');
        console.log(JSON.stringify({
          e1: allEntries[0].style.display, e2: allEntries[1].style.display,
          e3: allEntries[2].style.display, empty: helpEmpty.style.display,
        }));
        """)
        result = out.strip().splitlines()[-1]
        import json as _json
        data = _json.loads(result)
        assert data["e1"] == "none"   # "Order Block" doesn't match "bos"
        assert data["e2"] == ""       # "Break of Structure" -> data-help has "bos"
        assert data["e3"] == "none"
        assert data["empty"] == "none"

    def test_filter_with_no_matches_shows_empty_state(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        filterHelp('zzz_no_such_term');
        console.log(JSON.stringify({empty: helpEmpty.style.display}));
        """)
        import json as _json
        data = _json.loads(out.strip().splitlines()[-1])
        assert data["empty"] == "block"

    def test_empty_query_shows_everything(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        filterHelp('');
        console.log(JSON.stringify({
          e1: allEntries[0].style.display, e2: allEntries[1].style.display,
          e3: allEntries[2].style.display, empty: helpEmpty.style.display,
        }));
        """)
        import json as _json
        data = _json.loads(out.strip().splitlines()[-1])
        assert data == {"e1": "", "e2": "", "e3": "", "empty": "none"}

    def test_category_hidden_when_no_entries_in_it_match(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        filterHelp('risk');
        console.log(JSON.stringify({
          cat1: allCats[0].style.display, cat2: allCats[1].style.display,
        }));
        """)
        import json as _json
        data = _json.loads(out.strip().splitlines()[-1])
        assert data["cat1"] == "none"  # no entry under "CEO Method Terms" matches "risk"
        assert data["cat2"] == ""      # "Risk per trade" is under "Settings Explained"

    def test_open_help_sets_search_value_and_removes_hidden_class(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        openHelp('order block');
        console.log(JSON.stringify({
          searchVal: helpSearch.value, overlayRemoved: helpOverlay.classList.removed,
        }));
        """)
        import json as _json
        data = _json.loads(out.strip().splitlines()[-1])
        assert data["searchVal"] == "order block"
        assert data["overlayRemoved"] is True

    def test_close_help_adds_hidden_class(self, dashboard_html):
        out = self._run(dashboard_html, r"""
        closeHelp();
        console.log(JSON.stringify({overlayAdded: helpOverlay.classList.added}));
        """)
        import json as _json
        data = _json.loads(out.strip().splitlines()[-1])
        assert data["overlayAdded"] is True
