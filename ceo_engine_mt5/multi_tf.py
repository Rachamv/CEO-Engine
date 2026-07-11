"""
The CEO Protocol — Multi-Timeframe Signal Stack
====================================================
Fires a signal only when multiple timeframes agree simultaneously.

The CEO method is top-down:
    H4  → structural bias (trend direction, major swing context)
    H1  → intermediate structure (BOS, OB zones, Fib zones)
    M15 → entry timing (sweep, candle pattern confirmation)
    M5  → precision entry (optional fine-tuning)

A signal that appears on M15 alone is a base sweep.
A signal that appears on M15 AND H1 both show a valid setup
aligned with H4 bias is a genuinely different quality signal.

This module detects that cross-timeframe confirmation and scores it.

Architecture
------------
    MTFStack.check(symbol, tfs, conn) → MTFResult

    Each timeframe runs the full 225-column pipeline independently.
    The stack then checks whether the same direction is signalled
    across the required timeframes before returning a valid signal.

Confirmation modes
------------------
    "sweep"    — all TFs must show a base sweep in the same direction
    "ceo"      — all TFs must show ceo_long_valid / ceo_short_valid
    "bias"     — lower TF must have a signal; upper TFs must have
                 correct HTF bias (htf_bullish / htf_bearish)
    "cascade"  — each TF must confirm the one above it in sequence

Output
------
    MTFResult.valid        bool    — stack confirmed
    MTFResult.direction    str     — "LONG" or "SHORT"
    MTFResult.symbol       str
    MTFResult.tfs          list    — timeframes that confirmed
    MTFResult.score        float   — 0-100 confidence score
    MTFResult.entry_tf     str     — lowest TF (entry timing)
    MTFResult.entry_last   Series  — last bar of entry TF (for SL/TP)
    MTFResult.details      dict    — per-TF breakdown

Usage
-----
    from .multi_tf import MTFStack

    stack = MTFStack(
        tfs       = ["H4", "H1", "M15"],
        mode      = "bias",
        min_tfs   = 2,         # at least 2 TFs must confirm
        entry_tf  = "M15",     # lowest TF = entry
    )

    # In the monitoring loop (replaces check_symbol for multi-TF use)
    result = stack.check(symbol, conn, params, signal_params)
    if result.valid:
        print(result.summary())
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)

# Pipeline imports
from .indicators      import calc_all
from .signals         import build_all, build_confluence
from .candle_patterns import build_candle_patterns
from .ceo_structure   import build_ceo_structure
from .patterns        import build_patterns
from .session_filter  import add_session_columns


# ─────────────────────────────────────────────────────────────────────────────
# HTF auto-select (mirrors mt5_live._auto_htf)
# ─────────────────────────────────────────────────────────────────────────────

_HTF_MAP = {
    "m1":"m15","m2":"m15","m3":"m15","m5":"m15",
    "m15":"h1","m30":"h1","h1":"h4","h2":"h4",
    "h4":"d1","h6":"d1","h8":"d1","h12":"d1","d1":"w1",
    "1m":"15m","5m":"15m","15m":"1h","30m":"1h",
    "1h":"4h","4h":"1d","1d":"1w",
}

def _auto_htf(tf: str) -> str:
    return _HTF_MAP.get(tf.lower(), "h4")


# ─────────────────────────────────────────────────────────────────────────────
# Timeframe ordering (for cascade mode)
# ─────────────────────────────────────────────────────────────────────────────

_TF_RANK = {
    "m1":1,"m2":2,"m3":3,"m5":5,"m10":10,"m15":15,"m30":30,
    "h1":60,"h2":120,"h4":240,"h6":360,"h8":480,"h12":720,
    "d1":1440,"w1":10080,
    "1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440,"1w":10080,
}

def _tf_rank(tf: str) -> int:
    return _TF_RANK.get(tf.lower(), 60)

def _sort_tfs(tfs: List[str]) -> List[str]:
    """Sort timeframes from highest (HTF) to lowest (entry)."""
    return sorted(tfs, key=lambda t: _tf_rank(t), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-TF state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFState:
    """Enriched state for one timeframe."""
    tf:              str
    df:              pd.DataFrame
    last:            pd.Series

    # Signal columns
    base_long:       bool = False
    base_short:      bool = False
    ceo_long:        bool = False
    ceo_short:       bool = False
    htf_bullish:     bool = False
    htf_bearish:     bool = False
    bos_long:        bool = False
    bos_short:       bool = False
    in_discount:     bool = False
    in_premium:      bool = False
    quality_long:    float = 0.0
    quality_short:   float = 0.0
    cp_bull_confirm: bool = False
    cp_bear_confirm: bool = False
    pat_name:        str = ""
    regime:          str = ""
    ob_bull_active:  bool = False
    ob_bear_active:  bool = False

    def _b(self, col: str) -> bool:
        v = self.last.get(col, False)
        return bool(v) if not (isinstance(v, float) and np.isnan(v)) else False

    def _f(self, col: str) -> float:
        v = self.last.get(col, 0.0)
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0

    @classmethod
    def from_df(cls, tf: str, df: pd.DataFrame) -> "TFState":
        last = df.iloc[-1]
        s = cls(tf=tf, df=df, last=last)
        s.base_long       = s._b("base_long")
        s.base_short      = s._b("base_short")
        s.ceo_long        = s._b("ceo_long_valid")
        s.ceo_short       = s._b("ceo_short_valid")
        s.htf_bullish     = s._b("htf_bullish")
        s.htf_bearish     = s._b("htf_bearish")
        s.bos_long        = s._b("bos_long")
        s.bos_short       = s._b("bos_short")
        s.in_discount     = s._b("in_discount")
        s.in_premium      = s._b("in_premium")
        s.quality_long    = s._f("quality_long")
        s.quality_short   = s._f("quality_short")
        s.cp_bull_confirm = s._b("cp_bull_confirmation")
        s.cp_bear_confirm = s._b("cp_bear_confirmation")
        s.pat_name        = str(last.get("pat_name", "") or "")
        s.regime          = str(last.get("regime_name", "") or "")
        s.ob_bull_active  = s._b("ob_bull_active")
        s.ob_bear_active  = s._b("ob_bear_active")
        return s


# ─────────────────────────────────────────────────────────────────────────────
# MTF Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MTFResult:
    """Result from a multi-timeframe stack check."""
    valid:         bool
    direction:     str             # "LONG" | "SHORT" | ""
    symbol:        str
    tfs:           List[str]       # TFs that confirmed
    score:         float           # 0–100 composite confidence
    entry_tf:      str             # lowest TF = timing TF
    entry_last:    Optional[pd.Series] = None
    entry_df:      Optional[pd.DataFrame] = None
    details:       Dict[str, dict] = field(default_factory=dict)
    bar_time:      Optional[datetime] = None
    mode:          str = ""

    def summary(self) -> str:
        if not self.valid:
            return (f"[MTF] {self.symbol} — No cross-TF confirmation "
                    f"({', '.join(self.tfs) if self.tfs else 'none'})")
        arrow   = "▲" if self.direction == "LONG" else "▼"
        tf_str  = " + ".join(self.tfs)
        details = []
        for tf, d in self.details.items():
            flags = []
            if d.get("ceo_valid"):    flags.append("CEO✓")
            if d.get("bos"):          flags.append("BOS✓")
            if d.get("in_discount"):  flags.append("Disc✓")
            if d.get("ob_active"):    flags.append("OB✓")
            if d.get("cp_confirm"):   flags.append("CP✓")
            flag_str = " ".join(flags) if flags else "sweep"
            details.append(f"  {tf:<6}: Q={d.get('quality',0):.0f}  {flag_str}")
        return (
            f"\n{'═'*55}\n"
            f"  MTF STACK: {self.symbol}  {arrow} {self.direction}\n"
            f"  Mode      : {self.mode}  |  Score: {self.score:.1f}/100\n"
            f"  Confirmed : {tf_str}\n"
            f"  Entry TF  : {self.entry_tf}\n"
            + "\n".join(details) +
            f"\n{'═'*55}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (one TF)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(
    symbol:        str,
    tf:            str,
    conn,
    n_bars:        int          = 800,
    ind_params:    Optional[dict] = None,
    signal_params: Optional[dict] = None,
    sessions:      Optional[List[str]] = None,
    tick_size:     float        = 0.01,
) -> Optional[TFState]:
    """
    Fetch data and run the full 225-column pipeline for one TF.
    Returns TFState or None on failure.
    """
    sessions = sessions or ["london", "new_york", "overlap"]
    try:
        rates  = conn.fetch_rates(symbol, tf, n_bars=n_bars)
        if rates is None or len(rates) < 50:
            return None

        sym_info = conn.symbol_info(symbol)
        tick_sz  = sym_info.get("tick_size", tick_size)

        # Build base DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
        df = df.set_index("datetime")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.attrs["symbol"]    = symbol
        df.attrs["tick_size"] = tick_sz

        # HTF for bias
        htf_tf = _auto_htf(tf)
        htf_df = None
        try:
            htf_rates = conn.fetch_rates(symbol, htf_tf, n_bars=300)
            if htf_rates is not None and len(htf_rates) >= 50:
                htf_df = pd.DataFrame(htf_rates)
                htf_df["time"] = pd.to_datetime(htf_df["time"], unit="s", utc=True)
                htf_df = htf_df.rename(
                    columns={"time": "datetime", "tick_volume": "volume"})
                htf_df = htf_df.set_index("datetime")
                htf_df = htf_df[["open","high","low","close","volume"]].copy()
                htf_df.attrs["symbol"]    = symbol
                htf_df.attrs["tick_size"] = tick_sz
        except Exception as e:
            logger.warning("HTF fetch for multi-TF stack failed, continuing "
                            "without HTF bias: %s", e)

        # Full pipeline
        df = calc_all(df, params=ind_params)
        df = build_all(df, htf_df=htf_df, params=signal_params)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        # Kept consistent with every other full-pipeline call site even
        # though the MTF confirmation modes below don't read confluence
        # columns themselves -- see signals.build_all()'s docstring.
        df = build_confluence(df, params=signal_params)
        df = build_patterns(df)
        df = add_session_columns(df, allowed=sessions)

        return TFState.from_df(tf, df)

    except Exception as e:
        print(f"  ⚠️  MTF pipeline failed [{symbol} {tf}]: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation modes
# ─────────────────────────────────────────────────────────────────────────────

def _check_sweep(states: Dict[str, TFState], direction: str,
                 min_tfs: int) -> Tuple[bool, List[str]]:
    """All TFs must show base sweep in same direction."""
    col  = "base_long" if direction == "LONG" else "base_short"
    conf = [tf for tf, s in states.items() if getattr(s, col.replace("base_","base_"), False)
            or (direction == "LONG" and s.base_long)
            or (direction == "SHORT" and s.base_short)]
    # Simplify
    if direction == "LONG":
        conf = [tf for tf, s in states.items() if s.base_long]
    else:
        conf = [tf for tf, s in states.items() if s.base_short]
    return len(conf) >= min_tfs, conf


def _check_ceo(states: Dict[str, TFState], direction: str,
               min_tfs: int) -> Tuple[bool, List[str]]:
    """All TFs must show CEO full-sequence valid."""
    if direction == "LONG":
        conf = [tf for tf, s in states.items() if s.ceo_long]
    else:
        conf = [tf for tf, s in states.items() if s.ceo_short]
    return len(conf) >= min_tfs, conf


def _check_bias(states: Dict[str, TFState], direction: str,
                entry_tf: str, min_tfs: int,
                sorted_tfs: List[str]) -> Tuple[bool, List[str]]:
    """
    Entry TF must have a signal.
    Upper TFs must have correct HTF bias.
    This is the most practical mode for CEO method trading.

    Example (direction=LONG, tfs=[H4,H1,M15], entry_tf=M15):
        M15 must have base_long=True
        H1  must have htf_bullish=True  (or base_long, or ceo_long)
        H4  must have htf_bullish=True
    """
    entry_state = states.get(entry_tf)
    if entry_state is None:
        return False, []

    # Entry TF must have a base signal
    if direction == "LONG" and not entry_state.base_long:
        return False, []
    if direction == "SHORT" and not entry_state.base_short:
        return False, []

    confirmed = [entry_tf]

    # Upper TFs must confirm bias
    upper_tfs = [tf for tf in sorted_tfs if tf != entry_tf]
    for tf in upper_tfs:
        s = states.get(tf)
        if s is None:
            continue
        if direction == "LONG":
            # Bullish: HTF bias OR sweep OR CEO valid
            if s.htf_bullish or s.base_long or s.ceo_long:
                confirmed.append(tf)
        else:
            if s.htf_bearish or s.base_short or s.ceo_short:
                confirmed.append(tf)

    return len(confirmed) >= min_tfs, confirmed


def _check_cascade(states: Dict[str, TFState], direction: str,
                   sorted_tfs: List[str],
                   min_tfs: int) -> Tuple[bool, List[str]]:
    """
    Cascade: each TF must confirm the one above.
    H4 sets bias → H1 must agree with H4 → M15 must agree with H1.

    H4  : htf_bullish OR base_long
    H1  : base_long AND htf_bullish (H4 bias + own sweep)
    M15 : base_long AND htf_bullish (H1 bias + own sweep)
    M5  : base_long AND cp_bull_confirmation

    This is the strictest and highest-quality mode.
    """
    confirmed = []
    for i, tf in enumerate(sorted_tfs):
        s = states.get(tf)
        if s is None:
            continue

        if i == 0:
            # Highest TF — bias only
            if direction == "LONG" and (s.htf_bullish or s.base_long):
                confirmed.append(tf)
            elif direction == "SHORT" and (s.htf_bearish or s.base_short):
                confirmed.append(tf)
        else:
            # Lower TFs — must have their own sweep AND upper bias
            if direction == "LONG":
                upper_ok = any(states.get(sorted_tfs[j], TFState("","",pd.Series(),pd.Series())).htf_bullish
                               or states.get(sorted_tfs[j], TFState("","",pd.Series(),pd.Series())).base_long
                               for j in range(i))
                if s.base_long and upper_ok:
                    confirmed.append(tf)
            else:
                upper_ok = any(states.get(sorted_tfs[j], TFState("","",pd.Series(),pd.Series())).htf_bearish
                               or states.get(sorted_tfs[j], TFState("","",pd.Series(),pd.Series())).base_short
                               for j in range(i))
                if s.base_short and upper_ok:
                    confirmed.append(tf)

    return len(confirmed) >= min_tfs, confirmed


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

def _score(states: Dict[str, TFState], confirmed_tfs: List[str],
           direction: str, total_tfs: int) -> float:
    """
    Composite confidence score 0-100.

    Components:
        TF coverage    : confirmed / total × 40
        CEO quality    : avg quality across confirmed TFs × 0.35
        Structure bonus: BOS, OB, discount zone, CP confirmation
    """
    if not confirmed_tfs:
        return 0.0

    # TF coverage
    coverage = len(confirmed_tfs) / max(total_tfs, 1)
    score    = coverage * 40.0

    # Quality component
    q_vals = []
    struct_bonus = 0.0

    for tf in confirmed_tfs:
        s = states.get(tf)
        if s is None:
            continue
        if direction == "LONG":
            q_vals.append(s.quality_long)
            if s.bos_long:        struct_bonus += 5.0
            if s.in_discount:     struct_bonus += 3.0
            if s.ob_bull_active:  struct_bonus += 3.0
            if s.cp_bull_confirm: struct_bonus += 4.0
            if s.ceo_long:        struct_bonus += 6.0
        else:
            q_vals.append(s.quality_short)
            if s.bos_short:       struct_bonus += 5.0
            if s.in_premium:      struct_bonus += 3.0
            if s.ob_bear_active:  struct_bonus += 3.0
            if s.cp_bear_confirm: struct_bonus += 4.0
            if s.ceo_short:       struct_bonus += 6.0

    if q_vals:
        avg_q  = sum(q_vals) / len(q_vals)
        score += (avg_q / 100.0) * 35.0

    score += min(struct_bonus, 25.0)
    return min(round(score, 1), 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# MTFStack — main class
# ─────────────────────────────────────────────────────────────────────────────

class MTFStack:
    """
    Multi-timeframe signal stack.

    Parameters
    ----------
    tfs         : list of timeframes to monitor, e.g. ["H4","H1","M15"]
    mode        : confirmation mode
                  "bias"    — entry TF signal + upper TFs show correct bias
                  "sweep"   — all TFs must show a sweep in same direction
                  "ceo"     — all TFs must show CEO full-sequence valid
                  "cascade" — each TF confirms the one above it (strictest)
    min_tfs     : minimum number of TFs that must confirm (default: all)
    entry_tf    : the execution timeframe (lowest TF by default)
    sessions    : allowed sessions for session filter
    n_bars      : bars to fetch per TF
    min_score   : minimum composite score to return valid=True
    """

    def __init__(
        self,
        tfs:           List[str],
        mode:          str        = "bias",
        min_tfs:       Optional[int] = None,
        entry_tf:      Optional[str] = None,
        sessions:      Optional[List[str]] = None,
        n_bars:        int        = 800,
        min_score:     float      = 40.0,
        ind_params:    Optional[dict] = None,
        signal_params: Optional[dict] = None,
    ):
        self.tfs           = _sort_tfs(tfs)     # highest → lowest
        self.mode          = mode.lower()
        self.min_tfs       = min_tfs if min_tfs is not None else len(tfs)
        self.entry_tf      = entry_tf or self.tfs[-1]   # lowest TF
        self.sessions      = sessions or ["london","new_york","overlap"]
        self.n_bars        = n_bars
        self.min_score     = min_score
        self.ind_params    = ind_params
        self.signal_params = signal_params

        # Cache states to avoid re-running pipeline on same bar
        self._state_cache: Dict[str, Tuple[datetime, TFState]] = {}

    def _get_state(self, symbol: str, tf: str, conn) -> Optional[TFState]:
        """Get TFState, using cache if bar hasn't closed since last fetch."""
        state = _run_pipeline(
            symbol=symbol, tf=tf, conn=conn,
            n_bars=self.n_bars,
            ind_params=self.ind_params,
            signal_params=self.signal_params,
            sessions=self.sessions,
        )
        return state

    def check(
        self,
        symbol:    str,
        conn,
        verbose:   bool = False,
    ) -> MTFResult:
        """
        Run the full MTF stack check for a symbol.
        Returns MTFResult with valid=True if cross-TF confirmation found.
        """
        t0 = time.time()

        # Fetch all TF states
        states: Dict[str, TFState] = {}
        for tf in self.tfs:
            state = self._get_state(symbol, tf, conn)
            if state is not None:
                states[tf] = state

        if len(states) < self.min_tfs:
            return MTFResult(
                valid=False, direction="", symbol=symbol,
                tfs=[], score=0.0, entry_tf=self.entry_tf,
                details={"error": f"Only {len(states)}/{len(self.tfs)} TFs available"},
                mode=self.mode,
            )

        if verbose:
            elapsed = (time.time() - t0) * 1000
            print(f"  MTF [{symbol}]: {len(states)} TFs loaded in {elapsed:.0f}ms")

        # Check both directions
        for direction in ["LONG", "SHORT"]:
            conf_tfs = self._confirm(states, direction)

            if len(conf_tfs) < self.min_tfs:
                continue

            score = _score(states, conf_tfs, direction, len(self.tfs))

            if score < self.min_score:
                if verbose:
                    print(f"  MTF [{symbol}] {direction}: confirmed {conf_tfs} "
                          f"but score {score:.1f} < {self.min_score:.1f}")
                continue

            # Build per-TF details
            entry_state = states.get(self.entry_tf)
            details = {}
            for tf in conf_tfs:
                s = states[tf]
                details[tf] = {
                    "quality":    s.quality_long if direction=="LONG" else s.quality_short,
                    "ceo_valid":  s.ceo_long if direction=="LONG" else s.ceo_short,
                    "bos":        s.bos_long if direction=="LONG" else s.bos_short,
                    "in_discount":s.in_discount,
                    "ob_active":  s.ob_bull_active if direction=="LONG" else s.ob_bear_active,
                    "cp_confirm": s.cp_bull_confirm if direction=="LONG" else s.cp_bear_confirm,
                    "regime":     s.regime,
                    "pat_name":   s.pat_name,
                }

            result = MTFResult(
                valid     = True,
                direction = direction,
                symbol    = symbol,
                tfs       = conf_tfs,
                score     = score,
                entry_tf  = self.entry_tf,
                entry_last= entry_state.last if entry_state else None,
                details   = details,
                bar_time  = datetime.now(timezone.utc),
                mode      = self.mode,
            )

            if verbose:
                print(result.summary())

            return result

        entry_state = states.get(self.entry_tf)
        return MTFResult(
            valid=False, direction="", symbol=symbol,
            tfs=list(states.keys()), score=0.0,
            entry_tf=self.entry_tf,
            entry_last=entry_state.last if entry_state else None,
            entry_df=entry_state.df if entry_state else None,
            details={tf: {"quality": 0} for tf in states},
            mode=self.mode,
        )

    def _confirm(self, states: Dict[str, TFState],
                 direction: str) -> List[str]:
        """Apply the selected confirmation mode."""
        if self.mode == "sweep":
            _, conf = _check_sweep(states, direction, self.min_tfs)
        elif self.mode == "ceo":
            _, conf = _check_ceo(states, direction, self.min_tfs)
        elif self.mode == "cascade":
            _, conf = _check_cascade(states, direction, self.tfs, self.min_tfs)
        else:   # "bias" (default)
            _, conf = _check_bias(
                states, direction, self.entry_tf, self.min_tfs, self.tfs
            )
        return conf

    def summary_table(self, symbol: str, conn) -> str:
        """
        Print a diagnostic table of all TF states without needing a signal.
        Useful for manual analysis.
        """
        states = {tf: self._get_state(symbol, tf, conn)
                  for tf in self.tfs}

        lines = [f"\n  ── MTF State: {symbol} ──",
                 f"  {'TF':<8} {'BL':>4} {'BS':>4} {'CEO_L':>6} "
                 f"{'CEO_S':>6} {'HTF_B':>6} {'HTF_R':>6} "
                 f"{'BOS_L':>6} {'QL':>6} {'QS':>6} {'Regime':<12}"]
        lines.append("  " + "─" * 75)

        for tf in self.tfs:
            s = states.get(tf)
            if s is None:
                lines.append(f"  {tf:<8} {'—':>4} {'—':>4} {'—':>6} "
                              f"{'—':>6} {'—':>6} {'—':>6} {'—':>6} "
                              f"{'—':>6} {'—':>6} {'—':<12}")
                continue
            lines.append(
                f"  {tf:<8} "
                f"{'✓' if s.base_long  else '':>4} "
                f"{'✓' if s.base_short else '':>4} "
                f"{'✓' if s.ceo_long   else '':>6} "
                f"{'✓' if s.ceo_short  else '':>6} "
                f"{'✓' if s.htf_bullish else '':>6} "
                f"{'✓' if s.htf_bearish else '':>6} "
                f"{'✓' if s.bos_long   else '':>6} "
                f"{s.quality_long:>6.1f} "
                f"{s.quality_short:>6.1f} "
                f"{s.regime:<12}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper for mt5_live.py
# ─────────────────────────────────────────────────────────────────────────────

def make_mtf_stack(
    tfs:           List[str],
    mode:          str   = "bias",
    min_tfs:       Optional[int] = None,
    entry_tf:      Optional[str] = None,
    sessions:      Optional[List[str]] = None,
    min_score:     float = 40.0,
    ind_params:    Optional[dict] = None,
    signal_params: Optional[dict] = None,
) -> MTFStack:
    """Convenience constructor. Use in run_live() startup."""
    return MTFStack(
        tfs           = tfs,
        mode          = mode,
        min_tfs       = min_tfs,
        entry_tf      = entry_tf,
        sessions      = sessions,
        min_score     = min_score,
        ind_params    = ind_params,
        signal_params = signal_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (no MT5 needed — uses synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
