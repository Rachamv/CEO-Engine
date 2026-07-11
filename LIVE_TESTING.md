# CEO Engine MT5 — Live Testing Guide

## Pre-Flight Checklist

✅ **Dependencies**: All installed (pandas, numpy, MetaTrader5, Flask, matplotlib)
✅ **Python**: 3.14.2
✅ **MT5**: Connected and ready

---

## Quick Start: Run Live Signals

### Example 1: Monitor EURUSD on 1H
```bash
python run.py --symbol EURUSD --tf H1 --source mt5 --live
```

### Example 2: Monitor XAUUSD on 15M with sound alerts
```bash
python run.py --symbol XAUUSD --tf M15 --source mt5 --live --sound
```

### Example 3: Confluence mode (fire when 3+ models agree)
```bash
python run.py --symbol XAUUSD --tf H1 --source mt5 --live --confluence --min-models 3
```

---

## CLI Reference

| Flag | Description | Example |
|------|-------------|---------|
| `--symbol` | Trading pair | `EURUSD`, `XAUUSD`, `US30` |
| `--tf` | Timeframe | `M1`, `M5`, `M15`, `H1`, `H4`, `D1` |
| `--source` | Data source | `mt5` (live MT5 data) |
| `--live` | Live mode (prints signals on every new bar) | — |
| `--sound` | Play sound on signal | — |
| `--confluence` | Only fire when multiple models agree | — |
| `--min-models` | Minimum models for confluence | `3`, `4`, `5` |
| `--dashboard-port` | Flask dashboard port | `5000` (default) |
| `--no-htf` | Skip higher timeframe filter | — |

---

## Live Output Example

```
════════════════════════════════════════════════════════════
[2024-01-15 14:00:00 UTC]  EURUSD H1  ▲ LONG
════════════════════════════════════════════════════════════
  Model      : LQ + Trend
  Quality    : 72.5 / 100
  Regime     : Trend Up
  Alignment  : Aligned
  HTF Bias   : Bullish (H4)
  Confluence : 5 / 16 models
  ─────────────────────────────
  Entry ref  : 1.09245
  SL    ref  : 1.08980  (-26.5 pips)
  TP1   ref  : 1.09510  (+26.5 pips)
  TP2   ref  : 1.09775  (+53.0 pips)
  TP3   ref  : 1.10040  (+79.5 pips)
════════════════════════════════════════════════════════════
```

---

## Check Your Broker's Symbol Names

Symbol names vary by broker. Common formats:

| Instrument | Common Formats |
|------------|---------------|
| EURUSD | `EURUSD`, `EURUSD.m`, `EURUSDm` |
| Gold | `XAUUSD`, `GOLD`, `XAUUSD.m` |
| US30 (Dow) | `US30`, `DJ30`, `DJIA` |
| NAS100 | `NAS100`, `USTEC`, `NAS100m` |
| Brent Oil | `XBRUSD`, `OILm`, `BRENT` |
| Bitcoin | `BTCUSD`, `Bitcoin` |

**→ Check your MT5 MarketWatch for exact symbol names**

---

## Dashboard (Optional)

Run the web dashboard on port 5000:
```bash
python launcher.py
```

This launches a system tray app that:
- Starts the Flask dashboard server
- Opens browser to http://localhost:5000
- Allows real-time monitoring

---

## Testing Steps

1. **Open MetaTrader5 terminal** and log in to your account
2. **Choose a symbol** from your MarketWatch (e.g., EURUSD)
3. **Run a test command**:
   ```bash
   python run.py --symbol EURUSD --tf H1 --source mt5 --live
   ```
4. **Watch for signals** — the system will print on each new candle close
5. **Validate signals** against your chart (signals are reference levels only)

---

## Troubleshooting

### "MT5 Not Connected"
→ Ensure MetaTrader5 terminal is **open and logged in**

### "Symbol not found"
→ Check exact symbol name in MT5 MarketWatch (some brokers use `EURUSD.m` instead of `EURUSD`)

### "No candles found"
→ Ensure the timeframe is available on your broker for that symbol

### "Port 5000 already in use"
→ Use a different port: `python run.py ... --dashboard-port 5001`

---

## Next Steps

- Run a backtest first: `python run.py --symbol EURUSD --tf H1 --source mt5`
- Check performance stats
- Then move to live: `python run.py --symbol EURUSD --tf H1 --source mt5 --live`
