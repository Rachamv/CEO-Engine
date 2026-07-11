"""Quick MT5 connectivity check"""
import MetaTrader5 as mt5

try:
    if mt5.initialize():
        print("✓ MT5 Connected")
        info = mt5.terminal_info()
        print(f"  Account: {info.trade_mode}")
        print(f"  Server: {info.server}")
        mt5.shutdown()
    else:
        print("✗ MT5 Not Connected")
        print("  → Ensure MetaTrader5 terminal is OPEN and LOGGED IN")
except Exception as e:
    print(f"✗ Error: {e}")
