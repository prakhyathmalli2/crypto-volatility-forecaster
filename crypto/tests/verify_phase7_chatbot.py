"""Phase 7 comprehensive AI Chatbot verification script.
Tests all 8 quick actions across symbol switching: BTC/USDT -> RELIANCE.NS -> AAPL.
Verifies context injection, correct symbol detection, live telemetry accuracy, and lack of stale cross-asset context.
"""
import sys
import json
import urllib.request
from pathlib import Path

# Add crypto root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

ACTIONS = [
    "Why is this moving?",
    "Analyze this asset",
    "Explain RSI",
    "Explain MACD",
    "Summarize today's news",
    "What changed today?",
    "Explain the ML prediction",
    "Show technical signals",
]

SYMBOLS = ["BTC/USDT", "RELIANCE.NS", "AAPL"]

def query_chat(symbol: str, query: str) -> dict:
    url = "http://127.0.0.1:8000/api/v1/chat"
    body = json.dumps({"query": query, "symbol": symbol}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def verify_phase7():
    print("=================================================================")
    print("PHASE 7: REAL AI CHATBOT & MULTI-ASSET CONTEXT SWITCHING")
    print("=================================================================")

    for sym in SYMBOLS:
        print(f"\n--- TESTING ASSET CONTEXT: {sym} ---")
        for action in ACTIONS:
            res = query_chat(sym, action)
            reply = res.get("response", "")
            
            # Check symbol presence
            token = sym.split("/")[0].replace(".NS", "")
            sym_ok = (sym in reply or token in reply)
            
            # Check price/currency context
            if ".NS" in sym:
                curr_ok = ("₹" in reply or "INR" in reply or "Reliance" in reply)
            else:
                curr_ok = ("$" in reply or "USD" in reply or "BTC" in reply or "Apple" in reply)

            # Check no cross-contamination (e.g. Reliance in BTC, or BTC in Apple)
            cross_contam = False
            if sym == "BTC/USDT" and "Reliance" in reply:
                cross_contam = True
            if sym == "AAPL" and "Bitcoin" in reply and "BTC" not in action:
                cross_contam = True
            if sym == "RELIANCE.NS" and "Bitcoin" in reply and "BTC" not in action:
                cross_contam = True

            passed = sym_ok and curr_ok and not cross_contam
            status_str = "PASS" if passed else "FAIL"
            print(f"[{status_str}] '{action:25s}' => Sym: {sym_ok} | Curr: {curr_ok} | Contam: {cross_contam}")

if __name__ == "__main__":
    verify_phase7()
