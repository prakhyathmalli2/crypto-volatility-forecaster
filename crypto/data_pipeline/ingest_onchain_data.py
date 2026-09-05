"""Free Bitcoin on-chain ingestion using public Blockchain.com charts."""
from __future__ import annotations

import requests
import pandas as pd

from config import BLOCKCHAIN_CHARTS_BASE, RAW_DIR

CHARTS = {
    "active_addresses": "n-unique-addresses",
    "tx_count": "n-transactions",
    "hash_rate": "hash-rate",
    "difficulty": "difficulty",
    "miners_revenue": "miners-revenue",
}


def fetch_chart(chart_name: str, timespan: str = "2years") -> pd.DataFrame:
    url = f"{BLOCKCHAIN_CHARTS_BASE.rstrip('/')}/{chart_name}"
    r = requests.get(url, params={"timespan": timespan, "format": "json"}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    values = payload.get("values", [])
    rows = pd.DataFrame(values)
    if rows.empty:
        return pd.DataFrame(columns=["timestamp", chart_name])
    rows["timestamp"] = pd.to_datetime(rows["x"], unit="s", utc=True)
    return rows[["timestamp", "y"]].rename(columns={"y": chart_name})


def build_onchain_dataset(timespan: str = "2years") -> pd.DataFrame:
    frames = [fetch_chart(chart, timespan) for chart in CHARTS.values()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")
    return merged.sort_values("timestamp").ffill()


def save_onchain(timespan: str = "2years"):
    df = build_onchain_dataset(timespan)
    out = RAW_DIR / "onchain_btc.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = save_onchain()
    print(f"Saved {out}")
