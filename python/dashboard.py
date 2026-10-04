from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"

st.set_page_config(page_title="Stochastic Market Maker", layout="wide")
st.title("Stochastic Market Maker")
st.caption("Local dashboard for the checked-in research outputs.")

with st.sidebar:
    result_dir = st.text_input("Results directory", str(DEFAULT_RESULTS))
    result_path = Path(result_dir).expanduser().resolve()

json_path = result_path / "research_run.json"
policy_path = result_path / "policy_t0.csv"
events_path = result_path / "sample_hawkes_events.csv"

if not json_path.exists():
    st.error(f"Could not find {json_path}. Run the research pipeline first.")
    st.code("python scripts/run_research.py --runs 100 --output results/run")
    st.stop()

data = json.loads(json_path.read_text())
qvi = data["strategies"]["hjb_qvi"]
base = data["strategies"]["fixed_spread"]
reduction = 100.0 * (base["mean_rms_inventory"] - qvi["mean_rms_inventory"]) / base["mean_rms_inventory"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMS inventory reduction", f"{reduction:.2f}%")
c2.metric("Hawkes direction AUC", f"{data['hawkes']['next_sell_direction_auc']:.4f}")
c3.metric("Quote p99", f"{data['quote_engine']['p99_ns']:.2f} ns")
c4.metric("FDM solve", f"{data['hjb_fdm_solve_ms']:.4f} ms")

st.subheader("Inventory policy")
if policy_path.exists():
    rows = list(csv.DictReader(policy_path.read_text().splitlines()))
    inventory = [int(r["inventory"]) for r in rows]
    bid = [float(r["bid_delta"]) for r in rows]
    ask = [float(r["ask_delta"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(inventory, bid, label="Bid distance")
    ax.plot(inventory, ask, label="Ask distance")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Inventory")
    ax.set_ylabel("Quote distance")
    ax.set_title("HJB-QVI policy at t = 0")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
else:
    st.info("policy_t0.csv is not present in this results directory.")

st.subheader("Controller comparison")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(["HJB-QVI", "Fixed spread"], [qvi["mean_rms_inventory"], base["mean_rms_inventory"]])
ax.set_ylabel("Mean RMS inventory")
ax.set_title("Mean RMS inventory across shared paths")
st.pyplot(fig, clear_figure=True)

st.subheader("Hawkes event path")
if events_path.exists():
    times = []
    kinds = []
    with events_path.open() as fh:
        for row in csv.DictReader(fh):
            times.append(float(row["time"]))
            kinds.append(int(row["type"]))
    sell_t = [t for t, k in zip(times, kinds) if k == 0]
    buy_t = [t for t, k in zip(times, kinds) if k == 1]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.scatter(sell_t, [0] * len(sell_t), s=5, label="sell")
    ax.scatter(buy_t, [1] * len(buy_t), s=5, label="buy")
    ax.set_yticks([0, 1], ["sell", "buy"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Exact C++ Hawkes path used for the reference sample")
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)
else:
    st.info("sample_hawkes_events.csv is not present in this results directory.")

with st.expander("Raw run record"):
    st.json(data)
