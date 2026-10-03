#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    build = root / "build"
    out = root / "results"
    out.mkdir(exist_ok=True)

    subprocess.run(["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel"], check=True)
    subprocess.run([str(build / "research" / "smm_unit_tests")], check=True)

    raw = subprocess.run(
        [str(build / "research" / "smm_research"), str(args.runs), str(out)],
        check=True, capture_output=True, text=True
    )
    data = json.loads(raw.stdout)
    (out / "research_run.json").write_text(json.dumps(data, indent=2))

    import matplotlib.pyplot as plt
    policy = np.genfromtxt(out / "policy_t0.csv", delimiter=",", names=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(policy["inventory"], policy["bid_delta"], label="bid delta")
    plt.plot(policy["inventory"], policy["ask_delta"], label="ask delta")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Inventory")
    plt.ylabel("Quote distance")
    plt.title("HJB-QVI quote skew at t=0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "policy_skew.png", dpi=160)
    plt.close()

    ev = np.genfromtxt(out / "sample_hawkes_events.csv", delimiter=",", names=True)
    plt.figure(figsize=(9, 3.5))
    sell = ev["type"] == 0
    buy = ev["type"] == 1
    plt.scatter(ev["time"][sell], np.zeros(np.sum(sell)), s=8, label="sell MO")
    plt.scatter(ev["time"][buy], np.ones(np.sum(buy)), s=8, label="buy MO")
    plt.yticks([0, 1], ["sell", "buy"])
    plt.xlabel("time (s)")
    plt.title("Sample Hawkes order-flow path")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "hawkes_events.png", dpi=160)
    plt.close()

    q = data["strategies"]["hjb_qvi"]
    b = data["strategies"]["fixed_spread"]
    reduction = 100 * (b["mean_rms_inventory"] - q["mean_rms_inventory"]) / b["mean_rms_inventory"]
    summary = {
        "inventory_rms_reduction_pct": reduction,
        "hawkes_direction_auc": data["hawkes"]["next_sell_direction_auc"],
        "quote_p99_ns": data["quote_engine"]["p99_ns"],
        "fdm_solve_ms": data["hjb_fdm_solve_ms"],
    }
    (out / "python_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
