#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np


def find_executable(build_dir: pathlib.Path) -> pathlib.Path:
    candidates = [
        build_dir / "research" / "smm_research",
        build_dir / "research" / "Release" / "smm_research.exe",
        build_dir / "research" / "Debug" / "smm_research.exe",
        build_dir / "research" / "smm_research.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find smm_research after building. "
        "On Windows, check the selected CMake generator and Release configuration."
    )


def run_command(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required command not found: {cmd[0]}") from exc


def make_plots(out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policy = np.genfromtxt(out / "policy_t0.csv", delimiter=",", names=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(policy["inventory"], policy["bid_delta"], label="Bid distance")
    plt.plot(policy["inventory"], policy["ask_delta"], label="Ask distance")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Inventory")
    plt.ylabel("Quote distance")
    plt.title("HJB-QVI policy at t = 0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "policy_skew.png", dpi=160)
    plt.close()

    data = json.loads((out / "research_run.json").read_text())
    qvi = data["strategies"]["hjb_qvi"]
    base = data["strategies"]["fixed_spread"]

    plt.figure(figsize=(8, 4.5))
    names = ["HJB-QVI", "Fixed spread"]
    rms = [qvi["mean_rms_inventory"], base["mean_rms_inventory"]]
    plt.bar(names, rms)
    plt.ylabel("Mean RMS inventory")
    plt.title("Inventory risk comparison")
    plt.tight_layout()
    plt.savefig(out / "inventory_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    labels = ["Quote median (ns)", "Quote p99 (ns)", "FDM solve (ms)"]
    values_ns = [
        data["quote_engine"]["median_ns"],
        data["quote_engine"]["p99_ns"],
        data["hjb_fdm_solve_ms"] * 1_000_000.0,
    ]
    plt.bar(labels, values_ns)
    plt.yscale("log")
    plt.ylabel("Time (log scale, ns)")
    plt.title("Latency measurements")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "latency_benchmark.png", dpi=160)
    plt.close()

    events = np.genfromtxt(out / "sample_hawkes_events.csv", delimiter=",", names=True)
    sell = events["type"] == 0
    buy = events["type"] == 1

    plt.figure(figsize=(9, 3.5))
    plt.scatter(events["time"][sell], np.zeros(np.sum(sell)), s=6, label="Sell market order")
    plt.scatter(events["time"][buy], np.ones(np.sum(buy)), s=6, label="Buy market order")
    plt.yticks([0, 1], ["sell", "buy"])
    plt.xlabel("Time (s)")
    plt.title("Hawkes order-flow path, seed 1000")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "hawkes_events.png", dpi=160)
    plt.close()

    auc = data["hawkes"]["next_sell_direction_auc"]
    reduction = 100.0 * (base["mean_rms_inventory"] - qvi["mean_rms_inventory"]) / base["mean_rms_inventory"]
    summary = {
        "inventory_rms_reduction_pct": reduction,
        "hawkes_direction_auc": auc,
        "quote_p99_ns": data["quote_engine"]["p99_ns"],
        "fdm_solve_ms": data["hjb_fdm_solve_ms"],
    }
    (out / "python_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the stochastic market maker research pipeline.")
    parser.add_argument("--runs", type=int, default=100, help="Number of Monte Carlo paths.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/run"),
        help="Directory for generated experiment output.",
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=pathlib.Path("build"),
        help="CMake build directory.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing CMake build.")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    root = pathlib.Path(__file__).resolve().parents[1]
    build = (root / args.build_dir).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run_command(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
        if sys.platform.startswith("win"):
            run_command(["cmake", "--build", str(build), "--config", "Release", "--parallel"])
        else:
            run_command(["cmake", "--build", str(build), "--parallel"])

    if sys.platform.startswith("win"):
        run_command(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"])
    else:
        run_command(["ctest", "--test-dir", str(build), "--output-on-failure"])

    executable = find_executable(build)
    run_command([str(executable), str(args.runs), str(output)])
    make_plots(output)

    data = json.loads((output / "research_run.json").read_text())
    qvi = data["strategies"]["hjb_qvi"]
    base = data["strategies"]["fixed_spread"]
    reduction = 100.0 * (base["mean_rms_inventory"] - qvi["mean_rms_inventory"]) / base["mean_rms_inventory"]

    print("\nResearch run completed.")
    print(f"Output: {output}")
    print(f"Paths: {data['runs']}")
    print(f"Mean RMS inventory reduction: {reduction:.4f}%")
    print(f"Hawkes direction AUC: {data['hawkes']['next_sell_direction_auc']:.4f}")
    print(f"Quote p99: {data['quote_engine']['p99_ns']:.4f} ns")


if __name__ == "__main__":
    main()
