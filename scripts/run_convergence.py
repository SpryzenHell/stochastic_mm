#!/usr/bin/env python3
"""Check how Monte Carlo estimates change with nested sample sizes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_counts(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample sizes must be comma-separated integers") from exc
    if not values or any(n < 2 or n > 100000 for n in values):
        raise argparse.ArgumentTypeError("each sample size must be between 2 and 100000")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("sample sizes must be unique and strictly increasing")
    if len(values) > 8:
        raise argparse.ArgumentTypeError("at most eight sample sizes are supported")
    return values


def critical_t_95(n: int) -> float:
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
        20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
        26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048
    }
    if n in table:
        return table[n]
    z = 1.959963984540054
    df = float(n - 1)
    return (z + (z**3 + z) / (4.0 * df)
            + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
            + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3))


def ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    half = critical_t_95(n) * std / math.sqrt(n)
    return mean - half, mean + half


def locate_executable(build: pathlib.Path) -> pathlib.Path:
    for candidate in (
        build / "research" / "smm_research",
        build / "research" / "Release" / "smm_research.exe",
        build / "research" / "Debug" / "smm_research.exe",
        build / "research" / "smm_research.exe",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"smm_research not found under {build}; build first or set --build-dir")


def validate_result(folder: pathlib.Path, n: int) -> dict:
    path = folder / "research_run.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("runs") != n:
        raise ValueError(f"{path}: expected {n} paths")
    if abs(result["hjb"]["horizon"] - result["hawkes"]["horizon"]) > 1e-12:
        raise ValueError(f"{path}: HJB and Hawkes horizons do not match")
    hjb = result["hjb"]
    if hjb["time_steps"] != math.ceil(hjb["horizon"] / hjb["dt"]) + 1:
        raise ValueError(f"{path}: HJB grid size is inconsistent")
    with (folder / "policy_t0.csv").open(newline="", encoding="utf-8") as stream:
        policy = list(csv.DictReader(stream))
    if len(policy) != 2 * hjb["q_max"] + 1:
        raise ValueError(f"{folder}: policy grid has the wrong number of rows")
    for row in policy:
        for key in ("bid_delta", "ask_delta"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"{folder}/policy_t0.csv contains non-finite {key}")
    return result


def plot_series(out: pathlib.Path, rows: list[dict], filename: str, title: str, ylabel: str,
                key: str, baseline_key: str | None = None,
                low_key: str | None = None, high_key: str | None = None,
                reference_line: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    x = [r["runs"] for r in rows]
    if low_key and high_key:
        y = [r[key] for r in rows]
        low = [m - r[low_key] for m, r in zip(y, rows)]
        high = [r[high_key] - m for m, r in zip(y, rows)]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=4, linewidth=2, label="HJB-QVI")
    else:
        ax.plot(x, [r[key] for r in rows], marker="o", linewidth=2, label="HJB-QVI" if baseline_key else title)
    if baseline_key:
        ax.plot(x, [r[baseline_key] for r in rows], marker="s", linewidth=2, label="Fixed spread")
    if reference_line is not None:
        ax.axhline(reference_line, linestyle="--", linewidth=1.5, label=f"Reference {reference_line:g}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(n) for n in x])
    ax.set_xlabel("Monte Carlo path count")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(out / f"{filename}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run nested-seed sample-size convergence checks.")
    parser.add_argument("--run-counts", default="25,50,100,200",
                        help="Increasing sample sizes; default 25,50,100,200")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/convergence"))
    parser.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    try:
        counts = parse_counts(args.run_counts)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    root = pathlib.Path(__file__).resolve().parents[1]
    build = (root / args.build_dir).resolve()
    out = (root / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not args.skip_build:
            subprocess.run(["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
            command = ["cmake", "--build", str(build), "--config", "Release", "--parallel"] if sys.platform.startswith("win") else ["cmake", "--build", str(build), "--parallel"]
            subprocess.run(command, check=True)
        ctest = ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"] if sys.platform.startswith("win") else ["ctest", "--test-dir", str(build), "--output-on-failure"]
        subprocess.run(ctest, check=True)
        exe = locate_executable(build)

        rows = []
        previous_samples = -1
        for i, n in enumerate(counts, 1):
            folder = out / "runs" / f"n_{n}"
            folder.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(exe), str(n), str(folder), "0.05", "0.005"],
                           check=True, capture_output=True, text=True)
            result = validate_result(folder, n)
            samples = result["hawkes"]["prediction_samples"]
            if samples < previous_samples:
                raise ValueError("prediction samples decreased when the nested seed prefix grew")
            previous_samples = samples
            qvi = result["strategies"]["hjb_qvi"]
            fixed = result["strategies"]["fixed_spread"]
            qlow, qhigh = ci95(qvi["mean_pnl"], qvi["std_pnl"], n)
            flow, fhigh = ci95(fixed["mean_pnl"], fixed["std_pnl"], n)
            rms_drop = 100.0 * (fixed["mean_rms_inventory"] - qvi["mean_rms_inventory"]) / fixed["mean_rms_inventory"] if fixed["mean_rms_inventory"] else 0.0
            rows.append({
                "runs": n,
                "hjb_qvi_mean_pnl": qvi["mean_pnl"],
                "hjb_qvi_std_pnl": qvi["std_pnl"],
                "hjb_qvi_pnl_ci95_low": qlow,
                "hjb_qvi_pnl_ci95_high": qhigh,
                "fixed_spread_mean_pnl": fixed["mean_pnl"],
                "fixed_spread_std_pnl": fixed["std_pnl"],
                "fixed_spread_pnl_ci95_low": flow,
                "fixed_spread_pnl_ci95_high": fhigh,
                "hjb_qvi_mean_rms_inventory": qvi["mean_rms_inventory"],
                "fixed_spread_mean_rms_inventory": fixed["mean_rms_inventory"],
                "hjb_qvi_mean_abs_inventory": qvi["mean_abs_inventory"],
                "fixed_spread_mean_abs_inventory": fixed["mean_abs_inventory"],
                "inventory_rms_reduction_pct": rms_drop,
                "hawkes_direction_auc": result["hawkes"]["next_sell_direction_auc"],
                "prediction_samples": samples,
                "hjb_fdm_solve_ms": result["hjb_fdm_solve_ms"],
                "quote_median_ns": result["quote_engine"]["median_ns"],
                "quote_p99_ns": result["quote_engine"]["p99_ns"],
            })
            print(f"[{i}/{len(counts)}] n={n}; HJB PnL={qvi['mean_pnl']:.3f}; "
                  f"RMS inventory={qvi['mean_rms_inventory']:.3f}; AUC={rows[-1]['hawkes_direction_auc']:.5f} PASS",
                  flush=True)

        with (out / "convergence_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "experiment": "nested-seed Monte Carlo sample-size convergence",
            "run_counts": counts,
            "total_path_evaluations": sum(counts),
            "seed_start": 1000,
            "nested_seed_prefixes": True,
            "hjb_horizon_seconds": 60.0,
            "hawkes_horizon_seconds": 60.0,
            "confidence_interval": "approximate two-sided 95% Student-t interval for mean PnL",
            "rows": rows,
        }
        (out / "convergence_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot_series(out, rows, "pnl_convergence", "PnL estimate by sample size",
                    "Mean PnL (95% confidence interval)", "hjb_qvi_mean_pnl",
                    low_key="hjb_qvi_pnl_ci95_low", high_key="hjb_qvi_pnl_ci95_high")
        plot_series(out, rows, "inventory_convergence", "Inventory risk by sample size",
                    "Mean RMS inventory", "hjb_qvi_mean_rms_inventory",
                    baseline_key="fixed_spread_mean_rms_inventory")
        plot_series(out, rows, "auc_convergence", "Hawkes direction score by sample size",
                    "ROC AUC", "hawkes_direction_auc", reference_line=0.5)
        print(f"Completed counts {counts}; total path evaluations={sum(counts)}.")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"convergence experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
