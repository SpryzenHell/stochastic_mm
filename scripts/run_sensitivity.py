#!/usr/bin/env python3
"""Run a reproducible gamma/liquidation-cost sensitivity experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys

import numpy as np


def parse_values(raw: str, name: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError(f"{name} values must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} values must be unique")
    return values


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def find_executable(build_dir: pathlib.Path) -> pathlib.Path:
    names = [
        build_dir / "research" / "smm_research",
        build_dir / "research" / "Release" / "smm_research.exe",
        build_dir / "research" / "Debug" / "smm_research.exe",
        build_dir / "research" / "smm_research.exe",
    ]
    for item in names:
        if item.is_file():
            return item
    raise FileNotFoundError("smm_research was not found. Build the project first or select the right --build-dir.")


def t_critical_95(n: int) -> float:
    # Two-sided Student-t critical values for small sample sizes; normal limit thereafter.
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
        20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
        26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048,
    }
    if n in table:
        return table[n]
    # Cornish-Fisher approximation to the two-sided 95% Student-t quantile.
    z = 1.959963984540054
    df = float(n - 1)
    z2 = z * z
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    )


def ci95(mean: float, sd: float, n: int) -> tuple[float, float]:
    if n < 2:
        return (mean, mean)
    half = t_critical_95(n) * sd / math.sqrt(n)
    return mean - half, mean + half


def load_metrics(path: pathlib.Path, expected_runs: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("runs") != expected_runs:
        raise ValueError(f"{path}: expected {expected_runs} runs, found {data.get('runs')}")
    for branch in ("hjb_qvi", "fixed_spread"):
        if branch not in data.get("strategies", {}):
            raise ValueError(f"{path}: missing strategies.{branch}")
    return data


def pareto_rows(rows: list[dict]) -> list[dict]:
    # PnL is better when larger; inventory risk is better when smaller.
    keep = []
    for row in rows:
        dominated = any(
            other["qvi_mean_pnl"] >= row["qvi_mean_pnl"]
            and other["qvi_mean_rms_inventory"] <= row["qvi_mean_rms_inventory"]
            and (
                other["qvi_mean_pnl"] > row["qvi_mean_pnl"]
                or other["qvi_mean_rms_inventory"] < row["qvi_mean_rms_inventory"]
            )
            for other in rows
        )
        if not dominated:
            keep.append(row)
    return sorted(keep, key=lambda row: (row["qvi_mean_rms_inventory"], -row["qvi_mean_pnl"]))


def save_heatmap(root: pathlib.Path, rows: list[dict], gammas: list[float],
                 costs: list[float], field: str, title: str, color_label: str,
                 filename: str, cmap_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.full((len(gammas), len(costs)), np.nan, dtype=float)
    for row in rows:
        gi = gammas.index(row["gamma"])
        ci = costs.index(row["liquidation_cost"])
        matrix[gi, ci] = row[field]

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap_name)
    ax.set_xticks(range(len(costs)), [f"{x:g}" for x in costs])
    ax.set_yticks(range(len(gammas)), [f"{x:g}" for x in gammas])
    ax.set_xlabel("Liquidation cost per unit")
    ax.set_ylabel("Risk aversion (gamma)")
    ax.set_title(title, loc="left", fontweight="bold")
    threshold = float(np.nanmean(matrix))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            color = "white" if abs(value - threshold) > abs(threshold) * 0.15 else "black"
            ax.text(x, y, f"{value:.3g}", ha="center", va="center", fontsize=10, color=color)
    bar = fig.colorbar(image, ax=ax, shrink=0.86)
    bar.set_label(color_label)
    for ext in ("png", "svg"):
        fig.savefig(root / f"{filename}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(root: pathlib.Path, rows: list[dict], gammas: list[float],
                  costs: list[float], frontier: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(gammas), max(gammas))
    markers = ["o", "s", "^", "D", "P", "X"]
    for index, cost in enumerate(costs):
        selected = [r for r in rows if r["liquidation_cost"] == cost]
        ax.scatter(
            [r["qvi_mean_pnl"] for r in selected],
            [r["qvi_mean_rms_inventory"] for r in selected],
            c=[r["gamma"] for r in selected],
            cmap=cmap, norm=norm, marker=markers[index % len(markers)],
            s=100, edgecolors="black", linewidths=0.5, alpha=0.9,
        )
    if frontier:
        ax.plot([r["qvi_mean_pnl"] for r in frontier],
                [r["qvi_mean_rms_inventory"] for r in frontier],
                linestyle="--", linewidth=1.8, marker="*", markersize=11,
                color="#c43c39", label="Non-dominated settings")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Risk aversion (gamma)")
    handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], color="white",
               markeredgecolor="black", linestyle="None", markersize=8,
               label=f"Cost {cost:g}")
        for i, cost in enumerate(costs)
    ]
    if frontier:
        handles.append(Line2D([0], [0], color="#c43c39", linestyle="--",
                              marker="*", label="Non-dominated settings"))
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.set_xlabel("Mean PnL (larger is better)")
    ax.set_ylabel("Mean RMS inventory (smaller is better)")
    ax.set_title("Risk and PnL trade-off across parameter settings", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    for ext in ("png", "svg"):
        fig.savefig(root / f"sensitivity_tradeoff.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a paired parameter sweep for HJB-QVI risk aversion and liquidation cost."
    )
    parser.add_argument("--runs", type=int, default=25, help="Monte Carlo paths per parameter setting (default: 25)")
    parser.add_argument("--gamma-values", default="0.01,0.02,0.05,0.10,0.20",
                        help="Comma-separated positive risk-aversion values")
    parser.add_argument("--liquidation-costs", default="0.001,0.005,0.010,0.020",
                        help="Comma-separated positive liquidation costs")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/sensitivity"),
                        help="Output directory")
    parser.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build"),
                        help="CMake build directory")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing CMake build")
    args = parser.parse_args()

    if args.runs < 2:
        parser.error("--runs must be at least 2 to estimate a confidence interval")
    try:
        gammas = parse_values(args.gamma_values, "--gamma-values")
        costs = parse_values(args.liquidation_costs, "--liquidation-costs")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if len(gammas) * len(costs) > 100:
        parser.error("the parameter grid cannot exceed 100 settings")

    root = pathlib.Path(__file__).resolve().parents[1]
    build_dir = (root / args.build_dir).resolve()
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_build:
            run_command(["cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
            if sys.platform.startswith("win"):
                run_command(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
            else:
                run_command(["cmake", "--build", str(build_dir), "--parallel"])

        if sys.platform.startswith("win"):
            run_command(["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"])
        else:
            run_command(["ctest", "--test-dir", str(build_dir), "--output-on-failure"])

        exe = find_executable(build_dir)
        total = len(gammas) * len(costs)
        rows: list[dict] = []
        for gamma in gammas:
            for cost in costs:
                tag = f"gamma_{gamma:.6g}_cost_{cost:.6g}".replace(".", "p")
                run_dir = output_dir / "runs" / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [str(exe), str(args.runs), str(run_dir), f"{gamma:.12g}", f"{cost:.12g}"]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                result = load_metrics(run_dir / "research_run.json", args.runs)
                qvi = result["strategies"]["hjb_qvi"]
                base = result["strategies"]["fixed_spread"]
                qlo, qhi = ci95(qvi["mean_pnl"], qvi["std_pnl"], args.runs)
                blo, bhi = ci95(base["mean_pnl"], base["std_pnl"], args.runs)
                row = {
                    "gamma": gamma,
                    "liquidation_cost": cost,
                    "runs": args.runs,
                    "qvi_mean_pnl": qvi["mean_pnl"],
                    "qvi_std_pnl": qvi["std_pnl"],
                    "qvi_pnl_ci95_low": qlo,
                    "qvi_pnl_ci95_high": qhi,
                    "qvi_mean_rms_inventory": qvi["mean_rms_inventory"],
                    "qvi_mean_abs_inventory": qvi["mean_abs_inventory"],
                    "qvi_p95_abs_inventory": qvi["p95_abs_inventory"],
                    "fixed_mean_pnl": base["mean_pnl"],
                    "fixed_std_pnl": base["std_pnl"],
                    "fixed_pnl_ci95_low": blo,
                    "fixed_pnl_ci95_high": bhi,
                    "fixed_mean_rms_inventory": base["mean_rms_inventory"],
                    "fixed_mean_abs_inventory": base["mean_abs_inventory"],
                    "fixed_p95_abs_inventory": base["p95_abs_inventory"],
                    "inventory_rms_reduction_pct": 100.0 * (
                        base["mean_rms_inventory"] - qvi["mean_rms_inventory"]
                    ) / base["mean_rms_inventory"] if base["mean_rms_inventory"] else 0.0,
                    "hawkes_direction_auc": result["hawkes"]["next_sell_direction_auc"],
                    "hjb_fdm_solve_ms": result["hjb_fdm_solve_ms"],
                    "quote_median_ns": result["quote_engine"]["median_ns"],
                    "quote_p99_ns": result["quote_engine"]["p99_ns"],
                }
                rows.append(row)
                if completed.stderr.strip():
                    print(completed.stderr.strip())
                print(
                    f"[{len(rows):02d}/{total:02d}] gamma={gamma:g} cost={cost:g} "
                    f"PnL={row['qvi_mean_pnl']:.3f} RMS-inventory={row['qvi_mean_rms_inventory']:.3f} OK",
                    flush=True,
                )

        fields = list(rows[0].keys())
        with (output_dir / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        frontier = pareto_rows(rows)
        best_risk = min(rows, key=lambda row: row["qvi_mean_rms_inventory"])
        best_pnl = max(rows, key=lambda row: row["qvi_mean_pnl"])
        report = {
            "experiment": "HJB-QVI gamma and liquidation-cost sensitivity",
            "paths_per_setting": args.runs,
            "setting_count": total,
            "total_paths": args.runs * total,
            "random_seeds": list(range(1000, 1000 + args.runs)),
            "same_seed_set_used_for_every_setting": True,
            "gamma_values": gammas,
            "liquidation_costs": costs,
            "confidence_interval": "approximate two-sided 95% Student-t interval for mean PnL",
            "best_inventory_risk_setting": best_risk,
            "best_mean_pnl_setting": best_pnl,
            "non_dominated_settings": frontier,
            "rows": rows,
        }
        (output_dir / "sensitivity_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_heatmap(output_dir, rows, gammas, costs, "qvi_mean_rms_inventory",
                     "HJB-QVI inventory risk", "Mean RMS inventory", "inventory_rms_heatmap", "viridis")
        save_heatmap(output_dir, rows, gammas, costs, "qvi_mean_pnl",
                     "HJB-QVI mean PnL", "Mean PnL", "mean_pnl_heatmap", "coolwarm")
        save_heatmap(output_dir, rows, gammas, costs, "inventory_rms_reduction_pct",
                     "Inventory risk reduction vs fixed spread", "Reduction (%)",
                     "inventory_reduction_heatmap", "YlGnBu")
        save_tradeoff(output_dir, rows, gammas, costs, frontier)
        print(f"\nCompleted {total} settings / {args.runs * total} paths.")
        print(f"CSV: {output_dir / 'sensitivity_summary.csv'}")
        print(f"JSON: {output_dir / 'sensitivity_summary.json'}")
        print(f"Figures: {output_dir}/*heatmap.* and sensitivity_tradeoff.*")
        return 0
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"sensitivity run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
