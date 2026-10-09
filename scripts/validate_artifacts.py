#!/usr/bin/env python3
"""Validate checked-in numerical records and README visuals."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def walk_numbers(value, where="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            walk_numbers(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_numbers(item, f"{where}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(float(value)), f"non-finite number at {where}")


def read_json(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    walk_numbers(data, str(path))
    return data


def validate_svg(path: pathlib.Path, terminal: bool = False) -> None:
    root = ET.parse(path).getroot()
    require(root.tag.endswith("svg"), f"{path}: root is not SVG")
    width = float(re.sub(r"[^0-9.]", "", root.attrib.get("width", "0")))
    height = float(re.sub(r"[^0-9.]", "", root.attrib.get("height", "0")))
    require(width > 0 and height > 0, f"{path}: missing positive dimensions")
    texts = [node for node in root.iter() if node.tag.endswith("text")]
    require(bool(texts), f"{path}: no text nodes found")
    for node in texts:
        require("x" in node.attrib and "y" in node.attrib, f"{path}: text node lacks coordinates")
        x, y = float(node.attrib["x"]), float(node.attrib["y"])
        require(0 <= x < width and 0 <= y <= height, f"{path}: text starts outside the canvas")
    if not terminal:
        return
    font_sizes = []
    ys = []
    for node in texts:
        size = float(node.attrib.get("font-size", "0").replace("px", ""))
        require(size >= 16, f"{path}: terminal text is too small ({size:g}px)")
        font_sizes.append(size)
        ys.append(float(node.attrib["y"]))
    ordered = sorted(set(ys))
    for a, b in zip(ordered, ordered[1:]):
        require(b - a >= 1.05 * max(font_sizes), f"{path}: text baselines may overlap")
    for node in texts:
        x = float(node.attrib["x"])
        size = float(node.attrib.get("font-size", "0").replace("px", ""))
        estimated_width = len("".join(node.itertext())) * size * 0.62
        require(x + estimated_width <= width - 12, f"{path}: text may run off the right edge")


def check_png(path: pathlib.Path) -> None:
    data = path.read_bytes()
    require(data[:8] == bytes((137, 80, 78, 71, 13, 10, 26, 10)), f"{path}: invalid PNG signature")
    require(len(data) >= 24, f"{path}: truncated PNG")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    require(width > 0 and height > 0, f"{path}: invalid PNG dimensions")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repository reference and generated experiment files.")
    parser.add_argument("--run-dir", type=pathlib.Path, help="Also validate a fresh research-run output directory.")
    parser.add_argument("--sensitivity-dir", type=pathlib.Path, help="Also validate sensitivity sweep results.")
    parser.add_argument("--convergence-dir", type=pathlib.Path, help="Also validate sample-size convergence results.")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    require("main.png" not in lowered, "README still refers to the removed overview image")
    require(not (root / "main.png").exists(), "the incomplete overview image should be removed")

    images = re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", readme, re.IGNORECASE)
    images += re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    missing = []
    for ref in images:
        ref = ref.strip().split(" ")[0]
        if ref.startswith(("http://", "https://", "data:")) or ref.startswith("#"):
            continue
        path = unquote(ref.split("#", 1)[0].split("?", 1)[0])
        if path and not (root / path).exists():
            missing.append(path)
    require(not missing, "README image links are broken: " + ", ".join(missing))

    licenses = [p for p in root.rglob("*") if p.is_file() and re.fullmatch(
        r"(LICENSE|LICENSE\.[A-Za-z0-9._-]+|COPYING|COPYING\.[A-Za-z0-9._-]+)", p.name, re.IGNORECASE
    )]
    require(len(licenses) == 1, f"expected one repository license file, found {len(licenses)}")

    reference = read_json(root / "results/research_run.json")
    summary = read_json(root / "results/python_summary.json")
    require(reference.get("runs", 0) >= 1, "reference result has no paths")
    require(abs(reference["hjb"]["horizon"] - reference["hawkes"]["horizon"]) < 1e-12,
            "reference HJB and Hawkes horizons do not match")
    require(reference["hjb"]["time_steps"] == math.ceil(reference["hjb"]["horizon"] / reference["hjb"]["dt"]) + 1,
            "reference HJB time-step count is inconsistent")
    for name in ("hjb_qvi", "fixed_spread"):
        require(name in reference.get("strategies", {}), f"missing strategy {name}")
        item = reference["strategies"][name]
        for field in ("mean_pnl", "std_pnl", "mean_rms_inventory", "mean_abs_inventory", "p95_abs_inventory"):
            require(field in item, f"missing {name}.{field}")
    qvi = reference["strategies"]["hjb_qvi"]
    base = reference["strategies"]["fixed_spread"]
    require(base["mean_rms_inventory"] > 0.0, "fixed-spread RMS inventory must be positive")
    expected_reduction = 100.0 * (base["mean_rms_inventory"] - qvi["mean_rms_inventory"]) / base["mean_rms_inventory"]
    require(abs(expected_reduction - summary["inventory_rms_reduction_pct"]) < 1e-6,
            "Python summary does not match the checked-in reference JSON")
    require(abs(reference["hawkes"]["next_sell_direction_auc"] - summary["hawkes_direction_auc"]) < 1e-9,
            "direction AUC does not match the reference summary")

    with (root / "results/policy_t0.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "results/sensitivity_summary.csv").open(newline="", encoding="utf-8") as handle:
        sensitivity_rows = list(csv.DictReader(handle))
    require(len(sensitivity_rows) == 20, "reference sensitivity table must have 20 settings")
    sensitivity_pairs = {(float(row["gamma"]), float(row["liquidation_cost"])) for row in sensitivity_rows}
    require(len(sensitivity_pairs) == 20, "reference sensitivity table has duplicate settings")
    require(all(int(row["runs"]) == 25 for row in sensitivity_rows),
            "reference sensitivity table must use 25 paths per setting")
    with (root / "results/convergence_summary.csv").open(newline="", encoding="utf-8") as handle:
        convergence_rows = list(csv.DictReader(handle))
    convergence_counts = [int(row["runs"]) for row in convergence_rows]
    require(convergence_counts == [25, 50, 100, 200], "reference convergence sizes are inconsistent")
    require(sum(convergence_counts) == 375, "reference convergence path count is inconsistent")
    require(len(rows) == 2 * reference["hjb"]["q_max"] + 1, "policy CSV row count does not match q_max")
    inventories = [int(row["inventory"]) for row in rows]
    require(inventories == list(range(-reference["hjb"]["q_max"], reference["hjb"]["q_max"] + 1)),
            "policy inventory grid must be contiguous and ordered")
    for row in rows:
        for key in ("bid_delta", "ask_delta"):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0.0, f"invalid {key} in policy CSV")

    for name in ("policy_skew.svg", "inventory_comparison.svg", "latency_benchmark.svg",
                 "hawkes_events.svg", "research_summary.svg", "terminal_snapshot.svg",
                 "sensitivity_inventory_heatmap.svg", "sensitivity_pnl_heatmap.svg",
                 "sensitivity_reduction_heatmap.svg", "sensitivity_tradeoff.svg",
                 "pnl_convergence.svg", "inventory_convergence.svg", "auc_convergence.svg"):
        validate_svg(root / "results" / name, terminal=(name == "terminal_snapshot.svg"))

    if args.run_dir:
        run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
        run_record = read_json(run_dir / "research_run.json")
        require(run_record.get("runs", 0) >= 1, "generated run has no Monte Carlo paths")
        for name in ("policy_skew.png", "inventory_comparison.png",
                     "latency_benchmark.png", "hawkes_events.png"):
            check_png(run_dir / name)
        validate_svg(run_dir / "terminal_snapshot.svg", terminal=True)
        with (run_dir / "policy_t0.csv").open(newline="", encoding="utf-8") as handle:
            fresh_policy = list(csv.DictReader(handle))
        require(len(fresh_policy) == 2 * run_record["hjb"]["q_max"] + 1,
                "generated policy CSV has the wrong number of rows")
        fresh_inventory = [int(row["inventory"]) for row in fresh_policy]
        require(fresh_inventory == list(range(-run_record["hjb"]["q_max"], run_record["hjb"]["q_max"] + 1)),
                "generated policy inventory grid must be contiguous and ordered")
        for row in fresh_policy:
            for key in ("bid_delta", "ask_delta"):
                value = float(row[key])
                require(math.isfinite(value) and value >= 0.0,
                        f"invalid generated {key} in policy CSV")
        with (run_dir / "sample_hawkes_events.csv").open(newline="", encoding="utf-8") as handle:
            events = list(csv.DictReader(handle))
        last_time = -math.inf
        for event in events:
            event_time = float(event["time"])
            event_type = int(event["type"])
            require(math.isfinite(event_time) and event_time >= last_time,
                    "generated Hawkes event times are invalid or unsorted")
            require(event_type in (0, 1), "generated Hawkes event type is invalid")
            last_time = event_time

    if args.sensitivity_dir:
        sweep_dir = args.sensitivity_dir if args.sensitivity_dir.is_absolute() else root / args.sensitivity_dir
        report = read_json(sweep_dir / "sensitivity_summary.json")
        with (sweep_dir / "sensitivity_summary.csv").open(newline="", encoding="utf-8") as handle:
            sweep_rows = list(csv.DictReader(handle))
        require(len(sweep_rows) == report["setting_count"], "sensitivity CSV and JSON row counts differ")
        require(report["total_paths"] == report["setting_count"] * report["paths_per_setting"],
                "sensitivity total path count is inconsistent")
        combos = {(float(row["gamma"]), float(row["liquidation_cost"])) for row in sweep_rows}
        require(len(combos) == len(sweep_rows), "sensitivity grid contains duplicate settings")
        require(all(int(row["runs"]) == report["paths_per_setting"] for row in sweep_rows),
                "sensitivity rows have inconsistent run counts")
        for name in ("inventory_rms_heatmap", "mean_pnl_heatmap",
                     "inventory_reduction_heatmap", "sensitivity_tradeoff"):
            check_png(sweep_dir / f"{name}.png")
            ET.parse(sweep_dir / f"{name}.svg")
        for child in (sweep_dir / "runs").iterdir():
            if child.is_dir():
                record = read_json(child / "research_run.json")
                require(abs(record["hjb"]["horizon"] - record["hawkes"]["horizon"]) < 1e-12,
                        f"{child}: HJB/Hawkes horizons do not match")
                with (child / "policy_t0.csv").open(newline="", encoding="utf-8") as stream:
                    policy = list(csv.DictReader(stream))
                for row in policy:
                    for key in ("bid_delta", "ask_delta"):
                        require(math.isfinite(float(row[key])), f"{child}: non-finite {key}")

    if args.convergence_dir:
        conv_dir = args.convergence_dir if args.convergence_dir.is_absolute() else root / args.convergence_dir
        conv_report = read_json(conv_dir / "convergence_summary.json")
        with (conv_dir / "convergence_summary.csv").open(newline="", encoding="utf-8") as stream:
            conv_rows = list(csv.DictReader(stream))
        counts = [int(row["runs"]) for row in conv_rows]
        require(counts == conv_report["run_counts"], "convergence CSV counts do not match JSON")
        require(counts == sorted(set(counts)), "convergence sizes are not strictly increasing")
        require(conv_report["total_path_evaluations"] == sum(counts), "convergence path count is inconsistent")
        require(conv_report.get("nested_seed_prefixes") is True, "convergence report must identify nested seed prefixes")
        for name in ("pnl_convergence", "inventory_convergence", "auc_convergence"):
            check_png(conv_dir / f"{name}.png")
            ET.parse(conv_dir / f"{name}.svg")
        for count in counts:
            folder = conv_dir / "runs" / f"n_{count}"
            record = read_json(folder / "research_run.json")
            require(record["runs"] == count, f"{folder}: wrong path count")
            require(abs(record["hjb"]["horizon"] - record["hawkes"]["horizon"]) < 1e-12,
                    f"{folder}: horizons do not match")
            with (folder / "policy_t0.csv").open(newline="", encoding="utf-8") as stream:
                policy = list(csv.DictReader(stream))
            for row in policy:
                for key in ("bid_delta", "ask_delta"):
                    require(math.isfinite(float(row[key])), f"{folder}: non-finite {key}")

    details = f"{len(images)} README image links, {len(rows)} reference policy rows, {len(licenses)} license"
    if args.run_dir:
        details += ", generated run validated"
    if args.sensitivity_dir:
        details += f", {len(sweep_rows)} sensitivity settings and {report['total_paths']} paths validated"
    if args.convergence_dir:
        details += f", sample sizes {counts} and {conv_report['total_path_evaluations']} nested-seed path evaluations validated"
    print(f"artifact_validation: PASS ({details})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"artifact_validation: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
