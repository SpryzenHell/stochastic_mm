# Experiments and data analysis

This page describes the repeatable experiments supported by the C++ research runner.

## 1. Reproduce the reference run

From the repository root:

```bash
python scripts/run_research.py --runs 100 --output results/run_100
```

The runner configures and builds the C++ targets, runs CTest, simulates event paths, saves numerical records, and generates report figures. The new output is kept separate from the checked-in reference results.

## 2. Compare parameter settings

Run a paired sweep of risk aversion and liquidation cost:

```bash
python scripts/run_sensitivity.py --runs 25 --gamma-values 0.01,0.02,0.05,0.10,0.20 --liquidation-costs 0.001,0.005,0.010,0.020 --output results/sensitivity
```

This tests 20 parameter settings and simulates 500 paths. Every setting uses the same seed list, so the input paths are paired across the sweep.

The results directory contains:

- `sensitivity_summary.csv`: one row per setting, with PnL, inventory risk, direction AUC and timing values.
- `sensitivity_summary.json`: full results, confidence intervals, best-risk/best-PnL settings and non-dominated settings.
- `inventory_rms_heatmap.png/svg`: mean RMS inventory.
- `mean_pnl_heatmap.png/svg`: mean PnL.
- `inventory_reduction_heatmap.png/svg`: inventory risk change relative to the fixed-spread baseline.
- `sensitivity_tradeoff.png/svg`: PnL against RMS inventory.

The 95% intervals are approximate two-sided Student-t intervals for mean PnL. Small run counts are useful for checking the pipeline but are not enough to make strong statistical claims.

## 3. Sample-size stability

Run a nested-seed sample-size comparison:

```bash
python scripts/run_convergence.py --run-counts 25,50,100,200 --output results/convergence
```

The experiment compares 25, 50, 100 and 200 path estimates, reusing each smaller seed prefix within larger runs. It writes CSV and JSON summaries, PnL confidence intervals and separate PnL, inventory-risk and direction-AUC plots. It performs 375 path evaluations; because the seed prefixes overlap, they are not independent samples.

## 4. Hawkes order-flow checks

The `sample_hawkes_events.csv` file stores event time and event type. Type 0 is a sell market order / bid hit. Type 1 is a buy market order / ask lift. Regression tests check fixed-seed repeatability, event ordering, valid event types, finite stationary intensity for stable settings and explicit handling of supercritical settings.

The event figure is a short view of one path; the CSV is the full event record for that path.

## 5. Quote timing

The C++ runner reports the time spent solving the finite-difference grid and a hot-cache quote lookup benchmark. Quote timing is measured inside the process. It excludes market-data delivery, network time, exchange queues and order-management work.

Timing depends on the CPU, compiler, operating system and current system load. Compare timing runs on the same machine and build settings.

## 6. Read results carefully

The model uses synthetic event paths and does not download market data. It is not a profitability claim. The HJB implementation is a reduced inventory-space approximation, and each Hawkes event changes the synthetic mid price by one tick. A lower inventory measure should be considered alongside PnL and other metrics.
