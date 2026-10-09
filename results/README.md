# Research results

`research_run.json` is the numerical record for the checked-in reference run. `python_summary.json` holds derived metrics used by the README. `policy_t0.csv` contains the time-zero policy; `policy_skew.svg` plots the finite bid and ask distances, and `sample_hawkes_events.csv` contains the sample event path.

The SVG figures in this folder are readable vector images based on the reference data. `terminal_snapshot.svg` is a large-text, terminal-style summary created from the verified reference run values. It is a formatted view, not a screenshot from an actual terminal. Each new call to `scripts/run_research.py` writes a run-specific `terminal_snapshot.svg`.

Use `scripts/run_sensitivity.py` to create a CSV table, a full JSON report, three heatmaps and a PnL/risk trade-off plot. The default sweep checks 20 settings across 500 simulated paths.

Use `scripts/run_convergence.py` to compare 25, 50, 100 and 200 path estimates with nested seed prefixes. It writes CSV/JSON summaries and PnL, inventory-risk and AUC plots.

All default results are synthetic. Quote timing varies by machine, and the metrics should not be treated as live-market performance.
