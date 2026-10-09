# Research results

`research_run.json` is the numerical record for the checked-in reference run. `python_summary.json` holds derived metrics used by the README. `policy_t0.csv` contains the time-zero policy, and `sample_hawkes_events.csv` contains the sample event path.

The SVG figures in this folder are readable vector images based on the reference data. `cli_output.svg` is a large-text, terminal-style summary created from recorded values. It is a formatted view, not a screenshot from an actual terminal. Each new call to `scripts/run_research.py` writes `terminal_snapshot.svg` using that run's JSON file.

Use `scripts/run_sensitivity.py` to create a CSV table, a full JSON report, three heatmaps and a PnL/risk trade-off plot. The default sweep checks 20 settings across 500 simulated paths.

All default results are synthetic. Quote timing varies by machine, and the metrics should not be treated as live-market performance.
