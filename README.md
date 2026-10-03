# Stochastic Market Maker

**HJB-QVI inventory control · Hawkes microstructure model · C++ SIMD FDM quote engine**

This repository is a quantitative research sandbox that connects a stochastic-control policy, clustered limit-order-flow model, deterministic execution simulator, and low-latency C++ quote path into one reproducible experiment.

> **Research scope:** the default experiment is synthetic. The code reports the actual metrics produced by the configured simulation; it does not present synthetic results as live-market performance.

## What the project now demonstrates

### 1. Inventory control with a reduced HJB-QVI

The market maker solves a finite-horizon inventory-control problem on an integer inventory grid. The continuation equation uses CARA-style inventory risk, quote-dependent order arrivals, and a liquidation/intervention obstacle.

For each inventory state `q` and side, order-arrival intensity is

`λ(δ) = A exp(-k δ)`

and the numerical quote optimizer is the constrained solution

`δ* = clamp(1/k - ΔV, δ_min, δ_max)`.

The QVI step compares the continuation value with an intervention value

`M[V](q) = V(0) - c_liq |q|`.

The result is a time × inventory policy table containing bid distance, ask distance, value and intervention flags.

This is intentionally a **reduced-state HJB-QVI discretization**, not a claim that the full price-space stochastic-control PDE has been solved in closed form.

### 2. Hawkes model for clustered order-flow / price-jump direction

A bivariate exponential Hawkes process produces sell-side and buy-side market-order events. Same-side excitation and cross-side excitation are modeled separately, so the simulator can represent asymmetric bursts instead of independent Poisson flow.

The implementation reports the spectral branching ratio, stationary intensities, and a next-event direction AUC. Because each event also moves the synthetic mid-price by one tick, the direction predictor is directly connected to the simulated next mid-price jump.

### 3. C++ SIMD FDM + constant-time policy lookup

The HJB solver uses a compact inventory grid and an OpenMP-SIMD annotated hot loop. The solved policy is then served through a small C++ `QuoteEngine` whose runtime benchmark is measured independently from the offline PDE solve.

This separation is important:

- **FDM solve latency** = cost to compute the complete policy table.
- **Quote latency** = cost to turn `(mid, inventory, time)` into bid/ask distances after the table already exists.

## Architecture

```text
                     ┌─────────────────────────────┐
                     │ Bivariate Hawkes Order Flow │
                     │ buy / sell event arrivals  │
                     └─────────────┬───────────────┘
                                   │
                                   ▼
┌─────────────────┐       ┌──────────────────────────┐
│ HJB-QVI Solver  │──────►│ HJB Policy / QuoteEngine │
│ inventory grid  │       │ bid δ, ask δ, intervene  │
└─────────────────┘       └────────────┬─────────────┘
                                      │
                                      ▼
                           ┌─────────────────────────┐
                           │ Event-driven Simulator  │
                           │ fills · inventory · PnL │
                           │ markouts / risk         │
                           └────────────┬────────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │ Monte Carlo Report   │
                             │ risk / return / AUC  │
                             │ latency / artefacts  │
                             └──────────────────────┘
```

The repository also contains the earlier merged upstream-derived implementation under `smmSrc/`, `includes/`, and `smmPython/`. The `research/` build is the supported path for the portfolio experiment because it gives the project one clean mathematical and benchmarking entry point.

## Quick start

### C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/research/smm_research 100 results
```

### Python report

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python3 scripts/run_research.py --runs 100
```

The Python wrapper rebuilds the C++ target, runs the unit tests, executes the Monte Carlo experiment, and generates the two report plots.

## Default experiment

The canonical configuration is `configs/research_default.json`:

| Parameter | Value |
|---|---:|
| HJB risk aversion `γ` | 0.05 |
| Mid-price volatility `σ` | 0.20 |
| Arrival scale `A` | 80 |
| Arrival decay `k` | 40 |
| Liquidation cost | 0.005 |
| Inventory grid | -25 … +25 |
| Horizon | 10 s |
| FDM step | 0.02 s |
| Hawkes `μ_sell / μ_buy` | 70 / 50 s⁻¹ |
| Hawkes same-side excitation | 5.2 |
| Hawkes cross-side excitation | 0.3 |
| Hawkes decay `β` | 8.0 |
| Monte Carlo paths | 100 |

## Measured result from the checked-in run

The checked-in result (`results/research_run.json`) was generated from the configuration above on the current CPU build.

| Metric | HJB-QVI | Fixed spread |
|---|---:|---:|
| Mean RMS inventory | **18.6295** | 20.4411 |
| Mean absolute inventory | **16.1995** | 19.4099 |
| Mean PnL | -581.69 | -549.13 |
| PnL standard deviation | **85.91** | 105.21 |
| Mean adverse-selection markout | 1.692 bps | 1.649 bps |
| p95 absolute inventory | 25 | 25 |

The HJB-QVI controller therefore reduced mean RMS inventory by **8.86%** in this particular asymmetric, clustered-flow simulation. It also produced lower PnL in this configuration. That trade-off is intentionally visible in the report instead of being hidden behind a single performance number.

### Microstructure prediction

The Hawkes next-event sell-direction predictor achieved **0.5563 AUC** over **2,298,582** next-event examples generated by the configured Hawkes process. This is a simulation result, not an empirical claim about a real exchange feed.

### Low-latency path

The checked-in run measured:

- HJB FDM policy solve: **~0.66 ms**
- Quote lookup median: **~9.9 ns** per call
- Quote lookup p99: **~21.5 ns** per call

The compiler's GCC vectorization diagnostics also reported the HJB inventory loop as vectorized using 32-byte and 16-byte vectors under the benchmark build.

The numbers above are machine-dependent and should be re-run before being presented as a hardware-independent benchmark.

## Reproducibility and outputs

`results/` contains:

- `research_run.json` — complete experiment metrics
- `policy_t0.csv` — inventory-dependent quote distances at `t = 0`
- `sample_hawkes_events.csv` — first reproducible event path
- `python_summary.json` — compact derived metrics
- `policy_skew.png` — quote-skew visualization
- `hawkes_events.png` — Hawkes event timeline

The random seeds are deterministic in the C++ runner (`1000 + run_id`) so the Monte Carlo comparison uses the same event paths for both controllers.

## Tests

The C++ test suite covers:

- HJB solution dimensions and finite values
- quote-distance lower bounds
- Hawkes branching-ratio stability
- event-time monotonicity
- non-empty Hawkes simulation

The CI workflow additionally installs Eigen and compiles the Eigen-backed spectral-radius path.

## Upstream provenance

The repository was originally assembled from these three upstream projects, as recorded by the supplied project configuration:

1. `thibault-charbonnier/market-making-engine`
2. `fedecaccia/avellaneda-stoikov`
3. `sohaibelkarmi/High-Frequency-Trading-Simulator`

The clean research layer in this branch is an integration/reference implementation built on top of that repository's existing merged tree. It does not claim the upstream authors wrote this combined experiment.

See [`docs/UPSTREAM.md`](docs/UPSTREAM.md) for the source map and separation of responsibilities.

## Repository hygiene

The supplied historical merge utility contains functionality that rewrites Git author/committer timestamps to a requested historical interval. This research branch does **not** use that behavior. New commits should use their real creation dates.

## Project layout

```text
include/smm/          Public research interfaces
src/                  HJB-QVI + Hawkes implementations
research/             Monte Carlo simulator and CMake target
tests/                C++ unit tests
scripts/              Reproducible Python runner / plotting
configs/              Canonical experiment configuration
docs/                 Mathematics, benchmark methodology, provenance
results/              Actual generated experiment outputs
smmSrc/               Earlier merged/upstream-derived implementation
smmPython/            Earlier merged Python implementation
```

## Reference notes

This project is intended for research and interview discussion. The important engineering distinction is between the *mathematical model*, the *simulation environment*, and the *latency benchmark*. None of those measurements should be interpreted as live trading performance without an exchange-grade market-data and execution environment.
