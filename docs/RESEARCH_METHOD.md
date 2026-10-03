# Mathematical formulation

## 1. State and control

The reduced state is inventory `q` on an integer grid `[-Q, Q]`. The market maker chooses non-negative quote distances `δ_b` and `δ_a` from the current mid-price.

The synthetic fill intensity on either side is

`λ(δ) = A exp(-k δ)`.

Inventory changes by `+1` after a bid fill and `-1` after an ask fill.

## 2. Continuation equation

For a value function `V(t,q)`, the implementation uses the reduced continuation operator

`V_t + H_b(V) + H_a(V) - 1/2 γ σ² q²`.

For the bid side,

`H_b(V) = max_δ λ_b(δ) [δ + V(t,q+1) - V(t,q)]`,

and the ask side is analogous with `q-1`.

For `λ(δ)=A exp(-kδ)`, the unconstrained first-order optimizer is

`δ* = 1/k - ΔV`.

The numerical implementation clamps this to `[δ_min, δ_max]`.

## 3. QVI / intervention obstacle

Inventory can be forcibly flattened through the intervention operator

`M[V](q) = V(t,0) - c_liq |q|`.

The discretized QVI takes the pointwise envelope of continuation and intervention:

`V(t,q) = max(V_continuation(t,q), M[V](q))`.

At the hard inventory boundary `|q|=Q`, intervention is also enabled so the controller does not rely on a one-sided finite-difference stencil outside the grid.

## 4. Time discretization

The solver uses backward explicit stepping with

`Δt = horizon / ceil(horizon / dt)`.

The inventory dimension is kept deliberately small so the policy can be precomputed once and then served through a table lookup in the quote path.

## 5. Numerical limitation

This is a reduced inventory-space stochastic-control approximation. It omits a separate continuous price dimension and does not claim the full original stochastic-control problem has been solved exactly. The repository exposes the approximation explicitly so that assumptions are visible in an interview or research discussion.

# Hawkes microstructure model

## Event types

The experiment uses a two-dimensional event stream:

- `type=0`: sell market order / bid hit
- `type=1`: buy market order / ask lift

Each event changes the synthetic mid-price by one tick in the corresponding direction.

## Intensity model

For side `i`,

`λ_i(t) = μ_i + Σ_j α_ij Σ_{τ_k<t, type_k=j} exp(-β(t-τ_k))`.

The implementation uses separate same-side and cross-side coefficients. The default regime has more sell-side baseline flow than buy-side flow and stronger self-excitation than cross-excitation.

## Stability

The integrated kernel matrix is `K = α / β`. The branching ratio is the spectral radius `ρ(K)`. The experiment is configured with `ρ(K) = 0.6875`, which is below one.

When Eigen3 is available, the 2×2 spectral radius is evaluated through `Eigen::EigenSolver`; otherwise the mathematically equivalent closed-form 2×2 expression is used.

## Prediction experiment

After each event, the simulator computes the post-event sell-vs-buy intensity share. The next event's type is the binary label. ROC AUC measures how much information the Hawkes state carries about the direction of the next event.

Because the event type also drives the synthetic mid-price tick, this is a compact proxy for predicting the sign of the next microstructure jump.

The reported AUC is therefore a **model-generated result**, not an empirical claim about a real market.

# Benchmark methodology

## HJB FDM solve

The FDM benchmark measures wall-clock time for a complete backward solution across the full time × inventory grid. It is an offline computation performed once per policy configuration.

The implementation compiles the inventory loop with:

```text
-O3 -march=native -fopenmp-simd
```

The source contains an OpenMP SIMD annotation on the inventory sweep.

## Quote hot path

After the policy has been solved, `QuoteEngine::quote(mid, inventory, time)` performs clamping, a nearest time-index lookup, a nearest inventory lookup, and arithmetic to recover bid/ask prices.

The benchmark executes the quote function 1,000,000 times in grouped batches and reports the distribution of per-call group means. This reduces timing noise relative to taking a high-resolution clock sample around every individual call.

## Strategy comparison

Both controllers consume the **same Hawkes event path** for each Monte Carlo seed:

- HJB-QVI: dynamic policy from the precomputed value function.
- Fixed spread: symmetric `δ = 1/k` controller.

For every path we record:

- realized PnL after final inventory liquidation
- mean absolute inventory
- RMS inventory
- maximum absolute inventory
- adverse-selection markout over the next 10 events

The report then aggregates those values over the configured Monte Carlo paths.

## Interpreting the numbers

Latency numbers depend strongly on compiler flags, CPU microarchitecture, timer resolution, and operating-system noise. Risk/return numbers depend on the Hawkes parameter regime, inventory cap, liquidation cost and synthetic fill rule. Re-running under another environment can therefore change both classes of metrics.
