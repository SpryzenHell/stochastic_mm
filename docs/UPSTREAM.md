# Upstream provenance

The repository's original project configuration maps three upstream sources into the Stochastic Market Maker project:

1. https://github.com/thibault-charbonnier/market-making-engine.git
2. https://github.com/fedecaccia/avellaneda-stoikov.git
3. https://github.com/sohaibelkarmi/High-Frequency-Trading-Simulator.git

The source configuration supplied for this project specifies the output directory `./projects/stochastic_mm`, target folders such as `src`, `include`, `python`, `notebooks`, and `scripts`, plus the requested historical interval from October 1, 2025 through January 31, 2026. 

The current repository retains the earlier merged tree under `smmSrc/`, `includes/`, and `smmPython/`. The `research/` tree added in this branch is intentionally isolated so that the benchmarked mathematical experiment is understandable without relying on the merged code's naming/rewrite layer.

## Responsibility boundary

The clean research layer implements the integrated experiment directly. Upstream repositories are referenced as provenance for the earlier components, not as evidence that their authors implemented this integrated HJB-QVI + Hawkes + SIMD FDM system.

## License boundary

Consult each upstream repository for its original license terms. The repository-level `LICENSE` remains authoritative for this repository; this document does not replace it.
