#pragma once
#include "smm/hawkes.hpp"
#include "smm/hjb_qvi.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace smm::research {

struct StrategyRun {
    double pnl = 0.0;
    double mean_abs_inventory = 0.0;
    double rms_inventory = 0.0;
    int max_abs_inventory = 0;
    std::size_t fills = 0;
    std::size_t interventions = 0;
    double adverse_selection_bps = 0.0;
};

struct MonteCarloSummary {
    std::string strategy;
    std::size_t runs = 0;
    double mean_pnl = 0.0;
    double std_pnl = 0.0;
    double mean_rms_inventory = 0.0;
    double mean_abs_inventory = 0.0;
    double mean_adverse_selection_bps = 0.0;
    double p95_abs_inventory = 0.0;
};

std::vector<HawkesEvent> simulate_path(const BivariateHawkes& hawkes, std::uint64_t seed);
StrategyRun run_strategy(const std::vector<HawkesEvent>& events,
                         const QuoteEngine* hq,
                         double fixed_delta,
                         double initial_mid,
                         double tick_size,
                         double arrival_k,
                         double liquidation_cost);
MonteCarloSummary summarize(const std::string& name, const std::vector<StrategyRun>& runs);

} // namespace smm::research
