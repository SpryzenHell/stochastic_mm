#include "sim.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace smm::research {

std::vector<HawkesEvent> simulate_path(const BivariateHawkes& hawkes, std::uint64_t seed) {
    return hawkes.simulate(seed);
}

StrategyRun run_strategy(const std::vector<HawkesEvent>& events,
                         const QuoteEngine* hq,
                         double fixed_delta,
                         double initial_mid,
                         double tick_size,
                         double arrival_k,
                         double liquidation_cost) {
    StrategyRun r;
    if (events.empty()) return r;

    double mid = initial_mid;
    double cash = 0.0;
    int inventory = 0;
    constexpr int kInventoryLimit = 25;
    double abs_sum = 0.0, sq_sum = 0.0;
    std::vector<double> inv_abs;
    inv_abs.reserve(events.size());

    std::vector<int> jump_prefix(events.size() + 1, 0);
    for (std::size_t i = 0; i < events.size(); ++i) {
        jump_prefix[i + 1] = jump_prefix[i] + (events[i].type == 1 ? 1 : -1);
    }

    double adverse_sum = 0.0;
    std::size_t adverse_n = 0;

    for (std::size_t i = 0; i < events.size(); ++i) {
        const double t = events[i].time;
        bool intervene = false;
        double bid_delta = fixed_delta, ask_delta = fixed_delta;
        if (hq) {
            const Quote quote = hq->quote(mid, inventory, t);
            intervene = quote.intervention;
            bid_delta = quote.bid_delta;
            ask_delta = quote.ask_delta;
        }

        if (intervene && inventory != 0) {
            cash += static_cast<double>(inventory) * mid - liquidation_cost * std::abs(inventory);
            inventory = 0;
            ++r.interventions;
        }

        const double bid = mid - std::max(tick_size, bid_delta);
        const double ask = mid + std::max(tick_size, ask_delta);
        const double delta = events[i].type == 0 ? bid_delta : ask_delta;
        const double fill_probability = std::exp(-arrival_k * delta);
        const double u = static_cast<double>((i * 1103515245u + 12345u) & 0xFFFFu) / 65536.0;
        const bool fill = u < fill_probability;

        if (fill && events[i].type == 0 && inventory < kInventoryLimit) {
            cash -= bid;
            ++inventory;
            ++r.fills;
        } else if (fill && events[i].type == 1 && inventory > -kInventoryLimit) {
            cash += ask;
            --inventory;
            ++r.fills;
        }

        if (fill && ((events[i].type == 0 && inventory <= kInventoryLimit) ||
                     (events[i].type == 1 && inventory >= -kInventoryLimit))) {
            const std::size_t j = std::min(events.size() - 1, i + 10);
            const int signed_jumps = jump_prefix[j + 1] - jump_prefix[i];
            const double move_bps = (signed_jumps * tick_size / mid) * 1e4;
            const double adverse_bps = events[i].type == 0 ? -move_bps : move_bps;
            adverse_sum += adverse_bps;
            ++adverse_n;
        }

        mid += events[i].type == 1 ? tick_size : -tick_size;
        const double inv_abs_now = std::abs(static_cast<double>(inventory));
        abs_sum += inv_abs_now;
        sq_sum += static_cast<double>(inventory * inventory);
        inv_abs.push_back(inv_abs_now);
    }

    if (inventory != 0) {
        cash += static_cast<double>(inventory) * mid - liquidation_cost * std::abs(inventory);
    }
    r.pnl = cash;
    r.mean_abs_inventory = abs_sum / static_cast<double>(events.size());
    r.rms_inventory = std::sqrt(sq_sum / static_cast<double>(events.size()));
    r.max_abs_inventory = static_cast<int>(*std::max_element(inv_abs.begin(), inv_abs.end()));
    r.adverse_selection_bps = adverse_n ? adverse_sum / static_cast<double>(adverse_n) : 0.0;
    return r;
}

MonteCarloSummary summarize(const std::string& name, const std::vector<StrategyRun>& runs) {
    MonteCarloSummary s;
    s.strategy = name;
    s.runs = runs.size();
    if (runs.empty()) return s;
    std::vector<double> abs_inv;
    abs_inv.reserve(runs.size());
    s.mean_pnl = std::accumulate(runs.begin(), runs.end(), 0.0, [](double a, const auto& r) { return a + r.pnl; }) / runs.size();
    s.mean_rms_inventory = std::accumulate(runs.begin(), runs.end(), 0.0, [](double a, const auto& r) { return a + r.rms_inventory; }) / runs.size();
    s.mean_abs_inventory = std::accumulate(runs.begin(), runs.end(), 0.0, [](double a, const auto& r) { return a + r.mean_abs_inventory; }) / runs.size();
    s.mean_adverse_selection_bps = std::accumulate(runs.begin(), runs.end(), 0.0, [](double a, const auto& r) { return a + r.adverse_selection_bps; }) / runs.size();
    for (const auto& r : runs) abs_inv.push_back(static_cast<double>(r.max_abs_inventory));
    std::sort(abs_inv.begin(), abs_inv.end());
    s.p95_abs_inventory = abs_inv[static_cast<std::size_t>(0.95 * (abs_inv.size() - 1))];
    if (runs.size() > 1) {
        double ss = 0.0;
        for (const auto& r : runs) ss += (r.pnl - s.mean_pnl) * (r.pnl - s.mean_pnl);
        s.std_pnl = std::sqrt(ss / static_cast<double>(runs.size() - 1));
    }
    return s;
}

} // namespace smm::research
