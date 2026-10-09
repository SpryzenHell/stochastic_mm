#include "smm/hawkes.hpp"
#include "smm/hjb_qvi.hpp"
#include "smm/metrics.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int checks = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <typename F>
void throws_invalid(F&& fn, const std::string& message) {
    ++checks;
    try { fn(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

void test_default_horizons_align() {
    const smm::research::HjbQviConfig hjb;
    const smm::research::BivariateHawkesConfig hawkes;
    check(std::abs(hjb.horizon - hawkes.horizon) < 1e-12,
          "default HJB and Hawkes horizons match");
    check(std::abs(hjb.dt - 0.005) < 1e-12,
          "default HJB time step matches the stable research configuration");
    check(std::abs(hjb.gamma - 0.05) < 1e-12 &&
          std::abs(hjb.liquidation_cost - 0.005) < 1e-12,
          "default HJB parameters match the research configuration");
    const auto stable = smm::research::HjbQviSolver(hjb).solve();
    check(stable.time_steps == 12001, "default 60-second HJB grid size");
    check(stable.values.size() == stable.time_steps * stable.width(), "default HJB grid storage size");
    for (std::size_t i = 0; i < stable.values.size(); ++i) {
        check(std::isfinite(stable.values[i]), "default long-horizon value is finite");
        check(std::isfinite(stable.bid_delta[i]) && std::isfinite(stable.ask_delta[i]),
              "default long-horizon quote policy is finite");
        check(stable.bid_delta[i] >= hjb.min_delta && stable.bid_delta[i] <= hjb.max_delta,
              "default long-horizon bid is bounded");
        check(stable.ask_delta[i] >= hjb.min_delta && stable.ask_delta[i] <= hjb.max_delta,
              "default long-horizon ask is bounded");
    }
}

void test_hjb_grid_and_quotes() {
    smm::research::HjbQviConfig cfg;
    cfg.q_max = 8;
    cfg.horizon = 1.0;
    cfg.dt = 0.01;
    const auto sol = smm::research::HjbQviSolver(cfg).solve();
    check(sol.time_steps == 101, "HJB time-step count");
    check(sol.q_max == 8, "HJB inventory limit");
    check(std::abs(sol.dt - 0.01) < 1e-12, "HJB adjusted time step");
    check(sol.values.size() == 101u * 17u, "HJB value grid size");
    check(sol.bid_delta.size() == sol.values.size(), "bid policy grid size");
    check(sol.ask_delta.size() == sol.values.size(), "ask policy grid size");
    check(sol.intervene.size() == sol.values.size(), "intervention grid size");

    for (std::size_t t = 0; t < sol.time_steps; ++t) {
        for (int q = -cfg.q_max; q <= cfg.q_max; ++q) {
            check(std::isfinite(sol.value(t, q)), "finite HJB value");
            check(std::isfinite(sol.bid(t, q)), "finite bid distance");
            check(std::isfinite(sol.ask(t, q)), "finite ask distance");
            check(sol.bid(t, q) >= cfg.min_delta && sol.bid(t, q) <= cfg.max_delta,
                  "bid distance stays within configured bounds");
            check(sol.ask(t, q) >= cfg.min_delta && sol.ask(t, q) <= cfg.max_delta,
                  "ask distance stays within configured bounds");
        }
        check(sol.should_intervene(t, -cfg.q_max), "lower inventory boundary intervenes");
        check(sol.should_intervene(t, cfg.q_max), "upper inventory boundary intervenes");
    }

    smm::research::QuoteEngine engine(sol, 0.01);
    const auto low = engine.quote(100.0, -1000, -10.0);
    const auto high = engine.quote(100.0, 1000, 10.0);
    check(std::isfinite(low.bid) && std::isfinite(low.ask), "low-bound quote is finite");
    check(std::isfinite(high.bid) && std::isfinite(high.ask), "high-bound quote is finite");
    check(low.bid < low.ask && high.bid < high.ask, "bid is below ask");
    check(std::abs(low.bid_delta - sol.bid(0, -cfg.q_max)) < 1e-12,
          "inventory and time lower clamps");
    check(std::abs(high.ask_delta - sol.ask(sol.time_steps - 1, cfg.q_max)) < 1e-12,
          "inventory and time upper clamps");
    const auto nonfinite = engine.quote(std::numeric_limits<double>::quiet_NaN(), 0,
                                        std::numeric_limits<double>::quiet_NaN());
    check(std::isfinite(nonfinite.bid) && std::isfinite(nonfinite.ask),
          "non-finite quote inputs are handled safely");
    throws_invalid([&] { smm::research::QuoteEngine bad(sol, 0.0); }, "zero tick size must be rejected");
    throws_invalid([&] { smm::research::QuoteEngine bad(sol, -0.01); }, "negative tick size must be rejected");
    throws_invalid([&] { smm::research::QuoteEngine bad(sol, std::numeric_limits<double>::infinity()); },
                   "infinite tick size must be rejected");
}

void test_hjb_invalid_configs() {
    auto invalid = [](smm::research::HjbQviConfig c) {
        throws_invalid([&] { smm::research::HjbQviSolver solver(c); }, "invalid HJB config accepted");
    };
    smm::research::HjbQviConfig c;
    c.gamma = 0.0; invalid(c);
    c = {}; c.sigma = -0.1; invalid(c);
    c = {}; c.arrival_A = 0.0; invalid(c);
    c = {}; c.arrival_k = 0.0; invalid(c);
    c = {}; c.liquidation_cost = -0.1; invalid(c);
    c = {}; c.liquidation_cost = std::numeric_limits<double>::quiet_NaN(); invalid(c);
    c = {}; c.min_delta = 0.2; c.max_delta = 0.1; invalid(c);
    c = {}; c.horizon = 0.0; invalid(c);
    c = {}; c.dt = 0.0; invalid(c);
    c = {}; c.q_max = 0; invalid(c);
    c = {}; c.quote_stride = 0; invalid(c);
    c = {}; c.dt = 1e-10; invalid(c);
    c = {}; c.q_max = 1000001; invalid(c);
}

void test_hawkes() {
    smm::research::BivariateHawkesConfig cfg;
    smm::research::BivariateHawkes hawkes(cfg);
    const double branching = hawkes.branching_ratio();
    check(std::isfinite(branching) && branching > 0.0 && branching < 1.0,
          "default Hawkes process is stable");
    const auto stationary = hawkes.stationary_intensity();
    check(std::isfinite(stationary[0]) && stationary[0] > 0.0,
          "stationary sell intensity is finite and positive");
    check(std::isfinite(stationary[1]) && stationary[1] > 0.0,
          "stationary buy intensity is finite and positive");

    const auto first = hawkes.simulate(20261009);
    const auto second = hawkes.simulate(20261009);
    check(!first.empty(), "Hawkes sample path is not empty");
    check(first.size() == second.size(), "Hawkes simulation is deterministic in event count");
    for (std::size_t i = 0; i < first.size(); ++i) {
        check(first[i].time == second[i].time && first[i].type == second[i].type,
              "Hawkes simulation is deterministic for a fixed seed");
        check(std::isfinite(first[i].time) && first[i].time >= 0.0 && first[i].time < cfg.horizon,
              "event timestamp is within the simulation horizon");
        check(first[i].type == 0 || first[i].type == 1, "event type is 0 or 1");
        if (i > 0) check(first[i].time >= first[i - 1].time, "event timestamps are sorted");
    }

    const auto baseline = hawkes.intensity_after(first, first.size(), 0.1);
    check(baseline[0] == cfg.mu_sell && baseline[1] == cfg.mu_buy,
          "out-of-range intensity lookup returns baseline intensity");
    if (!first.empty()) {
        const auto after = hawkes.intensity_after(first, first.size() - 1, 0.0);
        check(std::isfinite(after[0]) && after[0] >= cfg.mu_sell, "post-event sell intensity is valid");
        check(std::isfinite(after[1]) && after[1] >= cfg.mu_buy, "post-event buy intensity is valid");
    }

    auto bad = cfg;
    bad.beta = 0.0;
    throws_invalid([&] { smm::research::BivariateHawkes x(bad); }, "zero Hawkes beta accepted");
    bad = cfg; bad.horizon = -1.0;
    throws_invalid([&] { smm::research::BivariateHawkes x(bad); }, "negative Hawkes horizon accepted");
    bad = cfg; bad.mu_sell = -1.0;
    throws_invalid([&] { smm::research::BivariateHawkes x(bad); }, "negative baseline intensity accepted");
    bad = cfg; bad.alpha_bs = -0.1;
    throws_invalid([&] { smm::research::BivariateHawkes x(bad); }, "negative excitation accepted");
    bad = cfg; bad.alpha_bb = std::numeric_limits<double>::infinity();
    throws_invalid([&] { smm::research::BivariateHawkes x(bad); }, "infinite excitation accepted");

    auto supercritical = cfg;
    supercritical.alpha_ss = 9.0;
    const smm::research::BivariateHawkes explosive(supercritical);
    check(explosive.branching_ratio() >= 1.0, "supercritical branching ratio is detected");
    const auto unstable = explosive.stationary_intensity();
    check(std::isinf(unstable[0]) && std::isinf(unstable[1]),
          "supercritical stationary intensity is reported as infinite");
}

void test_roc_auc() {
    using smm::research::roc_auc;
    check(std::abs(roc_auc({0.1, 0.4, 0.35, 0.8}, {0, 0, 1, 1}) - 0.75) < 1e-12,
          "ROC AUC for a hand-calculated example");
    check(std::abs(roc_auc({0.2, 0.2}, {0, 1}) - 0.5) < 1e-12,
          "ROC AUC average ranks tied scores");
    check(std::abs(roc_auc({0.1, 0.2, 0.3, 0.4}, {0, 0, 1, 1}) - 1.0) < 1e-12,
          "perfect ranking has AUC 1");
    check(std::abs(roc_auc({0.1, 0.2, 0.3, 0.4}, {1, 1, 0, 0}) - 0.0) < 1e-12,
          "reverse ranking has AUC 0");
    check(roc_auc({}, {}) == 0.5, "empty ROC AUC convention");
    check(roc_auc({0.2, 0.1}, {1, 1}) == 0.5, "single-class ROC AUC convention");
    throws_invalid([&] { (void)roc_auc({0.1}, {0, 1}); }, "AUC size mismatch accepted");
    throws_invalid([&] { (void)roc_auc({0.1}, {2}); }, "invalid AUC label accepted");
    throws_invalid([&] { (void)roc_auc({std::numeric_limits<double>::quiet_NaN()}, {1}); },
                   "non-finite AUC score accepted");
}
} // namespace

int main() {
    try {
        test_default_horizons_align();
        test_hjb_grid_and_quotes();
        test_hjb_invalid_configs();
        test_hawkes();
        test_roc_auc();
        std::cout << "smm_unit_tests: PASS (" << checks << " checks)\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "smm_unit_tests: FAIL after " << checks << " checks: "
                  << exc.what() << "\n";
        return 1;
    }
}
