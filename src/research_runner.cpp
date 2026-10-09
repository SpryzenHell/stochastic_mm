#include "smm/hawkes.hpp"
#include "smm/hjb_qvi.hpp"
#include "smm/metrics.hpp"
#include "sim.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace smm::research;

static void write_json(const std::string& path, const std::string& json) {
    std::ofstream f(path);
    f << json;
}


int main(int argc, char** argv) {
    if (argc > 5) {
        std::cerr << "Usage: smm_research [runs] [output_dir] [gamma] [liquidation_cost]\\n";
        return 2;
    }
    const std::size_t runs = argc > 1 ? static_cast<std::size_t>(std::stoul(argv[1])) : 100;
    if (runs == 0 || runs > 100000) {
        std::cerr << "runs must be between 1 and 100000\\n";
        return 2;
    }
    const std::string outdir = argc > 2 ? argv[2] : "results";
    std::filesystem::create_directories(outdir);

    BivariateHawkesConfig hh;
    hh.alpha_ss = 5.2;
    hh.alpha_sb = 0.3;
    hh.alpha_bs = 0.3;
    hh.alpha_bb = 5.2;
    hh.mu_sell = 70.0;
    hh.mu_buy = 50.0;

    HjbQviConfig hc;
    hc.gamma = 0.05;
    hc.dt = 0.02;
    hc.liquidation_cost = 0.005;
    hc.horizon = hh.horizon;
    if (argc > 3) hc.gamma = std::stod(argv[3]);
    if (argc > 4) hc.liquidation_cost = std::stod(argv[4]);

    const auto solve_start = std::chrono::steady_clock::now();
    HjbQviSolution sol = HjbQviSolver(hc).solve();
    const auto solve_end = std::chrono::steady_clock::now();
    const double solve_ms =
        std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
    QuoteEngine quote_engine(sol, 0.01);

    {
        std::ofstream pf(outdir + "/policy_t0.csv");
        pf << "inventory,bid_delta,ask_delta,intervention\n";
        for (int q = -hc.q_max; q <= hc.q_max; ++q) {
            pf << q << "," << sol.bid(0, q) << "," << sol.ask(0, q) << ","
               << (sol.should_intervene(0, q) ? 1 : 0) << "\n";
        }
    }

    BivariateHawkes hawkes(hh);
    const auto stationary = hawkes.stationary_intensity();

    std::vector<StrategyRun> qvi_runs, base_runs;
    qvi_runs.reserve(runs);
    base_runs.reserve(runs);

    std::vector<double> hawkes_scores;
    std::vector<int> hawkes_labels;
    hawkes_scores.reserve(runs * 500);
    hawkes_labels.reserve(runs * 500);

    for (std::size_t run = 0; run < runs; ++run) {
        const auto events = simulate_path(hawkes, 1000 + run);

        double e0 = 0.0, e1 = 0.0, last_t = 0.0;
        for (std::size_t i = 0; i + 1 < events.size(); ++i) {
            const double dt = events[i].time - last_t;
            const double decay = std::exp(-hh.beta * dt);
            e0 *= decay;
            e1 *= decay;
            if (events[i].type == 0) {
                e0 += hh.alpha_ss;
                e1 += hh.alpha_bs;
            } else {
                e0 += hh.alpha_sb;
                e1 += hh.alpha_bb;
            }
            const double ls = hh.mu_sell + e0;
            const double lb = hh.mu_buy + e1;
            hawkes_scores.push_back(ls / (ls + lb));
            hawkes_labels.push_back(events[i + 1].type == 0 ? 1 : 0);
            last_t = events[i].time;
        }

        qvi_runs.push_back(run_strategy(
            events, &quote_engine, 1.0 / hc.arrival_k, 100.0, 0.01,
            hc.arrival_k, hc.liquidation_cost));
        base_runs.push_back(run_strategy(
            events, nullptr, 1.0 / hc.arrival_k, 100.0, 0.01,
            hc.arrival_k, hc.liquidation_cost));

        if (run == 0) {
            std::ofstream f(outdir + "/sample_hawkes_events.csv");
            f << "time,type\n";
            for (const auto& e : events) {
                f << std::fixed << std::setprecision(9)
                  << e.time << "," << int(e.type) << "\n";
            }
        }
    }

    const auto qvi = summarize("hjb_qvi", qvi_runs);
    const auto base = summarize("fixed_spread", base_runs);

    std::vector<double> group_ns;
    group_ns.reserve(200);
    volatile double sink = 0.0;

    for (int g = 0; g < 200; ++g) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5000; ++i) {
            const auto q = quote_engine.quote(
                100.0 + (i % 101) * 0.0001,
                (i % 31) - 15,
                (static_cast<std::size_t>(i) % sol.time_steps) * sol.dt);
            sink += q.bid + q.ask;
        }
        const auto t1 = std::chrono::steady_clock::now();
        group_ns.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count() / 5000.0);
    }

    std::sort(group_ns.begin(), group_ns.end());
    const double quote_median_ns = group_ns[group_ns.size() / 2];
    const double quote_p99_ns =
        group_ns[static_cast<std::size_t>(0.99 * (group_ns.size() - 1))];
    (void)sink;

    std::ostringstream j;
    j << std::setprecision(10);
    j << "{\n"
      << "  \"runs\": " << runs << ",\n"
      << "  \"seed_start\": 1000,\n"
      << "  \"hjb\": {\"gamma\": " << hc.gamma
      << ", \"sigma\": " << hc.sigma
      << ", \"arrival_A\": " << hc.arrival_A
      << ", \"arrival_k\": " << hc.arrival_k
      << ", \"liquidation_cost\": " << hc.liquidation_cost
      << ", \"min_delta\": " << hc.min_delta
      << ", \"max_delta\": " << hc.max_delta
      << ", \"horizon\": " << hc.horizon
      << ", \"dt\": " << sol.dt
      << ", \"q_max\": " << hc.q_max
      << ", \"time_steps\": " << sol.time_steps << "},\n"
      << "  \"hjb_fdm_solve_ms\": " << solve_ms << ",\n"
      << "  \"hawkes\": {\"mu_sell\": " << hh.mu_sell
      << ", \"mu_buy\": " << hh.mu_buy
      << ", \"alpha_ss\": " << hh.alpha_ss
      << ", \"alpha_sb\": " << hh.alpha_sb
      << ", \"alpha_bs\": " << hh.alpha_bs
      << ", \"alpha_bb\": " << hh.alpha_bb
      << ", \"beta\": " << hh.beta
      << ", \"horizon\": " << hh.horizon
      << ", \"branching_ratio\": " << hawkes.branching_ratio()
      << ", \"stationary_sell_intensity\": " << stationary[0]
      << ", \"stationary_buy_intensity\": " << stationary[1]
      << ", \"next_sell_direction_auc\": "
      << smm::research::roc_auc(hawkes_scores, hawkes_labels)
      << ", \"prediction_samples\": " << hawkes_labels.size() << "},\n"
      << "  \"quote_engine\": {\"median_ns\": " << quote_median_ns
      << ", \"p99_ns\": " << quote_p99_ns << "},\n"
      << "  \"strategies\": {\n"
      << "    \"hjb_qvi\": {\"mean_pnl\": " << qvi.mean_pnl
      << ", \"std_pnl\": " << qvi.std_pnl
      << ", \"mean_rms_inventory\": " << qvi.mean_rms_inventory
      << ", \"mean_abs_inventory\": " << qvi.mean_abs_inventory
      << ", \"mean_adverse_selection_bps\": "
      << qvi.mean_adverse_selection_bps
      << ", \"p95_abs_inventory\": " << qvi.p95_abs_inventory << "},\n"
      << "    \"fixed_spread\": {\"mean_pnl\": " << base.mean_pnl
      << ", \"std_pnl\": " << base.std_pnl
      << ", \"mean_rms_inventory\": " << base.mean_rms_inventory
      << ", \"mean_abs_inventory\": " << base.mean_abs_inventory
      << ", \"mean_adverse_selection_bps\": "
      << base.mean_adverse_selection_bps
      << ", \"p95_abs_inventory\": " << base.p95_abs_inventory << "}\n"
      << "  }\n"
      << "}\n";

    write_json(outdir + "/research_run.json", j.str());
    std::cout << j.str();
    return 0;
}
