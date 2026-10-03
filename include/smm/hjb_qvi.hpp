#pragma once
#include <cstddef>
#include <vector>

namespace smm::research {

struct HjbQviConfig {
    double gamma = 0.01;                 // CARA risk aversion
    double sigma = 0.20;                 // mid-price volatility scale
    double arrival_A = 80.0;             // baseline order-flow scale
    double arrival_k = 40.0;             // intensity decay in quote distance
    double liquidation_cost = 0.02;      // cash penalty per unit when flattening
    double min_delta = 0.0005;            // admissible quote distance
    double max_delta = 0.50;              // numerical cap
    double horizon = 10.0;               // seconds
    double dt = 0.01;                    // FDM time step
    int q_max = 25;                       // inventory grid [-q_max, q_max]
    int quote_stride = 1;                // retain every FDM policy row
};

struct HjbQviSolution {
    std::size_t time_steps = 0;
    int q_max = 0;
    double dt = 0.0;
    double terminal_time = 0.0;
    std::vector<double> values;           // row-major [time][q]
    std::vector<double> bid_delta;        // row-major [time][q]
    std::vector<double> ask_delta;        // row-major [time][q]
    std::vector<unsigned char> intervene; // row-major [time][q]

    std::size_t width() const noexcept { return static_cast<std::size_t>(2 * q_max + 1); }
    std::size_t index(std::size_t t, int q) const noexcept {
        return t * width() + static_cast<std::size_t>(q + q_max);
    }

    double value(std::size_t t, int q) const noexcept { return values[index(t, q)]; }
    double bid(std::size_t t, int q) const noexcept { return bid_delta[index(t, q)]; }
    double ask(std::size_t t, int q) const noexcept { return ask_delta[index(t, q)]; }
    bool should_intervene(std::size_t t, int q) const noexcept { return intervene[index(t, q)] != 0; }
};

class HjbQviSolver {
public:
    explicit HjbQviSolver(HjbQviConfig config);
    HjbQviSolution solve() const;

private:
    HjbQviConfig config_;
};

struct Quote {
    double bid = 0.0;
    double ask = 0.0;
    double bid_delta = 0.0;
    double ask_delta = 0.0;
    bool intervention = false;
};

class QuoteEngine {
public:
    QuoteEngine(const HjbQviSolution& solution, double tick_size = 0.01);
    Quote quote(double mid, int inventory, double time) const noexcept;

private:
    const HjbQviSolution* solution_;
    double tick_size_;
};

} // namespace smm::research
