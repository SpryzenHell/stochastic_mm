#include "smm/hjb_qvi.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace smm::research {

HjbQviSolver::HjbQviSolver(HjbQviConfig config) : config_(config) {
    if (config_.gamma <= 0.0 || config_.sigma < 0.0 || config_.arrival_A <= 0.0 || config_.arrival_k <= 0.0) {
        throw std::invalid_argument("HJB-QVI parameters must be positive (except sigma >= 0)");
    }
    if (config_.horizon <= 0.0 || config_.dt <= 0.0 || config_.q_max < 1) {
        throw std::invalid_argument("horizon, dt and q_max must be positive");
    }
    if (config_.min_delta < 0.0 || config_.max_delta < config_.min_delta) {
        throw std::invalid_argument("invalid quote distance bounds");
    }
}

HjbQviSolution HjbQviSolver::solve() const {
    const int qmin = -config_.q_max;
    const int qmax = config_.q_max;
    const std::size_t width = static_cast<std::size_t>(2 * config_.q_max + 1);
    const std::size_t steps = static_cast<std::size_t>(std::ceil(config_.horizon / config_.dt));

    HjbQviSolution out;
    out.time_steps = steps + 1;
    out.q_max = config_.q_max;
    out.dt = config_.horizon / static_cast<double>(steps);
    out.terminal_time = config_.horizon;
    const std::size_t total = out.time_steps * width;
    out.values.assign(total, 0.0);
    out.bid_delta.assign(total, config_.min_delta);
    out.ask_delta.assign(total, config_.min_delta);
    out.intervene.assign(total, 0);

    for (int q = qmin; q <= qmax; ++q) {
        out.values[out.index(steps, q)] = -config_.liquidation_cost * std::abs(static_cast<double>(q));
    }

    std::vector<double> next(width), current(width);
    for (std::size_t j = 0; j < width; ++j) next[j] = out.values[out.index(steps, qmin + static_cast<int>(j))];

    // Reduced CARA HJB-QVI in inventory space:
    // max{ V_t + H(V) - 0.5*gamma*sigma^2*q^2, M[V]-V } = 0
    // with exponential fill intensity lambda(delta) = A exp(-k delta).
    for (std::size_t ti = steps; ti-- > 0;) {
        std::fill(current.begin(), current.end(), 0.0);
        #pragma omp simd
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(width); ++j) {
            const int q = qmin + static_cast<int>(j);
            const std::size_t idx = static_cast<std::size_t>(j);
            const std::size_t up = std::min<std::size_t>(idx + 1, width - 1);
            const std::size_t dn = idx == 0 ? 0 : idx - 1;

            const double dv_bid = next[up] - next[idx];
            const double dv_ask = next[dn] - next[idx];
            const double db = std::clamp(1.0 / config_.arrival_k - dv_bid,
                                         config_.min_delta, config_.max_delta);
            const double da = std::clamp(1.0 / config_.arrival_k - dv_ask,
                                         config_.min_delta, config_.max_delta);

            const double lb = config_.arrival_A * std::exp(-config_.arrival_k * db);
            const double la = config_.arrival_A * std::exp(-config_.arrival_k * da);
            const double hb = lb * (db + dv_bid);
            const double ha = la * (da + dv_ask);
            const double risk = -0.5 * config_.gamma * config_.sigma * config_.sigma *
                                static_cast<double>(q * q);

            const double continuation = next[idx] + out.dt * (hb + ha + risk);
            const double intervention = next[static_cast<std::size_t>(config_.q_max)]
                                      - config_.liquidation_cost * std::abs(static_cast<double>(q));
            const bool do_intervene = (std::abs(q) >= config_.q_max) || (intervention > continuation);
            current[idx] = std::max(continuation, intervention);

            const std::size_t out_idx = out.index(ti, q);
            out.bid_delta[out_idx] = db;
            out.ask_delta[out_idx] = da;
            out.intervene[out_idx] = static_cast<unsigned char>(do_intervene);
        }

        for (std::size_t j = 0; j < width; ++j) {
            next[j] = current[j];
            out.values[out.index(ti, qmin + static_cast<int>(j))] = current[j];
        }
    }
    return out;
}

QuoteEngine::QuoteEngine(const HjbQviSolution& solution, double tick_size)
    : solution_(&solution), tick_size_(tick_size) {}

Quote QuoteEngine::quote(double mid, int inventory, double time) const noexcept {
    const int q = std::clamp(inventory, -solution_->q_max, solution_->q_max);
    const double clamped_t = std::clamp(time, 0.0, solution_->terminal_time);
    const std::size_t t = static_cast<std::size_t>(std::llround(clamped_t / solution_->dt));
    const std::size_t ti = std::min(t, solution_->time_steps - 1);
    const double bd = solution_->bid(ti, q);
    const double ad = solution_->ask(ti, q);
    Quote out;
    out.bid_delta = bd;
    out.ask_delta = ad;
    out.bid = mid - std::max(tick_size_, bd);
    out.ask = mid + std::max(tick_size_, ad);
    out.intervention = solution_->should_intervene(ti, q);
    return out;
}

} // namespace smm::research
