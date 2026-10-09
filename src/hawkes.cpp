#include "smm/hawkes.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(SMM_HAS_EIGEN)
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#endif

namespace smm::research {

BivariateHawkes::BivariateHawkes(BivariateHawkesConfig config) : config_(config) {
    const double params[] = {
        config_.mu_sell, config_.mu_buy, config_.alpha_ss, config_.alpha_sb,
        config_.alpha_bs, config_.alpha_bb, config_.beta, config_.horizon
    };
    for (double x : params) {
        if (!std::isfinite(x)) throw std::invalid_argument("Hawkes parameters must be finite");
    }
    if (config_.mu_sell < 0.0 || config_.mu_buy < 0.0) {
        throw std::invalid_argument("Hawkes baseline intensities must be non-negative");
    }
    if (config_.alpha_ss < 0.0 || config_.alpha_sb < 0.0 ||
        config_.alpha_bs < 0.0 || config_.alpha_bb < 0.0) {
        throw std::invalid_argument("Hawkes excitation parameters must be non-negative");
    }
    if (config_.beta <= 0.0 || config_.horizon <= 0.0) {
        throw std::invalid_argument("Hawkes beta and horizon must be positive");
    }
}

double BivariateHawkes::branching_ratio() const noexcept {
#if defined(SMM_HAS_EIGEN)
    Eigen::Matrix2d K;
    K << config_.alpha_ss, config_.alpha_sb,
         config_.alpha_bs, config_.alpha_bb;
    K /= config_.beta;
    Eigen::EigenSolver<Eigen::Matrix2d> es(K, false);
    const auto ev = es.eigenvalues();
    return std::max(std::abs(ev[0]), std::abs(ev[1]));
#else
    const double a = config_.alpha_ss / config_.beta;
    const double d = config_.alpha_bb / config_.beta;
    const double bc = (config_.alpha_sb * config_.alpha_bs) / (config_.beta * config_.beta);
    const double disc = std::sqrt(std::max(0.0, (a - d) * (a - d) + 4.0 * bc));
    return 0.5 * (a + d + disc);
#endif
}

std::array<double, 2> BivariateHawkes::stationary_intensity() const noexcept {
    const double r = branching_ratio();
    if (r >= 1.0) return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    const double a = config_.alpha_ss / config_.beta;
    const double b = config_.alpha_sb / config_.beta;
    const double c = config_.alpha_bs / config_.beta;
    const double d = config_.alpha_bb / config_.beta;
    const double det = (1.0 - a) * (1.0 - d) - b * c;
    if (std::abs(det) < 1e-12) return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    const double s0 = ((1.0 - d) * config_.mu_sell + b * config_.mu_buy) / det;
    const double s1 = (c * config_.mu_sell + (1.0 - a) * config_.mu_buy) / det;
    return {s0, s1};
}

std::vector<HawkesEvent> BivariateHawkes::simulate(std::uint64_t seed) const {
    std::vector<HawkesEvent> out;
    out.reserve(static_cast<std::size_t>((config_.mu_sell + config_.mu_buy) * config_.horizon * 2.0));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double t = 0.0;
    double e0 = 0.0, e1 = 0.0;
    double lambda0 = config_.mu_sell, lambda1 = config_.mu_buy;
    while (t < config_.horizon) {
        const double upper = std::max(lambda0 + lambda1, 1e-12);
        const double u = std::max(uni(rng), 1e-15);
        const double dt = -std::log(u) / upper;
        t += dt;
        if (t >= config_.horizon) break;

        const double decay = std::exp(-config_.beta * dt);
        e0 *= decay;
        e1 *= decay;

        const double cand0 = config_.mu_sell + e0;
        const double cand1 = config_.mu_buy + e1;
        const double cand_total = cand0 + cand1;
        if (cand_total <= 0.0) continue;
        if (uni(rng) <= cand_total / upper) {
            const bool sell = uni(rng) < cand0 / cand_total;
            out.push_back({t, static_cast<std::uint8_t>(sell ? 0 : 1)});
            if (sell) {
                e0 += config_.alpha_ss;
                e1 += config_.alpha_bs;
            } else {
                e0 += config_.alpha_sb;
                e1 += config_.alpha_bb;
            }
        }

        lambda0 = config_.mu_sell + e0;
        lambda1 = config_.mu_buy + e1;
    }
    return out;
}

std::array<double, 2> BivariateHawkes::intensity_after(const std::vector<HawkesEvent>& events,
                                                         std::size_t event_index,
                                                         double lookahead) const noexcept {
    if (event_index >= events.size()) return {config_.mu_sell, config_.mu_buy};
    double e0 = 0.0, e1 = 0.0;
    double last = 0.0;
    for (std::size_t i = 0; i <= event_index; ++i) {
        const double dt = events[i].time - last;
        const double decay = std::exp(-config_.beta * std::max(0.0, dt));
        e0 *= decay;
        e1 *= decay;
        if (events[i].type == 0) {
            e0 += config_.alpha_ss;
            e1 += config_.alpha_bs;
        } else {
            e0 += config_.alpha_sb;
            e1 += config_.alpha_bb;
        }
        last = events[i].time;
    }
    const double decay = std::exp(-config_.beta * std::max(0.0, lookahead));
    return {config_.mu_sell + e0 * decay, config_.mu_buy + e1 * decay};
}

} // namespace smm::research
