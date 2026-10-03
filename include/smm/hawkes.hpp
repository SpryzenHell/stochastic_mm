#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace smm::research {

struct HawkesEvent {
    double time = 0.0;
    std::uint8_t type = 0; // 0 = sell market order / bid hit, 1 = buy market order / ask hit
};

struct BivariateHawkesConfig {
    double mu_sell = 40.0;
    double mu_buy = 40.0;
    double alpha_ss = 3.2;
    double alpha_sb = 0.9;
    double alpha_bs = 0.9;
    double alpha_bb = 3.2;
    double beta = 8.0;
    double horizon = 60.0;
};

class BivariateHawkes {
public:
    explicit BivariateHawkes(BivariateHawkesConfig config);

    double branching_ratio() const noexcept;
    std::array<double, 2> stationary_intensity() const noexcept;
    std::vector<HawkesEvent> simulate(std::uint64_t seed) const;
    std::array<double, 2> intensity_after(const std::vector<HawkesEvent>& events,
                                           std::size_t event_index,
                                           double lookahead) const noexcept;

private:
    BivariateHawkesConfig config_;
};

struct AdverseSelectionStats {
    std::size_t bid_fills = 0;
    std::size_t ask_fills = 0;
    double mean_bid_markout_bps = 0.0;
    double mean_ask_markout_bps = 0.0;
    double mean_adverse_selection_bps = 0.0;
};

} // namespace smm::research
