#include "smm/hawkes.hpp"
#include "smm/hjb_qvi.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    smm::research::HjbQviConfig c;
    c.q_max = 8;
    c.horizon = 1.0;
    c.dt = 0.01;
    auto s = smm::research::HjbQviSolver(c).solve();

    assert(s.time_steps == 101);
    for (int q = -8; q <= 8; ++q) {
        assert(std::isfinite(s.value(0, q)));
        assert(s.bid(0, q) >= c.min_delta);
        assert(s.ask(0, q) >= c.min_delta);
    }

    smm::research::BivariateHawkes h({});
    assert(h.branching_ratio() < 1.0);
    auto e = h.simulate(7);
    assert(!e.empty());
    for (std::size_t i = 1; i < e.size(); ++i) assert(e[i].time >= e[i - 1].time);
    std::cout << "smm_unit_tests: PASS
";
}
