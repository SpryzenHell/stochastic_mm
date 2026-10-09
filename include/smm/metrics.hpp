#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace smm::research {

// Mann-Whitney ROC AUC. Equal scores receive their average rank.
// Returns 0.5 when the input has only one label class.
inline double roc_auc(const std::vector<double>& scores, const std::vector<int>& labels) {
    if (scores.size() != labels.size()) {
        throw std::invalid_argument("ROC AUC scores and labels must have equal lengths");
    }
    if (scores.empty()) return 0.5;

    std::size_t positive = 0;
    std::size_t negative = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) throw std::invalid_argument("ROC AUC scores must be finite");
        if (labels[i] == 1) ++positive;
        else if (labels[i] == 0) ++negative;
        else throw std::invalid_argument("ROC AUC labels must be 0 or 1");
    }
    if (positive == 0 || negative == 0) return 0.5;

    std::vector<std::size_t> order(scores.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return scores[a] < scores[b];
    });

    double positive_rank_sum = 0.0;
    std::size_t first = 0;
    while (first < order.size()) {
        std::size_t last = first + 1;
        while (last < order.size() && scores[order[last]] == scores[order[first]]) ++last;
        const double average_rank =
            (static_cast<double>(first + 1) + static_cast<double>(last)) / 2.0;
        for (std::size_t j = first; j < last; ++j) {
            if (labels[order[j]] == 1) positive_rank_sum += average_rank;
        }
        first = last;
    }

    const double n_pos = static_cast<double>(positive);
    const double n_neg = static_cast<double>(negative);
    return (positive_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
}

} // namespace smm::research
