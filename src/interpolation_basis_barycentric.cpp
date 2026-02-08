#include "mixed_approximation/interpolation_basis.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mixed_approx {

void InterpolationBasis::compute_barycentric_weights_standard() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double weight = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == k) continue;
            weight *= (nodes[k] - nodes[j]);
        }
        barycentric_weights[k] = 1.0 / weight;
    }
    
    double max_abs = 0.0;
    for (double w : barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
        weight_scale = max_abs;
    }
}

void InterpolationBasis::compute_barycentric_weights_logarithmic() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    std::vector<double> log_abs_weights(m, 0.0);
    std::vector<int> sign_weights(m, 1);
    
    for (int e = 0; e < m; ++e) {
        double log_sum = 0.0;
        int sign = 1;
        
        for (int k = 0; k < m; ++k) {
            if (k == e) continue;
            double diff = nodes[e] - nodes[k];
            double abs_diff = std::abs(diff);
            
            if (abs_diff < 1e-15) {
                compute_barycentric_weights_standard();
                return;
            }
            
            log_sum -= std::log(abs_diff);
            if (diff < 0) sign = -sign;
        }
        
        log_abs_weights[e] = log_sum;
        sign_weights[e] = sign;
    }
    
    double max_log = *std::max_element(log_abs_weights.begin(), log_abs_weights.end());
    
    double max_abs = 0.0;
    for (int e = 0; e < m; ++e) {
        double abs_w = std::exp(log_abs_weights[e] - max_log);
        barycentric_weights[e] = sign_weights[e] * abs_w;
        max_abs = std::max(max_abs, std::abs(barycentric_weights[e]));
    }
    
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
    }
    weight_scale = std::exp(max_log);
}

void InterpolationBasis::compute_chebyshev_weights() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        barycentric_weights[k] = sign * std::sin(M_PI * (2.0*k + 1.0) / (2.0 * m));
    }
    
    double max_abs = 0.0;
    for (double w : barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
    }
    weight_scale = max_abs;
}

void InterpolationBasis::compute_barycentric_weights() {
    if (nodes.size() == 1) {
        barycentric_weights = {1.0};
        weighted_values = {values[0]};
        weight_scale = 1.0;
        return;
    }
    
    if (nodes.size() == 2) {
        barycentric_weights.resize(2);
        double diff = nodes[1] - nodes[0];
        barycentric_weights[0] = 1.0 / diff;
        barycentric_weights[1] = -barycentric_weights[0];
        weight_scale = 1.0;
        precompute_weighted_values();
        return;
    }
    
    if (detect_chebyshev_nodes()) {
        compute_chebyshev_weights();
        precompute_weighted_values();
        return;
    }
    
    compute_barycentric_weights_logarithmic();
    precompute_weighted_values();
}

void InterpolationBasis::precompute_weighted_values() {
    weighted_values.resize(barycentric_weights.size());
    for (size_t i = 0; i < barycentric_weights.size(); ++i) {
        weighted_values[i] = barycentric_weights[i] * values[i];
    }
}

double InterpolationBasis::evaluate_barycentric(double x) const {
    int m = static_cast<int>(nodes.size());
    
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            return values[k];
        }
    }
    
    const double* wf = weighted_values.empty() ? nullptr : weighted_values.data();
    
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int k = 0; k < m; ++k) {
        double diff = x - nodes[k];
        double inv_diff = 1.0 / diff;
        denominator += barycentric_weights[k] * inv_diff;
        if (wf) {
            numerator += wf[k] * inv_diff;
        } else {
            numerator += barycentric_weights[k] * values[k] * inv_diff;
        }
    }
    
    if (std::abs(denominator) < 1e-14) {
        return 0.0;
    }
    
    return numerator / denominator;
}

} // namespace mixed_approx
