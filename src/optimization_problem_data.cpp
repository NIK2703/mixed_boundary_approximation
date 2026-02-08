#include "mixed_approximation/optimization_problem_data.h"
#include "mixed_approximation/polynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mixed_approx {

OptimizationProblemData::OptimizationProblemData(const ApproximationConfig& config)
    : gamma(config.gamma)
    , interval_a(config.interval_start)
    , interval_b(config.interval_end)
    , epsilon(config.epsilon)
{
    // Copy approximation points
    approx_x.reserve(config.approx_points.size());
    approx_f.reserve(config.approx_points.size());
    approx_weight.reserve(config.approx_points.size());
    for (const auto& p : config.approx_points) {
        approx_x.push_back(p.x);
        approx_f.push_back(p.value);
        approx_weight.push_back(1.0 / p.weight);
    }
    
    // Copy repulsion points
    repel_y.reserve(config.repel_points.size());
    repel_forbidden.reserve(config.repel_points.size());
    repel_weight.reserve(config.repel_points.size());
    for (const auto& p : config.repel_points) {
        repel_y.push_back(p.x);
        repel_forbidden.push_back(p.y_forbidden);
        repel_weight.push_back(p.weight);
    }
    
    // Copy interpolation nodes
    interp_z.reserve(config.interp_nodes.size());
    interp_f.reserve(config.interp_nodes.size());
    for (const auto& n : config.interp_nodes) {
        interp_z.push_back(n.x);
        interp_f.push_back(n.value);
    }
}

bool OptimizationProblemData::is_valid() const {
    // Check consistency
    if (approx_x.size() != approx_f.size() || approx_x.size() != approx_weight.size()) {
        return false;
    }
    if (repel_y.size() != repel_forbidden.size() || repel_y.size() != repel_weight.size()) {
        return false;
    }
    if (interp_z.size() != interp_f.size()) {
        return false;
    }
    
    // Check parameters
    if (interval_a >= interval_b) return false;
    if (gamma < 0) return false;
    if (epsilon <= 0) return false;
    
    // Check weights
    for (double w : approx_weight) {
        if (!std::isfinite(w) || w <= 0) return false;
    }
    for (double w : repel_weight) {
        if (!std::isfinite(w) || w <= 0) return false;
    }
    
    return true;
}

void OptimizationCache::clear() {
    P_at_x.clear();
    W_at_x.clear();
    phi_at_x.clear();
    P_at_y.clear();
    W_at_y.clear();
    phi_at_y.clear();
    
    quad_points.clear();
    quad_weights.clear();
    W_at_quad.clear();
    W1_at_quad.clear();
    W2_at_quad.clear();
    P2_at_quad.clear();
    
    stiffness_matrix.clear();
    stiffness_dim = 0;
    
    basis_cache.clear();
    
    data_cache_valid = false;
    quad_cache_valid = false;
    basis_cache_valid = false;
    stiffness_valid = false;
}

bool OptimizationCache::is_ready(int /*n_free*/) const {
    return data_cache_valid && basis_cache_valid;
}

} // namespace mixed_approx
