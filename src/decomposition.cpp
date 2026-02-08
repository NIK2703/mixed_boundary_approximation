#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/polynomial.h"
#include <numeric>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mixed_approx {

// ============== DecompositionResult implementation ==============

Polynomial DecompositionResult::build_polynomial(const std::vector<double>& q_coeffs) const {
    if (!metadata.is_valid) {
        throw std::invalid_argument("Cannot build polynomial from invalid decomposition: " + metadata.validation_message);
    }
    
    int n_free = metadata.n_free;
    if (static_cast<int>(q_coeffs.size()) != n_free) {
        throw std::invalid_argument("Invalid number of coefficients for Q(x): expected " +
                                    std::to_string(n_free) + ", got " + std::to_string(q_coeffs.size()));
    }
    
    Polynomial Q(q_coeffs.empty() ? 0 : static_cast<int>(q_coeffs.size() - 1));
    if (!q_coeffs.empty()) {
        Q.setCoefficients(q_coeffs);
    }
    
    Polynomial W(weight_multiplier.degree());
    if (!weight_multiplier.coeffs.empty()) {
        W.setCoefficients(weight_multiplier.coeffs);
    } else {
        std::vector<InterpolationNode> roots_as_nodes;
        for (double r : weight_multiplier.roots) {
            roots_as_nodes.emplace_back(r, 0.0);
        }
        W = build_interpolation_multiplier(roots_as_nodes);
    }
    
    Polynomial QW = Q * W;
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < interpolation_basis.nodes.size(); ++i) {
        double node_x = interpolation_basis.nodes[i];
        if (interpolation_basis.is_normalized) {
            node_x = node_x * interpolation_basis.x_scale + interpolation_basis.x_center;
        }
        interp_nodes.emplace_back(node_x, interpolation_basis.values[i]);
    }
    Polynomial P_int = build_lagrange_polynomial(interp_nodes);
    
    return P_int + QW;
}

double DecompositionResult::evaluate(double x, const std::vector<double>& q_coeffs) const {
    if (!metadata.is_valid) {
        throw std::invalid_argument("Cannot evaluate from invalid decomposition: " + metadata.validation_message);
    }
    
    double p_int_val = interpolation_basis.evaluate(x);
    double qw_val = weight_multiplier.evaluate_product(x, q_coeffs);
    
    return p_int_val + qw_val;
}

void DecompositionResult::build_caches(const std::vector<double>& points_x,
                                       const std::vector<double>& points_y) {
    if (!metadata.is_valid) {
        throw std::invalid_argument("Cannot build caches from invalid decomposition: " + metadata.validation_message);
    }
    
    clear_caches();
    
    for (size_t i = 0; i < points_x.size(); ++i) {
        double x = points_x[i];
        cache_W_x.push_back(weight_multiplier.evaluate(x));
        cache_W1_x.push_back(weight_multiplier.evaluate_derivative(x, 1));
        cache_W2_x.push_back(weight_multiplier.evaluate_derivative(x, 2));
    }
    
    for (size_t i = 0; i < points_y.size(); ++i) {
        double y = points_y[i];
        cache_W_y.push_back(weight_multiplier.evaluate(y));
        cache_W1_y.push_back(weight_multiplier.evaluate_derivative(y, 1));
        cache_W2_y.push_back(weight_multiplier.evaluate_derivative(y, 2));
    }
    
    caches_built = true;
}

void DecompositionResult::clear_caches() {
    cache_W_x.clear();
    cache_W_y.clear();
    cache_W1_x.clear();
    cache_W1_y.clear();
    cache_W2_x.clear();
    cache_W2_y.clear();
    caches_built = false;
}

bool DecompositionResult::verify_interpolation(double tolerance) const {
    if (!metadata.is_valid) {
        return false;
    }
    
    for (size_t i = 0; i < interpolation_basis.nodes.size(); ++i) {
        double z_e = interpolation_basis.nodes[i];
        double f_z = interpolation_basis.values[i];
        
        if (interpolation_basis.is_normalized) {
            z_e = z_e * interpolation_basis.x_scale + interpolation_basis.x_center;
        }
        
        std::vector<double> q_zero(metadata.n_free, 0.0);
        double F_at_z = evaluate(z_e, q_zero);
        
        if (std::abs(F_at_z - f_z) > tolerance * std::max(1.0, std::abs(f_z))) {
            return false;
        }
    }
    
    return true;
}

// ============== Decomposer implementation ==============

DecompositionResult Decomposer::decompose(const Parameters& params) {
    DecompositionResult result;
    int n = params.polynomial_degree;
    std::vector<double> nodes;
    std::vector<double> values;
    
    for (const auto& node : params.interp_nodes) {
        nodes.push_back(node.x);
        values.push_back(node.value);
    }
    
    int m = static_cast<int>(nodes.size());
    
    if (!check_degree_condition(n, m)) {
        result.metadata.is_valid = false;
        result.metadata.validation_message =
            "Insufficient polynomial degree: n=" + std::to_string(n) +
            " < m-1=" + std::to_string(m-1) +
            ". Need at least degree " + std::to_string(m-1) + " to interpolate all nodes.";
        return result;
    }
    
    double interval_length = params.interval_end - params.interval_start;
    std::vector<std::pair<int, int>> duplicate_pairs;
    if (!check_unique_nodes(nodes, interval_length, params.epsilon_unique, &duplicate_pairs)) {
        result.metadata.is_valid = false;
        std::ostringstream oss;
        oss << "Duplicate interpolation nodes detected (within tolerance "
            << params.epsilon_unique * interval_length << "):\n";
        for (auto& p : duplicate_pairs) {
            oss << "  nodes[" << p.first << "]=" << nodes[p.first]
                << " and nodes[" << p.second << "]=" << nodes[p.second] << "\n";
        }
        for (auto& p : duplicate_pairs) {
            if (std::abs(values[p.first] - values[p.second]) > 1e-12) {
                oss << "  CONFLICT: f(z) values differ: " << values[p.first]
                    << " vs " << values[p.second] << "\n";
            }
        }
        result.metadata.validation_message = oss.str();
        return result;
    }
    
    std::vector<int> out_of_bounds;
    if (!check_nodes_in_interval(nodes, params.interval_start, params.interval_end,
                                 params.epsilon_bound, &out_of_bounds)) {
        result.metadata.is_valid = false;
        std::ostringstream oss;
        oss << "Interpolation nodes outside interval [" << params.interval_start
            << ", " << params.interval_end << "]:\n";
        for (int idx : out_of_bounds) {
            oss << "  nodes[" << idx << "]=" << nodes[idx] << "\n";
        }
        result.metadata.validation_message = oss.str();
        return result;
    }
    
    if (!check_rank_solvency(nodes, params.epsilon_rank)) {
        result.metadata.is_valid = false;
        result.metadata.validation_message =
            "Linear dependence detected in constraint system. Check for duplicate or nearly duplicate nodes.";
        return result;
    }
    
    result.metadata.is_valid = true;
    result.metadata.n_total = n;
    result.metadata.m_constraints = m;
    result.metadata.m_eff = m;
    result.metadata.n_free = n - m + 1;
    result.metadata.validation_message = "Decomposition successful";
    
    InterpolationBasis::sort_nodes_and_values(nodes, values);
    
    result.weight_multiplier.build_from_roots(
        nodes,
        params.interval_start,
        params.interval_end,
        true
    );
    result.metadata.min_root_distance = result.weight_multiplier.min_root_distance;
    result.metadata.requires_normalization = result.weight_multiplier.is_normalized;
    
    if (result.metadata.requires_normalization) {
        result.metadata.validation_message +=
            "\nInfo: Weight multiplier was normalized (shift=" + std::to_string(result.weight_multiplier.shift) +
            ", scale=" + std::to_string(result.weight_multiplier.scale) + ").";
    }
    
    double scale_W = estimate_weight_multiplier_scale(nodes);
    if (scale_W > 1e150 || scale_W < 1e-150) {
        result.metadata.validation_message +=
            "\nWarning: Weight multiplier has extreme scale: " + std::to_string(scale_W) +
            ". This may cause numerical issues.";
    }
    
    result.interpolation_basis.build(
        nodes, 
        values, 
        InterpolationMethod::BARYCENTRIC,
        params.interval_start,
        params.interval_end,
        true,
        true
    );
    
    double value_range = analyze_value_range(values);
    if (!values.empty() && value_range < 1e-12 * std::max({1.0, std::abs(values[0]), std::abs(values.back())})) {
        result.metadata.validation_message +=
            "\nWarning: Very small range of interpolation values. Task may be degenerate.";
    }
    
    if (m <= 4) {
        int detected_degree = detect_low_degree_polynomial(nodes, values);
        if (detected_degree > 0 && detected_degree < m - 1) {
            result.metadata.validation_message +=
                "\nNote: Interpolation points lie on a polynomial of degree " +
                std::to_string(detected_degree) + ". Consider using simpler basis.";
        }
    }
    
    return result;
}

bool Decomposer::check_rank_solvency(const std::vector<double>& nodes,
                                      double tolerance,
                                      std::vector<int>* conflict_indices) {
    int m = static_cast<int>(nodes.size());
    if (m <= 1) return true;
    
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            if (std::abs(nodes[i] - nodes[j]) < tolerance) {
                if (conflict_indices) {
                    conflict_indices->push_back(i);
                    conflict_indices->push_back(j);
                }
                return false;
            }
        }
    }
    
    return true;
}

bool Decomposer::check_degree_condition(int n, int m) {
    return n >= m - 1;
}

bool Decomposer::check_unique_nodes(const std::vector<double>& nodes,
                                     double interval_length,
                                     double tolerance,
                                     std::vector<std::pair<int, int>>* duplicate_pairs) {
    int m = static_cast<int>(nodes.size());
    double abs_tol = tolerance * interval_length;
    
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int i, int j) { return nodes[i] < nodes[j]; });
    
    bool all_unique = true;
    for (int i = 1; i < m; ++i) {
        if (std::abs(nodes[indices[i]] - nodes[indices[i-1]]) <= abs_tol) {
            all_unique = false;
            if (duplicate_pairs) {
                duplicate_pairs->emplace_back(indices[i-1], indices[i]);
            }
        }
    }
    
    return all_unique;
}

bool Decomposer::check_nodes_in_interval(const std::vector<double>& nodes,
                                           double a, double b,
                                           double tolerance,
                                           std::vector<int>* out_of_bounds) {
    double abs_tol = tolerance * (b - a);
    bool all_inside = true;
    
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (nodes[i] < a - abs_tol || nodes[i] > b + abs_tol) {
            all_inside = false;
            if (out_of_bounds) {
                out_of_bounds->push_back(i);
            }
        }
    }
    
    return all_inside;
}

void Decomposer::prepare_sorted_nodes_and_values(
    const std::vector<InterpolationNode>& input_nodes,
    std::vector<double>* sorted_nodes,
    std::vector<double>* sorted_values) {
    
    sorted_nodes->resize(input_nodes.size());
    sorted_values->resize(input_nodes.size());
    
    for (size_t i = 0; i < input_nodes.size(); ++i) {
        (*sorted_nodes)[i] = input_nodes[i].x;
        (*sorted_values)[i] = input_nodes[i].value;
    }
    
    InterpolationBasis::sort_nodes_and_values(*sorted_nodes, *sorted_values);
}

double Decomposer::estimate_weight_multiplier_scale(const std::vector<double>& roots) {
    double scale = 1.0;
    for (double root : roots) {
        scale *= std::max(std::abs(root), 1.0);
        if (scale > 1e200 || scale < 1e-200) {
            break;
        }
    }
    return scale;
}

double Decomposer::analyze_value_range(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    double vmin = *std::min_element(values.begin(), values.end());
    double vmax = *std::max_element(values.begin(), values.end());
    return vmax - vmin;
}

int Decomposer::detect_low_degree_polynomial(const std::vector<double>& nodes,
                                              const std::vector<double>& values) {
    int m = static_cast<int>(nodes.size());
    if (m <= 2) return m;
    
    std::vector<InterpolationNode> nodes_vec;
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes_vec.emplace_back(nodes[i], values[i]);
    }
    
    if (m == 3) {
        double slope12 = (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double slope23 = (values[2] - values[1]) / (nodes[2] - nodes[1]);
        if (std::abs(slope12 - slope23) < 1e-10) {
            return 1;
        }
        return 2;
    } else if (m == 4) {
        double slope12 = (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double slope23 = (values[2] - values[1]) / (nodes[2] - nodes[1]);
        double slope34 = (values[3] - values[2]) / (nodes[3] - nodes[2]);
        if (std::abs(slope12 - slope23) < 1e-10 && std::abs(slope23 - slope34) < 1e-10) {
            return 1;
        }
        double diff1 = (values[2] - values[1]) / (nodes[2] - nodes[1]) - 
                       (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double diff2 = (values[3] - values[2]) / (nodes[3] - nodes[2]) - 
                       (values[2] - values[1]) / (nodes[2] - nodes[1]);
        double avg_second_diff = (diff1 + diff2) / 2.0;
        if (std::abs(diff1 - avg_second_diff) < 1e-8 && std::abs(diff2 - avg_second_diff) < 1e-8) {
            return 2;
        }
        return 3;
    }
    
    return 0;
}

} // namespace mixed_approx
