#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace mixed_approx {

void InterpolationBasis::build(const std::vector<double>& nodes_vec,
                               const std::vector<double>& values_vec,
                               InterpolationMethod meth,
                               double interval_start,
                               double interval_end,
                               bool enable_normalization,
                               bool enable_node_merging) {
    if (nodes_vec.empty()) {
        is_valid = false;
        error_message = "Empty nodes array";
        return;
    }
    
    if (nodes_vec.size() != values_vec.size()) {
        is_valid = false;
        error_message = "Nodes and values size mismatch";
        return;
    }
    
    method = meth;
    x_center = (interval_start + interval_end) / 2.0;
    x_scale = (interval_end - interval_start) / 2.0;
    is_normalized = false;
    weight_scale = 1.0;
    is_valid = false;
    error_message.clear();
    
    nodes = nodes_vec;
    nodes_original = nodes_vec;
    values = values_vec;
    
    if (enable_normalization && x_scale > 0) {
        normalize_nodes(interval_start, interval_end);
        is_normalized = true;
    }
    
    sort_nodes_and_values(nodes, values);
    
    if (enable_node_merging && nodes.size() > 1) {
        double interval_length = interval_end - interval_start;
        auto merged = merge_close_nodes(nodes, values, interval_length);
        
        if (merged.size() < nodes.size()) {
            nodes.clear();
            values.clear();
            for (const auto& mn : merged) {
                nodes.push_back(mn.x);
                values.push_back(mn.value);
            }
        }
    }
    
    m_eff = static_cast<int>(nodes.size());
    
    if (!are_nodes_unique(nodes, 1e-14)) {
        is_valid = false;
        error_message = "Non-unique nodes remain after merging";
        return;
    }
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        compute_barycentric_weights();
        precompute_weighted_values();
        compute_divided_differences();
    } else if (method == InterpolationMethod::NEWTON) {
        compute_divided_differences();
    }
    
    is_valid = true;
}

void InterpolationBasis::normalize_nodes(double a, double b) {
    double center = (a + b) / 2.0;
    double scale = (b - a) / 2.0;
    if (scale == 0) return;
    
    for (double& node : nodes) {
        node = (node - center) / scale;
    }
    x_center = center;
    x_scale = scale;
}

std::vector<InterpolationBasis::MergedNode> InterpolationBasis::merge_close_nodes(
    const std::vector<double>& nodes_input,
    const std::vector<double>& values_input,
    double interval_length) {
    
    std::vector<MergedNode> merged;
    int m = static_cast<int>(nodes_input.size());
    
    double epsilon_close = std::max(1e-12, 1e-4 / m);
    double abs_tol = epsilon_close * interval_length;
    
    for (int i = 0; i < m; ) {
        MergedNode current;
        current.x = nodes_input[i];
        current.value = values_input[i];
        current.count = 1;
        
        int j = i + 1;
        while (j < m) {
            if (std::abs(nodes_input[j] - current.x) < abs_tol) {
                current.value = (current.value * current.count + values_input[j]) / (current.count + 1);
                current.count++;
                j++;
            } else {
                break;
            }
        }
        
        merged.push_back(current);
        i = j;
    }
    
    return merged;
}

bool InterpolationBasis::are_nodes_unique(const std::vector<double>& nodes_vec, double tolerance) {
    if (nodes_vec.empty()) return true;
    
    std::vector<double> sorted = nodes_vec;
    std::sort(sorted.begin(), sorted.end());
    
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (std::abs(sorted[i] - sorted[i-1]) <= tolerance) {
            return false;
        }
    }
    return true;
}

void InterpolationBasis::sort_nodes_and_values(std::vector<double>& nodes_vec,
                                               std::vector<double>& values_vec) {
    if (nodes_vec.size() != values_vec.size() || nodes_vec.empty()) {
        return;
    }
    
    std::vector<size_t> indices(nodes_vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return nodes_vec[i] < nodes_vec[j]; });
    
    std::vector<double> sorted_nodes(nodes_vec.size());
    std::vector<double> sorted_values(values_vec.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_nodes[i] = nodes_vec[indices[i]];
        sorted_values[i] = values_vec[indices[i]];
    }
    
    nodes_vec = sorted_nodes;
    values_vec = sorted_values;
}

bool InterpolationBasis::verify_interpolation(double tolerance) const {
    if (!is_valid) return false;
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        double computed = evaluate(nodes[i]);
        if (std::abs(computed - values[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::string InterpolationBasis::get_info() const {
    std::ostringstream oss;
    oss << "InterpolationBasis info:\n";
    oss << "  m_eff: " << m_eff << "\n";
    oss << "  method: ";
    switch (method) {
        case InterpolationMethod::LAGRANGE: oss << "Lagrange"; break;
        case InterpolationMethod::BARYCENTRIC: oss << "Barycentric"; break;
        case InterpolationMethod::NEWTON: oss << "Newton"; break;
    }
    oss << "\n";
    oss << "  normalized: " << (is_normalized ? "yes" : "no") << "\n";
    if (is_normalized) {
        oss << "  x_center: " << x_center << ", x_scale: " << x_scale << "\n";
    }
    oss << "  weight_scale: " << weight_scale << "\n";
    oss << "  valid: " << (is_valid ? "yes" : "no") << "\n";
    if (!error_message.empty()) {
        oss << "  error: " << error_message << "\n";
    }
    return oss.str();
}

bool InterpolationBasis::detect_equally_spaced_nodes(double tolerance) const {
    if (nodes.size() < 2) return false;
    
    double step = nodes[1] - nodes[0];
    for (size_t i = 2; i < nodes.size(); ++i) {
        if (std::abs((nodes[i] - nodes[i-1]) - step) > tolerance) {
            return false;
        }
    }
    return true;
}

bool InterpolationBasis::detect_chebyshev_nodes(double tolerance) const {
    if (nodes.size() < 2) return false;
    int m = static_cast<int>(nodes.size());
    
    for (int k = 0; k < m; ++k) {
        double expected = std::cos(M_PI * (2.0*k + 1.0) / (2.0 * m));
        if (std::abs(nodes[k] - expected) > tolerance) {
            return false;
        }
    }
    return true;
}

} // namespace mixed_approx
