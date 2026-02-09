#include "mixed_approximation/edge_case_handler.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace mixed_approx {

// ============== Классификация крайних случаев ==============

EdgeCaseLevel EdgeCaseHandler::classify_case(EdgeCaseType type) {
    switch (type) {
        case EdgeCaseType::OVERCONSTRAINED:
            return EdgeCaseLevel::CRITICAL;
        case EdgeCaseType::ZERO_NODES:
        case EdgeCaseType::FULL_INTERPOLATION:
        case EdgeCaseType::MULTIPLE_ROOTS:
            return EdgeCaseLevel::SPECIAL;
        case EdgeCaseType::CLOSE_NODES:
        case EdgeCaseType::HIGH_DEGREE:
            return EdgeCaseLevel::WARNING;
        case EdgeCaseType::DEGENERATE_DATA:
        case EdgeCaseType::CONSTANT_VALUES:
        case EdgeCaseType::LINEAR_DEPENDENCE:
            return EdgeCaseLevel::WARNING;
        case EdgeCaseType::NUMERICAL_OVERFLOW:
        case EdgeCaseType::GRADIENT_EXPLOSION:
        case EdgeCaseType::OSCILLATION:
            return EdgeCaseLevel::RECOVERABLE;
        default:
            return EdgeCaseLevel::WARNING;
    }
}

// ============== Вспомогательные методы ==============

bool EdgeCaseHandler::is_constant(const std::vector<double>& values, double threshold) {
    if (values.empty()) return true;
    
    double spread = compute_value_spread(values);
    double max_abs = *std::max_element(values.begin(), values.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    
    double reference = std::max(max_abs, 1.0);
    return spread < threshold * reference;
}

double EdgeCaseHandler::compute_value_spread(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    auto min_max = std::minmax_element(values.begin(), values.end());
    return *(min_max.second) - *(min_max.first);
}

std::vector<NodeCluster> EdgeCaseHandler::cluster_close_nodes(const std::vector<double>& coords,
                                                             const std::vector<double>& values,
                                                             double epsilon) {
    std::vector<NodeCluster> clusters;
    
    if (coords.empty()) return clusters;
    
    int n = static_cast<int>(coords.size());
    std::vector<bool> visited(n, false);
    
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        
        NodeCluster cluster;
        cluster.indices.push_back(i);
        visited[i] = true;
        
        double sum_x = coords[i];
        double sum_val = values[i];
        
        for (int j = i + 1; j < n; ++j) {
            if (visited[j]) continue;
            
            double dist = std::abs(coords[j] - coords[i]);
            if (dist < epsilon) {
                cluster.indices.push_back(j);
                visited[j] = true;
                sum_x += coords[j];
                sum_val += values[j];
            }
        }
        
        cluster.center = sum_x / cluster.indices.size();
        cluster.value_center = sum_val / cluster.indices.size();
        
        double max_diff = 0.0;
        for (int idx : cluster.indices) {
            double diff = std::abs(values[idx] - cluster.value_center);
            max_diff = std::max(max_diff, diff);
        }
        cluster.value_spread = max_diff;
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

bool EdgeCaseHandler::should_merge_cluster(const NodeCluster& cluster, double epsilon_value) {
    return cluster.value_spread < epsilon_value * std::max(1.0, std::abs(cluster.value_center));
}

std::vector<std::string> EdgeCaseHandler::generate_high_degree_recommendations(int n) {
    std::vector<std::string> recs;
    
    std::ostringstream oss;
    oss << "WARNING: High polynomial degree (n = " << n << ").\n";
    oss << "         Risk of numerical instability and Runge oscillations.\n";
    recs.push_back(oss.str());
    
    recs.push_back("Recommended alternatives:");
    recs.push_back("  - Use cubic splines (recommended)");
    recs.push_back("  - Use B-splines of degree 3-5");
    recs.push_back("  - Split interval into segments and use local polynomials");
    
    return recs;
}

} // namespace mixed_approx
