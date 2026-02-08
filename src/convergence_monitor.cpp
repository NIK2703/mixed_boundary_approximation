#include "mixed_approximation/convergence_monitor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace mixed_approx {

void ConvergenceMonitor::reset() {
    current_iteration_ = 0;
    oscillation_count_ = 0;
    plateau_count_ = 0;
    barrier_proximity_count_ = 0;
    numerical_anomaly_count_ = 0;
    is_diverging_ = false;
    objective_history_.clear();
    gradient_history_.clear();
}

bool ConvergenceMonitor::is_converged(double gradient_norm, double objective_value,
                                       double objective_change, double step_size) {
    // Gradient norm criterion
    if (gradient_norm < tol_gradient * std::max(1.0, std::abs(objective_value))) {
        return true;
    }
    
    // Objective change criterion
    if (objective_change < tol_objective * std::max(1.0, std::abs(objective_value))) {
        return true;
    }
    
    // Small step criterion
    if (step_size < tol_step) {
        return true;
    }
    
    return false;
}

bool ConvergenceMonitor::detect_oscillation(const std::vector<double>& history) {
    if (history.size() < 4) return false;
    
    size_t n = history.size();
    double diff1 = history[n-1] - history[n-3];
    double diff2 = history[n-2] - history[n-4];
    
    if (diff1 * diff2 < 0) {
        oscillation_count_++;
        return oscillation_count_ >= max_oscillation_count;
    }
    
    oscillation_count_ = 0;
    return false;
}

bool ConvergenceMonitor::detect_plateau(double current_objective) {
    objective_history_.push_back(current_objective);
    
    if (objective_history_.size() < static_cast<size_t>(max_plateau_count + 10)) {
        return false;
    }
    
    size_t start_idx = objective_history_.size() - max_plateau_count - 1;
    double old_value = objective_history_[start_idx];
    double relative_change = std::abs(current_objective - old_value) / std::max(1.0, std::abs(current_objective));
    
    if (relative_change < 1e-10) {
        plateau_count_++;
        return plateau_count_ >= 2;
    }
    
    plateau_count_ = 0;
    return false;
}

bool ConvergenceMonitor::detect_divergence(double current_objective) {
    if (objective_history_.empty()) {
        return false;
    }
    
    double old_value = objective_history_.back();
    if (current_objective > 10.0 * old_value && current_objective > old_value) {
        is_diverging_ = true;
        return true;
    }
    
    return false;
}

std::string ConvergenceMonitor::get_diagnostic_info() const {
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "Convergence Monitor Status:\n"
             "  Iteration: %d\n"
             "  Oscillation count: %d\n"
             "  Plateau count: %d\n"
             "  Barrier proximity count: %d\n"
             "  Numerical anomaly count: %d\n"
             "  Is diverging: %s",
             current_iteration_, oscillation_count_, plateau_count_,
             barrier_proximity_count_, numerical_anomaly_count_,
             is_diverging_ ? "yes" : "no");
    return std::string(buffer);
}

} // namespace mixed_approx
