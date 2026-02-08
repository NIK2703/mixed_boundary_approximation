#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace mixed_approx {

// ============== OptimizationProblemData ==============

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

// ============== OptimizationCache ==============

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

// ============== ConvergenceMonitor ==============

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

// ============== ObjectiveFunctor ==============

ObjectiveFunctor::ObjectiveFunctor(const CompositePolynomial& param,
                                   const OptimizationProblemData& data)
    : param_(param)
    , problem_data_(data)
    , caches_built_(false)
{
}

bool ObjectiveFunctor::is_valid() const {
    return param_.is_valid() && problem_data_.is_valid();
}

void ObjectiveFunctor::build_caches() {
    if (caches_built_) return;
    
    int n_free = param_.num_free_parameters();
    cache_.clear();
    cache_.stiffness_dim = n_free;
    
    // Precompute for approximation points
    if (problem_data_.num_approx_points() > 0) {
        cache_.P_at_x.resize(problem_data_.num_approx_points());
        cache_.W_at_x.resize(problem_data_.num_approx_points());
        
        for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
            double x = problem_data_.approx_x[i];
            cache_.P_at_x[i] = param_.interpolation_basis.evaluate(x);
            cache_.W_at_x[i] = param_.weight_multiplier.evaluate(x);
        }
        
        cache_.data_cache_valid = true;
    }
    
    // Precompute for repulsion points
    if (problem_data_.num_repel_points() > 0) {
        cache_.P_at_y.resize(problem_data_.num_repel_points());
        cache_.W_at_y.resize(problem_data_.num_repel_points());
        
        for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
            double y = problem_data_.repel_y[j];
            cache_.P_at_y[j] = param_.interpolation_basis.evaluate(y);
            cache_.W_at_y[j] = param_.weight_multiplier.evaluate(y);
        }
        
        cache_.data_cache_valid = true;
    }
    
    // Build basis cache
    update_basis_cache(n_free);
    
    caches_built_ = true;
}

void ObjectiveFunctor::update_basis_cache(int n_free) {
    if (n_free <= 0) return;
    
    BasisType basis_type = param_.correction_poly.get_basis_type();
    
    if (basis_type == BasisType::MONOMIAL) {
        // Cache for approximation points
        if (problem_data_.num_approx_points() > 0) {
            cache_.phi_at_x.resize(problem_data_.num_approx_points());
            for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
                cache_.phi_at_x[i].resize(n_free);
                double x = problem_data_.approx_x[i];
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    cache_.phi_at_x[i][k] = power;
                    power *= x;
                }
            }
        }
        
        // Cache for repulsion points
        if (problem_data_.num_repel_points() > 0) {
            cache_.phi_at_y.resize(problem_data_.num_repel_points());
            for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
                cache_.phi_at_y[j].resize(n_free);
                double y = problem_data_.repel_y[j];
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    cache_.phi_at_y[j][k] = power;
                    power *= y;
                }
            }
        }
    }
    
    cache_.basis_cache_valid = true;
}

double ObjectiveFunctor::value(const std::vector<double>& q) const {
    double J = 0.0;
    
    // Approximation term
    if (problem_data_.num_approx_points() > 0) {
        J += compute_approx_term(q);
    }
    
    // Repulsion term
    if (problem_data_.num_repel_points() > 0) {
        J += compute_repel_term(q);
    }
    
    // Regularization term
    if (problem_data_.gamma > 0) {
        J += compute_reg_term(q);
    }
    
    return J;
}

void ObjectiveFunctor::gradient(const std::vector<double>& q, std::vector<double>& grad) const {
    int n_free = param_.num_free_parameters();
    grad.assign(n_free, 0.0);
    
    // Approximation gradient
    if (problem_data_.num_approx_points() > 0) {
        compute_approx_gradient(q, grad);
    }
    
    // Repulsion gradient
    if (problem_data_.num_repel_points() > 0) {
        compute_repel_gradient(q, grad);
    }
    
    // Regularization gradient
    if (problem_data_.gamma > 0) {
        compute_reg_gradient(q, grad);
    }
}

void ObjectiveFunctor::value_and_gradient(const std::vector<double>& q,
                                           double& f,
                                           std::vector<double>& grad) const {
    int n_free = param_.num_free_parameters();
    
    // Compute F(x) at all points
    std::vector<double> F_at_x;
    if (problem_data_.num_approx_points() > 0) {
        F_at_x.resize(problem_data_.num_approx_points());
        for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
            double phi_q = 0.0;
            if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    phi_q += q[k] * power;
                    power *= problem_data_.approx_x[i];
                }
            }
            F_at_x[i] = cache_.P_at_x[i] + phi_q * cache_.W_at_x[i];
        }
    }
    
    std::vector<double> F_at_y;
    if (problem_data_.num_repel_points() > 0) {
        F_at_y.resize(problem_data_.num_repel_points());
        for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
            double phi_q = 0.0;
            if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    phi_q += q[k] * power;
                    power *= problem_data_.repel_y[j];
                }
            }
            F_at_y[j] = cache_.P_at_y[j] + phi_q * cache_.W_at_y[j];
        }
    }
    
    // Compute objective value
    f = 0.0;
    
    if (problem_data_.num_approx_points() > 0) {
        for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
            double residual = F_at_x[i] - problem_data_.approx_f[i];
            f += problem_data_.approx_weight[i] * residual * residual;
        }
    }
    
    if (problem_data_.num_repel_points() > 0) {
        for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
            double distance = std::abs(problem_data_.repel_forbidden[j] - F_at_y[j]);
            double safe_distance = std::max(distance, problem_data_.epsilon);
            f += problem_data_.repel_weight[j] / (safe_distance * safe_distance);
        }
    }
    
    if (problem_data_.gamma > 0) {
        double reg = 0.0;
        for (int k = 0; k < n_free; ++k) {
            for (int l = 0; l < n_free; ++l) {
                reg += q[k] * q[l] * param_.correction_poly.get_stiffness_element(k, l);
            }
        }
        f += problem_data_.gamma * reg;
    }
    
    // Compute gradient
    grad.assign(n_free, 0.0);
    
    if (problem_data_.num_approx_points() > 0) {
        for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
            double residual = F_at_x[i] - problem_data_.approx_f[i];
            double w = problem_data_.approx_weight[i] * cache_.W_at_x[i];
            
            if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    grad[k] += 2.0 * residual * w * power;
                    power *= problem_data_.approx_x[i];
                }
            }
        }
    }
    
    if (problem_data_.num_repel_points() > 0) {
        for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
            double distance = std::abs(problem_data_.repel_forbidden[j] - F_at_y[j]);
            double safe_distance = std::max(distance, problem_data_.epsilon);
            double sign = (problem_data_.repel_forbidden[j] - F_at_y[j]) >= 0 ? 1.0 : -1.0;
            double factor = 2.0 * problem_data_.repel_weight[j] / (safe_distance * safe_distance * safe_distance);
            factor *= sign;
            double w = cache_.W_at_y[j];
            
            if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    grad[k] -= factor * w * power;
                    power *= problem_data_.repel_y[j];
                }
            }
        }
    }
    
    if (problem_data_.gamma > 0) {
        for (int k = 0; k < n_free; ++k) {
            double sum = 0.0;
            for (int l = 0; l < n_free; ++l) {
                sum += q[l] * param_.correction_poly.get_stiffness_element(k, l);
            }
            grad[k] += 2.0 * problem_data_.gamma * sum;
        }
    }
}

ObjectiveFunctor::Components ObjectiveFunctor::compute_components(const std::vector<double>& q) const {
    Components comp;
    comp.approx = 0.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    if (problem_data_.num_approx_points() > 0) {
        comp.approx = compute_approx_term(q);
    }
    
    if (problem_data_.num_repel_points() > 0) {
        comp.repel = compute_repel_term(q);
    }
    
    if (problem_data_.gamma > 0) {
        comp.reg = compute_reg_term(q);
    }
    
    comp.total = comp.approx + comp.repel + comp.reg;
    return comp;
}

// ============== Functional components ==============

double ObjectiveFunctor::compute_approx_term(const std::vector<double>& q) const {
    double J = 0.0;
    int n_free = param_.num_free_parameters();
    
    for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
        double phi_q = 0.0;
        
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += q[k] * power;
                power *= problem_data_.approx_x[i];
            }
        }
        
        double F_x = cache_.P_at_x[i] + phi_q * cache_.W_at_x[i];
        double residual = F_x - problem_data_.approx_f[i];
        
        // Overflow protection
        double square = residual * residual;
        if (std::isfinite(square)) {
            J += problem_data_.approx_weight[i] * square;
        }
    }
    
    return J;
}

double ObjectiveFunctor::compute_repel_term(const std::vector<double>& q) const {
    double J = 0.0;
    int n_free = param_.num_free_parameters();
    
    for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
        double phi_q = 0.0;
        
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += q[k] * power;
                power *= problem_data_.repel_y[j];
            }
        }
        
        double F_y = cache_.P_at_y[j] + phi_q * cache_.W_at_y[j];
        double distance = std::abs(problem_data_.repel_forbidden[j] - F_y);
        double safe_distance = std::max(distance, problem_data_.epsilon);
        
        // Explosion protection
        double term = problem_data_.repel_weight[j] / (safe_distance * safe_distance);
        if (std::isfinite(term)) {
            J += term;
        } else {
            J += std::numeric_limits<double>::max();
        }
    }
    
    return J;
}

double ObjectiveFunctor::compute_reg_term(const std::vector<double>& q) const {
    if (problem_data_.gamma <= 0) return 0.0;
    
    int n_free = param_.num_free_parameters();
    double J = 0.0;
    
    // J_reg = gamma * sum_k sum_l q_k * K[k][l] * q_l
    for (int k = 0; k < n_free; ++k) {
        for (int l = 0; l < n_free; ++l) {
            double K = param_.correction_poly.get_stiffness_element(k, l);
            J += problem_data_.gamma * q[k] * K * q[l];
        }
    }
    
    return J;
}

// ============== Gradients ==============

void ObjectiveFunctor::compute_approx_gradient(const std::vector<double>& q,
                                                std::vector<double>& grad) const {
    int n_free = param_.num_free_parameters();
    
    for (size_t i = 0; i < problem_data_.num_approx_points(); ++i) {
        double phi_q = 0.0;
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += q[k] * power;
                power *= problem_data_.approx_x[i];
            }
        }
        
        double F_x = cache_.P_at_x[i] + phi_q * cache_.W_at_x[i];
        double residual = F_x - problem_data_.approx_f[i];
        double factor = 2.0 * problem_data_.approx_weight[i] * residual;
        double w = cache_.W_at_x[i];
        
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                grad[k] += factor * w * power;
                power *= problem_data_.approx_x[i];
            }
        }
    }
}

void ObjectiveFunctor::compute_repel_gradient(const std::vector<double>& q,
                                              std::vector<double>& grad) const {
    int n_free = param_.num_free_parameters();
    
    for (size_t j = 0; j < problem_data_.num_repel_points(); ++j) {
        double phi_q = 0.0;
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += q[k] * power;
                power *= problem_data_.repel_y[j];
            }
        }
        
        double F_y = cache_.P_at_y[j] + phi_q * cache_.W_at_y[j];
        double distance = std::abs(problem_data_.repel_forbidden[j] - F_y);
        double safe_distance = std::max(distance, problem_data_.epsilon);
        
        double sign = (problem_data_.repel_forbidden[j] - F_y) >= 0 ? 1.0 : -1.0;
        double factor = -2.0 * problem_data_.repel_weight[j] / (safe_distance * safe_distance * safe_distance);
        
        // Factor clamping for explosion protection
        factor = std::copysign(std::min(std::abs(factor), 1e8), factor);
        
        double w = cache_.W_at_y[j];
        
        if (param_.correction_poly.get_basis_type() == BasisType::MONOMIAL) {
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                grad[k] += factor * w * power;
                power *= problem_data_.repel_y[j];
            }
        }
    }
}

void ObjectiveFunctor::compute_reg_gradient(const std::vector<double>& q,
                                              std::vector<double>& grad) const {
    if (problem_data_.gamma <= 0) return;
    
    int n_free = param_.num_free_parameters();
    
    for (int k = 0; k < n_free; ++k) {
        double sum = 0.0;
        for (int l = 0; l < n_free; ++l) {
            double K = param_.correction_poly.get_stiffness_element(k, l);
            sum += q[l] * K;
        }
        grad[k] += 2.0 * problem_data_.gamma * sum;
    }
}

// ============== Numerical anomaly protection ==============

double ObjectiveFunctor::safe_barrier_distance(double poly_value, double forbidden_value) const {
    double distance = std::abs(forbidden_value - poly_value);
    return std::max(distance, problem_data_.epsilon);
}

bool ObjectiveFunctor::has_numerical_anomaly(double value) const {
    return !std::isfinite(value) || std::abs(value) > 1e150;
}

// ============== InitializationStrategySelector ==============

InitializationStrategy InitializationStrategySelector::select(
    const CompositePolynomial& param,
    const OptimizationProblemData& data)
{
    if (data.num_approx_points() == 0) {
        return InitializationStrategy::ZERO;
    }
    
    if (data.num_repel_points() == 0) {
        return InitializationStrategy::LEAST_SQUARES;
    }
    
    for (double w : data.repel_weight) {
        if (w > 100.0) {
            return InitializationStrategy::ZERO;
        }
    }
    
    return InitializationStrategy::LEAST_SQUARES;
}

InitializationResult InitializationStrategySelector::initialize(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationStrategy strategy = select(param, data);
    
    switch (strategy) {
        case InitializationStrategy::ZERO:
            return zero_initialization(param);
        case InitializationStrategy::LEAST_SQUARES:
            return least_squares_initialization(param, data, functor);
        case InitializationStrategy::MULTI_START:
            return multi_start_initialization(param, data, functor);
        default:
            return zero_initialization(param);
    }
}

InitializationResult InitializationStrategySelector::zero_initialization(
    const CompositePolynomial& param)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::ZERO;
    
    int n_free = param.num_free_parameters();
    result.initial_coeffs.assign(n_free, 0.0);
    
    result.success = true;
    result.message = "Zero initialization: all coefficients set to zero";
    
    return result;
}

InitializationResult InitializationStrategySelector::least_squares_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::LEAST_SQUARES;
    
    int n_free = param.num_free_parameters();
    
    size_t N = data.num_approx_points();
    if (N == 0) {
        return zero_initialization(param);
    }
    
    std::vector<std::vector<double>> A(N, std::vector<double>(n_free, 0.0));
    
    for (size_t i = 0; i < N; ++i) {
        double x = data.approx_x[i];
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            A[i][k] = power * functor.parameterization().weight_multiplier.evaluate(x);
            power *= x;
        }
    }
    
    std::vector<double> b(N);
    for (size_t i = 0; i < N; ++i) {
        double F_base = functor.parameterization().interpolation_basis.evaluate(data.approx_x[i]);
        b[i] = data.approx_weight[i] * (data.approx_f[i] - F_base);
    }
    
    std::vector<std::vector<double>> ATA(n_free, std::vector<double>(n_free, 0.0));
    for (size_t i = 0; i < N; ++i) {
        double w = data.approx_weight[i];
        for (int k = 0; k < n_free; ++k) {
            for (int l = 0; l < n_free; ++l) {
                ATA[k][l] += w * A[i][k] * A[i][l];
            }
        }
    }
    
    std::vector<double> ATb(n_free, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (int k = 0; k < n_free; ++k) {
            ATb[k] += A[i][k] * b[i];
        }
    }
    
    double lambda = 1e-8;
    for (int k = 0; k < n_free; ++k) {
        ATA[k][k] += lambda;
    }
    
    result.initial_coeffs = ATb;
    for (int k = 0; k < n_free; ++k) {
        int max_row = k;
        for (int i = k + 1; i < n_free; ++i) {
            if (std::abs(ATA[i][k]) > std::abs(ATA[max_row][k])) {
                max_row = i;
            }
        }
        
        if (std::abs(ATA[max_row][k]) < 1e-15) {
            result.initial_coeffs[k] = 0.0;
            continue;
        }
        
        if (max_row != k) {
            std::swap(ATA[k], ATA[max_row]);
            std::swap(result.initial_coeffs[k], result.initial_coeffs[max_row]);
        }
        
        double piv = ATA[k][k];
        for (int j = k; j < n_free; ++j) {
            ATA[k][j] /= piv;
        }
        result.initial_coeffs[k] /= piv;
        
        for (int i = k + 1; i < n_free; ++i) {
            double factor = ATA[i][k];
            if (std::abs(factor) > 0) {
                for (int j = k; j < n_free; ++j) {
                    ATA[i][j] -= factor * ATA[k][j];
                }
                result.initial_coeffs[i] -= factor * result.initial_coeffs[k];
            }
        }
    }
    
    for (int k = n_free - 2; k >= 0; --k) {
        for (int i = k + 1; i < n_free; ++i) {
            result.initial_coeffs[k] -= ATA[k][i] * result.initial_coeffs[i];
        }
    }
    
    // Barrier protection
    if (data.num_repel_points() > 0) {
        functor.build_caches();
        double min_distance = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double phi_q = 0.0;
            double y = data.repel_y[j];
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += result.initial_coeffs[k] * power;
                power *= y;
            }
            double F_y = functor.parameterization().interpolation_basis.evaluate(y) + 
                         phi_q * functor.parameterization().weight_multiplier.evaluate(y);
            double distance = std::abs(data.repel_forbidden[j] - F_y);
            min_distance = std::min(min_distance, distance);
        }
        
        double d_safe = 0.01;
        if (min_distance < d_safe) {
            double scale = (d_safe / min_distance) * 0.5;
            for (int k = 0; k < n_free; ++k) {
                result.initial_coeffs[k] *= scale;
            }
        }
    }
    
    functor.build_caches();
    result.initial_objective = functor.value(result.initial_coeffs);
    result.success = true;
    result.message = "Least squares initialization with barrier protection";
    
    return result;
}

InitializationResult InitializationStrategySelector::multi_start_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::MULTI_START;
    
    int n_free = param.num_free_parameters();
    if (n_free > 10) {
        return least_squares_initialization(param, data, functor);
    }
    
    functor.build_caches();
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> best_coeffs(n_free, 0.0);
    double best_objective = functor.value(best_coeffs);
    
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<double> trial_coeffs(n_free);
        for (int k = 0; k < n_free; ++k) {
            trial_coeffs[k] = dist(rng);
        }
        
        std::vector<double> coeffs = trial_coeffs;
        double step = 0.01;
        
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<double> grad;
            double obj_value;
            functor.value_and_gradient(coeffs, obj_value, grad);
            
            for (int k = 0; k < n_free; ++k) {
                coeffs[k] -= step * grad[k];
            }
        }
        
        double objective = functor.value(coeffs);
        if (objective < best_objective) {
            best_objective = objective;
            best_coeffs = coeffs;
        }
    }
    
    result.initial_coeffs = best_coeffs;
    result.initial_objective = best_objective;
    result.success = true;
    result.message = "Multi-start initialization: best of 10 trials";
    
    return result;
}

// ============== OptimizationPostProcessor ==============

OptimizationPostProcessor::OptimizationPostProcessor(
    const CompositePolynomial& param,
    const OptimizationProblemData& data)
    : param_(param)
    , data_(data)
{
}

PostOptimizationReport OptimizationPostProcessor::generate_report(
    const std::vector<double>& final_coeffs,
    double final_objective)
{
    PostOptimizationReport report;
    
    // Check interpolation conditions
    report.max_interpolation_error = 0.0;
    for (size_t e = 0; e < data_.num_interp_nodes(); ++e) {
        double F_z = param_.evaluate(data_.interp_z[e]);
        double error = std::abs(F_z - data_.interp_f[e]);
        report.max_interpolation_error = std::max(report.max_interpolation_error, error);
    }
    report.interpolation_satisfied = report.max_interpolation_error < 1e-10;
    
    // Check barrier safety
    report.min_barrier_distance = std::numeric_limits<double>::max();
    for (size_t j = 0; j < data_.num_repel_points(); ++j) {
        double F_y = param_.evaluate(data_.repel_y[j]);
        double distance = std::abs(data_.repel_forbidden[j] - F_y);
        report.min_barrier_distance = std::min(report.min_barrier_distance, distance);
    }
    report.barrier_constraints_satisfied = report.min_barrier_distance > data_.epsilon * 10;
    
    // Compute components
    ObjectiveFunctor functor(param_, data_);
    functor.build_caches();
    auto comps = functor.compute_components(final_coeffs);
    
    report.approx_percentage = comps.total > 0 ? 100.0 * comps.approx / comps.total : 0.0;
    report.repel_percentage = comps.total > 0 ? 100.0 * comps.repel / comps.total : 0.0;
    report.reg_percentage = comps.total > 0 ? 100.0 * comps.reg / comps.total : 0.0;
    
    // Generate recommendations
    if (report.repel_percentage > 90.0) {
        report.recommendations.push_back(
            "Strong barriers (J_repel >> J_approx). "
            "Consider decreasing B_j by 10x and re-running optimization.");
    }
    if (report.approx_percentage > 90.0 && !report.barrier_constraints_satisfied) {
        report.recommendations.push_back(
            "Barriers ineffective, violations detected. "
            "Consider increasing B_j by 10x.");
    }
    if (report.reg_percentage > 90.0) {
        report.recommendations.push_back(
            "Excessive regularization (J_reg >> J_approx + J_repel). "
            "Consider decreasing gamma by 5x.");
    }
    if (report.interpolation_satisfied && report.barrier_constraints_satisfied) {
        report.recommendations.push_back(
            "Good component balance, parameters are optimal.");
    }
    
    return report;
}

std::string OptimizationPostProcessor::generate_text_report(
    const PostOptimizationReport& report)
{
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
             "OPTIMIZATION REPORT\n"
             "==================\n"
             "Solution Quality:\n"
             "  Max interpolation error: %.2e %s\n"
             "  Min barrier distance: %.2e %s\n"
             "\n"
             "Component Balance:\n"
             "  Approximation: %.1f%%\n"
             "  Repulsion: %.1f%%\n"
             "  Regularization: %.1f%%\n",
             report.max_interpolation_error,
             report.interpolation_satisfied ? "[OK]" : "[FAIL]",
             report.min_barrier_distance,
             report.barrier_constraints_satisfied ? "[OK]" : "[FAIL]",
             report.approx_percentage,
             report.repel_percentage,
             report.reg_percentage);
    
    std::string result(buffer);
    
    if (!report.recommendations.empty()) {
        result += "\nRecommendations:\n";
        for (const auto& rec : report.recommendations) {
            result += "  - " + rec + "\n";
        }
    }
    
    return result;
}

void OptimizationPostProcessor::suggest_parameter_corrections(
    const PostOptimizationReport& report,
    ApproximationConfig& config)
{
    if (report.repel_percentage > 100.0 && config.repel_points.size() > 0) {
        for (auto& p : config.repel_points) {
            p.weight /= 10.0;
        }
    }
    
    if (report.approx_percentage > 100.0 && !report.barrier_constraints_satisfied) {
        for (auto& p : config.repel_points) {
            p.weight *= 10.0;
        }
    }
    
    if (report.reg_percentage > 10.0 * (report.approx_percentage + report.repel_percentage)) {
        config.gamma /= 5.0;
    }
}

} // namespace mixed_approx
