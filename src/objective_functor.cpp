#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace mixed_approx {

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

} // namespace mixed_approx
