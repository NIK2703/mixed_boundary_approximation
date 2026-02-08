#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/gauss_quadrature.h"
#include <sstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

namespace {
    const double EPSILON_ROOT = 1e-12;
    const double MAX_ANALYTIC_DEGREE = 15;
    const double REGULARIZATION_THRESHOLD = 1e-100;
}  // anonymous namespace

void CompositePolynomial::build(const InterpolationBasis& basis,
                               const WeightMultiplier& W,
                               const CorrectionPolynomial& Q,
                               double interval_start,
                               double interval_end,
                               EvaluationStrategy strategy) {
    interpolation_basis = basis;
    weight_multiplier = W;
    correction_poly = Q;
    
    interval_a = interval_start;
    interval_b = interval_end;
    eval_strategy = strategy;
    
    total_degree = correction_poly.degree + weight_multiplier.degree();
    num_constraints = static_cast<int>(weight_multiplier.roots.size());
    num_free_params = correction_poly.n_free;
    
    if (total_degree < 0 || num_constraints < 0) {
        validation_message = "Invalid polynomial structure: negative degree or constraints";
        return;
    }
    
    if (num_constraints > total_degree + 1) {
        validation_message = "Too many constraints: m = " + std::to_string(num_constraints) +
                            " > n + 1 = " + std::to_string(total_degree + 1);
        return;
    }
    
    if (!interpolation_basis.is_valid) {
        validation_message = "Interpolation basis is not valid: " + interpolation_basis.error_message;
        return;
    }
    
    clear_caches();
    analytic_coeffs.clear();
    analytic_coeffs_valid = false;
    
    validation_message = "CompositePolynomial built successfully";
}

double CompositePolynomial::evaluate(double x) const {
    double p_int_val = interpolation_basis.evaluate(x);
    
    if (num_constraints == 0) {
        return p_int_val + correction_poly.evaluate_Q(x);
    }
    
    if (num_constraints == total_degree + 1) {
        return p_int_val;
    }
    
    double q_val = correction_poly.evaluate_Q(x);
    double w_val = weight_multiplier.evaluate(x);
    
    return p_int_val + q_val * w_val;
}

double CompositePolynomial::evaluate_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("CompositePolynomial::evaluate_derivative: order must be 1 or 2");
    }
    
    double p_int_val = interpolation_basis.evaluate_derivative(x, order);
    
    if (num_constraints == 0) {
        if (order == 1) {
            return p_int_val + correction_poly.evaluate_Q_derivative(x, 1);
        } else {
            return p_int_val + correction_poly.evaluate_Q_derivative(x, 2);
        }
    }
    
    if (num_constraints == total_degree + 1) {
        return p_int_val;
    }
    
    double q_val = correction_poly.evaluate_Q(x);
    double q1_val = correction_poly.evaluate_Q_derivative(x, 1);
    double q2_val = (order == 2) ? correction_poly.evaluate_Q_derivative(x, 2) : 0.0;
    
    double w_val = weight_multiplier.evaluate(x);
    double w1_val = weight_multiplier.evaluate_derivative(x, 1);
    double w2_val = (order == 2) ? weight_multiplier.evaluate_derivative(x, 2) : 0.0;
    
    if (order == 1) {
        double result = p_int_val + q1_val * w_val + q_val * w1_val;
        return result;
    } else {
        double result = p_int_val + q2_val * w_val + 2.0 * q1_val * w1_val + q_val * w2_val;
        return result;
    }
}

void CompositePolynomial::evaluate_batch(const std::vector<double>& points, 
                                          std::vector<double>& results) const {
    results.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        results[i] = evaluate(points[i]);
    }
}

bool CompositePolynomial::is_valid() const {
    if (total_degree < 0 || num_constraints < 0) {
        return false;
    }
    
    if (num_constraints > total_degree + 1) {
        return false;
    }
    
    if (!interpolation_basis.is_valid) {
        return false;
    }
    
    if (!correction_poly.is_initialized) {
        return false;
    }
    
    return true;
}

double CompositePolynomial::transform_quadrature_node(double t) const {
    return 0.5 * (interval_b - interval_a) * t + 0.5 * (interval_a + interval_b);
}

std::string CompositePolynomial::get_diagnostic_info() const {
    std::ostringstream oss;
    oss << "CompositePolynomial diagnostic:\n";
    oss << "  degree: " << total_degree << "\n";
    oss << "  constraints (m): " << num_constraints << "\n";
    oss << "  free params (n-m+1): " << num_free_params << "\n";
    oss << "  interval: [" << interval_a << ", " << interval_b << "]\n";
    oss << "  eval_strategy: ";
    switch (eval_strategy) {
        case EvaluationStrategy::LAZY: oss << "LAZY"; break;
        case EvaluationStrategy::ANALYTIC: oss << "ANALYTIC"; break;
        case EvaluationStrategy::HYBRID: oss << "HYBRID"; break;
    }
    oss << "\n";
    oss << "  analytic_coeffs: " << (analytic_coeffs_valid ? "valid" : "not built") << "\n";
    oss << "  caches: " << (caches_built ? "built" : "not built") << "\n";
    oss << "  message: " << validation_message << "\n";
    
    if (analytic_coeffs_valid && !analytic_coeffs.empty()) {
        oss << "  coeffs: [";
        int max_show = std::min(10, static_cast<int>(analytic_coeffs.size()));
        for (int i = 0; i < max_show; ++i) {
            oss << analytic_coeffs[i];
            if (i < max_show - 1) oss << ", ";
        }
        if (static_cast<int>(analytic_coeffs.size()) > max_show) {
            oss << ", ...";
        }
        oss << "]\n";
    }
    
    return oss.str();
}

bool CompositePolynomial::verify_assembly(double tolerance) {
    if (num_constraints > 0) {
        for (double root : weight_multiplier.roots) {
            double W_val = weight_multiplier.evaluate(root);
            if (std::abs(W_val) > tolerance) {
                validation_message = "W(z_e) not close to zero at z_e = " + std::to_string(root);
                return false;
            }
        }
        
        for (double z_e : weight_multiplier.roots) {
            double F_val = evaluate(z_e);
            double P_int_val = interpolation_basis.evaluate(z_e);
            double abs_tol = tolerance * std::max(1.0, std::abs(P_int_val));
            if (std::abs(F_val - P_int_val) > abs_tol) {
                validation_message = "Interpolation condition failed at z_e = " + std::to_string(z_e) + 
                                    ": F=" + std::to_string(F_val) + ", P_int=" + std::to_string(P_int_val);
                return false;
            }
        }
    }
    
    if (analytic_coeffs_valid && !analytic_coeffs.empty()) {
        std::vector<double> test_points = {
            interval_a, 
            (interval_a + interval_b) * 0.5, 
            interval_b,
            (interval_a + interval_b) * 0.25,
            (interval_a + interval_b) * 0.75
        };
        
        for (double x : test_points) {
            double lazy_val = evaluate(x);
            double analytic_val = evaluate_analytic(x);
            double rel_diff = std::abs(lazy_val - analytic_val) / 
                             (std::abs(lazy_val) + std::numeric_limits<double>::epsilon());
            
            if (rel_diff > 1e-8) {
                validation_message = "Discrepancy between lazy and analytic evaluation at x = " +
                                    std::to_string(x);
                return false;
            }
        }
    }
    
    validation_message = "Assembly verification passed";
    return true;
}

bool CompositePolynomial::verify_representations_consistency(int num_test_points,
                                                            double relative_tolerance) const {
    if (!analytic_coeffs_valid) {
        return true;
    }
    
    std::vector<double> test_points;
    double step = (interval_b - interval_a) / (num_test_points + 1);
    for (int i = 1; i <= num_test_points; ++i) {
        test_points.push_back(interval_a + i * step);
    }
    
    test_points.push_back(interval_a);
    test_points.push_back(interval_b);
    
    for (double x : test_points) {
        double lazy_val = evaluate(x);
        double analytic_val = evaluate_analytic(x);
        
        double max_abs = std::max(std::abs(lazy_val), std::abs(analytic_val));
        if (max_abs < std::numeric_limits<double>::epsilon()) {
            max_abs = 1.0;
        }
        
        double rel_diff = std::abs(lazy_val - analytic_val) / max_abs;
        
        if (rel_diff > relative_tolerance) {
            return false;
        }
    }
    
    return true;
}

double compute_regularization_via_components(const CompositePolynomial& F, double gamma) {
    return F.compute_regularization_term(gamma);
}

} // namespace mixed_approx
