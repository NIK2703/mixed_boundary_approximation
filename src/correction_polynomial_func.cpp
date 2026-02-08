#include "mixed_approximation/correction_polynomial.h"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace mixed_approx {

double CorrectionPolynomial::evaluate_Q_with_coeffs(double x, const std::vector<double>& q) const {
    if (q.empty()) {
        return 0.0;
    }
    
    double result = 0.0;
    double x_work = x;
    
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    
    if (basis_type == BasisType::MONOMIAL) {
        double power = 1.0;
        for (int k = 0; k < static_cast<int>(q.size()); ++k) {
            result += q[k] * power;
            power *= x_work;
        }
    } else {
        int deg = static_cast<int>(q.size()) - 1;
        if (deg >= 0) {
            double T_prev = 1.0;
            result += q[0] * T_prev;
            if (deg >= 1) {
                double T_curr = x_work;
                result += q[1] * T_curr;
                for (int k = 2; k <= deg; ++k) {
                    double T_next = 2.0 * x_work * T_curr - T_prev;
                    result += q[k] * T_next;
                    T_prev = T_curr;
                    T_curr = T_next;
                }
            }
        }
    }
    
    return result;
}

double CorrectionPolynomial::evaluate_Q_derivative_with_coeffs(double x, const std::vector<double>& q, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("CorrectionPolynomial::evaluate_Q_derivative_with_coeffs: order must be 1 or 2");
    }
    if (q.empty()) {
        return 0.0;
    }
    
    double x_work = x;
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    
    double result = 0.0;
    for (int k = 0; k < static_cast<int>(q.size()); ++k) {
        double phi_k_deriv = compute_basis_derivative_with_coeffs(x_work, q, k, order);
        result += q[k] * phi_k_deriv;
    }
    
    if (basis_type == BasisType::CHEBYSHEV && x_scale != 0.0) {
        result /= std::pow(x_scale, order);
    }
    
    return result;
}

double CorrectionPolynomial::compute_basis_function_with_coeffs(double x, const std::vector<double>& q, int k) const {
    double x_work = x;
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    return compute_basis_function(x_work, k);
}

double CorrectionPolynomial::compute_basis_derivative_with_coeffs(double x, const std::vector<double>& q, int k, int order) const {
    double x_work = x;
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    return compute_basis_derivative(x_work, k, order);
}

} // namespace mixed_approx
