#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace mixed_approx {

double CorrectionPolynomial::evaluate_Q(double x) const {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    
    double result = 0.0;
    double x_work = x;
    
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    
    if (basis_type == BasisType::MONOMIAL) {
        double power = 1.0;
        for (int k = 0; k <= degree; ++k) {
            result += coeffs[k] * power;
            power *= x_work;
        }
    } else {
        if (degree >= 0) {
            double T_prev = 1.0;
            result += coeffs[0] * T_prev;
            if (degree >= 1) {
                double T_curr = x_work;
                result += coeffs[1] * T_curr;
                for (int k = 2; k <= degree; ++k) {
                    double T_next = 2.0 * x_work * T_curr - T_prev;
                    result += coeffs[k] * T_next;
                    T_prev = T_curr;
                    T_curr = T_next;
                }
            }
        }
    }
    
    return result;
}

double CorrectionPolynomial::evaluate_Q_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("CorrectionPolynomial::evaluate_Q_derivative: order must be 1 or 2");
    }
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    
    double x_work = x;
    if (basis_type == BasisType::CHEBYSHEV) {
        x_work = normalize_x(x);
    }
    
    double result = 0.0;
    for (int k = 0; k <= degree; ++k) {
        double phi_k_deriv = compute_basis_derivative(x_work, k, order);
        result += coeffs[k] * phi_k_deriv;
    }
    
    if (basis_type == BasisType::CHEBYSHEV && x_scale != 0.0) {
        result /= std::pow(x_scale, order);
    }
    
    return result;
}

} // namespace mixed_approx
