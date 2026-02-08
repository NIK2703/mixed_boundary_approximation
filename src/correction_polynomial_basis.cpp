#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/gauss_quadrature.h"
#include <cmath>
#include <vector>

namespace mixed_approx {

double CorrectionPolynomial::compute_basis_function(double x_norm, int k) const {
    if (basis_type == BasisType::MONOMIAL) {
        if (k == 0) return 1.0;
        return std::pow(x_norm, k);
    } else {
        if (k == 0) return 1.0;
        if (k == 1) return x_norm;
        
        double T_prev = 1.0;
        double T_curr = x_norm;
        for (int i = 2; i <= k; ++i) {
            double T_next = 2.0 * x_norm * T_curr - T_prev;
            T_prev = T_curr;
            T_curr = T_next;
        }
        return T_curr;
    }
}

double CorrectionPolynomial::compute_basis_derivative(double x_norm, int k, int order) const {
    if (basis_type == BasisType::MONOMIAL) {
        if (order == 1) {
            if (k == 0) return 0.0;
            return k * std::pow(x_norm, k - 1);
        } else {
            if (k == 0) return 0.0;
            if (k == 1) return 0.0;
            return k * (k - 1) * std::pow(x_norm, k - 2);
        }
    } else {
        std::vector<double> T, T1, T2;
        chebyshev_derivatives(x_norm, k, T, T1, T2);
        if (order == 1) {
            return T1[k];
        } else {
            return T2[k];
        }
    }
}

void CorrectionPolynomial::chebyshev_polynomials(double t, int max_k, std::vector<double>& T) {
    T.resize(max_k + 1);
    if (max_k >= 0) T[0] = 1.0;
    if (max_k >= 1) T[1] = t;
    for (int k = 2; k <= max_k; ++k) {
        T[k] = 2.0 * t * T[k-1] - T[k-2];
    }
}

void CorrectionPolynomial::chebyshev_derivatives(double t, int max_k, std::vector<double>& T, 
                                                  std::vector<double>& T1, std::vector<double>& T2) {
    T.resize(max_k + 1);
    T1.resize(max_k + 1);
    T2.resize(max_k + 1);
    
    if (max_k >= 0) {
        T[0] = 1.0;
        T1[0] = 0.0;
        T2[0] = 0.0;
    }
    if (max_k >= 1) {
        T[1] = t;
        T1[1] = 1.0;
        T2[1] = 0.0;
    }
    
    for (int k = 2; k <= max_k; ++k) {
        T[k] = 2.0 * t * T[k-1] - T[k-2];
        T1[k] = 2.0 * T[k-1] + 2.0 * t * T1[k-1] - T1[k-2];
        T2[k] = 4.0 * T1[k-1] + 2.0 * t * T2[k-1] - T2[k-2];
    }
}

} // namespace mixed_approx
