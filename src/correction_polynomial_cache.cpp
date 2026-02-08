#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/gauss_quadrature.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace mixed_approx {

void CorrectionPolynomial::build_caches(const std::vector<double>& points_x,
                                        const std::vector<double>& points_y) {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    
    basis_cache_x.clear();
    basis_cache_y.clear();
    basis2_cache_x.clear();
    basis2_cache_y.clear();
    
    basis_cache_x.resize(points_x.size(), std::vector<double>(n_free));
    basis2_cache_x.resize(points_x.size(), std::vector<double>(n_free));
    for (size_t i = 0; i < points_x.size(); ++i) {
        double x = points_x[i];
        double x_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(x) : x;
        for (int k = 0; k < n_free; ++k) {
            basis_cache_x[i][k] = compute_basis_function(x_work, k);
            basis2_cache_x[i][k] = compute_basis_derivative(x_work, k, 2);
        }
    }
    
    basis_cache_y.resize(points_y.size(), std::vector<double>(n_free));
    basis2_cache_y.resize(points_y.size(), std::vector<double>(n_free));
    for (size_t j = 0; j < points_y.size(); ++j) {
        double y = points_y[j];
        double y_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(y) : y;
        for (int k = 0; k < n_free; ++k) {
            basis_cache_y[j][k] = compute_basis_function(y_work, k);
            basis2_cache_y[j][k] = compute_basis_derivative(y_work, k, 2);
        }
    }
}

void CorrectionPolynomial::clear_caches() {
    basis_cache_x.clear();
    basis_cache_y.clear();
    basis2_cache_x.clear();
    basis2_cache_y.clear();
    stiffness_matrix.clear();
    stiffness_matrix_computed = false;
}

void CorrectionPolynomial::compute_stiffness_matrix(double a, double b, const WeightMultiplier& W, int gauss_points) {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    
    std::vector<double> gauss_nodes, gauss_weights;
    get_gauss_legendre_quadrature(gauss_points, gauss_nodes, gauss_weights);
    
    int n = n_free;
    stiffness_matrix.assign(n, std::vector<double>(n, 0.0));
    
    for (size_t idx = 0; idx < gauss_nodes.size(); ++idx) {
        double t = gauss_nodes[idx];
        double weight = gauss_weights[idx];
        double x = transform_to_standard_interval(t, a, b);
        
        std::vector<double> phi2(n);
        for (int k = 0; k < n; ++k) {
            double x_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(x) : x;
            phi2[k] = compute_basis_derivative(x_work, k, 2);
            if (basis_type == BasisType::CHEBYSHEV && x_scale != 0.0) {
                phi2[k] /= std::pow(x_scale, 2);
            }
        }
        
        double W_val = W.evaluate(x);
        double W2 = W_val * W_val;
        
        for (int k = 0; k < n; ++k) {
            for (int l = 0; l < n; ++l) {
                stiffness_matrix[k][l] += phi2[k] * phi2[l] * W2 * weight * 0.5 * (b - a);
            }
        }
    }
    
    stiffness_matrix_computed = true;
}

} // namespace mixed_approx
