#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/gauss_quadrature.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mixed_approx {

void CorrectionPolynomial::initialize_least_squares(const std::vector<WeightedPoint>& approx_points,
                                                    const InterpolationBasis& p_int,
                                                    const WeightMultiplier& W) {
    if (approx_points.empty()) {
        throw std::invalid_argument("Cannot perform least squares: no approximation points provided");
    }
    
    int N = static_cast<int>(approx_points.size());
    int n_free = this->n_free;
    
    std::vector<std::vector<double>> A(n_free, std::vector<double>(n_free, 0.0));
    std::vector<double> b(n_free, 0.0);
    
    std::vector<std::vector<double>> phi_k_x(N, std::vector<double>(n_free));
    std::vector<double> W_x(N);
    std::vector<double> P_int_x(N);
    std::vector<double> r_i(N);
    
    for (int i = 0; i < N; ++i) {
        double x = approx_points[i].x;
        double sigma = approx_points[i].weight;
        double f_x = approx_points[i].value;
        
        P_int_x[i] = p_int.evaluate(x);
        r_i[i] = P_int_x[i] - f_x;
        
        W_x[i] = W.evaluate(x);
        
        double x_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(x) : x;
        for (int k = 0; k < n_free; ++k) {
            phi_k_x[i][k] = compute_basis_function(x_work, k);
        }
    }
    
    for (int i = 0; i < N; ++i) {
        double weight = 1.0 / approx_points[i].weight;
        double W_val = W_x[i];
        double W2 = W_val * W_val;
        
        for (int k = 0; k < n_free; ++k) {
            double phi_k = phi_k_x[i][k];
            double term_k = phi_k * W_val * weight;
            
            b[k] += r_i[i] * term_k;
            
            for (int l = 0; l < n_free; ++l) {
                double phi_l = phi_k_x[i][l];
                A[k][l] += phi_k * phi_l * W2 * weight;
            }
        }
    }
    
    for (double& val : b) {
        val = -val;
    }
    
    int n = n_free;
    
    for (int col = 0; col < n; ++col) {
        int max_row = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                max_row = row;
            }
        }
        
        if (max_val < 1e-14) {
            for (int i = 0; i < n; ++i) {
                A[i][i] += 1e-8;
            }
            max_row = col;
            max_val = std::abs(A[col][col]);
            for (int row = col + 1; row < n; ++row) {
                if (std::abs(A[row][col]) > max_val) {
                    max_val = std::abs(A[row][col]);
                    max_row = row;
                }
            }
        }
        
        if (max_row != col) {
            std::swap(A[col], A[max_row]);
            std::swap(b[col], b[max_row]);
        }
        
        double pivot = A[col][col];
        if (std::abs(pivot) < 1e-14) {
            std::fill(coeffs.begin(), coeffs.end(), 0.0);
            validation_message = "Least squares failed: singular matrix, falling back to zero initialization";
            return;
        }
        
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row][col] / pivot;
            b[row] -= factor * b[col];
            for (int j = col; j < n; ++j) {
                A[row][j] -= factor * A[col][j];
            }
        }
    }
    
    coeffs.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * coeffs[j];
        }
        coeffs[i] = sum / A[i][i];
    }
    
    validation_message = "Least squares initialization";
}

bool CorrectionPolynomial::verify_initialization(const std::vector<WeightedPoint>& approx_points,
                                                 const std::vector<RepulsionPoint>& repel_points,
                                                 const InterpolationBasis& p_int,
                                                 const WeightMultiplier& W) {
    int n = n_free;
    std::vector<double> test_points;
    double a = x_center - x_scale;
    double b = x_center + x_scale;
    for (int i = 0; i < n; ++i) {
        double t = -1.0 + 2.0 * (i + 0.5) / n;
        double x = transform_to_standard_interval(t, a, b);
        test_points.push_back(x);
    }
    
    if (static_cast<int>(test_points.size()) >= n) {
        std::vector<std::vector<double>> G(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i) {
            double x_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(test_points[i]) : test_points[i];
            for (int k = 0; k < n; ++k) {
                G[i][k] = compute_basis_function(x_work, k);
            }
        }
        
        if (n <= 10) {
            double det = 1.0;
            std::vector<std::vector<double>> M = G;
            for (int col = 0; col < n; ++col) {
                int max_row = col;
                double max_val = std::abs(M[col][col]);
                for (int row = col + 1; row < n; ++row) {
                    if (std::abs(M[row][col]) > max_val) {
                        max_val = std::abs(M[row][col]);
                        max_row = row;
                    }
                }
                if (max_val < 1e-14) {
                    validation_message = "Basis functions are linearly dependent (det ~ 0)";
                    return false;
                }
                if (max_row != col) {
                    std::swap(M[col], M[max_row]);
                    det = -det;
                }
                double pivot = M[col][col];
                det *= pivot;
                for (int row = col + 1; row < n; ++row) {
                    double factor = M[row][col] / pivot;
                    for (int j = col; j < n; ++j) {
                        M[row][j] -= factor * M[col][j];
                    }
                }
            }
            if (std::abs(det) < 1e-10) {
                validation_message = "Basis functions are linearly dependent (det too small)";
                return false;
            }
        }
    }
    
    if (!repel_points.empty()) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& rp : repel_points) {
            double p_int_val = p_int.evaluate(rp.x);
            double q_val = evaluate_Q(rp.x);
            double w_val = W.evaluate(rp.x);
            double F_y = p_int_val + q_val * w_val;
            double dist = std::abs(rp.y_forbidden - F_y);
            min_dist = std::min(min_dist, dist);
        }
        
        double char_dist = std::numeric_limits<double>::max();
        for (const auto& rp : repel_points) {
            for (const auto& ap : approx_points) {
                double d = std::abs(ap.value - rp.y_forbidden);
                char_dist = std::min(char_dist, d);
            }
        }
        if (char_dist == std::numeric_limits<double>::max()) {
            char_dist = 1.0;
        }
        
        double safe_dist = 0.1 * char_dist;
        if (min_dist < safe_dist) {
            validation_message = "Initial coefficients too close to repulsion barriers";
            return false;
        }
    }
    
    validation_message = "Initialization verification passed";
    return true;
}

double CorrectionPolynomial::compute_objective(const std::vector<WeightedPoint>& approx_points,
                                               const InterpolationBasis& p_int,
                                               const WeightMultiplier& W) const {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    if (approx_points.empty()) {
        return 0.0;
    }
    
    int N = approx_points.size();
    double sum_sq = 0.0;
    
    for (int i = 0; i < N; ++i) {
        double x = approx_points[i].x;
        double f = approx_points[i].value;
        double sigma = approx_points[i].weight;
        
        double p_int_val = p_int.evaluate(x);
        double q_val = evaluate_Q(x);
        double w_val = W.evaluate(x);
        double residual = p_int_val + q_val * w_val - f;
        
        sum_sq += (residual * residual) / sigma;
    }
    
    double obj = sum_sq;
    
    if (regularization_lambda > 0.0) {
        if (!stiffness_matrix_computed) {
            throw std::runtime_error("Stiffness matrix not computed. Call compute_stiffness_matrix first.");
        }
        double reg = 0.0;
        for (int k = 0; k < n_free; ++k) {
            for (int l = 0; l < n_free; ++l) {
                reg += coeffs[k] * coeffs[l] * stiffness_matrix[k][l];
            }
        }
        obj += regularization_lambda * reg;
    }
    
    return obj;
}

std::vector<double> CorrectionPolynomial::compute_gradient(const std::vector<WeightedPoint>& approx_points,
                                                            const InterpolationBasis& p_int,
                                                            const WeightMultiplier& W) const {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial not initialized");
    }
    if (approx_points.empty()) {
        return std::vector<double>(n_free, 0.0);
    }
    
    std::vector<double> grad(n_free, 0.0);
    int N = approx_points.size();
    
    for (int i = 0; i < N; ++i) {
        double x = approx_points[i].x;
        double f = approx_points[i].value;
        double sigma = approx_points[i].weight;
        
        double p_int_val = p_int.evaluate(x);
        double q_val = evaluate_Q(x);
        double w_val = W.evaluate(x);
        double residual = p_int_val + q_val * w_val - f;
        
        double x_work = (basis_type == BasisType::CHEBYSHEV) ? normalize_x(x) : x;
        for (int k = 0; k < n_free; ++k) {
            double phi_k = compute_basis_function(x_work, k);
            grad[k] += 2.0 * residual * phi_k * w_val / sigma;
        }
    }
    
    if (regularization_lambda > 0.0) {
        if (!stiffness_matrix_computed) {
            throw std::runtime_error("Stiffness matrix not computed.");
        }
        std::vector<double> reg_grad(n_free, 0.0);
        for (int k = 0; k < n_free; ++k) {
            for (int l = 0; l < n_free; ++l) {
                reg_grad[k] += stiffness_matrix[k][l] * coeffs[l];
            }
        }
        for (int k = 0; k < n_free; ++k) {
            grad[k] += 2.0 * regularization_lambda * reg_grad[k];
        }
    }
    
    return grad;
}

} // namespace mixed_approx
