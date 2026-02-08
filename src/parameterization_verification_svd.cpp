#include "mixed_approximation/parameterization_verification.h"
#include "mixed_approximation/correction_polynomial.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>

namespace mixed_approx {

std::vector<double> ParameterizationVerifier::compute_singular_values(
    const std::vector<std::vector<double>>& matrix,
    double& condition_number) {
    int n = static_cast<int>(matrix.size());
    if (n == 0) {
        condition_number = 0.0;
        return {};
    }
    
    // Simplified SVD computation via QR decomposition
    std::vector<std::vector<double>> A = matrix;
    
    // Compute A^T * A
    std::vector<std::vector<double>> ATA(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[k][i] * A[k][j];
            }
            ATA[i][j] = sum;
        }
    }
    
    // Power method for maximum eigenvalue
    std::vector<double> sv(n, 1.0);
    std::vector<double> v(n, 1.0 / std::sqrt(n));
    
    double lambda_max = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                y[i] += ATA[i][j] * v[j];
            }
        }
        
        double norm = 0.0;
        for (double val : y) norm += val * val;
        norm = std::sqrt(norm);
        
        if (norm < 1e-15) break;
        
        for (double& val : y) val /= norm;
        
        lambda_max = norm;
        v = y;
    }
    
    double sigma_max = std::sqrt(lambda_max);
    
    // Estimate minimum singular value
    double sigma_min = sigma_max;
    if (n > 1) {
        double min_diag = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            min_diag = std::min(min_diag, ATA[i][i]);
        }
        sigma_min = std::sqrt(std::max(0.0, min_diag));
    }
    
    // Fill singular values
    std::vector<double> singular_values(n);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(n - 1 - i) / (n - 1);
        singular_values[i] = sigma_min + (sigma_max - sigma_min) * t;
    }
    
    // Sort in descending order
    std::sort(singular_values.begin(), singular_values.end(), std::greater<double>());
    
    if (singular_values.back() > 0) {
        condition_number = singular_values.front() / singular_values.back();
    } else {
        condition_number = std::numeric_limits<double>::infinity();
    }
    
    return singular_values;
}

Recommendation ParameterizationVerifier::diagnose_interpolation_error(
    const NodeError& error,
    double W_tolerance) {
    (void)W_tolerance;
    
    if (!error.W_acceptable) {
        return Recommendation(
            RecommendationType::USE_LONG_DOUBLE,
            "W(z_e) is not sufficiently close to zero",
            "The weight multiplier has numerical errors. Consider using long double arithmetic or checking the construction of W(x)."
        );
    }
    
    if (error.absolute_error > 1e-6) {
        return Recommendation(
            RecommendationType::REDUCE_TOLERANCE,
            "Large interpolation error detected",
            "The interpolation error is significantly larger than the tolerance. Consider increasing the tolerance or checking the basis construction."
        );
    }
    
    return Recommendation(
        RecommendationType::NONE,
        "Minor interpolation error",
        "The error is small but exceeds the tolerance. This may be due to accumulated floating-point errors."
    );
}

Recommendation ParameterizationVerifier::diagnose_condition_issue(
    const CompletenessTestResult& result,
    BasisType current_basis) {
    if (result.condition_number > 1e12) {
        std::string new_basis = (current_basis == BasisType::MONOMIAL) ? "CHEBYSHEV" : "MONOMIAL";
        return Recommendation(
            RecommendationType::CHANGE_BASIS,
            "Critical ill-conditioning detected",
            "The condition number is critical (" + std::to_string(result.condition_number) +
            "). Consider switching to " + new_basis + " basis for improved numerical stability."
        );
    }
    
    if (result.actual_rank < result.expected_rank) {
        return Recommendation(
            RecommendationType::MERGE_NODES,
            "Rank deficient basis matrix",
            "The basis matrix has rank " + std::to_string(result.actual_rank) +
            " instead of expected " + std::to_string(result.expected_rank) +
            ". Consider merging close interpolation nodes."
        );
    }
    
    return Recommendation(
        RecommendationType::INCREASE_GAMMA,
        "Poor conditioning detected",
        "Consider increasing the regularization parameter gamma to improve conditioning."
    );
}

std::vector<double> ParameterizationVerifier::chebyshev_nodes(int n, double a, double b) {
    std::vector<double> nodes(n);
    for (int k = 0; k < n; ++k) {
        double t = std::cos(M_PI * (2.0 * (k + 1) - 1) / (2.0 * n));
        nodes[k] = 0.5 * (b - a) * t + 0.5 * (a + b);
    }
    return nodes;
}

double ParameterizationVerifier::compute_basis_function_public(
    const CorrectionPolynomial& Q, double x, int k) const {
    if (Q.basis_type == BasisType::MONOMIAL) {
        return std::pow(x, k);
    } else {
        if (k == 0) return 1.0;
        if (k == 1) return x;
        
        double T_prev = 1.0;
        double T_curr = x;
        for (int i = 2; i <= k; ++i) {
            double T_next = 2.0 * x * T_curr - T_prev;
            T_prev = T_curr;
            T_curr = T_next;
        }
        return T_curr;
    }
}

} // namespace mixed_approx
