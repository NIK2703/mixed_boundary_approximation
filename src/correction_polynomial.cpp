#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/interpolation_basis.h"
#include <numeric>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <random>
#include <stdexcept>

namespace mixed_approx {

BasisType CorrectionPolynomial::choose_basis_type(int deg) {
    if (deg <= 5) {
        return BasisType::MONOMIAL;
    } else {
        return BasisType::CHEBYSHEV;
    }
}

void CorrectionPolynomial::initialize(int deg, BasisType basis, double interval_center, double interval_scale) {
    degree = deg;
    n_free = deg + 1;
    basis_type = basis;
    x_center = interval_center;
    x_scale = interval_scale;
    is_initialized = true;
    validation_message.clear();
    
    coeffs.assign(n_free, 0.0);
    
    clear_caches();
    
    stiffness_matrix.clear();
    stiffness_matrix_computed = false;
    
    regularization_lambda = 0.0;
}

void CorrectionPolynomial::initialize_coefficients(InitializationMethod method,
                                                   const std::vector<WeightedPoint>& approx_points,
                                                   const std::vector<RepulsionPoint>& repel_points,
                                                   const InterpolationBasis& p_int,
                                                   const WeightMultiplier& W,
                                                   double interval_start,
                                                   double interval_end) {
    if (!is_initialized) {
        throw std::runtime_error("CorrectionPolynomial: must call initialize() before setting coefficients");
    }
    
    init_method = method;
    
    switch (method) {
        case InitializationMethod::ZERO:
            initialize_zero();
            break;
        case InitializationMethod::LEAST_SQUARES:
            initialize_least_squares(approx_points, p_int, W);
            break;
        case InitializationMethod::RANDOM:
            initialize_random();
            break;
        default:
            throw std::invalid_argument("Unknown initialization method");
    }
    
    if (!repel_points.empty() && method != InitializationMethod::ZERO) {
        apply_barrier_protection(repel_points);
    }
}

void CorrectionPolynomial::initialize_zero() {
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
    init_method = InitializationMethod::ZERO;
    validation_message = "Zero initialization";
}

void CorrectionPolynomial::initialize_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    
    for (double& coeff : coeffs) {
        coeff = dist(gen);
    }
    init_method = InitializationMethod::RANDOM;
    validation_message = "Random initialization in [-0.01, 0.01]";
}

void CorrectionPolynomial::apply_barrier_protection(const std::vector<RepulsionPoint>& repel_points,
                                                    double safe_distance_factor) {
    (void)safe_distance_factor;
    if (repel_points.empty() || coeffs.empty()) {
        return;
    }
    validation_message += "\nWarning: Barrier protection not fully implemented yet";
}

std::string CorrectionPolynomial::get_diagnostic_info() const {
    std::ostringstream oss;
    oss << "CorrectionPolynomial info:\n";
    oss << "  degree: " << degree << "\n";
    oss << "  n_free: " << n_free << "\n";
    oss << "  basis_type: " << (basis_type == BasisType::MONOMIAL ? "MONOMIAL" : "CHEBYSHEV") << "\n";
    oss << "  initialized: " << (is_initialized ? "yes" : "no") << "\n";
    if (is_initialized) {
        oss << "  init_method: ";
        switch (init_method) {
            case InitializationMethod::ZERO: oss << "ZERO"; break;
            case InitializationMethod::LEAST_SQUARES: oss << "LEAST_SQUARES"; break;
            case InitializationMethod::RANDOM: oss << "RANDOM"; break;
        }
        oss << "\n";
        oss << "  x_center: " << x_center << ", x_scale: " << x_scale << "\n";
        oss << "  coeffs: ";
        for (size_t i = 0; i < coeffs.size(); ++i) {
            oss << coeffs[i];
            if (i + 1 < coeffs.size()) oss << ", ";
        }
        oss << "\n";
        oss << "  caches: x=" << basis_cache_x.size() << " points, y=" << basis_cache_y.size() << " points\n";
        oss << "  stiffness_matrix: " << (stiffness_matrix_computed ? "computed" : "not computed") << "\n";
    }
    if (!validation_message.empty()) {
        oss << "  message: " << validation_message << "\n";
    }
    return oss.str();
}

} // namespace mixed_approx
