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

// Вспомогательные функции для квадратуры Гаусса-Лежандра
namespace {
    
const std::vector<double> GAUSS_LEGENDRE_NODES_10 = {
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.14887433898163122, 0.14887433898163122,
    0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
    0.9739065285171717
};

const std::vector<double> GAUSS_LEGENDRE_WEIGHTS_10 = {
    0.06667134430868814, 0.1494513491505806, 0.21908636251598204,
    0.26926671930999635, 0.29552422471475287, 0.29552422471475287,
    0.26926671930999635, 0.21908636251598204, 0.1494513491505806,
    0.06667134430868814
};

const std::vector<double> GAUSS_LEGENDRE_NODES_20 = {
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.636053680726515,
    -0.5108670019508271, -0.37370608871541955, -0.22778585114164507,
    -0.07652652113349733, 0.07652652113349733, 0.22778585114164507,
    0.37370608871541955, 0.5108670019508271, 0.636053680726515,
    0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
    0.9639719272779138, 0.9931285991850949
};

const std::vector<double> GAUSS_LEGENDRE_WEIGHTS_20 = {
    0.017614007139152118, 0.04060142980038694, 0.06267204833410906,
    0.08327674157670475, 0.10193011981724044, 0.11819453196151842,
    0.13168863844917664, 0.14209610931838205, 0.14917298647260374,
    0.15275338713072585, 0.15275338713072585, 0.14917298647260374,
    0.14209610931838205, 0.13168863844917664, 0.11819453196151842,
    0.10193011981724044, 0.08327674157670475, 0.06267204833410906,
    0.04060142980038694, 0.017614007139152118
};

void get_gauss_legendre_quadrature(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    if (n <= 10) {
        nodes = GAUSS_LEGENDRE_NODES_10;
        weights = GAUSS_LEGENDRE_WEIGHTS_10;
        if (n < 10) {
            nodes.resize(n);
            weights.resize(n);
        }
    } else if (n <= 20) {
        nodes = GAUSS_LEGENDRE_NODES_20;
        weights = GAUSS_LEGENDRE_WEIGHTS_20;
        if (n < 20) {
            nodes.resize(n);
            weights.resize(n);
        }
    } else {
        nodes = GAUSS_LEGENDRE_NODES_20;
        weights = GAUSS_LEGENDRE_WEIGHTS_20;
    }
}

double transform_to_standard_interval(double t, double a, double b) {
    return 0.5 * (b - a) * t + 0.5 * (a + b);
}

} // anonymous namespace

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

void CorrectionPolynomial::apply_barrier_protection(const std::vector<RepulsionPoint>& repel_points,
                                                      double safe_distance_factor) {
    if (repel_points.empty() || coeffs.empty()) {
        return;
    }
    validation_message += "\nWarning: Barrier protection not fully implemented yet";
}

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

void CorrectionPolynomial::chebyshev_derivatives(double t, int max_k, std::vector<double>& T, std::vector<double>& T1, std::vector<double>& T2) {
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

// ============== Методы для FunctionalEvaluator (с внешними коэффициентами) ==============

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
