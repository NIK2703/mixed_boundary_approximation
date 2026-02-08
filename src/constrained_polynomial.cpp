#include "mixed_approximation/constrained_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include <sstream>
#include <cmath>
#include <algorithm>

namespace mixed_approx {

ConstrainedPolynomial::ConstrainedPolynomial()
    : interval_a_(0.0)
    , interval_b_(1.0)
    , interval_center_(0.5)
    , interval_scale_(0.5)
    , value_cache_(1024, true, 1e-9)
    , W_cache_(1024, true, 1e-9)
    , Q_cache_(1024, true, 1e-9)
    , basis_cache_(1024, true, 1e-9)
    , last_evaluated_x_index_(0)
    , evaluation_count_(0)
    , cache_hits_(0)
    , is_valid_(false)
    , validation_message_("Default constructed") {}

ConstrainedPolynomial::ConstrainedPolynomial(const std::vector<InterpolationNode>& nodes,
                                             int deg_Q,
                                             double interval_start,
                                             double interval_end)
    : ConstrainedPolynomial(nodes, deg_Q, BasisType::MONOMIAL, interval_start, interval_end) {}

ConstrainedPolynomial::ConstrainedPolynomial(const std::vector<InterpolationNode>& nodes,
                                             int deg_Q,
                                             BasisType basis_type,
                                             double interval_start,
                                             double interval_end)
    : interval_a_(interval_start)
    , interval_b_(interval_end)
    , interval_center_((interval_start + interval_end) / 2.0)
    , interval_scale_((interval_end - interval_start) / 2.0)
    , value_cache_(1024, true, 1e-9)
    , W_cache_(1024, true, 1e-9)
    , Q_cache_(1024, true, 1e-9)
    , basis_cache_(1024, true, 1e-9)
    , last_evaluated_x_index_(0)
    , evaluation_count_(0)
    , cache_hits_(0)
    , is_valid_(false)
    , validation_message_("Not built") {
    
    if (nodes.empty()) {
        validation_message_ = "Empty nodes array";
        return;
    }
    
    if (deg_Q < 0) {
        validation_message_ = "Invalid degree for correction polynomial";
        return;
    }
    
    // Извлекаем узлы и значения
    std::vector<double> node_x;
    std::vector<double> node_values;
    node_x.reserve(nodes.size());
    node_values.reserve(nodes.size());
    
    for (const auto& node : nodes) {
        node_x.push_back(node.x);
        node_values.push_back(node.value);
    }
    
    // Строим интерполяционный базис
    p_int_.build(node_x, node_values,
                 InterpolationMethod::BARYCENTRIC,
                 interval_start, interval_end,
                 true, true);
    
    if (!p_int_.is_valid) {
        validation_message_ = "Failed to build interpolation basis: " + p_int_.error_message;
        return;
    }
    
    // Строим весовой множитель
    W_.build_from_roots(node_x, interval_start, interval_end, true);
    
    if (!W_.verify_construction(1e-10)) {
        validation_message_ = "Failed to build weight multiplier";
        return;
    }
    
    // Инициализируем корректирующий полином
    Q_.initialize(deg_Q, basis_type, interval_center_, interval_scale_);
    
    is_valid_ = true;
    validation_message_ = "Successfully constructed";
}

double ConstrainedPolynomial::compute_W(double x) const {
    double cached_value;
    if (W_cache_.try_get(x, cached_value)) {
        return cached_value;
    }
    double value = W_.evaluate(x);
    W_cache_.put(x, value);
    return value;
}

double ConstrainedPolynomial::compute_Q(double x) const {
    double cached_value;
    if (Q_cache_.try_get(x, cached_value)) {
        return cached_value;
    }
    double value = Q_.evaluate_Q(x);
    Q_cache_.put(x, value);
    return value;
}

double ConstrainedPolynomial::compute_W_derivative(double x, int order) const {
    return W_.evaluate_derivative(x, order);
}

double ConstrainedPolynomial::compute_Q_derivative(double x, int order) const {
    return Q_.evaluate_Q_derivative(x, order);
}

bool ConstrainedPolynomial::is_near_node(double x, double& node_value) const {
    // Проверяем близость к узлам интерполяции
    for (size_t i = 0; i < p_int_.nodes_original.size(); ++i) {
        if (std::abs(x - p_int_.nodes_original[i]) < 1e-10) {
            node_value = p_int_.values[i];
            return true;
        }
    }
    return false;
}

double ConstrainedPolynomial::evaluate(double x) const noexcept {
    evaluation_count_++;
    
    // Проверяем кэш
    std::array<double, 3> cached;
    if (value_cache_.try_get(x, cached)) {
        cache_hits_++;
        return cached[0];
    }
    
    double node_value = 0.0;
    if (is_near_node(x, node_value)) {
        // В узле интерполяции возвращаем точное значение
        value_cache_.put(x, {node_value, 0.0, 0.0});
        return node_value;
    }
    
    // F(x) = P_int(x) + Q(x) * W(x)
    double P_int = p_int_.evaluate(x);
    double Q_val = Q_.evaluate_Q(x);
    double W_val = W_.evaluate(x);
    
    double result = P_int + Q_val * W_val;
    
    value_cache_.put(x, {result, 0.0, 0.0});
    return result;
}

bool ConstrainedPolynomial::evaluate_safe(double x, double& result) const noexcept {
    if (!std::isfinite(x)) {
        return false;
    }
    try {
        result = evaluate(x);
        return std::isfinite(result);
    } catch (...) {
        return false;
    }
}

void ConstrainedPolynomial::derivatives(double x, double& f, double& f1, double& f2) const noexcept {
    evaluation_count_++;
    
    // Проверяем кэш
    std::array<double, 3> cached;
    if (value_cache_.try_get(x, cached)) {
        cache_hits_++;
        f = cached[0];
        f1 = cached[1];
        f2 = cached[2];
        return;
    }
    
    double node_value = 0.0;
    if (is_near_node(x, node_value)) {
        f = node_value;
        f1 = 0.0;
        f2 = 0.0;
        value_cache_.put(x, {f, f1, f2});
        return;
    }
    
    // Вычисляем компоненты
    double P_int = p_int_.evaluate(x);
    double P_int_1 = p_int_.evaluate_derivative(x, 1);
    double P_int_2 = p_int_.evaluate_derivative(x, 2);
    
    double Q_val = Q_.evaluate_Q(x);
    double Q_val_1 = Q_.evaluate_Q_derivative(x, 1);
    double Q_val_2 = Q_.evaluate_Q_derivative(x, 2);
    
    double W_val = W_.evaluate(x);
    double W_val_1 = W_.evaluate_derivative(x, 1);
    double W_val_2 = W_.evaluate_derivative(x, 2);
    
    // F(x) = P_int(x) + Q(x) * W(x)
    // F'(x) = P_int'(x) + Q'(x) * W(x) + Q(x) * W'(x)
    // F''(x) = P_int''(x) + Q''(x) * W(x) + 2 * Q'(x) * W'(x) + Q(x) * W''(x)
    
    f = P_int + Q_val * W_val;
    f1 = P_int_1 + Q_val_1 * W_val + Q_val * W_val_1;
    f2 = P_int_2 + Q_val_2 * W_val + 2.0 * Q_val_1 * W_val_1 + Q_val * W_val_2;
    
    value_cache_.put(x, {f, f1, f2});
}

EvaluationResult ConstrainedPolynomial::evaluate_with_derivatives(double x) const noexcept {
    EvaluationResult result;
    derivatives(x, result.value, result.first_deriv, result.second_deriv);
    return result;
}

double ConstrainedPolynomial::first_derivative(double x) const noexcept {
    double f, f1, f2;
    derivatives(x, f, f1, f2);
    return f1;
}

double ConstrainedPolynomial::second_derivative(double x) const noexcept {
    double f, f1, f2;
    derivatives(x, f, f1, f2);
    return f2;
}

std::size_t ConstrainedPolynomial::degree() const noexcept {
    // deg(F) = max(deg(P_int), deg(Q) + deg(W))
    std::size_t deg_P_int = static_cast<std::size_t>(p_int_.m_eff > 0 ? p_int_.m_eff - 1 : 0);
    std::size_t deg_Q = static_cast<std::size_t>(Q_.degree);
    std::size_t deg_W = static_cast<std::size_t>(W_.degree());
    return std::max(deg_P_int, deg_Q + deg_W);
}

std::size_t ConstrainedPolynomial::num_parameters() const noexcept {
    return static_cast<std::size_t>(Q_.n_free);
}

double ConstrainedPolynomial::parameter(std::size_t index) const {
    if (index >= Q_.n_free) {
        throw std::out_of_range("Parameter index out of range");
    }
    return Q_.coeffs[index];
}

void ConstrainedPolynomial::set_parameter(std::size_t index, double value) {
    if (index >= Q_.n_free) {
        throw std::out_of_range("Parameter index out of range");
    }
    if (!std::isfinite(value)) {
        throw std::domain_error("Parameter value must be finite");
    }
    Q_.coeffs[index] = value;
    reset_cache();
}

std::vector<double> ConstrainedPolynomial::parameters() const {
    return Q_.coeffs;
}

void ConstrainedPolynomial::set_parameters(const std::vector<double>& params) {
    if (params.size() != Q_.n_free) {
        throw std::invalid_argument("Parameters size mismatch");
    }
    Q_.coeffs = params;
    reset_cache();
}

double ConstrainedPolynomial::basis_function(std::size_t k, double x) const {
    if (k >= Q_.n_free) {
        throw std::out_of_range("Basis function index out of range");
    }
    
    // ∂F/∂q_k = φ_k(x) · W(x)
    double phi_k = Q_.compute_basis_function_with_coeffs(x, Q_.coeffs, static_cast<int>(k));
    double W_val = W_.evaluate(x);
    
    return phi_k * W_val;
}

double ConstrainedPolynomial::basis_derivative(std::size_t k, double x, int order) const {
    if (k >= Q_.n_free) {
        throw std::out_of_range("Basis function index out of range");
    }
    
    // ∂²F/∂q_k∂x = φ_k'(x) · W(x) + φ_k(x) · W'(x) для первой производной
    // ∂³F/∂q_k∂x² = φ_k''(x) · W(x) + 2·φ_k'(x) · W'(x) + φ_k(x) · W''(x) для второй
    double phi = Q_.compute_basis_function_with_coeffs(x, Q_.coeffs, static_cast<int>(k));
    double phi_1 = (order >= 1) ? Q_.compute_basis_derivative_with_coeffs(x, Q_.coeffs, static_cast<int>(k), 1) : 0.0;
    double phi_2 = (order == 2) ? Q_.compute_basis_derivative_with_coeffs(x, Q_.coeffs, static_cast<int>(k), 2) : 0.0;
    
    double W_val = W_.evaluate(x);
    double W_val_1 = (order >= 1) ? W_.evaluate_derivative(x, 1) : 0.0;
    double W_val_2 = (order == 2) ? W_.evaluate_derivative(x, 2) : 0.0;
    
    if (order == 0) {
        return phi * W_val;
    } else if (order == 1) {
        return phi_1 * W_val + phi * W_val_1;
    } else { // order == 2
        return phi_2 * W_val + 2.0 * phi_1 * W_val_1 + phi * W_val_2;
    }
}

std::vector<double> ConstrainedPolynomial::gradient(double x) const {
    std::vector<double> grad(Q_.n_free);
    for (std::size_t k = 0; k < Q_.n_free; ++k) {
        grad[k] = basis_function(k, x);
    }
    return grad;
}

std::string ConstrainedPolynomial::to_string() const {
    std::ostringstream oss;
    oss << "ConstrainedPolynomial {\n";
    oss << "  interpolation_nodes: " << p_int_.m_eff << "\n";
    oss << "  degree: " << degree() << "\n";
    oss << "  num_parameters: " << num_parameters() << "\n";
    oss << "  valid: " << (is_valid_ ? "yes" : "no") << "\n";
    oss << "  validation_message: " << validation_message_ << "\n";
    oss << "  Q coeffs: [";
    for (size_t i = 0; i < Q_.coeffs.size(); ++i) {
        oss << Q_.coeffs[i];
        if (i + 1 < Q_.coeffs.size()) oss << ", ";
    }
    oss << "]\n";
    oss << "}";
    return oss.str();
}

bool ConstrainedPolynomial::validate() const {
    if (!is_valid_) {
        return false;
    }
    
    // Проверяем P_int
    if (!p_int_.verify_interpolation(1e-10)) {
        validation_message_ = "Interpolation basis validation failed";
        return false;
    }
    
    // Проверяем W
    if (!W_.verify_construction(1e-10)) {
        validation_message_ = "Weight multiplier validation failed";
        return false;
    }
    
    // Проверяем интерполяционные условия
    if (!check_interpolation_conditions(1e-8)) {
        validation_message_ = "Interpolation conditions not satisfied";
        return false;
    }
    
    validation_message_ = "Validation passed";
    return true;
}

void ConstrainedPolynomial::reset_cache() {
    value_cache_.clear();
    W_cache_.clear();
    Q_cache_.clear();
    basis_cache_.clear();
}

BasisType ConstrainedPolynomial::basis_type() const noexcept {
    return Q_.basis_type;
}

std::array<double, 2> ConstrainedPolynomial::interval() const noexcept {
    return {interval_a_, interval_b_};
}

bool ConstrainedPolynomial::check_interpolation_conditions(double tolerance) const {
    if (!p_int_.is_valid) return false;
    
    for (size_t i = 0; i < p_int_.nodes_original.size(); ++i) {
        double z_e = p_int_.nodes_original[i];
        double expected = p_int_.values[i];
        double computed = evaluate(z_e);
        if (std::abs(computed - expected) > tolerance) {
            return false;
        }
    }
    return true;
}

void ConstrainedPolynomial::build_caches(const std::vector<double>& points) {
    for (double x : points) {
        double f, f1, f2;
        derivatives(x, f, f1, f2);
    }
}

} // namespace mixed_approx
