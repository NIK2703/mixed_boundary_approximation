#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include <numeric>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mixed_approx {

void InterpolationBasis::build(const std::vector<double>& nodes_vec,
                               const std::vector<double>& values_vec,
                               InterpolationMethod meth,
                               double interval_start,
                               double interval_end,
                               bool enable_normalization,
                               bool enable_node_merging) {
    if (nodes_vec.empty()) {
        is_valid = false;
        error_message = "Empty nodes array";
        return;
    }
    
    if (nodes_vec.size() != values_vec.size()) {
        is_valid = false;
        error_message = "Nodes and values size mismatch";
        return;
    }
    
    // Инициализация
    method = meth;
    x_center = (interval_start + interval_end) / 2.0;
    x_scale = (interval_end - interval_start) / 2.0;
    is_normalized = false;
    weight_scale = 1.0;
    is_valid = false;
    error_message.clear();
    
    // 1. Копируем исходные данные в члены класса
    nodes = nodes_vec;
    values = values_vec;
    
    // 2. Нормализация координат (если включено)
    if (enable_normalization && x_scale > 0) {
        normalize_nodes(interval_start, interval_end);
        is_normalized = true;
    }
    
    // 3. Сортировка узлов
    sort_nodes_and_values(nodes, values);
    
    // 4. Объединение близких узлов (если включено)
    if (enable_node_merging && nodes.size() > 1) {
        double interval_length = interval_end - interval_start;
        auto merged = merge_close_nodes(nodes, values, interval_length);
        
        if (merged.size() < nodes.size()) {
            nodes.clear();
            values.clear();
            for (const auto& mn : merged) {
                nodes.push_back(mn.x);
                values.push_back(mn.value);
            }
        }
    }
    
    m_eff = static_cast<int>(nodes.size());
    
    // 5. Проверка уникальности после объединения
    if (!are_nodes_unique(nodes, 1e-14)) {
        is_valid = false;
        error_message = "Non-unique nodes remain after merging";
        return;
    }
    
    // 6. Вычисление барицентрических весов (если нужно)
    if (method == InterpolationMethod::BARYCENTRIC) {
        compute_barycentric_weights();
        precompute_weighted_values();
        // Вычисляем divided differences для устойчивого вычисления производных
        compute_divided_differences();
    } else if (method == InterpolationMethod::NEWTON) {
        compute_divided_differences();
    }
    
    is_valid = true;
}

void InterpolationBasis::normalize_nodes(double a, double b) {
    double center = (a + b) / 2.0;
    double scale = (b - a) / 2.0;
    if (scale == 0) return;
    
    for (double& node : nodes) {
        node = (node - center) / scale;
    }
    x_center = center;
    x_scale = scale;
}

std::vector<InterpolationBasis::MergedNode> InterpolationBasis::merge_close_nodes(
    const std::vector<double>& nodes_input,
    const std::vector<double>& values_input,
    double interval_length) {
    
    std::vector<MergedNode> merged;
    int m = static_cast<int>(nodes_input.size());
    
    double epsilon_close = std::max(1e-12, 1e-4 / m);
    double abs_tol = epsilon_close * interval_length;
    
    for (int i = 0; i < m; ) {
        MergedNode current;
        current.x = nodes_input[i];
        current.value = values_input[i];
        current.count = 1;
        
        int j = i + 1;
        while (j < m) {
            if (std::abs(nodes_input[j] - current.x) < abs_tol) {
                current.value = (current.value * current.count + values_input[j]) / (current.count + 1);
                current.count++;
                j++;
            } else {
                break;
            }
        }
        
        merged.push_back(current);
        i = j;
    }
    
    return merged;
}

bool InterpolationBasis::are_nodes_unique(const std::vector<double>& nodes_vec, double tolerance) {
    if (nodes_vec.empty()) return true;
    
    std::vector<double> sorted = nodes_vec;
    std::sort(sorted.begin(), sorted.end());
    
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (std::abs(sorted[i] - sorted[i-1]) <= tolerance) {
            return false;
        }
    }
    return true;
}

void InterpolationBasis::sort_nodes_and_values(std::vector<double>& nodes_vec,
                                               std::vector<double>& values_vec) {
    if (nodes_vec.size() != values_vec.size() || nodes_vec.empty()) {
        return;
    }
    
    std::vector<size_t> indices(nodes_vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return nodes_vec[i] < nodes_vec[j]; });
    
    std::vector<double> sorted_nodes(nodes_vec.size());
    std::vector<double> sorted_values(values_vec.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_nodes[i] = nodes_vec[indices[i]];
        sorted_values[i] = values_vec[indices[i]];
    }
    
    nodes_vec = sorted_nodes;
    values_vec = sorted_values;
}

void InterpolationBasis::compute_barycentric_weights_standard() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double weight = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == k) continue;
            weight *= (nodes[k] - nodes[j]);
        }
        barycentric_weights[k] = 1.0 / weight;
    }
    
    double max_abs = 0.0;
    for (double w : barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
        weight_scale = max_abs;
    }
}

void InterpolationBasis::compute_barycentric_weights_logarithmic() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    std::vector<double> log_abs_weights(m, 0.0);
    std::vector<int> sign_weights(m, 1);
    
    for (int e = 0; e < m; ++e) {
        double log_sum = 0.0;
        int sign = 1;
        
        for (int k = 0; k < m; ++k) {
            if (k == e) continue;
            double diff = nodes[e] - nodes[k];
            double abs_diff = std::abs(diff);
            
            if (abs_diff < 1e-15) {
                compute_barycentric_weights_standard();
                return;
            }
            
            log_sum -= std::log(abs_diff);
            if (diff < 0) sign = -sign;
        }
        
        log_abs_weights[e] = log_sum;
        sign_weights[e] = sign;
    }
    
    double max_log = *std::max_element(log_abs_weights.begin(), log_abs_weights.end());
    
    double max_abs = 0.0;
    for (int e = 0; e < m; ++e) {
        double abs_w = std::exp(log_abs_weights[e] - max_log);
        barycentric_weights[e] = sign_weights[e] * abs_w;
        max_abs = std::max(max_abs, std::abs(barycentric_weights[e]));
    }
    
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
    }
    weight_scale = std::exp(max_log);
}

void InterpolationBasis::compute_divided_differences() {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return;
    
    divided_differences.resize(m);
    
    for (int i = 0; i < m; ++i) {
        divided_differences[i] = values[i];
    }
    
    for (int level = 1; level < m; ++level) {
        for (int i = m - 1; i >= level; --i) {
            double denom = nodes[i] - nodes[i - level];
            if (std::abs(denom) < 1e-14) {
                divided_differences[i] = 0.0;
            } else {
                divided_differences[i] = (divided_differences[i] - divided_differences[i-1]) / denom;
            }
        }
    }
}

void InterpolationBasis::precompute_weighted_values() {
    weighted_values.resize(barycentric_weights.size());
    for (size_t i = 0; i < barycentric_weights.size(); ++i) {
        weighted_values[i] = barycentric_weights[i] * values[i];
    }
}

double InterpolationBasis::evaluate(double x) const {
    if (!is_valid || nodes.empty()) {
        return 0.0;
    }
    
    double x_norm = x;
    if (is_normalized) {
        x_norm = (x - x_center) / x_scale;
    }
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        return evaluate_barycentric(x_norm);
    } else if (method == InterpolationMethod::NEWTON) {
        return evaluate_newton(x_norm);
    } else {
        return evaluate_lagrange(x_norm);
    }
}

double InterpolationBasis::evaluate_barycentric(double x) const {
    int m = static_cast<int>(nodes.size());
    
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            return values[k];
        }
    }
    
    const double* wf = weighted_values.empty() ? nullptr : weighted_values.data();
    
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int k = 0; k < m; ++k) {
        double diff = x - nodes[k];
        double inv_diff = 1.0 / diff;
        denominator += barycentric_weights[k] * inv_diff;
        if (wf) {
            numerator += wf[k] * inv_diff;
        } else {
            numerator += barycentric_weights[k] * values[k] * inv_diff;
        }
    }
    
    if (std::abs(denominator) < 1e-14) {
        return 0.0;
    }
    
    return numerator / denominator;
}

double InterpolationBasis::evaluate_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("Derivative order must be 1 or 2");
    }
    
    if (!is_valid || nodes.empty()) {
        return 0.0;
    }
    
    double x_norm = x;
    if (is_normalized) {
        x_norm = (x - x_center) / x_scale;
    }
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        return evaluate_barycentric_derivative(x_norm, order);
    } else {
        const double h = 1e-6;
        if (order == 1) {
            double fp = evaluate(x + h);
            double fm = evaluate(x - h);
            return (fp - fm) / (2.0 * h);
        } else {
            double fp = evaluate(x + h);
            double f = evaluate(x);
            double fm = evaluate(x - h);
            return (fp - 2.0 * f + fm) / (h * h);
        }
    }
}

double InterpolationBasis::evaluate_newton_derivative(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    // Производная полинома Ньютона:
    // P'(x) = f[z_0,z_1] + f[z_0,z_1,z_2] * (x-z_0) + ...
    // где f[z_i,...,z_j] - разделенные разности
    
    if (m == 1) {
        // Константный полином - производная = 0
        return 0.0;
    }
    
    // Производная первого порядка:
    double deriv = divided_differences[1];
    double product = 1.0;
    
    for (int level = 2; level < m; ++level) {
        product *= (x - nodes[level-1]);
        deriv += divided_differences[level] * product;
    }
    
    return deriv;
}

double InterpolationBasis::evaluate_barycentric_derivative(double x, int order) const {
    int m = static_cast<int>(nodes.size());
    
    // Для m <= 2 барицентрическая формула дает нестабильные результаты
    // Используем evaluate_newton_derivative как fallback
    if (m <= 2) {
        return evaluate_newton_derivative(x);
    }
    
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            // В узлах используем численную производную для устойчивости
            const double h = 1e-8;
            double fp = evaluate_barycentric(x + h);
            double fm = evaluate_barycentric(x - h);
            return (fp - fm) / (2.0 * h);
        }
    }
    
    if (order == 1) {
        // Формула для первой производной в барицентрическом интерполировании:
        // P'(x) = [Σ w_j*f_j/(x-x_j) * Σ w_k/(x-x_k) - Σ w_j/(x-x_j)^2 * Σ w_k*f_k/(x-x_k)] / (Σ w_k/(x-x_k))^2
        // = [Σ w_j*f_j/(x-x_j) - P(x) * Σ w_k/(x-x_k)] / (Σ w_k/(x-x_k))^2 * (Σ w_k/(x-x_k))
        // = [Σ w_j*f_j/(x-x_j) - P(x) * Σ w_k/(x-x_k)] / (Σ w_k/(x-x_k))^2
        // что эквивалентно: (Σ w_j*f_j/(x-x_j) - P(x) * Σ w_k/(x-x_k)) / (Σ w_k/(x-x_k))^2
        double sum_w_over_dx = 0.0;
        double sum_wf_over_dx = 0.0;
        double sum_w_over_dx2 = 0.0;
        
        for (int k = 0; k < m; ++k) {
            double diff = x - nodes[k];
            double inv_diff = 1.0 / diff;
            double inv_diff2 = inv_diff * inv_diff;
            
            sum_w_over_dx += barycentric_weights[k] * inv_diff;
            sum_wf_over_dx += weighted_values.empty() ? 
                barycentric_weights[k] * values[k] * inv_diff : 
                weighted_values[k] * inv_diff;
            sum_w_over_dx2 += barycentric_weights[k] * inv_diff2;
        }
        
        double denominator = sum_w_over_dx * sum_w_over_dx;
        if (std::abs(denominator) < 1e-14) {
            return 0.0;
        }
        
        // F'(x) = [Σ w_j*f_j/(x-x_j) - P(x) * Σ w_k/(x-x_k)] / (Σ w_k/(x-x_k))^2
        double p_x = evaluate_barycentric(x);
        double numerator = sum_wf_over_dx - p_x * sum_w_over_dx;
        return numerator / denominator;
    } else {
        const double h = 1e-6;
        double fp = evaluate_derivative(x + h, 1);
        double fm = evaluate_derivative(x - h, 1);
        return (fp - fm) / (2.0 * h);
    }
}

bool InterpolationBasis::verify_interpolation(double tolerance) const {
    if (!is_valid) return false;
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        double computed = evaluate(nodes[i]);
        if (std::abs(computed - values[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::string InterpolationBasis::get_info() const {
    std::ostringstream oss;
    oss << "InterpolationBasis info:\n";
    oss << "  m_eff: " << m_eff << "\n";
    oss << "  method: ";
    switch (method) {
        case InterpolationMethod::LAGRANGE: oss << "Lagrange"; break;
        case InterpolationMethod::BARYCENTRIC: oss << "Barycentric"; break;
        case InterpolationMethod::NEWTON: oss << "Newton"; break;
    }
    oss << "\n";
    oss << "  normalized: " << (is_normalized ? "yes" : "no") << "\n";
    if (is_normalized) {
        oss << "  x_center: " << x_center << ", x_scale: " << x_scale << "\n";
    }
    oss << "  weight_scale: " << weight_scale << "\n";
    oss << "  valid: " << (is_valid ? "yes" : "no") << "\n";
    if (!error_message.empty()) {
        oss << "  error: " << error_message << "\n";
    }
    return oss.str();
}

bool InterpolationBasis::detect_equally_spaced_nodes(double tolerance) const {
    if (nodes.size() < 2) return false;
    
    double step = nodes[1] - nodes[0];
    for (size_t i = 2; i < nodes.size(); ++i) {
        if (std::abs((nodes[i] - nodes[i-1]) - step) > tolerance) {
            return false;
        }
    }
    return true;
}

bool InterpolationBasis::detect_chebyshev_nodes(double tolerance) const {
    if (nodes.size() < 2) return false;
    int m = static_cast<int>(nodes.size());
    
    for (int k = 0; k < m; ++k) {
        double expected = std::cos(M_PI * (2.0*k + 1.0) / (2.0 * m));
        if (std::abs(nodes[k] - expected) > tolerance) {
            return false;
        }
    }
    return true;
}

void InterpolationBasis::compute_chebyshev_weights() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        barycentric_weights[k] = sign * std::sin(M_PI * (2.0*k + 1.0) / (2.0 * m));
    }
    
    double max_abs = 0.0;
    for (double w : barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    if (max_abs > 0) {
        for (double& w : barycentric_weights) {
            w /= max_abs;
        }
    }
    weight_scale = max_abs;
}

void InterpolationBasis::compute_barycentric_weights() {
    if (nodes.size() == 1) {
        barycentric_weights = {1.0};
        weighted_values = {values[0]};
        weight_scale = 1.0;
        return;
    }
    
    if (nodes.size() == 2) {
        barycentric_weights.resize(2);
        double diff = nodes[1] - nodes[0];
        barycentric_weights[0] = 1.0 / diff;
        barycentric_weights[1] = -barycentric_weights[0];
        weight_scale = 1.0;
        precompute_weighted_values();
        return;
    }
    
    if (detect_chebyshev_nodes()) {
        compute_chebyshev_weights();
        precompute_weighted_values();
        return;
    }
    
    compute_barycentric_weights_logarithmic();
    precompute_weighted_values();
}

double InterpolationBasis::evaluate_newton(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    double result = divided_differences[0];
    double product = 1.0;
    
    for (int level = 1; level < m; ++level) {
        product *= (x - nodes[level-1]);
        result += divided_differences[level] * product;
    }
    
    return result;
}

double InterpolationBasis::evaluate_lagrange(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    double result = 0.0;
    
    for (int e = 0; e < m; ++e) {
        double Le = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == e) continue;
            double denom = nodes[e] - nodes[j];
            if (std::abs(denom) < 1e-14) {
                return values[e];
            }
            Le *= (x - nodes[j]) / denom;
        }
        result += values[e] * Le;
    }
    
    return result;
}

} // namespace mixed_approx
