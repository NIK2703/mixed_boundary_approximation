#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/polynomial.h"
#include <numeric>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>

namespace mixed_approx {

// ============== WeightMultiplier implementation ==============

double WeightMultiplier::evaluate(double x) const {
    if (use_direct_evaluation) {
        // Прямое вычисление через произведение (x - z_e)
        double result = 1.0;
        for (double root : roots) {
            result *= (x - root);
        }
        return result;
    } else {
        // Использование коэффициентов полинома (если построен)
        if (coeffs.empty()) {
            throw std::runtime_error("WeightMultiplier: coefficients not built");
        }
        // Используем схему Горнера
        double result = 0.0;
        for (double coeff : coeffs) {
            result = result * x + coeff;
        }
        return result;
    }
}

void WeightMultiplier::build_from_roots(const std::vector<double>& roots_vec) {
    roots = roots_vec;
    std::sort(roots.begin(), roots.end());
    
    // Вычисляем минимальное расстояние между корнями
    min_root_distance = std::numeric_limits<double>::max();
    for (size_t i = 1; i < roots.size(); ++i) {
        double dist = roots[i] - roots[i-1];
        if (dist < min_root_distance) {
            min_root_distance = dist;
        }
    }
    if (roots.size() < 2) {
        min_root_distance = 0.0;
    }
    
    // Строим коэффициенты полинома W(x) = ∏(x - z_e)
    int m = static_cast<int>(roots.size());
    coeffs.clear();
    coeffs.resize(m + 1, 0.0);
    coeffs[0] = 1.0;  // старший коэффициент = 1
    
    for (double root : roots) {
        // Умножаем текущий полином на (x - root)
        // Проходим с конца к началу, чтобы не перезаписывать
        for (int i = m - 1; i >= 0; --i) {
            coeffs[i + 1] -= root * coeffs[i];
        }
        // coeffs[0] остаётся 1.0
    }
    
    use_direct_evaluation = false;
}

// ============== InterpolationBasis implementation ==============

// Старые реализации удалены, заменены на новые методы ниже

// ============== Новые методы для шага 2.1.2 ==============

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
        normalize_nodes(interval_start, interval_end);  // нормализует nodes
        is_normalized = true;
    }
    
    // 3. Сортировка узлов
    sort_nodes_and_values(nodes, values);
    
    // 4. Объединение близких узлов (если включено)
    if (enable_node_merging && nodes.size() > 1) {
        double interval_length = interval_end - interval_start;
        auto merged = merge_close_nodes(nodes, values, interval_length);
        
        if (merged.size() < nodes.size()) {
            // Было объединение
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
        // Кэширование weighted_values
        precompute_weighted_values();
    } else if (method == InterpolationMethod::NEWTON) {
        compute_divided_differences();
    }
    // Для Лагранжа ничего дополнительно не нужно
    
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
    
    // Порог близости: ε_close = max(1e-12, 1e-4 / m)
    double epsilon_close = std::max(1e-12, 1e-4 / m);
    double abs_tol = epsilon_close * interval_length;
    
    // Группируем близкие узлы
    for (int i = 0; i < m; ) {
        MergedNode current;
        current.x = nodes_input[i];
        current.value = values_input[i];
        current.count = 1;
        
        // Проверяем последующие узлы
        int j = i + 1;
        while (j < m) {
            if (std::abs(nodes_input[j] - current.x) < abs_tol) {
                // Объединяем
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

// Статические методы
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
    
    // Создаём индексы и сортируем их
    std::vector<size_t> indices(nodes_vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return nodes_vec[i] < nodes_vec[j]; });
    
    // Применяем перестановку
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
    
    // Нормализация весов для улучшения масштаба
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
                // Слишком близкие узлы - это ошибка, но мы уже объединили
                // Если всё же остались, используем стандартный метод
                compute_barycentric_weights_standard();
                return;
            }
            
            log_sum -= std::log(abs_diff);
            if (diff < 0) sign = -sign;
        }
        
        log_abs_weights[e] = log_sum;
        sign_weights[e] = sign;
    }
    
    // Находим максимальный логарифм для нормализации
    double max_log = *std::max_element(log_abs_weights.begin(), log_abs_weights.end());
    
    // Преобразуем обратно и нормализуем
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
    
    // Выделяем память под разделённые разности
    divided_differences.resize(m);
    
    // Копируем значения в первый столбец
    for (int i = 0; i < m; ++i) {
        divided_differences[i] = values[i];
    }
    
    // Вычисляем разделённые разности по схеме:
    // f[z_i,...,z_j] = (f[z_{i+1},...,z_j] - f[z_i,...,z_{j-1}]) / (z_j - z_i)
    // Храним в виде: divided_differences[i] = f[z_i,...,z_{level}] где level меняется
    for (int level = 1; level < m; ++level) {
        // Обратный проход для переиспользования памяти
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
    
    // Преобразуем в нормализованные координаты, если нужно
    double x_norm = x;
    if (is_normalized) {
        x_norm = (x - x_center) / x_scale;
    }
    
    // Выбор метода
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
    
    // Проверка совпадения с узлом
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            return values[k];
        }
    }
    
    // Используем кэшированные weighted_values если есть
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
        // Для других методов используем численное дифференцирование
        // (можно позже добавить аналитические формулы)
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

double InterpolationBasis::evaluate_barycentric_derivative(double x, int order) const {
    int m = static_cast<int>(nodes.size());
    
    // Проверка совпадения с узлом
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            // В узле производная может быть определена через предельный переход
            // Пока возвращаем 0 (или можно вычислить через формулу Ньютона)
            return 0.0;
        }
    }
    
    // Аналитическая формула для первой производной:
    // P'(x) = Σ w_k f_k / (x - z_k)^2 * Σ w_k/(x - z_k) - Σ w_k f_k/(x - z_k) * Σ w_k/(x - z_k)^2
    //           all over (Σ w_k/(x - z_k))^2
    
    if (order == 1) {
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
        
        double numerator = sum_wf_over_dx * sum_w_over_dx - 
                          sum_w_over_dx2 * sum_wf_over_dx;
        return numerator / denominator;
    }
    else { // order == 2
        // Вторая производная - более сложное выражение
        // Пока используем численное дифференцирование
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

// Специальные случаи

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
    
    // Узлы Чебышёва на [-1, 1]: z_k = cos(π*(2k-1)/(2m)), k=1..m
    for (int k = 0; k < m; ++k) {
        double expected = std::cos(M_PI * (2.0*k + 1.0) / (2.0 * m));
        if (std::abs(nodes[k] - expected) > tolerance) {
            return false;
        }
    }
    return true;
}

void InterpolationBasis::compute_chebyshev_weights() {
    // Для узлов Чебышёва на равномерной сетке в [-1, 1]
    // Веса: w_k = (-1)^k * sin(π*(2k-1)/(2m))
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        barycentric_weights[k] = sign * std::sin(M_PI * (2.0*k + 1.0) / (2.0 * m));
    }
    
    // Нормализация
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

// Замена старого метода compute_barycentric_weights
void InterpolationBasis::compute_barycentric_weights() {
    // Проверяем специальные случаи
    if (nodes.size() == 1) {
        barycentric_weights = {1.0};
        weighted_values = {values[0]};
        weight_scale = 1.0;
        return;
    }
    
    if (nodes.size() == 2) {
        // Для двух узлов можно использовать прямую формулу
        barycentric_weights.resize(2);
        double diff = nodes[1] - nodes[0];
        barycentric_weights[0] = 1.0 / diff;
        barycentric_weights[1] = -barycentric_weights[0];
        weight_scale = 1.0;
        precompute_weighted_values();
        return;
    }
    
    // Проверяем, равноотстоящие ли узлы
    if (detect_equally_spaced_nodes()) {
        // Для равноотстоящих узлов можно использовать оптимизированные веса
        // Но пока используем общий метод
    }
    
    // Проверяем, узлы Чебышёва ли
    if (detect_chebyshev_nodes()) {
        compute_chebyshev_weights();
        precompute_weighted_values();
        return;
    }
    
    // Стандартный логарифмический метод
    compute_barycentric_weights_logarithmic();
    precompute_weighted_values();
}


double InterpolationBasis::evaluate_newton(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    // Формула Ньютона: P(x) = f[z_0] + f[z_0,z_1]*(x-z_0) + f[z_0,z_1,z_2]*(x-z_0)*(x-z_1) + ...
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
                return values[e];  // x совпадает с узлом
            }
            Le *= (x - nodes[j]) / denom;
        }
        result += values[e] * Le;
    }
    
    return result;
}

// ============== DecompositionResult implementation ==============

Polynomial DecompositionResult::build_polynomial(const std::vector<double>& q_coeffs) const {
    if (!metadata.is_valid) {
        throw std::invalid_argument("Cannot build polynomial from invalid decomposition: " + metadata.validation_message);
    }
    
    int n_free = metadata.n_free;
    if (static_cast<int>(q_coeffs.size()) != n_free) {
        throw std::invalid_argument("Invalid number of coefficients for Q(x): expected " +
                                    std::to_string(n_free) + ", got " + std::to_string(q_coeffs.size()));
    }
    
    // Строим Q(x) как полином степени n_free - 1
    Polynomial Q(q_coeffs.empty() ? 0 : static_cast<int>(q_coeffs.size() - 1));
    if (!q_coeffs.empty()) {
        Q.setCoefficients(q_coeffs);
    }
    
    // Строим W(x) из коэффициентов (уже построены в weight_multiplier)
    Polynomial W(weight_multiplier.degree());
    if (!weight_multiplier.coeffs.empty()) {
        W.setCoefficients(weight_multiplier.coeffs);
    } else {
        // Маловероятно, но на всякий случай строим из корней
        std::vector<InterpolationNode> roots_as_nodes;
        for (double r : weight_multiplier.roots) {
            roots_as_nodes.emplace_back(r, 0.0);
        }
        W = build_interpolation_multiplier(roots_as_nodes);
    }
    
    // Вычисляем Q(x) * W(x)
    Polynomial QW = Q * W;
    
    // Получаем P_int(x) через интерполяционный базис
    // Важно: P_int(x) должен быть в исходных координатах x, а не в нормализованных.
    // interpolation_basis.nodes могут быть нормализованными, если включена нормализация.
    // Поэтому преобразуем узлы обратно к исходным координатам, если нужно.
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < interpolation_basis.nodes.size(); ++i) {
        double node_x = interpolation_basis.nodes[i];
        // Если узлы нормализованы, преобразуем обратно в исходные координаты
        if (interpolation_basis.is_normalized) {
            node_x = node_x * interpolation_basis.x_scale + interpolation_basis.x_center;
        }
        interp_nodes.emplace_back(node_x, interpolation_basis.values[i]);
    }
    Polynomial P_int = build_lagrange_polynomial(interp_nodes);
    
    // F(x) = P_int(x) + Q(x)·W(x)
    return P_int + QW;
}

double DecompositionResult::evaluate(double x, const std::vector<double>& q_coeffs) const {
    if (!metadata.is_valid) {
        throw std::invalid_argument("Cannot evaluate from invalid decomposition: " + metadata.validation_message);
    }
    
    // Вычисляем P_int(x)
    double p_int_val = interpolation_basis.evaluate(x);
    
    // Вычисляем W(x)
    double w_val = weight_multiplier.evaluate(x);
    
    // Вычисляем Q(x)
    double q_val = 0.0;
    int n_free = q_coeffs.size();
    for (int i = 0; i < n_free; ++i) {
        // q_coeffs[i] соответствует коэффициенту при x^{n_free-1-i}
        // Но для вычисления значения удобнее хранить в порядке возрастания степеней
        // Принимаем, что q_coeffs заданы в порядке [q_{n_free-1}, ..., q_0]
        q_val = q_val * x + q_coeffs[n_free - 1 - i];
    }
    
    return p_int_val + q_val * w_val;
}

// ============== Decomposer implementation ==============

DecompositionResult Decomposer::decompose(const Parameters& params) {
    DecompositionResult result;
    int n = params.polynomial_degree;
    std::vector<double> nodes;
    std::vector<double> values;
    
    // Извлекаем узлы и значения
    for (const auto& node : params.interp_nodes) {
        nodes.push_back(node.x);
        values.push_back(node.value);
    }
    
    int m = static_cast<int>(nodes.size());
    
    // 1. Проверка условия n ≥ m - 1
    if (!check_degree_condition(n, m)) {
        result.metadata.is_valid = false;
        result.metadata.validation_message =
            "Insufficient polynomial degree: n=" + std::to_string(n) +
            " < m-1=" + std::to_string(m-1) +
            ". Need at least degree " + std::to_string(m-1) + " to interpolate all nodes.";
        return result;
    }
    
    // 2. Проверка уникальности узлов
    double interval_length = params.interval_end - params.interval_start;
    std::vector<std::pair<int, int>> duplicate_pairs;
    if (!check_unique_nodes(nodes, interval_length, params.epsilon_unique, &duplicate_pairs)) {
        result.metadata.is_valid = false;
        std::ostringstream oss;
        oss << "Duplicate interpolation nodes detected (within tolerance "
            << params.epsilon_unique * interval_length << "):\n";
        for (auto& p : duplicate_pairs) {
            oss << "  nodes[" << p.first << "]=" << nodes[p.first]
                << " and nodes[" << p.second << "]=" << nodes[p.second] << "\n";
        }
        // Проверяем значения в дублирующихся узлах
        for (auto& p : duplicate_pairs) {
            if (std::abs(values[p.first] - values[p.second]) > 1e-12) {
                oss << "  CONFLICT: f(z) values differ: " << values[p.first]
                    << " vs " << values[p.second] << "\n";
            }
        }
        result.metadata.validation_message = oss.str();
        return result;
    }
    
    // 3. Проверка расположения узлов в интервале
    std::vector<int> out_of_bounds;
    if (!check_nodes_in_interval(nodes, params.interval_start, params.interval_end,
                                 params.epsilon_bound, &out_of_bounds)) {
        result.metadata.is_valid = false;
        std::ostringstream oss;
        oss << "Interpolation nodes outside interval [" << params.interval_start
            << ", " << params.interval_end << "]:\n";
        for (int idx : out_of_bounds) {
            oss << "  nodes[" << idx << "]=" << nodes[idx] << "\n";
        }
        result.metadata.validation_message = oss.str();
        return result;
    }
    
    // 4. Проверка ранга системы ограничений
    if (!check_rank_solvency(nodes, params.epsilon_rank)) {
        result.metadata.is_valid = false;
        result.metadata.validation_message =
            "Linear dependence detected in constraint system. Check for duplicate or nearly duplicate nodes.";
        return result;
    }
    
    // Все проверки пройдены
    result.metadata.is_valid = true;
    result.metadata.n_total = n;
    result.metadata.m_constraints = m;
    result.metadata.m_eff = m;  // будет обновлено после построения interpolation_basis
    result.metadata.n_free = n - m + 1;
    result.metadata.validation_message = "Decomposition successful";
    
    // 5. Предварительные вычисления
    
    // Сортируем узлы и значения
    InterpolationBasis::sort_nodes_and_values(nodes, values);
    
    // 6. Строим весовой множитель W(x)
    result.weight_multiplier.build_from_roots(nodes);
    result.metadata.min_root_distance = result.weight_multiplier.min_root_distance;
    
    // 7. Оценка масштаба
    double scale_W = estimate_weight_multiplier_scale(nodes);
    if (scale_W > 1e150 || scale_W < 1e-150) {
        result.metadata.requires_normalization = true;
        result.metadata.validation_message +=
            "\nWarning: Weight multiplier scale is " + std::to_string(scale_W) +
            ". Consider coordinate normalization.";
    }
    
    // 8. Строим базисный интерполяционный полином P_int(x) с расширенными возможностями
    result.interpolation_basis.build(
        nodes, 
        values, 
        InterpolationMethod::BARYCENTRIC,
        params.interval_start,
        params.interval_end,
        true,   // enable_normalization
        true    // enable_node_merging
    );
    
    // 9. Анализ диапазона значений
    double value_range = analyze_value_range(values);
    if (!values.empty() && value_range < 1e-12 * std::max({1.0, std::abs(values[0]), std::abs(values.back())})) {
        result.metadata.validation_message +=
            "\nWarning: Very small range of interpolation values. Task may be degenerate.";
    }
    
    // 10. Проверка на полиномиальную зависимость (опционально)
    if (m <= 4) {
        int detected_degree = detect_low_degree_polynomial(nodes, values);
        if (detected_degree > 0 && detected_degree < m - 1) {
            result.metadata.validation_message +=
                "\nNote: Interpolation points lie on a polynomial of degree " +
                std::to_string(detected_degree) + ". Consider using simpler basis.";
        }
    }
    
    return result;
}

bool Decomposer::check_rank_solvency(const std::vector<double>& nodes,
                                     double tolerance,
                                     std::vector<int>* conflict_indices) {
    // Проверяем линейную независимость строк матрицы Вандермонда
    // V = [1, z_e, z_e^2, ..., z_e^n] для всех e
    // Для этого проверяем, что все узлы различны (в матрице Вандермонда линейная зависимость
    // возникает именно при совпадении узлов)
    // Более точная проверка: вычислить определитель или ранг через SVD,
    // но для нашей цели достаточно проверки уникальности узлов
    
    int m = static_cast<int>(nodes.size());
    if (m <= 1) return true;
    
    // Проверяем попарные расстояния
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            if (std::abs(nodes[i] - nodes[j]) < tolerance) {
                if (conflict_indices) {
                    conflict_indices->push_back(i);
                    conflict_indices->push_back(j);
                }
                return false;
            }
        }
    }
    
    return true;
}

bool Decomposer::check_degree_condition(int n, int m) {
    return n >= m - 1;
}

bool Decomposer::check_unique_nodes(const std::vector<double>& nodes,
                                     double interval_length,
                                     double tolerance,
                                     std::vector<std::pair<int, int>>* duplicate_pairs) {
    int m = static_cast<int>(nodes.size());
    double abs_tol = tolerance * interval_length;
    
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int i, int j) { return nodes[i] < nodes[j]; });
    
    bool all_unique = true;
    for (int i = 1; i < m; ++i) {
        if (std::abs(nodes[indices[i]] - nodes[indices[i-1]]) <= abs_tol) {
            all_unique = false;
            if (duplicate_pairs) {
                duplicate_pairs->emplace_back(indices[i-1], indices[i]);
            }
        }
    }
    
    return all_unique;
}

bool Decomposer::check_nodes_in_interval(const std::vector<double>& nodes,
                                          double a, double b,
                                          double tolerance,
                                          std::vector<int>* out_of_bounds) {
    double abs_tol = tolerance * (b - a);
    bool all_inside = true;
    
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        if (nodes[i] < a - abs_tol || nodes[i] > b + abs_tol) {
            all_inside = false;
            if (out_of_bounds) {
                out_of_bounds->push_back(i);
            }
        }
    }
    
    return all_inside;
}

void Decomposer::prepare_sorted_nodes_and_values(
    const std::vector<InterpolationNode>& input_nodes,
    std::vector<double>* sorted_nodes,
    std::vector<double>* sorted_values) {
    
    sorted_nodes->resize(input_nodes.size());
    sorted_values->resize(input_nodes.size());
    
    for (size_t i = 0; i < input_nodes.size(); ++i) {
        (*sorted_nodes)[i] = input_nodes[i].x;
        (*sorted_values)[i] = input_nodes[i].value;
    }
    
    InterpolationBasis::sort_nodes_and_values(*sorted_nodes, *sorted_values);
}

double Decomposer::estimate_weight_multiplier_scale(const std::vector<double>& roots) {
    // Оценка масштаба: ∏_{e=1..m} max(|z_e|, 1.0)
    double scale = 1.0;
    for (double root : roots) {
        scale *= std::max(std::abs(root), 1.0);
        // Если scale становится слишком большим или маленьким, прерываем
        if (scale > 1e200 || scale < 1e-200) {
            break;
        }
    }
    return scale;
}

double Decomposer::analyze_value_range(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    double vmin = *std::min_element(values.begin(), values.end());
    double vmax = *std::max_element(values.begin(), values.end());
    return vmax - vmin;
}

int Decomposer::detect_low_degree_polynomial(const std::vector<double>& nodes,
                                              const std::vector<double>& values) {
    int m = static_cast<int>(nodes.size());
    if (m <= 2) return m;  // 2 точки всегда определяют линейный полином
    
    // Пробуем подобрать полином степени 1, 2, 3
    // Для простоты используем метод наименьших квадратов для малых m
    // и проверяем, насколько хорошо аппроксимируют
    
    std::vector<InterpolationNode> nodes_vec;
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes_vec.emplace_back(nodes[i], values[i]);
    }
    
    // Пробуем построить полином Лагранжа и посмотреть его степень
    // Но это сложно, потому что Lagrange всегда даёт степень m-1
    // Вместо этого, проверим, лежат ли точки на линии, параболе и т.д.
    
    if (m == 3) {
        // Проверка на линейность: (f2-f1)/(z2-z1) ≈ (f3-f2)/(z3-z2)
        double slope12 = (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double slope23 = (values[2] - values[1]) / (nodes[2] - nodes[1]);
        if (std::abs(slope12 - slope23) < 1e-10) {
            return 1;
        }
        return 2;  // три точки всегда лежат на параболе
    } else if (m == 4) {
        // Проверка на линейность
        double slope12 = (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double slope23 = (values[2] - values[1]) / (nodes[2] - nodes[1]);
        double slope34 = (values[3] - values[2]) / (nodes[3] - nodes[2]);
        if (std::abs(slope12 - slope23) < 1e-10 && std::abs(slope23 - slope34) < 1e-10) {
            return 1;
        }
        // Проверка на квадратичность: вторые разности постоянны
        double diff1 = (values[2] - values[1]) / (nodes[2] - nodes[1]) - 
                       (values[1] - values[0]) / (nodes[1] - nodes[0]);
        double diff2 = (values[3] - values[2]) / (nodes[3] - nodes[2]) - 
                       (values[2] - values[1]) / (nodes[2] - nodes[1]);
        double avg_second_diff = (diff1 + diff2) / 2.0;
        if (std::abs(diff1 - avg_second_diff) < 1e-8 && std::abs(diff2 - avg_second_diff) < 1e-8) {
            return 2;
        }
        return 3;  // 4 точки определяют кубический полином
    }
    
    return 0;  // не обнаружено полинома степени < m-1
}

} // namespace mixed_approx
