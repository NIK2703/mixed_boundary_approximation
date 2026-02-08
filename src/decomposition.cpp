#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/polynomial.h"
#include <numeric>
#include <iomanip>

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

void InterpolationBasis::build(const std::vector<double>& nodes_vec,
                               const std::vector<double>& values_vec,
                               InterpolationMethod meth) {
    nodes = nodes_vec;
    values = values_vec;
    method = meth;
    
    // Сортируем узлы и значения для устойчивости
    sort_nodes_and_values(nodes, values);
    
    switch (method) {
        case InterpolationMethod::BARYCENTRIC:
            compute_barycentric_weights();
            break;
        case InterpolationMethod::NEWTON:
            compute_divided_differences();
            break;
        case InterpolationMethod::LAGRANGE:
            // Для Лагранжа ничего дополнительно не нужно
            break;
    }
}

double InterpolationBasis::evaluate(double x) const {
    if (nodes.empty()) {
        return 0.0;
    }
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        return evaluate_barycentric(x);
    } else if (method == InterpolationMethod::NEWTON) {
        return evaluate_newton(x);
    } else {
        return evaluate_lagrange(x);
    }
}

bool InterpolationBasis::are_nodes_unique(const std::vector<double>& nodes_vec, double tolerance) {
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

void InterpolationBasis::compute_barycentric_weights() {
    int m = static_cast<int>(nodes.size());
    barycentric_weights.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double weight = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == k) continue;
            weight /= (nodes[k] - nodes[j]);
        }
        barycentric_weights[k] = weight;
    }
}

double InterpolationBasis::evaluate_barycentric(double x) const {
    int m = static_cast<int>(nodes.size());
    
    // Проверяем, совпадает ли x с одним из узлов
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-14) {
            return values[k];  // Точное значение в узле
        }
    }
    
    // Барицентрическая формула:
    // P(x) = Σ w_k * f_k / (x - z_k) / Σ w_k / (x - z_k)
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int k = 0; k < m; ++k) {
        double diff = x - nodes[k];
        double term = barycentric_weights[k] / diff;
        numerator += term * values[k];
        denominator += term;
    }
    
    if (std::abs(denominator) < 1e-14) {
        // Математически это возможно только при совпадении x с узлом,
        // но на всякий случай
        return 0.0;
    }
    
    return numerator / denominator;
}

void InterpolationBasis::compute_divided_differences() {
    int m = static_cast<int>(nodes.size());
    divided_differences = values;  // начинаем с f[z_k]
    
    // Вычисляем разделённые разности
    for (int level = 1; level < m; ++level) {
        for (int k = m - 1; k >= level; --k) {
            double denom = nodes[k] - nodes[k - level];
            if (std::abs(denom) < 1e-14) {
                divided_differences[k] = 0.0;
            } else {
                divided_differences[k] = (divided_differences[k] - divided_differences[k-1]) / denom;
            }
        }
    }
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
    // P_int(x) уже построен в interpolation_basis, просто вычислим его значение в виде полинома
    // Для простоты строим заново через Лагранжа (так как это не происходит часто)
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < interpolation_basis.nodes.size(); ++i) {
        interp_nodes.emplace_back(interpolation_basis.nodes[i], interpolation_basis.values[i]);
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
    
    // 8. Строим базисный интерполяционный полином P_int(x)
    result.interpolation_basis.build(nodes, values, InterpolationMethod::BARYCENTRIC);
    
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
