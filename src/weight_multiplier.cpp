#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/polynomial.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mixed_approx {

double WeightMultiplier::evaluate(double x) const {
    if (use_direct_evaluation) {
        // Прямое вычисление через произведение (x - z_e) в нормализованных координатах
        double x_norm = (x - shift) / scale;
        double result = 1.0;
        for (double root : roots_norm) {
            result *= (x_norm - root);
        }
        // Масштабируем обратно: W(x) = W_norm(x_norm) * scale^m
        result *= std::pow(scale, static_cast<double>(roots.size()));
        return result;
    } else {
        // Использование коэффициентов полинома (в исходных координатах)
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

double WeightMultiplier::evaluate_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("WeightMultiplier::evaluate_derivative: order must be 1 or 2");
    }
    
    int m = degree();
    if (m == 0) {
        return 0.0;  // W(x) = 1, производная = 0
    }
    
    // Преобразуем в нормализованные координаты, если нужно
    double x_work = x;
    if (is_normalized && scale != 0.0) {
        x_work = (x - shift) / scale;
    }
    
    const std::vector<double>& working_roots = is_normalized ? roots_norm : roots;
    
    // Проверяем, совпадает ли x_work с каким-либо корнем (для простых корней)
    bool at_root = false;
    int root_index = -1;
    for (size_t i = 0; i < working_roots.size(); ++i) {
        if (std::abs(x_work - working_roots[i]) < 1e-12) {
            at_root = true;
            root_index = static_cast<int>(i);
            break;
        }
    }
    
    if (at_root && order == 1) {
        // Вычисляем W'(z_e) аналитически: W'(z_e) = ∏_{k≠e} (z_e - z_k)
        double result = 1.0;
        for (size_t k = 0; k < working_roots.size(); ++k) {
            if (static_cast<int>(k) == root_index) continue;
            result *= (x_work - working_roots[k]);
        }
        // Если нормализация, применяем масштаб: W'(x) = W_norm'(x_norm) * scale^{m-1}
        if (is_normalized) {
            result *= std::pow(scale, static_cast<double>(m - 1));
        }
        return result;
    }
    
    if (at_root && order == 2) {
        // W''(z_e) = 2 * (∏_{j≠e} (z_e - z_j)) * (Σ_{j≠e} 1/(z_e - z_j))
        double P = 1.0;  // произведение (z_e - z_j) для j≠e
        double S = 0.0;  // сумма 1/(z_e - z_j) для j≠e
        for (size_t k = 0; k < working_roots.size(); ++k) {
            if (static_cast<int>(k) == root_index) continue;
            double diff = x_work - working_roots[k];
            P *= diff;
            S += 1.0 / diff;
        }
        double result = 2.0 * P * S;
        if (is_normalized) {
            result *= std::pow(scale, static_cast<double>(m - 2));
        }
        return result;
    }
    
    // Обычный случай (x не совпадает с корнями)
    if (order == 1) {
        // W'(x) = W(x) * Σ_{e=1..m} 1/(x - z_e)
        double W_val = 1.0;
        double sum_inv = 0.0;
        
        for (double root : working_roots) {
            double diff = x_work - root;
            W_val *= diff;
            sum_inv += 1.0 / diff;
        }
        
        double result = W_val * sum_inv;
        
        // Если нормализация была, применяем цепное правило
        if (is_normalized) {
            result *= std::pow(scale, static_cast<double>(m - 1));
        }
        
        return result;
    } else { // order == 2
        // W''(x) = W(x) * ([Σ 1/(x - z_e)]² - Σ 1/(x - z_e)²)
        double W_val = 1.0;
        double sum_inv = 0.0;
        double sum_inv2 = 0.0;
        
        for (double root : working_roots) {
            double diff = x_work - root;
            W_val *= diff;
            double inv = 1.0 / diff;
            sum_inv += inv;
            sum_inv2 += inv * inv;
        }
        
        double result = W_val * (sum_inv * sum_inv - sum_inv2);
        
        // Нормализация
        if (is_normalized) {
            result *= std::pow(scale, static_cast<double>(m - 2));
        }
        
        return result;
    }
}

void WeightMultiplier::build_from_roots(const std::vector<double>& roots_vec,
                                        double interval_start,
                                        double interval_end,
                                        bool enable_normalization) {
    if (roots_vec.empty()) {
        // Специальный случай: m = 0
        roots.clear();
        roots_norm.clear();
        coeffs = {1.0};
        min_root_distance = 0.0;
        use_direct_evaluation = false;
        is_normalized = false;
        shift = 0.0;
        scale = 1.0;
        return;
    }
    
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
    
    // Определяем, требуется ли нормализация
    is_normalized = false;
    shift = 0.0;
    scale = 1.0;
    
    if (enable_normalization) {
        double interval_length = interval_end - interval_start;
        if (interval_length > 0) {
            // Оценим разброс корней
            double min_root = *std::min_element(roots.begin(), roots.end());
            double max_root = *std::max_element(roots.begin(), roots.end());
            double range = max_root - min_root;
            double max_abs = std::max(std::abs(min_root), std::abs(max_root));
            
            // Если разброс слишком большой или корни далеко от нуля, нормализуем
            if (range > 0.1 * interval_length || max_abs > 2.0 * interval_length) {
                is_normalized = true;
                shift = (interval_start + interval_end) / 2.0;
                scale = interval_length / 2.0;
            }
        }
    }
    
    // Нормализуем корни если нужно
    if (is_normalized) {
        roots_norm.resize(roots.size());
        for (size_t i = 0; i < roots.size(); ++i) {
            roots_norm[i] = (roots[i] - shift) / scale;
        }
    }
    
    // Строим коэффициенты полинома W(x) = ∏(x - z_e) В ИСХОДНЫХ КООРДИНАТАХ
    int m = static_cast<int>(roots.size());
    coeffs.clear();
    coeffs.resize(m + 1, 0.0);
    coeffs[0] = 1.0;  // старший коэффициент = 1
    
    // Умножаем на линейные множители используя ИСХОДНЫЕ корни
    for (double root : roots) {
        for (int i = m - 1; i >= 0; --i) {
            coeffs[i + 1] -= root * coeffs[i];
        }
    }
    
    use_direct_evaluation = is_normalized;
    caches_ready = false;
}

bool WeightMultiplier::verify_construction(double tolerance) const {
    int m = degree();
    if (m == 0) {
        return coeffs.size() == 1 && std::abs(coeffs[0] - 1.0) < tolerance;
    }
    
    // 1. Проверка моничности: старший коэффициент должен быть 1
    if (coeffs.empty() || std::abs(coeffs[0] - 1.0) > tolerance) {
        return false;
    }
    
    // 2. Проверка W(z_e) ≈ 0 для всех корней
    for (size_t i = 0; i < roots.size(); ++i) {
        double z_e = roots[i];
        double W_at_z = evaluate(z_e);
        double abs_tol = tolerance * std::max(1.0, std::pow(std::abs(z_e), static_cast<double>(m)));
        if (std::abs(W_at_z) > abs_tol) {
            return false;
        }
    }
    
    // 3. Проверка согласованности представлений
    if (!coeffs.empty()) {
        std::vector<double> test_points = {0.0, 0.5, 1.0, -0.5, -1.0};
        for (double x : test_points) {
            double val_coeffs = 0.0;
            for (double coeff : coeffs) {
                val_coeffs = val_coeffs * x + coeff;
            }
            double val_direct = 1.0;
            for (double root : roots) {
                val_direct *= (x - root);
            }
            if (std::abs(val_coeffs - val_direct) > tolerance * std::max(1.0, std::abs(val_direct))) {
                return false;
            }
        }
    }
    
    return true;
}

void WeightMultiplier::build_caches(const std::vector<double>& points_x,
                                    const std::vector<double>& points_y) {
    cache_x_vals.clear();
    cache_x_deriv1.clear();
    cache_x_deriv2.clear();
    cache_y_vals.clear();
    cache_y_deriv1.clear();
    cache_y_deriv2.clear();
    
    for (double x : points_x) {
        cache_x_vals.push_back(evaluate(x));
        cache_x_deriv1.push_back(evaluate_derivative(x, 1));
        cache_x_deriv2.push_back(evaluate_derivative(x, 2));
    }
    
    for (double y : points_y) {
        cache_y_vals.push_back(evaluate(y));
        cache_y_deriv1.push_back(evaluate_derivative(y, 1));
        cache_y_deriv2.push_back(evaluate_derivative(y, 2));
    }
    
    caches_ready = true;
}

void WeightMultiplier::clear_caches() {
    cache_x_vals.clear();
    cache_x_deriv1.clear();
    cache_x_deriv2.clear();
    cache_y_vals.clear();
    cache_y_deriv1.clear();
    cache_y_deriv2.clear();
    caches_ready = false;
}

std::vector<double> WeightMultiplier::multiply_by_Q(const std::vector<double>& q_coeffs) const {
    if (coeffs.empty()) {
        throw std::runtime_error("WeightMultiplier: coefficients not built");
    }
    if (q_coeffs.empty()) {
        return {};
    }
    
    int deg_W = degree();
    int deg_Q = static_cast<int>(q_coeffs.size()) - 1;
    int deg_result = deg_W + deg_Q;
    
    std::vector<double> result(deg_result + 1, 0.0);
    
    for (int i = 0; i <= deg_Q; ++i) {
        for (int j = 0; j <= deg_W; ++j) {
            int k = i + j;
            result[k] += q_coeffs[i] * coeffs[j];
        }
    }
    
    return result;
}

double WeightMultiplier::evaluate_product(double x, const std::vector<double>& q_coeffs) const {
    if (q_coeffs.empty()) {
        return 0.0;
    }
    
    double q_val = 0.0;
    for (double coeff : q_coeffs) {
        q_val = q_val * x + coeff;
    }
    
    double w_val = evaluate(x);
    return q_val * w_val;
}

} // namespace mixed_approx
