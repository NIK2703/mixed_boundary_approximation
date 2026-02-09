#include "mixed_approximation/variable_normalizer.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>
#include <iostream>

namespace mixed_approx {

// ============== Вспомогательные функции для нормализации ==============

namespace {
    double compute_median(std::vector<double> values) {
        if (values.empty()) return 0.0;
        
        std::sort(values.begin(), values.end());
        size_t n = values.size();
        
        if (n % 2 == 0) {
            return 0.5 * (values[n/2 - 1] + values[n/2]);
        } else {
            return values[n/2];
        }
    }

    double compute_robust_std(const std::vector<double>& values) {
        if (values.empty()) return 1.0;
        
        double median = compute_median(values);
        
        // Вычисление MAD (Median Absolute Deviation)
        std::vector<double> abs_deviations;
        abs_deviations.reserve(values.size());
        
        for (double v : values) {
            abs_deviations.push_back(std::abs(v - median));
        }
        
        double mad = compute_median(abs_deviations);
        
        // MAD × 1.4826 для оценки стандартного отклонения
        double robust_std = mad * 1.4826;
        
        // Минимум 1.0 для избежания деления на ноль
        return std::max(robust_std, 1.0);
    }

    std::vector<double> apply_log_transform(const std::vector<double>& values) {
        std::vector<double> result;
        result.reserve(values.size());
        
        double min_val = *std::min_element(values.begin(), values.end());
        double offset = (min_val <= 0) ? -min_val + 1.0 : 0.0;
        
        for (double v : values) {
            result.push_back(std::log(v + offset + 1.0));
        }
        
        return result;
    }

    bool needs_log_transform(const std::vector<double>& values) {
        if (values.empty()) return false;
        
        double min_val = *std::min_element(values.begin(), values.end());
        double max_val = *std::max_element(values.begin(), values.end());
        
        return (min_val > 0) && (max_val / min_val > 100.0);
    }
}

// ============== Конструкторы ==============

VariableNormalizer::VariableNormalizer(const ApproximationConfig& config)
    : config_(config)
    , x_strategy_(XNormalizationStrategy::LINEAR_TO_M1_1)
    , y_strategy_(YNormalizationStrategy::STANDARDIZE) {
}

VariableNormalizer::VariableNormalizer()
    : config_()
    , x_strategy_(XNormalizationStrategy::LINEAR_TO_M1_1)
    , y_strategy_(YNormalizationStrategy::STANDARDIZE) {
}

// ============== Основные методы ==============

NormalizationResult VariableNormalizer::normalize() {
    NormalizationResult result;
    result.success = false;
    
    // Шаг 1: Вычисление параметров нормализации для обеих осей (ДО проверок)
    compute_x_params();
    compute_y_params();
    
    // Шаг 2: Проверка на вырожденные случая (после инициализации params_)
    auto [is_degenerate, message] = check_degenerate_cases();
    
    if (is_degenerate) {
        result.params.status = NormalizationStatus::DEGENERATE_X_RANGE;
        result.params.message = message;
        result.diagnostic_report = generate_diagnostic_report();
        return result;
    }
    
    // Шаг 3: Проверка на константные значения по Y
    if (!params_.y_params.is_valid) {
        handle_degenerate_cases(result);
        result.diagnostic_report = generate_diagnostic_report();
        return result;
    }
    
    // Шаг 4: Вычисление нормализованных значений
    compute_normalized_values(result);
    
    // Шаг 5: Проверка валидности
    if (params_.x_params.is_valid && params_.y_params.is_valid) {
        params_.is_ready = true;
        result.success = true;
        // Копируем только необходимые поля вручную
        result.params.x_params.center = params_.x_params.center;
        result.params.x_params.scale = params_.x_params.scale;
        result.params.x_params.shift = params_.x_params.shift;
        result.params.x_params.t_scale = params_.x_params.t_scale;
        result.params.x_params.original_min = params_.x_params.original_min;
        result.params.x_params.original_max = params_.x_params.original_max;
        result.params.x_params.is_valid = params_.x_params.is_valid;
        result.params.x_params.uses_log = params_.x_params.uses_log;
        result.params.x_params.log_base = params_.x_params.log_base;
        
        result.params.y_params.center = params_.y_params.center;
        result.params.y_params.scale = params_.y_params.scale;
        result.params.y_params.shift = params_.y_params.shift;
        result.params.y_params.t_scale = params_.y_params.t_scale;
        result.params.y_params.original_min = params_.y_params.original_min;
        result.params.y_params.original_max = params_.y_params.original_max;
        result.params.y_params.is_valid = params_.y_params.is_valid;
        result.params.y_params.uses_log = params_.y_params.uses_log;
        result.params.y_params.log_base = params_.y_params.log_base;
        
        result.params.interval_a = params_.interval_a;
        result.params.interval_b = params_.interval_b;
        result.params.norm_a = params_.norm_a;
        result.params.norm_b = params_.norm_b;
        result.params.gamma_correction_factor = params_.gamma_correction_factor;
        result.params.x_strategy = params_.x_strategy;
        result.params.y_strategy = params_.y_strategy;
        result.params.status = params_.status;
        result.params.message = params_.message;
        result.params.warnings = params_.warnings;
        result.params.is_ready = params_.is_ready;
        
        result.diagnostic_report = generate_diagnostic_report();
    }
    
    return result;
}

void VariableNormalizer::compute_x_params() {
    AxisNormalizationParams& p = params_.x_params;
    
    // Сбор всех x-координат из всех наборов данных
    double x_min = config_.interval_start;
    double x_max = config_.interval_end;
    
    // Аппроксимирующие точки
    for (const auto& pt : config_.approx_points) {
        x_min = std::min(x_min, pt.x);
        x_max = std::max(x_max, pt.x);
    }
    
    // Отталкивающие точки
    for (const auto& pt : config_.repel_points) {
        x_min = std::min(x_min, pt.x);
        x_max = std::max(x_max, pt.x);
    }
    
    // Интерполяционные узлы
    for (const auto& node : config_.interp_nodes) {
        x_min = std::min(x_min, node.x);
        x_max = std::max(x_max, node.x);
    }
    
    // Сохранение исходных границ
    p.original_min = x_min;
    p.original_max = x_max;
    params_.interval_a = config_.interval_start;
    params_.interval_b = config_.interval_end;
    
    // Вычисление центра и диапазона
    p.center = 0.5 * (x_min + x_max);
    double x_range = x_max - x_min;
    
    // Адаптивный порог для определения вырожденности
    double adaptive_eps = EPS_RANGE * std::max(1.0, std::abs(p.center));
    x_range = std::max(x_range, adaptive_eps);
    
    // Выбор стратегии нормализации
    switch (x_strategy_) {
        case XNormalizationStrategy::LINEAR_TO_M1_1:
            // Преобразование в [-1, 1]
            p.t_scale = 2.0 / x_range;
            p.shift = -p.center * p.t_scale;
            p.scale = x_range / 2.0;
            params_.norm_a = -1.0;
            params_.norm_b = 1.0;
            break;
            
        case XNormalizationStrategy::LINEAR_TO_0_1:
            // Преобразование в [0, 1]
            p.t_scale = 1.0 / x_range;
            p.shift = -x_min * p.t_scale;
            p.scale = x_range;
            params_.norm_a = 0.0;
            params_.norm_b = 1.0;
            break;
            
        case XNormalizationStrategy::ADAPTIVE:
            // Адаптивная стратегия: выбираем на основе симметрии данных
            if (std::abs(p.center) < 0.1 * x_range) {
                // Данные симметричны относительно нуля → [-1, 1]
                p.t_scale = 2.0 / x_range;
                p.shift = -p.center * p.t_scale;
                p.scale = x_range / 2.0;
                params_.norm_a = -1.0;
                params_.norm_b = 1.0;
            } else {
                // Данные несимметричны → [0, 1]
                p.t_scale = 1.0 / x_range;
                p.shift = -x_min * p.t_scale;
                p.scale = x_range;
                params_.norm_a = 0.0;
                params_.norm_b = 1.0;
            }
            break;
    }
    
    p.is_valid = true;
    
    // Вычисление коррекции для γ
    // γ_normalized = γ_original * (x_range/2)^3
    double x_range_over_2 = x_range / 2.0;
    params_.gamma_correction_factor = std::pow(x_range_over_2, 3.0);
}

void VariableNormalizer::compute_y_params() {
    AxisNormalizationParams& p = params_.y_params;
    
    // Сбор всех y-значений
    std::vector<double> y_values;
    
    // Аппроксимирующие точки: f(x_i)
    for (const auto& pt : config_.approx_points) {
        y_values.push_back(pt.value);
    }
    
    // Отталкивающие точки: y_j^*
    for (const auto& pt : config_.repel_points) {
        y_values.push_back(pt.y_forbidden);
    }
    
    // Интерполяционные узлы: f(z_e)
    for (const auto& node : config_.interp_nodes) {
        y_values.push_back(node.value);
    }
    
    if (y_values.empty()) {
        p.is_valid = false;
        return;
    }
    
    // Проверка на необходимость логарифмического преобразования
    bool apply_log = needs_log_transform(y_values);
    
    if (apply_log) {
        p.uses_log = true;
        y_values = apply_log_transform(y_values);
    }
    
    // Вычисление параметров в зависимости от стратегии
    switch (y_strategy_) {
        case YNormalizationStrategy::STANDARDIZE:
            // Стандартизация с использованием медианы и MAD
            p.center = compute_median(y_values);
            p.scale = compute_robust_std(y_values);
            break;
            
        case YNormalizationStrategy::LINEAR:
            // Простое линейное преобразование
            p.center = compute_median(y_values);
            p.scale = (p.scale > EPS_SCALE) ? p.scale : 1.0;
            break;
            
        case YNormalizationStrategy::LOG_LINEAR:
            // Логарифмическое преобразование уже применено выше
            p.center = compute_median(y_values);
            p.scale = compute_robust_std(y_values);
            break;
    }
    
    // Адаптивная коррекция масштаба
    double adaptive_eps = EPS_SCALE * std::max(1.0, std::abs(p.center));
    p.scale = std::max(p.scale, adaptive_eps);
    
    // Вычисление параметров преобразования
    p.t_scale = 1.0 / p.scale;
    p.shift = -p.center * p.t_scale;
    
    // Сохранение исходных границ
    if (!y_values.empty()) {
        p.original_min = *std::min_element(y_values.begin(), y_values.end());
        p.original_max = *std::max_element(y_values.begin(), y_values.end());
    }
    
    // Проверка на константные значения
    if (std::abs(p.original_max - p.original_min) < adaptive_eps) {
        p.is_valid = false;
        params_.status = NormalizationStatus::CONSTANT_Y_VALUES;
        params_.message = "Все значения Y константны - задача тривиальна";
    } else {
        p.is_valid = true;
    }
}

void VariableNormalizer::compute_normalized_values(NormalizationResult& result) {
    // Нормализация аппроксимирующих точек
    for (const auto& pt : config_.approx_points) {
        result.approx_x_norm.push_back(params_.transform_x(pt.x));
        result.approx_f_norm.push_back(params_.transform_y(pt.value));
        // Веса масштабируются пропорционально масштабу Y
        result.approx_weight_norm.push_back(pt.weight * params_.y_params.t_scale);
    }
    
    // Нормализация отталкивающих точек
    for (const auto& pt : config_.repel_points) {
        result.repel_y_norm.push_back(params_.transform_x(pt.x));
        result.repel_forbidden_norm.push_back(params_.transform_y(pt.y_forbidden));
        // Веса B_j НЕ масштабируются - они управляют относительной силой барьера
    }
    
    // Нормализация интерполяционных узлов
    for (const auto& node : config_.interp_nodes) {
        result.interp_z_norm.push_back(params_.transform_x(node.x));
        result.interp_f_norm.push_back(params_.transform_y(node.value));
    }
    
    // Коррекция параметра регуляризации
    result.gamma_normalized = config_.gamma * params_.gamma_correction_factor;
    
    result.success = true;
}

void VariableNormalizer::handle_degenerate_cases(NormalizationResult& result) {
    auto [is_degenerate, message] = check_degenerate_cases();
    
    if (is_degenerate) {
        result.success = false;
        result.params.status = NormalizationStatus::DEGENERATE_X_RANGE;
        result.params.message = message;
        return;
    }
    
    // Обработка константных значений Y
    if (!params_.y_params.is_valid) {
        result.success = true;  // Это не ошибка, а тривиальный случай
        
        // Для константных данных возвращаем "нулевой" результат
        // (оптимизация не требуется - результат будет константный полином)
        result.params.status = NormalizationStatus::CONSTANT_Y_VALUES;
        result.params.message = "Все значения Y константны - возвращаем константный полином";
        
        // Все нормализованные значения устанавливаем в 0
        for (size_t i = 0; i < config_.approx_points.size(); ++i) {
            result.approx_x_norm.push_back(params_.transform_x(config_.approx_points[i].x));
            result.approx_f_norm.push_back(0.0);
            result.approx_weight_norm.push_back(config_.approx_points[i].weight);
        }
        
        for (size_t i = 0; i < config_.repel_points.size(); ++i) {
            result.repel_y_norm.push_back(params_.transform_x(config_.repel_points[i].x));
            result.repel_forbidden_norm.push_back(0.0);
        }
        
        result.gamma_normalized = config_.gamma;
    }
}

// ============== Обратное преобразование коэффициентов ==============

std::vector<double> VariableNormalizer::inverse_transform_coefficients(
    const std::vector<double>& normalized_coeffs) const {
    return inverse_transform_monomial(normalized_coeffs);
}

std::vector<double> VariableNormalizer::inverse_transform_monomial(
    const std::vector<double>& normalized_coeffs) const {
    
    if (!params_.is_ready || normalized_coeffs.empty()) {
        return normalized_coeffs;
    }
    
    int n = static_cast<int>(normalized_coeffs.size()) - 1;  // степень полинома
    
    // Предварительно вычисляем биномиальные коэффициенты C(k, l)
    auto binom = compute_binomial_coefficients(n);
    
    // Предварительно вычисляем степени t_shift и t_scale
    auto shift_powers = compute_powers(params_.x_params.shift, n);
    auto scale_powers = compute_powers(params_.x_params.t_scale, n);
    
    // Временный вектор для накопления коэффициентов до масштабирования Y
    std::vector<double> temp_coeffs(n + 1, 0.0);
    
    // Алгоритм из документации:
    // для l от 0 до n:
    //     a_l_temp = Σ_{k=l..n} g_k · C(k,l) · t_scale^l · t_shift^{k-l}
    
    for (int l = 0; l <= n; ++l) {
        double sum = 0.0;
        for (int k = l; k <= n; ++k) {
            double gk = normalized_coeffs[n - k];  // коэффициенты хранятся [a_n, ..., a_0]
            double Ckl = binom[k][l];
            double scale_factor = scale_powers[l];
            double shift_factor = shift_powers[k - l];
            sum += gk * Ckl * scale_factor * shift_factor;
        }
        temp_coeffs[l] = sum;
    }
    
    // Применение масштаба ординаты Y
    std::vector<double> final_coeffs(n + 1, 0.0);
    for (int l = 0; l <= n; ++l) {
        final_coeffs[l] = params_.y_params.scale * temp_coeffs[l];
    }
    
    // Добавление сдвига для константного члена
    final_coeffs[n] += params_.y_params.center;
    
    return final_coeffs;
}

std::vector<double> VariableNormalizer::forward_transform_coefficients(
    const std::vector<double>& original_coeffs) const {
    
    if (!params_.is_ready || original_coeffs.empty()) {
        return original_coeffs;
    }
    
    int n = static_cast<int>(original_coeffs.size()) - 1;
    
    // Шаг 1: Удаление y_center (центрирование)
    // F(x) = y_center + y_scale * G(t), где G(t) - полином в нормализованных координатах
    // G(t) = (F(x) - y_center) / y_scale
    std::vector<double> centered_coeffs = original_coeffs;
    centered_coeffs[n] -= params_.y_params.center;
    
    // Шаг 2: Деление на y_scale
    for (auto& c : centered_coeffs) {
        c /= params_.y_params.scale;
    }
    
    // Шаг 3: Преобразование x -> t (полиномиальное)
    // x = (t - shift) / t_scale = t/t_scale - shift/t_scale
    // t = t_scale * x + shift
    // G(t_scale * x + shift) = sum_{k=0}^n a_k * (t_scale * x + shift)^k
    //
    // Для мономиального базиса используем биномиальное разложение:
    // (t_scale * x + shift)^k = sum_{l=0}^k C(k,l) * (t_scale * x)^l * shift^{k-l}
    // = sum_{l=0}^k C(k,l) * t_scale^l * shift^{k-l} * x^l
    
    auto binom = compute_binomial_coefficients(n);
    auto shift_powers = compute_powers(params_.x_params.shift, n);
    auto scale_powers = compute_powers(params_.x_params.t_scale, n);
    
    std::vector<double> result(n + 1, 0.0);
    
    // result[l] = sum_{k=l..n} a_k * C(k,l) * t_scale^l * shift^{k-l}
    for (int l = 0; l <= n; ++l) {
        double sum = 0.0;
        for (int k = l; k <= n; ++k) {
            double ak = centered_coeffs[k];  // [a_n, ..., a_0]
            double Ckl = binom[k][l];
            double scale_factor = scale_powers[l];       // t_scale^l
            double shift_factor = shift_powers[k - l];   // shift^{k-l}
            sum += ak * Ckl * scale_factor * shift_factor;
        }
        result[l] = sum;
    }
    
    return result;
}

// ============== Вспомогательные методы вычисления ==============

std::vector<std::vector<double>> VariableNormalizer::compute_binomial_coefficients(int n) const {
    if (n < 0) {
        return {{1.0}};  // Для константного полинома степени 0
    }
    
    std::vector<std::vector<double>> binom(n + 1, std::vector<double>(n + 1, 0.0));
    
    for (int i = 0; i <= n; ++i) {
        binom[i][0] = binom[i][i] = 1.0;
        for (int j = 1; j < i; ++j) {
            binom[i][j] = binom[i-1][j-1] + binom[i-1][j];
        }
    }
    
    return binom;
}

std::vector<double> VariableNormalizer::compute_powers(double base, int max_power) const {
    std::vector<double> powers(max_power + 1, 1.0);
    for (int i = 1; i <= max_power; ++i) {
        powers[i] = powers[i-1] * base;
    }
    return powers;
}

// ============== Валидация и диагностика ==============

std::pair<bool, std::string> VariableNormalizer::check_degenerate_cases() const {
    // Проверка на вырожденный диапазон X
    double x_min = params_.x_params.original_min;
    double x_max = params_.x_params.original_max;
    double x_center = params_.x_params.center;
    double adaptive_eps = EPS_RANGE * std::max(1.0, std::abs(x_center));
    
    if (std::abs(x_max - x_min) < adaptive_eps) {
        return {true, "Вырожденный диапазон по оси X: все точки имеют одинаковую абсциссу"};
    }
    
    return {false, ""};
}

bool VariableNormalizer::validate_normalization() {
    // Выполняем нормализацию если ещё не выполнена
    if (!params_.is_ready) {
        auto result = normalize();
        if (!result.success) {
            return false;
        }
    }
    
    // Запуск тестов
    bool test1 = test_interpolation_invariance();
    bool test2 = test_barrier_distance_invariance();
    bool test3 = test_coefficient_precision();
    
    return test1 && test2 && test3;
}

bool VariableNormalizer::test_interpolation_invariance() const {
    if (!params_.is_ready) return false;
    
    constexpr double tol = 1e-10;
    
    for (size_t i = 0; i < config_.interp_nodes.size(); ++i) {
        double t = params_.transform_x(config_.interp_nodes[i].x);
        
        // Проверяем, что t в пределах нормализованного диапазона
        if (t < params_.norm_a - 1e-9 || t > params_.norm_b + 1e-9) {
            return false;
        }
        
        // Проверяем симметрию для [-1, 1]
        if (std::abs(params_.norm_a - (-1.0)) < 1e-9 && std::abs(params_.norm_b - 1.0) < 1e-9) {
            // Для симметричного преобразования: t_a = -1, t_b = 1
            if (std::abs(params_.transform_x(params_.interval_a) - (-1.0)) > tol ||
                std::abs(params_.transform_x(params_.interval_b) - 1.0) > tol) {
                return false;
            }
        }
    }
    
    return true;
}

bool VariableNormalizer::test_barrier_distance_invariance() const {
    if (!params_.is_ready) return false;
    
    constexpr double tol = 1e-8;
    
    for (size_t i = 0; i < config_.repel_points.size(); ++i) {
        double y_original = config_.repel_points[i].y_forbidden;
        double y_normalized = params_.transform_y(y_original);
        
        // Проверяем, что преобразование Y корректно
        double y_recovered = params_.inverse_transform_y(y_normalized);
        if (std::abs(y_original - y_recovered) > tol * std::max(1.0, std::abs(y_original))) {
            return false;
        }
    }
    
    return true;
}

bool VariableNormalizer::test_coefficient_precision() const {
    if (!params_.is_ready || config_.approx_points.empty()) return true;
    
    constexpr double tol = 1e-10;
    
    // Создаём тестовые коэффициенты
    std::vector<double> test_coeffs = {1.0, 0.5, 0.25, 0.125};
    
    // Прямое преобразование
    std::vector<double> normalized = forward_transform_coefficients(test_coeffs);
    
    // Обратное преобразование
    std::vector<double> recovered = inverse_transform_coefficients(normalized);
    
    // Сравнение
    for (size_t i = 0; i < test_coeffs.size(); ++i) {
        double rel_error = std::abs(test_coeffs[i] - recovered[i]) / 
                           std::max(std::abs(test_coeffs[i]), 1.0);
        if (rel_error > tol) {
            return false;
        }
    }
    
    return true;
}

std::string VariableNormalizer::generate_diagnostic_report() const {
    std::ostringstream oss;
    
    oss << "=== Отчёт о нормализации переменных (Шаг 5.2) ===\n\n";
    
    // Параметры X
    oss << "Ось X (абсцисса):\n";
    oss << "  Исходный диапазон: [" << params_.x_params.original_min << ", " 
        << params_.x_params.original_max << "]\n";
    oss << "  Центр: " << params_.x_params.center << "\n";
    oss << "  Масштаб (t_scale): " << params_.x_params.t_scale << "\n";
    oss << "  Сдвиг (t_shift): " << params_.x_params.shift << "\n";
    oss << "  Нормализованный диапазон: [" << params_.norm_a << ", " << params_.norm_b << "]\n";
    
    // Параметры Y
    oss << "\nОсь Y (ордината):\n";
    oss << "  Центр: " << params_.y_params.center << "\n";
    oss << "  Масштаб: " << params_.y_params.scale << "\n";
    oss << "  Масштаб (v_scale): " << params_.y_params.t_scale << "\n";
    oss << "  Сдвиг (v_shift): " << params_.y_params.shift << "\n";
    if (params_.y_params.uses_log) {
        oss << "  Логарифмическое преобразование: применено\n";
    }
    
    // Коррекция γ
    oss << "\nКоррекция регуляризации:\n";
    oss << "  Корректирующий фактор: " << params_.gamma_correction_factor << "\n";
    
    // Статус
    oss << "\nСтатус: ";
    switch (params_.status) {
        case NormalizationStatus::SUCCESS:
            oss << "Успех";
            break;
        case NormalizationStatus::DEGENERATE_X_RANGE:
            oss << "Вырожденный диапазон X";
            break;
        case NormalizationStatus::CONSTANT_Y_VALUES:
            oss << "Константные значения Y";
            break;
        case NormalizationStatus::LOGARITHMIC_APPLIED:
            oss << "Применено логарифмическое преобразование";
            break;
        case NormalizationStatus::ADAPTIVE_SCALING_USED:
            oss << "Использована адаптивная стратегия";
            break;
        case NormalizationStatus::WARNING:
            oss << "Предупреждения";
            break;
    }
    oss << "\n";
    
    if (!params_.message.empty()) {
        oss << "Сообщение: " << params_.message << "\n";
    }
    
    // Статистика данных
    oss << "\nСтатистика данных:\n";
    oss << "  Аппроксимирующих точек: " << config_.approx_points.size() << "\n";
    oss << "  Отталкивающих точек: " << config_.repel_points.size() << "\n";
    oss << "  Интерполяционных узлов: " << config_.interp_nodes.size() << "\n";
    
    return oss.str();
}

std::string VariableNormalizer::get_diagnostic_report() const {
    return generate_diagnostic_report();
}

// ============== Реализация вспомогательных функций NormalizationUtils ==============

namespace NormalizationUtils {

double compute_median(std::vector<double> values) {
    return ::mixed_approx::compute_median(std::move(values));
}

double compute_robust_std(const std::vector<double>& values) {
    return ::mixed_approx::compute_robust_std(values);
}

bool is_degenerate_range(double min_val, double max_val, double center) {
    double adaptive_eps = 1e-12 * std::max(1.0, std::abs(center));
    return std::abs(max_val - min_val) < adaptive_eps;
}

bool is_constant_values(const std::vector<double>& values, double center) {
    if (values.empty()) return true;
    
    double adaptive_eps = 1e-12 * std::max(1.0, std::abs(center));
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    
    return std::abs(max_val - min_val) < adaptive_eps;
}

std::vector<double> apply_log_transform(const std::vector<double>& values) {
    return ::mixed_approx::apply_log_transform(values);
}

bool needs_log_transform(const std::vector<double>& values) {
    return ::mixed_approx::needs_log_transform(values);
}

} // namespace NormalizationUtils

} // namespace mixed_approx
