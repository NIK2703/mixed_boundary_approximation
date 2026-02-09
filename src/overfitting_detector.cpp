#include "mixed_approximation/overfitting_detector.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/mixed_approximation.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>

namespace mixed_approx {

// ============== Реализация структур результатов ==============

std::string OverfittingDiagnostics::format_report() const {
    std::ostringstream oss;
    oss << "=== OVERFITTING DIAGNOSTICS REPORT ===\n\n";
    
    // Кривизна
    oss << "1. Curvature Metric (κ):\n";
    oss << "   Normalized curvature: " << std::fixed << std::setprecision(4) << curvature.normalized_curvature << "\n";
    oss << "   Threshold: " << curvature.threshold << "\n";
    oss << "   Status: " << (curvature.is_over_threshold ? "EXCEEDED" : "OK") << "\n\n";
    
    // Осцилляции
    oss << "2. Oscillation Metric:\n";
    oss << "   Total extrema: " << oscillation.total_extrema << "\n";
    oss << "   Extrema in empty regions: " << oscillation.extrema_in_empty_regions << "\n";
    oss << "   Oscillation score: " << std::fixed << std::setprecision(4) << oscillation.oscillation_score << "\n\n";
    
    // Кросс-валидация
    oss << "3. Cross-Validation Metric:\n";
    oss << "   Train error (RMS): " << std::fixed << std::setprecision(6) << cross_validation.train_error << "\n";
    oss << "   CV error (RMS): " << cross_validation.cv_error << "\n";
    oss << "   Generalization ratio: " << std::fixed << std::setprecision(4) << cross_validation.generalization_ratio << "\n\n";
    
    // Чувствительность
    oss << "4. Sensitivity Metric:\n";
    oss << "   Sensitivity score: " << std::fixed << std::setprecision(6) << sensitivity.sensitivity_score << "\n";
    oss << "   Perturbations: " << sensitivity.perturbation_count << "\n\n";
    
    // Итоговый риск
    oss << "=== FINAL RISK ASSESSMENT ===\n";
    oss << "   Overall risk score: " << std::fixed << std::setprecision(4) << risk_score << "/1.0\n";
    oss << "   Risk level: ";
    switch (risk_level) {
        case OverfittingRiskLevel::LOW: oss << "LOW"; break;
        case OverfittingRiskLevel::MODERATE: oss << "MODERATE"; break;
        case OverfittingRiskLevel::HIGH: oss << "HIGH"; break;
    }
    oss << "\n\n";
    
    // Рекомендации
    if (!recommendations.empty()) {
        oss << "RECOMMENDATIONS:\n";
        for (size_t i = 0; i < recommendations.size(); ++i) {
            oss << "  " << (i + 1) << ". " << recommendations[i] << "\n";
        }
    }
    
    return oss.str();
}

// ============== Реализация OverfittingDetector ==============

OverfittingDetector::OverfittingDetector(const OverfittingDetectorConfig& config)
    : config_(config) {}

// ============== Основной метод диагностики ==============

OverfittingDiagnostics OverfittingDetector::diagnose(const Polynomial& poly,
                                                     const OptimizationProblemData& data,
                                                     double gamma) {
    OverfittingDiagnostics diagnostics;
    
    // 1. Метрика кривизны
    diagnostics.curvature = compute_curvature_metric(poly, data, gamma);
    
    // 2. Метрика осцилляций
    diagnostics.oscillation = compute_oscillation_metric(poly, data);
    
    // 3. Кросс-валидация (опционально)
    if (config_.enable_cross_validation && data.approx_x.size() >= config_.min_cv_folds) {
        diagnostics.cross_validation = compute_cross_validation_metric(poly, data, gamma, optimizer_callback_);
    }
    
    // 4. Чувствительность к шуму (опционально)
    if (config_.enable_sensitivity_analysis) {
        diagnostics.sensitivity = compute_sensitivity_metric(poly, data);
    }
    
    // 5. Комплексная оценка риска
    diagnostics.risk_score = compute_risk_score(diagnostics);
    diagnostics.risk_level = assess_risk_level(diagnostics.risk_score);
    
    // 6. Рекомендация стратегии коррекции
    diagnostics.recommended_strategy = recommend_correction_strategy(diagnostics);
    
    // 7. Генерация рекомендаций
    diagnostics.recommendations = generate_recommendations(diagnostics);
    
    // 8. Параметры для коррекции
    diagnostics.suggested_gamma_multiplier = config_.gamma_boost_factor;
    diagnostics.suggested_degree_reduction = static_cast<int>(
        std::max(1.0, poly.degree() * config_.degree_reduction_factor * diagnostics.risk_score));
    diagnostics.suggested_degree_reduction = std::max(0, diagnostics.suggested_degree_reduction);
    
    // 9. Обнаружение выбросов
    diagnostics.outlier_indices = detect_outliers(poly, data);
    
    return diagnostics;
}

// ============== Метрика 1: Нормированная кривизна ==============

CurvatureMetricResult OverfittingDetector::compute_curvature_metric(const Polynomial& poly,
                                                                   const OptimizationProblemData& data,
                                                                   double gamma) {
    CurvatureMetricResult result;
    
    // gamma используется для информационных целей, не直接影响 вычисление кривизны
    (void)gamma;
    
    // Вычисляем ||F''||² через интегрирование (F''(x))² на [a, b]
    double a = data.interval_a;
    double b = data.interval_b;
    
    // Используем квадратуру для вычисления интеграла
    int n = 200;
    double h = (b - a) / n;
    double integral = 0.0;
    
    for (int i = 0; i <= n; ++i) {
        double x = a + i * h;
        double weight = (i == 0 || i == n) ? 0.5 : 1.0;
        double second_deriv = poly.second_derivative(x);
        integral += weight * second_deriv * second_deriv * h;
    }
    
    result.second_deriv_norm = integral;
    
    // Вычисляем ожидаемый масштаб кривизны
    result.expected_curvature_scale = compute_expected_curvature_scale(data);
    
    if (result.expected_curvature_scale > 0) {
        result.normalized_curvature = result.second_deriv_norm / result.expected_curvature_scale;
    } else {
        result.normalized_curvature = 0.0;
    }
    
    // Вычисляем адаптивный порог
    result.threshold = compute_adaptive_curvature_threshold(data);
    
    // Проверяем превышение
    result.is_over_threshold = result.normalized_curvature > result.threshold;
    
    return result;
}

double OverfittingDetector::compute_expected_curvature_scale(const OptimizationProblemData& data) const {
    // scale_curvature_expected = max|f(x_i)| / (b - a)²
    double a = data.interval_a;
    double b = data.interval_b;
    double range = b - a;
    
    if (range <= 0) return 1.0;
    
    double max_abs_f = 0.0;
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        max_abs_f = std::max(max_abs_f, std::abs(data.approx_f[i]));
    }
    
    // Для линейной функции кривизна = 0, для параболы ~ amplitude / range²
    double scale = max_abs_f / (range * range);
    
    // Минимальный масштаб для численной стабильности
    return std::max(scale, 1e-12);
}

double OverfittingDetector::compute_adaptive_curvature_threshold(const OptimizationProblemData& data) const {
    // Учёт плотности данных: ρ = N_approx / (b - a)
    double a = data.interval_a;
    double b = data.interval_b;
    double range = b - a;
    
    if (range <= 0 || data.approx_x.empty()) {
        return config_.curvature_threshold_base;
    }
    
    double density = data.approx_x.size() / range;
    
    // Коррекция порога: κ_threshold = 5.0 · max(1.0, log10(1.0 + n / ρ))
    double correction = std::log10(1.0 + data.approx_x.size() / density);
    correction = std::max(1.0, correction);
    
    return config_.curvature_threshold_base * correction;
}

// ============== Вспомогательная функция для создания полинома-производной ==============

Polynomial create_derivative(const Polynomial& poly) {
    if (poly.degree() <= 0) {
        return Polynomial(0);
    }
    
    const auto& coeffs = poly.coefficients();
    int n = poly.degree();
    
    // Если P(x) = a_n*x^n + ... + a_0, то P'(x) = n*a_n*x^{n-1} + ... + a_1
    std::vector<double> deriv_coeffs;
    deriv_coeffs.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        int power = n - i;
        deriv_coeffs.push_back(power * coeffs[i]);
    }
    
    return Polynomial(deriv_coeffs);
}

// ============== Метрика 2: Осцилляции ==============

OscillationMetricResult OverfittingDetector::compute_oscillation_metric(const Polynomial& poly,
                                                                       const OptimizationProblemData& data) {
    OscillationMetricResult result;
    
    double a = data.interval_a;
    double b = data.interval_b;
    
    // Находим корни первой производной (экстремумы)
    Polynomial deriv = create_derivative(poly);
    std::vector<double> roots = find_derivative_roots(deriv, a, b, config_.num_grid_points);
    
    result.total_extrema = static_cast<int>(roots.size());
    result.extremum_positions = roots;
    result.is_suspicious.resize(roots.size(), false);
    
    if (roots.empty()) {
        return result;
    }
    
    // Вычисляем медианное расстояние между точками данных
    double median_spacing = compute_median_spacing(data.approx_x);
    
    // Для каждого экстремума проверяем, находится ли он в "пустой" области
    for (size_t i = 0; i < roots.size(); ++i) {
        double x_ext = roots[i];
        
        // Находим ближайшие точки данных
        double d_left = std::numeric_limits<double>::infinity();
        double d_right = std::numeric_limits<double>::infinity();
        
        for (const auto& x_data : data.approx_x) {
            double dist = std::abs(x_ext - x_data);
            if (x_data < x_ext) {
                d_left = std::min(d_left, dist);
            } else {
                d_right = std::min(d_right, dist);
            }
        }
        
        // Также проверяем интерполяционные узлы
        for (const auto& z : data.interp_z) {
            double dist = std::abs(x_ext - z);
            if (z < x_ext) {
                d_left = std::min(d_left, dist);
            } else {
                d_right = std::min(d_right, dist);
            }
        }
        
        // Проверяем отталкивающие точки
        for (const auto& y : data.repel_y) {
            double dist = std::abs(x_ext - y);
            if (y < x_ext) {
                d_left = std::min(d_left, dist);
            } else {
                d_right = std::min(d_right, dist);
            }
        }
        
        double min_distance = std::min(d_left, d_right);
        double gap_ratio = (median_spacing > 0) ? min_distance / median_spacing : 0.0;
        
        // Если gap_ratio > threshold, это "пустая" область
        if (gap_ratio > config_.gap_ratio_threshold) {
            result.extrema_in_empty_regions++;
            result.is_suspicious[i] = true;
            
            // Добавляем вклад в oscillation_score
            result.oscillation_score += 1.0 / (1.0 + gap_ratio);
        }
    }
    
    return result;
}

std::vector<double> OverfittingDetector::find_derivative_roots(const Polynomial& deriv,
                                                              double a, double b,
                                                              int num_grid_points) const {
    std::vector<double> roots;
    
    if (num_grid_points < 2) return roots;
    
    double h = (b - a) / num_grid_points;
    std::vector<double> grid_points(num_grid_points + 1);
    std::vector<double> grid_values(num_grid_points + 1);
    
    for (int i = 0; i <= num_grid_points; ++i) {
        grid_points[i] = a + i * h;
        grid_values[i] = deriv.evaluate(grid_points[i]);
    }
    
    // Поиск смены знака
    for (int i = 0; i < num_grid_points; ++i) {
        if (grid_values[i] == 0.0) {
            // Точка на узле - считаем за корень
            roots.push_back(grid_points[i]);
        } else if (grid_values[i] * grid_values[i + 1] < 0.0) {
            // Смена знака - уточняем методом Ньютона
            double root = refine_root_newton(deriv,
                                            grid_points[i],
                                            grid_points[i + 1],
                                            config_.num_extremum_refinements,
                                            1e-10);
            if (std::isfinite(root) && root >= a && root <= b) {
                // Проверяем, что это новый корень (не слишком близок к существующему)
                bool is_new = true;
                for (double existing : roots) {
                    if (std::abs(root - existing) < 1e-6) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    roots.push_back(root);
                }
            }
        }
    }
    
    return roots;
}

double OverfittingDetector::refine_root_newton(const Polynomial& deriv,
                                               double x_low, double x_high,
                                               int max_iterations,
                                               double tolerance) const {
    double x = (x_low + x_high) / 2.0;  // Начальное приближение - середина
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double f = deriv.evaluate(x);
        
        if (std::abs(f) < tolerance) {
            return x;
        }
        
        // Первая производная для уточнения направления
        double f_prime = deriv.derivative(x);
        
        if (std::abs(f_prime) < 1e-15) {
            // Производная слишком мала - возвращаем текущее приближение
            return x;
        }
        
        // Шаг Ньютона
        double x_new = x - f / f_prime;
        
        // Проверяем, что остаёмся в интервале
        if (x_new < x_low || x_new > x_high) {
            // Если Ньютон ушёл за границы, используем бисекцию
            x = (x_low + x_high) / 2.0;
            if (x_high - x_low < tolerance) {
                return x;
            }
        } else {
            x = x_new;
        }
    }
    
    return x;
}

int OverfittingDetector::classify_extremum(const Polynomial& poly, double x) const {
    // Вторая производная: F''(x) > 0 => минимум, F''(x) < 0 => максимум
    double second_deriv = poly.second_derivative(x);
    
    if (second_deriv > 0) {
        return 1;  // Минимум
    } else if (second_deriv < 0) {
        return -1; // Максимум
    }
    return 0;      // Точка перегиба или седловая точка
}

double OverfittingDetector::compute_median_spacing(const std::vector<double>& points) const {
    if (points.size() < 2) return 1.0;
    
    std::vector<double> spacings;
    spacings.reserve(points.size() - 1);
    
    for (size_t i = 1; i < points.size(); ++i) {
        spacings.push_back(points[i] - points[i - 1]);
    }
    
    std::sort(spacings.begin(), spacings.end());
    
    // Медиана
    size_t mid = spacings.size() / 2;
    if (spacings.size() % 2 == 0) {
        return (spacings[mid - 1] + spacings[mid]) / 2.0;
    }
    return spacings[mid];
}

// ============== Метрика 3: Кросс-валидация ==============

CrossValidationMetricResult OverfittingDetector::compute_cross_validation_metric(
    const Polynomial& poly,
    const OptimizationProblemData& data,
    double gamma,
    std::function<Polynomial(const std::vector<double>&)> optimizer_callback) {
    
    // Параметры зарезервированы для расширенной версии с переоптимизацией
    (void)gamma;
    (void)optimizer_callback;
    
    CrossValidationMetricResult result;
    
    // Определяем число фолдов
    int num_approx = static_cast<int>(data.approx_x.size());
    result.num_folds = std::min(config_.max_cv_folds, num_approx);
    result.num_folds = std::max(config_.min_cv_folds, result.num_folds);
    
    if (num_approx < result.num_folds) {
        // Недостаточно точек для кросс-валидации
        result.generalization_ratio = 1.0;
        return result;
    }
    
    // Вычисляем train_error на полном наборе
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        double F_x = poly.evaluate(data.approx_x[i]);
        double error = (F_x - data.approx_f[i]) / data.approx_weight[i];
        sum_squared_error += error * error;
    }
    result.train_error = std::sqrt(sum_squared_error / num_approx);
    
    // Разбиваем на фолды
    std::vector<int> indices(num_approx);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Простое разбиение (последовательное)
    int fold_size = num_approx / result.num_folds;
    double total_cv_error = 0.0;
    result.fold_errors.resize(result.num_folds, 0.0);
    
    // Упрощённая кросс-валидация - только оценка без переоптимизации
    for (int fold = 0; fold < result.num_folds; ++fold) {
        int val_start = fold * fold_size;
        int val_end = (fold == result.num_folds - 1) ? num_approx : (fold + 1) * fold_size;
        
        double fold_error = 0.0;
        for (int i = val_start; i < val_end; ++i) {
            double F_x = poly.evaluate(data.approx_x[i]);
            double error = (F_x - data.approx_f[i]) / data.approx_weight[i];
            fold_error += error * error;
        }
        result.fold_errors[fold] = std::sqrt(fold_error / (val_end - val_start));
        total_cv_error += fold_error;
    }
    
    // CV_error - средняя ошибка по фолдам
    result.cv_error = std::sqrt(total_cv_error / num_approx);
    
    // Отношение обобщения
    if (result.train_error > 0) {
        result.generalization_ratio = result.cv_error / result.train_error;
    } else {
        result.generalization_ratio = 1.0;
    }
    
    return result;
}

// ============== Метрика 4: Чувствительность к шуму ==============

SensitivityMetricResult OverfittingDetector::compute_sensitivity_metric(const Polynomial& poly,
                                                                       const OptimizationProblemData& data) {
    SensitivityMetricResult result;
    
    result.perturbation_count = config_.num_perturbations;
    
    double a = data.interval_a;
    double b = data.interval_b;
    double range = b - a;
    
    if (range <= 0 || data.approx_x.empty()) {
        return result;
    }
    
    // Контрольная сетка для оценки вариации
    const int num_test_points = 100;
    std::vector<double> test_points(num_test_points);
    for (int i = 0; i < num_test_points; ++i) {
        test_points[i] = a + i * range / (num_test_points - 1);
    }
    
    double max_F = 0.0;
    for (int p = 0; p < num_test_points; ++p) {
        max_F = std::max(max_F, std::abs(poly.evaluate(test_points[p])));
    }
    result.max_value = max_F;
    
    // Вычисляем остатки
    double max_residual = 0.0;
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        double residual = std::abs(poly.evaluate(data.approx_x[i]) - data.approx_f[i]);
        max_residual = std::max(max_residual, residual);
    }
    
    double median_spacing = compute_median_spacing(data.approx_x);
    if (median_spacing > 0 && max_F > 0) {
        // Оценка чувствительности через отношение остатка к расстоянию между точками
        result.sensitivity_score = max_residual / median_spacing / max_F;
    } else {
        result.sensitivity_score = 0.0;
    }
    
    // Масштабируем для получения интерпретируемого скора
    result.sensitivity_score *= 100.0;
    
    return result;
}

std::vector<WeightedPoint> OverfittingDetector::generate_perturbed_points(
    const std::vector<WeightedPoint>& points,
    double delta_x, double delta_y) const {
    
    std::vector<WeightedPoint> perturbed;
    perturbed.reserve(points.size());
    
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (const auto& p : points) {
        double x_pert = p.x + delta_x * normal(rng);
        double f_pert = p.value + delta_y * normal(rng);
        perturbed.emplace_back(x_pert, f_pert, p.weight);
    }
    
    return perturbed;
}

// ============== Комплексная оценка риска ==============

double OverfittingDetector::compute_risk_score(const OverfittingDiagnostics& diagnostics) const {
    // Веса метрик
    double w1 = config_.weight_curvature;
    double w2 = config_.weight_oscillation;
    double w3 = config_.weight_generalization;
    double w4 = config_.weight_sensitivity;
    
    // Корректируем вес осцилляций, если данные предположительно осциллирующие
    if (config_.assume_oscillating_data) {
        w2 = 0.05;  // Значительно снижаем вес
        // Перенормируем остальные веса
        double total = w1 + w3 + w4;
        w1 = w1 / total * 0.95;
        w3 = w3 / total * 0.95;
        w4 = w4 / total * 0.95;
    }
    
    // Нормировка кривизны
    double norm_curvature = normalize_metric(
        diagnostics.curvature.normalized_curvature,
        diagnostics.curvature.threshold,
        diagnostics.curvature.threshold * 10.0);
    
    // Нормировка осцилляций
    double norm_oscillation = normalize_metric(
        diagnostics.oscillation.oscillation_score,
        config_.oscillation_threshold_low,
        config_.oscillation_threshold_high);
    
    // Нормировка обобщения
    double norm_generalization = normalize_metric(
        diagnostics.cross_validation.generalization_ratio,
        config_.generalization_threshold_low,
        config_.generalization_threshold_high);
    
    // Нормировка чувствительности
    double norm_sensitivity = normalize_metric(
        diagnostics.sensitivity.sensitivity_score,
        config_.sensitivity_threshold_low,
        config_.sensitivity_threshold_high);
    
    // Комплексная оценка
    double risk = w1 * norm_curvature + w2 * norm_oscillation
                + w3 * norm_generalization + w4 * norm_sensitivity;
    
    // Ограничиваем диапазон [0, 1]
    return std::max(0.0, std::min(1.0, risk));
}

double OverfittingDetector::normalize_metric(double value, double low, double high) const {
    // normalize(metric) = (metric - threshold_low) / (threshold_high - threshold_low)
    double range = high - low;
    if (range <= 0) return 0.0;
    
    double normalized = (value - low) / range;
    return std::max(0.0, std::min(1.0, normalized));
}

OverfittingRiskLevel OverfittingDetector::assess_risk_level(double risk_score) const {
    if (risk_score < 0.3) {
        return OverfittingRiskLevel::LOW;
    } else if (risk_score < 0.7) {
        return OverfittingRiskLevel::MODERATE;
    }
    return OverfittingRiskLevel::HIGH;
}

CorrectionStrategy OverfittingDetector::recommend_correction_strategy(const OverfittingDiagnostics& diagnostics) const {
    // Определяем по преобладающей метрике
    double norm_curvature = diagnostics.curvature.normalized_curvature / diagnostics.curvature.threshold;
    double norm_osc = diagnostics.oscillation.oscillation_score / config_.oscillation_threshold_high;
    double norm_gen = diagnostics.cross_validation.generalization_ratio / config_.generalization_threshold_high;
    double norm_sens = diagnostics.sensitivity.sensitivity_score / config_.sensitivity_threshold_high;
    
    if (diagnostics.risk_score < 0.3) {
        return CorrectionStrategy::NONE;
    }
    
    // Если высокая кривизна - усиление регуляризации
    if (norm_curvature > norm_osc && norm_curvature > norm_gen && norm_curvature > norm_sens) {
        return CorrectionStrategy::REGULARIZATION;
    }
    
    // Если много осцилляций - возможно, снижение степени
    if (norm_osc > norm_curvature && norm_osc > norm_gen && norm_osc > norm_sens) {
        return CorrectionStrategy::DEGREE_REDUCTION;
    }
    
    // Если плохое обобщение и чувствительность - возможно, выбросы
    if (!diagnostics.outlier_indices.empty() && norm_sens > 0.5) {
        return CorrectionStrategy::WEIGHT_CORRECTION;
    }
    
    // По умолчанию - усиление регуляризации
    return CorrectionStrategy::REGULARIZATION;
}

// ============== Стратегии коррекции ==============

OverfittingCorrectionResult OverfittingDetector::apply_correction(const OverfittingDiagnostics& diagnostics,
                                                                 double current_gamma,
                                                                 int current_degree,
                                                                 std::vector<double> weights) {
    OverfittingCorrectionResult result;
    
    result.risk_before = diagnostics.risk_score;
    result.strategy_used = diagnostics.recommended_strategy;
    
    switch (diagnostics.recommended_strategy) {
        case CorrectionStrategy::REGULARIZATION: {
            // γ_new = γ_current · (1.0 + α · risk_score)
            double new_gamma = current_gamma * (1.0 + config_.gamma_boost_factor * diagnostics.risk_score);
            
            // Ограничения
            double gamma_min = config_.min_gamma;
            double gamma_max = current_gamma * config_.gamma_max_multiplier;
            new_gamma = std::max(gamma_min, std::min(gamma_max, new_gamma));
            
            result.new_gamma = new_gamma;
            result.correction_applied = true;
            result.message = "Applied regularization boost: γ = " + std::to_string(new_gamma);
            break;
        }
        
        case CorrectionStrategy::DEGREE_REDUCTION: {
            // Δn = max(1, floor(0.2 · n_current · risk_score))
            int max_reduction = static_cast<int>(
                std::max(1.0, std::floor(0.2 * current_degree * diagnostics.risk_score)));
            int new_degree = std::max(config_.min_degree_from_constraints, current_degree - max_reduction);
            
            result.new_degree = new_degree;
            result.correction_applied = true;
            result.message = "Applied degree reduction: n = " + std::to_string(new_degree);
            break;
        }
        
        case CorrectionStrategy::WEIGHT_CORRECTION: {
            // Коррекция весов для потенциальных выбросов
            result.corrected_weights = weights;
            double beta = 0.5;  // Коэффициент ослабления
            
            for (int idx : diagnostics.outlier_indices) {
                if (idx < static_cast<int>(weights.size())) {
                    weights[idx] *= (1.0 + beta * 0.5);
                }
            }
            result.corrected_weights = weights;
            result.correction_applied = true;
            result.message = "Applied weight correction to " + 
                            std::to_string(diagnostics.outlier_indices.size()) + " outliers";
            break;
        }
        
        case CorrectionStrategy::MANUAL_REVIEW:
            result.message = "Manual review recommended - cannot auto-correct";
            break;
            
        case CorrectionStrategy::NONE:
            result.message = "No correction needed - solution is acceptable";
            break;
    }
    
    // Если auto_correction включён, симулируем улучшение риска
    if (config_.enable_auto_correction && result.correction_applied) {
        result.risk_after = diagnostics.risk_score * 0.7;  // Упрощённая оценка
    } else {
        result.risk_after = diagnostics.risk_score;
    }
    
    return result;
}

// ============== Генерация рекомендаций ==============

std::vector<std::string> OverfittingDetector::generate_recommendations(const OverfittingDiagnostics& diagnostics) const {
    std::vector<std::string> recommendations;
    
    std::ostringstream oss;
    
    // Рекомендации по кривизне
    if (diagnostics.curvature.is_over_threshold) {
        oss.str("");
        oss << "Increase regularization parameter γ (current threshold exceeded by "
            << std::fixed << std::setprecision(2)
            << diagnostics.curvature.normalized_curvature / diagnostics.curvature.threshold << "x)";
        recommendations.push_back(oss.str());
    }
    
    // Рекомендации по осцилляциям
    if (diagnostics.oscillation.oscillation_score >= config_.oscillation_threshold_high) {
        oss.str("");
        oss << "High oscillation detected (" << diagnostics.oscillation.extrema_in_empty_regions
            << " extrema in sparse regions). Consider reducing polynomial degree or adding more data points.";
        recommendations.push_back(oss.str());
    }
    
    // Рекомендации по обобщению
    if (diagnostics.cross_validation.generalization_ratio >= config_.generalization_threshold_high) {
        oss.str("");
        oss << "Poor generalization (ratio: "
            << std::fixed << std::setprecision(2)
            << diagnostics.cross_validation.generalization_ratio
            << "). Possible overfitting or outliers in data.";
        recommendations.push_back(oss.str());
    }
    
    // Рекомендации по чувствительности
    if (diagnostics.sensitivity.sensitivity_score >= config_.sensitivity_threshold_high) {
        oss.str("");
        oss << "High sensitivity to noise. Consider increasing regularization or checking for outliers.";
        recommendations.push_back(oss.str());
    }
    
    // Общие рекомендации
    if (diagnostics.risk_level == OverfittingRiskLevel::MODERATE) {
        recommendations.push_back("Moderate overfitting detected - review is recommended.");
    } else if (diagnostics.risk_level == OverfittingRiskLevel::HIGH) {
        recommendations.push_back("CRITICAL: Significant overfitting - corrective action required!");
        recommendations.push_back("建议: Increase γ by " +
                                 std::to_string(static_cast<int>(config_.gamma_boost_factor * 100)) +
                                 "% or reduce polynomial degree.");
    }
    
    return recommendations;
}

// ============== Обнаружение выбросов ==============

std::vector<int> OverfittingDetector::detect_outliers(const Polynomial& poly,
                                                      const OptimizationProblemData& data,
                                                      double residual_threshold) {
    std::vector<int> outliers;
    
    // Вычисляем остатки
    std::vector<double> residuals;
    residuals.reserve(data.approx_x.size());
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        double residual = poly.evaluate(data.approx_x[i]) - data.approx_f[i];
        residuals.push_back(residual / data.approx_weight[i]);
        sum_sq += residuals.back() * residuals.back();
    }
    
    double rms_residual = std::sqrt(sum_sq / residuals.size());
    
    // Стандартизованные остатки
    for (size_t i = 0; i < residuals.size(); ++i) {
        double standardized = std::abs(residuals[i]) / rms_residual;
        if (standardized > residual_threshold) {
            outliers.push_back(static_cast<int>(i));
        }
    }
    
    return outliers;
}

} // namespace mixed_approx
