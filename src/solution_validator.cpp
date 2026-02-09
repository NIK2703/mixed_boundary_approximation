#include "mixed_approximation/solution_validator.h"
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/composite_polynomial.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace mixed_approx {

// ============== Вспомогательные функции ==============

namespace {

/**
 * @brief Вычисление адаптивного порога для интерполяции (шаг 6.1.2)
 */
double compute_adaptive_interpolation_threshold_impl(const OptimizationProblemData& data, double base_epsilon) {
    double epsilon_interp = base_epsilon;
    
    // Масштабирование под характеристики задачи: scale_y = max(1.0, 0.1 * max|f|)
    double max_abs_f = 0.0;
    for (double f : data.approx_f) {
        max_abs_f = std::max(max_abs_f, std::abs(f));
    }
    for (double f : data.interp_f) {
        max_abs_f = std::max(max_abs_f, std::abs(f));
    }
    
    double scale_y = std::max(1.0, 0.1 * max_abs_f);
    epsilon_interp = std::max(epsilon_interp, 1e-12 * scale_y);
    
    // Коррекция на степень полинома: для высоких степеней ошибка округления растёт
    int estimated_degree = static_cast<int>(data.approx_x.size()) > 0 
                          ? static_cast<int>(data.approx_x.size()) - 1 
                          : 5;
    if (estimated_degree > 15) {
        epsilon_interp *= (1.0 + 0.05 * (estimated_degree - 15));
    }
    
    return epsilon_interp;
}

/**
 * @brief Вычисление адаптивного порога безопасности для барьеров (шаг 6.1.3)
 */
double compute_adaptive_safety_threshold_impl(const OptimizationProblemData& data, double base_epsilon) {
    double epsilon_safe = base_epsilon;
    
    // Масштабирование
    double max_abs_f = 0.0;
    for (double f : data.approx_f) {
        max_abs_f = std::max(max_abs_f, std::abs(f));
    }
    for (double y_forbidden : data.repel_forbidden) {
        max_abs_f = std::max(max_abs_f, std::abs(y_forbidden));
    }
    
    double scale_y = std::max(1.0, 0.1 * max_abs_f);
    epsilon_safe = std::max(epsilon_safe, 1e-10 * scale_y);
    
    // Динамическая коррекция на основе силы барьера
    double max_weight = 0.0;
    for (double w : data.repel_weight) {
        max_weight = std::max(max_weight, w);
    }
    
    if (max_weight > 1000.0) {
        epsilon_safe *= 10.0;  // Сильный барьер требует большего запаса
    } else if (max_weight < 10.0) {
        epsilon_safe *= 0.5;    // Слабый барьер допускает меньший запас
    }
    
    return epsilon_safe;
}

/**
 * @brief Вычисление ожидаемого масштаба кривизны
 */
double compute_expected_curvature_scale(const OptimizationProblemData& data) {
    double a = data.interval_a;
    double b = data.interval_b;
    double range = b - a;
    
    if (range <= 0) return 1.0;
    
    double max_abs_f = 0.0;
    for (double f : data.approx_f) {
        max_abs_f = std::max(max_abs_f, std::abs(f));
    }
    
    // scale_curvature_expected = max|f(x_i)| / (b - a)^2
    double scale = max_abs_f / (range * range);
    return std::max(scale, 1e-12);
}

/**
 * @brief Нормализация метрики в диапазон [0, 1]
 */
double normalize_metric_impl(double value, double low_threshold, double high_threshold) {
    double range = high_threshold - low_threshold;
    if (range <= 0) return 0.0;
    
    double normalized = (value - low_threshold) / range;
    return std::max(0.0, std::min(1.0, normalized));
}

/**
 * @brief Поиск корней производной полинома (экстремумов)
 */
std::vector<double> find_derivative_roots(const Polynomial& poly, double a, double b, int num_grid_points) {
    std::vector<double> roots;
    
    if (num_grid_points < 2 || poly.degree() <= 0) return roots;
    
    // Создаём полином производной
    std::vector<double> deriv_coeffs;
    const auto& coeffs = poly.coefficients();
    int n = poly.degree();
    
    for (int i = 0; i < n; ++i) {
        int power = n - i;
        deriv_coeffs.push_back(power * coeffs[i]);
    }
    
    Polynomial deriv(deriv_coeffs);
    
    // Сетка для поиска смены знака
    double h = (b - a) / num_grid_points;
    std::vector<double> grid_values(num_grid_points + 1);
    
    for (int i = 0; i <= num_grid_points; ++i) {
        double x = a + i * h;
        grid_values[i] = deriv.evaluate(x);
    }
    
    // Поиск интервалов со сменой знака
    for (int i = 0; i < num_grid_points; ++i) {
        if (grid_values[i] == 0.0) {
            roots.push_back(a + i * h);
        } else if (grid_values[i] * grid_values[i + 1] < 0.0) {
            // Уточнение методом Ньютона
            double x_low = a + i * h;
            double x_high = a + (i + 1) * h;
            double x_mid = (x_low + x_high) / 2.0;
            
            for (int iter = 0; iter < 10; ++iter) {
                double f = deriv.evaluate(x_mid);
                double f_prime = deriv.derivative(x_mid);
                
                if (std::abs(f_prime) < 1e-15) break;
                
                double x_new = x_mid - f / f_prime;
                if (x_new < x_low || x_new > x_high) break;
                x_mid = x_new;
            }
            
            if (x_mid >= a && x_mid <= b) {
                bool is_new = true;
                for (double r : roots) {
                    if (std::abs(x_mid - r) < 1e-6) {
                        is_new = false;
                        break;
                    }
                }
                if (is_new) {
                    roots.push_back(x_mid);
                }
            }
        }
    }
    
    return roots;
}

} // anonymous namespace

// ============== Конструктор ==============

SolutionValidator::SolutionValidator(double eps_safe, double interp_tol)
    : epsilon_safe(eps_safe)
    , interp_tolerance(interp_tol)
    , max_value_factor(100.0)
    , num_check_points(1000) {
}

// ============== Основной метод валидации ==============

ValidationResult SolutionValidator::validate(const Polynomial& poly, 
                                           const OptimizationProblemData& data) const {
    ValidationResult result;
    result.is_valid = true;
    result.numerical_correct = true;
    result.interpolation_ok = true;
    result.barriers_safe = true;
    result.physically_plausible = true;
    result.numerically_stable = true;
    result.message = "Validation passed";
    
    // 1. Численная корректность
    result.numerical_correct = check_numerical_correctness(poly, data);
    if (!result.numerical_correct) {
        result.is_valid = false;
        result.message = "Numerical correctness check failed";
        result.warnings.push_back("NaN or Inf detected in polynomial evaluation");
    }
    
    // 2. Интерполяционные условия
    double interp_error;
    result.interpolation_ok = check_interpolation(poly, data, interp_error);
    result.max_interpolation_error = interp_error;
    if (!result.interpolation_ok) {
        result.is_valid = false;
        result.message = "Interpolation conditions not satisfied";
    }
    
    // 3. Безопасность барьеров
    double min_distance;
    result.barriers_safe = check_barrier_safety(poly, data, min_distance);
    
    // Всегда добавляем информацию о расстоянии до барьера
    if (min_distance < std::numeric_limits<double>::infinity()) {
        double threshold = compute_adaptive_safety_threshold(data);
        std::ostringstream warn;
        warn << "Barrier safety: distance " << std::scientific << std::setprecision(2) 
             << min_distance << " (threshold: " << threshold << ")";
        result.warnings.push_back(warn.str());
    }
    
    if (!result.barriers_safe) {
        result.is_valid = false;
        result.message = "Barrier safety violated";
    }
    
    // 4. Физическая правдоподобность
    double max_val;
    result.physically_plausible = check_physical_plausibility(poly, data, max_val);
    if (!result.physically_plausible) {
        result.is_valid = false;
        result.message = "Solution lacks physical plausibility";
    }
    
    // 5. Численная стабильность
    result.numerically_stable = check_numerical_stability(poly, data);
    if (!result.numerically_stable) {
        result.is_valid = false;
        result.message = "Numerical instability detected";
    }
    
    // Классификация статуса
    if (result.is_valid) {
        double adaptive_threshold = compute_adaptive_interpolation_threshold(data);
        if (interp_error < 0.1 * adaptive_threshold && min_distance > 10.0 * epsilon_safe) {
            result.status = 1; // VERIFICATION_OK
        } else {
            result.status = 2; // VERIFICATION_WARNING
        }
    } else {
        result.status = 3; // VERIFICATION_CRITICAL
    }
    
    return result;
}

// ============== Полная верификация ==============

ExtendedVerificationResult SolutionValidator::verify_full(const Polynomial& poly,
                                                        const OptimizationProblemData& data,
                                                        FunctionalDiagnostics& func_diag) const {
    ExtendedVerificationResult result;
    
    // 1. Базовая валидация
    result.validation = validate(poly, data);
    
    // 2. Анализ баланса компонент функционала
    analyze_functional_balance(poly, data, func_diag);
    result.diagnostics = func_diag;
    
    // 3. Вычисление оценки качества
    result.quality_score = compute_quality_score(result.validation, func_diag);
    result.quality = classify_quality(result.quality_score);
    
    // 4. Определение статуса верификации
    if (result.validation.is_valid) {
        if (result.quality == SolutionQuality::EXCELLENT) {
            result.status = VerificationStatus::VERIFICATION_OK;
        } else {
            result.status = VerificationStatus::VERIFICATION_WARNING;
        }
    } else {
        result.status = VerificationStatus::VERIFICATION_CRITICAL;
    }
    
    // 5. Генерация рекомендаций
    result.recommendations = generate_quality_recommendations(result.quality);
    
    return result;
}

// ============== Методы проверки ==============

bool SolutionValidator::check_interpolation(const Polynomial& poly,
                                          const OptimizationProblemData& data,
                                          double& max_error) const {
    if (data.interp_z.empty()) {
        max_error = 0.0;
        return true;
    }
    
    max_error = 0.0;
    double threshold = compute_adaptive_interpolation_threshold(data);
    
    for (size_t e = 0; e < data.interp_z.size(); ++e) {
        double z = data.interp_z[e];
        double f_z = data.interp_f[e];
        double F_z = poly.evaluate(z);
        double error = std::abs(F_z - f_z);
        
        max_error = std::max(max_error, error);
    }
    
    return max_error < threshold;
}

bool SolutionValidator::check_barrier_safety(const Polynomial& poly,
                                            const OptimizationProblemData& data,
                                            double& min_distance) const {
    if (data.repel_y.empty()) {
        min_distance = std::numeric_limits<double>::infinity();
        return true;
    }
    
    min_distance = std::numeric_limits<double>::infinity();
    double threshold = compute_adaptive_safety_threshold(data);
    
    for (size_t j = 0; j < data.repel_y.size(); ++j) {
        double y = data.repel_y[j];
        double y_forbidden = data.repel_forbidden[j];
        double F_y = poly.evaluate(y);
        double distance = std::abs(F_y - y_forbidden);
        
        min_distance = std::min(min_distance, distance);
    }
    
    return min_distance >= threshold;
}

bool SolutionValidator::check_numerical_correctness(const Polynomial& poly,
                                                  const OptimizationProblemData& data) const {
    // Проверка коэффициентов
    for (const double& coeff : poly.coefficients()) {
        if (!std::isfinite(coeff)) {
            return false;
        }
    }
    
    // Проверка в контрольных точках
    std::vector<double> test_points = {
        data.interval_a,
        data.interval_b,
        (data.interval_a + data.interval_b) / 2.0
    };
    
    for (double x : test_points) {
        double val = poly.evaluate(x);
        if (!std::isfinite(val)) {
            return false;
        }
    }
    
    // Проверка в точках данных
    for (double x : data.approx_x) {
        double val = poly.evaluate(x);
        if (!std::isfinite(val)) {
            return false;
        }
    }
    
    for (double y : data.repel_y) {
        double val = poly.evaluate(y);
        if (!std::isfinite(val)) {
            return false;
        }
    }
    
    for (double z : data.interp_z) {
        double val = poly.evaluate(z);
        if (!std::isfinite(val)) {
            return false;
        }
    }
    
    return true;
}

bool SolutionValidator::check_physical_plausibility(const Polynomial& poly,
                                                  const OptimizationProblemData& data,
                                                  double& max_value) const {
    max_value = 0.0;
    
    // Генерация контрольной сетки
    int N_grid = std::max(100, 10 * poly.degree());
    double a = data.interval_a;
    double b = data.interval_b;
    double h = (b - a) / N_grid;
    
    // Вычисление масштаба данных
    double data_scale = 1.0;
    for (double f : data.approx_f) {
        data_scale = std::max(data_scale, std::abs(f));
    }
    
    int extreme_count = 0;
    
    for (int i = 0; i <= N_grid; ++i) {
        double x = a + i * h;
        double F_x = poly.evaluate(x);
        double F2_x = poly.second_derivative(x);
        
        max_value = std::max(max_value, std::abs(F_x));
        
        // Проверка на экстремальные значения
        if (std::abs(F_x) > max_value_factor * data_scale) {
            extreme_count++;
        }
        
        // Проверка на чрезмерную кривизну
        if (std::abs(F2_x) > 1e6 * data_scale / ((b - a) * (b - a))) {
            extreme_count++;
        }
    }
    
    // Решение считается правдоподобным, если число экстремумов < 10% от числа точек
    return extreme_count < 0.1 * N_grid;
}

bool SolutionValidator::check_numerical_stability(const Polynomial& poly,
                                                const OptimizationProblemData& data) const {
    int N_grid = std::max(100, 10 * poly.degree());
    double a = data.interval_a;
    double b = data.interval_b;
    double h = (b - a) / N_grid;
    
    double prev_F = poly.evaluate(a);
    if (std::isnan(prev_F) || std::isinf(prev_F)) {
        return false;
    }
    
    int anomalies = 0;
    
    for (int i = 1; i <= N_grid; ++i) {
        double x = a + i * h;
        double F = poly.evaluate(x);
        
        if (std::isnan(F) || std::isinf(F)) {
            anomalies++;
            continue;
        }
        
        // Проверка на резкие скачки
        double max_val = std::max(std::abs(prev_F), std::abs(F));
        if (max_val > 0 && std::abs(F - prev_F) > 1e6 * max_val) {
            anomalies++;
        }
        
        prev_F = F;
    }
    
    // Допускаем до 1% аномалий
    return anomalies < 0.01 * N_grid;
}

// ============== Анализ баланса компонент (шаг 6.1.5) ==============

bool SolutionValidator::analyze_functional_balance(const Polynomial& poly,
                                                  const OptimizationProblemData& data,
                                                  FunctionalDiagnostics& diagnostics) const {
    // Создаём конфигурацию для функционала
    ApproximationConfig config;
    config.polynomial_degree = poly.degree();
    config.interval_start = data.interval_a;
    config.interval_end = data.interval_b;
    config.gamma = data.gamma;
    
    // Заполняем точки
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        config.approx_points.emplace_back(data.approx_x[i], data.approx_f[i], 
                                         data.approx_weight.size() > i ? data.approx_weight[i] : 1.0);
    }
    
    for (size_t j = 0; j < data.repel_y.size(); ++j) {
        config.repel_points.emplace_back(data.repel_y[j], 
                                        data.repel_forbidden.size() > j ? data.repel_forbidden[j] : 0.0,
                                        data.repel_weight.size() > j ? data.repel_weight[j] : 1.0);
    }
    
    Functional functional(config);
    auto components = functional.get_components(poly);
    
    diagnostics.raw_approx = components.approx_component;
    diagnostics.raw_repel = components.repel_component;
    diagnostics.raw_reg = components.reg_component;
    diagnostics.total_functional = components.total;
    
    double total = components.total;
    if (total > 0) {
        diagnostics.share_approx = 100.0 * components.approx_component / total;
        diagnostics.share_repel = 100.0 * components.repel_component / total;
        diagnostics.share_reg = 100.0 * components.reg_component / total;
    } else {
        diagnostics.share_approx = diagnostics.share_repel = diagnostics.share_reg = 0.0;
    }
    
    return true;
}

std::string SolutionValidator::classify_balance(double share_approx, 
                                              double share_repel, 
                                              double share_reg) const {
    // Идеальный баланс: все доли в диапазоне [20%, 60%]
    bool ideal = (share_approx >= 20.0 && share_approx <= 60.0 &&
                  share_repel >= 20.0 && share_repel <= 60.0 &&
                  share_reg >= 20.0 && share_reg <= 60.0);
    
    // Вычисляем разницу между максимальной и минимальной долей
    double max_share = std::max({share_approx, share_repel, share_reg});
    double min_share = std::min({share_approx, share_repel, share_reg});
    double imbalance_gap = max_share - min_share;
    
    // Сильный дисбаланс: разрыв > 50% или максимальная доля > 75%
    bool strong = (max_share > 75.0) || (imbalance_gap > 50.0);
    
    // Умеренный дисбаланс: разрыв > 30% или максимальная доля > 60%
    bool moderate = (max_share > 60.0) || (imbalance_gap > 30.0);
    
    if (ideal) {
        return "IDEAL BALANCE";
    } else if (strong) {
        return "STRONG IMBALANCE";
    } else if (moderate) {
        return "MODERATE IMBALANCE";
    }
    return "MINOR IMBALANCE";
}

// ============== Оценка качества (шаг 6.1.7) ==============

double SolutionValidator::compute_quality_score(const ValidationResult& validation,
                                               const FunctionalDiagnostics& diagnostics) const {
    // Нормированная оценка интерполяции: (1 - normalized_interp_error)
    double adaptive_threshold = interp_tolerance;
    double normalized_interp = normalize_metric_impl(
        validation.max_interpolation_error, 
        0.0, 
        10.0 * adaptive_threshold
    );
    double score_interp = 1.0 - normalized_interp;
    
    // Оценка безопасности барьеров (бинарная)
    double score_barrier = validation.barriers_safe ? 1.0 : 0.0;
    
    // Оценка стабильности (бинарная)
    double score_stability = validation.numerically_stable ? 1.0 : 0.0;
    
    // Оценка баланса
    double balance_score = 1.0;
    double total_share = diagnostics.share_approx + diagnostics.share_repel + diagnostics.share_reg;
    if (total_share > 0) {
        double max_share = std::max({diagnostics.share_approx, 
                                     diagnostics.share_repel, 
                                     diagnostics.share_reg});
        if (max_share > 99.0) {
            balance_score = 0.0;
        } else if (max_share > 70.0) {
            balance_score = 0.5;
        } else {
            balance_score = 1.0;
        }
    }
    
    // Взвешенная сумма (шаг 6.1.7)
    // качество = 0.4 * интерполяция + 0.3 * барьеры + 0.2 * стабильность + 0.1 * баланс
    double quality = 0.4 * score_interp + 0.3 * score_barrier + 
                     0.2 * score_stability + 0.1 * balance_score;
    
    return std::max(0.0, std::min(1.0, quality));
}

SolutionQuality SolutionValidator::classify_quality(double quality_score) const {
    // Классификация по шкале качества (шаг 6.1.7)
    if (quality_score >= 0.95) {
        return SolutionQuality::EXCELLENT;
    } else if (quality_score >= 0.85) {
        return SolutionQuality::GOOD;
    } else if (quality_score >= 0.70) {
        return SolutionQuality::SATISFACTORY;
    }
    return SolutionQuality::UNACCEPTABLE;
}

std::vector<std::string> SolutionValidator::generate_quality_recommendations(
    SolutionQuality quality) const {
    std::vector<std::string> recommendations;
    
    std::ostringstream oss;
    
    switch (quality) {
        case SolutionQuality::EXCELLENT:
            recommendations.push_back("Solution passed all checks. Ready for use.");
            recommendations.push_back("No modifications required.");
            break;
            
        case SolutionQuality::GOOD:
            recommendations.push_back("Solution is acceptable.");
            recommendations.push_back("Visual inspection recommended for final confirmation.");
            break;
            
        case SolutionQuality::SATISFACTORY:
            recommendations.push_back("Moderate violations detected.");
            oss << "Consider adjusting regularization parameter gamma by +/- 20%.";
            recommendations.push_back(oss.str());
            recommendations.push_back("Check data quality for potential outliers.");
            break;
            
        case SolutionQuality::UNACCEPTABLE:
            recommendations.push_back("CRITICAL: Solution does not meet requirements.");
            recommendations.push_back("Correction required (step 6.2).");
            recommendations.push_back("Consider increasing gamma or adjusting barrier weights.");
            recommendations.push_back("May need to reduce polynomial degree.");
            break;
    }
    
    return recommendations;
}

// ============== Проекционная коррекция ==============

bool SolutionValidator::apply_projection_correction(Polynomial& poly,
                                                  const OptimizationProblemData& data) const {
    if (data.interp_z.empty()) {
        return false;
    }
    
    // Построение интерполяционного полинома по узлам
    std::vector<InterpolationNode> nodes;
    nodes.reserve(data.interp_z.size());
    for (size_t e = 0; e < data.interp_z.size(); ++e) {
        nodes.emplace_back(data.interp_z[e], data.interp_f[e]);
    }
    
    try {
        // Построение интерполяционного полинома Лагранжа
        Polynomial P_int = build_lagrange_polynomial(nodes);
        
        // Заменяем коэффициенты полинома на P_int (гарантирует интерполяцию)
        poly = P_int;
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// ============== Генерация отчётов (шаг 6.1.6) ==============

std::string SolutionValidator::generate_report(const ValidationResult& result) const {
    std::ostringstream oss;
    oss << "=== SOLUTION VERIFICATION REPORT ===\n\n";
    
    oss << "Overall Status: " << (result.is_valid ? "PASSED" : "FAILED") << "\n";
    oss << "Verification Status Code: " << result.status << "\n\n";
    
    oss << "Checks:\n";
    oss << "  Numerical correctness:  " << (result.numerical_correct ? "PASS" : "FAIL") << "\n";
    oss << "  Interpolation:         " << (result.interpolation_ok ? "PASS" : "FAIL");
    if (!result.interpolation_ok) {
        oss << " (max error: " << std::scientific << std::setprecision(2) 
            << result.max_interpolation_error << ")";
    }
    oss << "\n";
    oss << "  Barrier safety:        " << (result.barriers_safe ? "PASS" : "FAIL") << "\n";
    oss << "  Physical plausibility: " << (result.physically_plausible ? "PASS" : "FAIL") << "\n";
    oss << "  Numerical stability:   " << (result.numerically_stable ? "PASS" : "FAIL") << "\n";
    
    if (!result.warnings.empty()) {
        oss << "\nWarnings (" << result.warnings.size() << "):\n";
        for (const auto& w : result.warnings) {
            oss << "  - " << w << "\n";
        }
    }
    
    if (!result.message.empty()) {
        oss << "\nMessage: " << result.message << "\n";
    }
    
    return oss.str();
}

std::string SolutionValidator::generate_extended_report(
    const ExtendedVerificationResult& result) const {
    std::ostringstream oss;
    
    oss << "=== SOLUTION VERIFICATION REPORT ===\n";
    oss << "(Extended Analysis - Step 6.1)\n\n";
    
    // 1. Интерполяционные условия
    oss << "1. INTERPOLATION CONDITIONS:\n";
    double threshold = interp_tolerance;
    oss << "   Status: " << (result.validation.interpolation_ok ? "SATISFACTORY" : "FAILED") << "\n";
    oss << "   Max deviation: " << std::scientific << std::setprecision(2) 
        << result.validation.max_interpolation_error << "\n";
    oss << "   Adaptive threshold: " << std::scientific << std::setprecision(2) << threshold << "\n";
    
    // Классификация точности
    double rel_error = result.validation.max_interpolation_error / threshold;
    if (rel_error < 0.1) {
        oss << "   Classification: IDEAL (error < 0.1 * threshold)\n";
    } else if (rel_error < 1.0) {
        oss << "   Classification: SATISFACTORY (error < threshold)\n";
    } else if (rel_error < 10.0) {
        oss << "   Classification: WARNING (error near threshold)\n";
    } else {
        oss << "   Classification: CRITICAL (error >> threshold)\n";
    }
    oss << "\n";
    
    // 2. Безопасность барьеров
    oss << "2. BARRIER SAFETY:\n";
    oss << "   Status: " << (result.validation.barriers_safe ? "SAFE" : "UNSAFE") << "\n";
    oss << "   Base threshold: " << epsilon_safe << "\n";
    oss << "   Adaptive threshold: " << epsilon_safe << "\n\n";
    
    // 3. Численная стабильность
    oss << "3. NUMERICAL STABILITY:\n";
    oss << "   Status: " << (result.validation.numerically_stable ? "STABLE" : "UNSTABLE") << "\n";
    oss << "   Numerical correctness: " << (result.validation.numerical_correct ? "OK" : "FAILED") << "\n";
    oss << "   Physical plausibility: " << (result.validation.physically_plausible ? "OK" : "FAILED") << "\n";
    oss << "   Condition number: " << std::scientific << std::setprecision(2) 
        << result.validation.condition_number << "\n\n";
    
    // 4. Баланс компонент
    oss << "4. FUNCTIONAL BALANCE:\n";
    oss << "   Approximation:  " << std::fixed << std::setprecision(1) 
        << result.diagnostics.share_approx << "%\n";
    oss << "   Repulsion:       " << std::fixed << std::setprecision(1) 
        << result.diagnostics.share_repel << "%\n";
    oss << "   Regularization:  " << std::fixed << std::setprecision(1) 
        << result.diagnostics.share_reg << "%\n";
    
    std::string balance_status = classify_balance(
        result.diagnostics.share_approx,
        result.diagnostics.share_repel,
        result.diagnostics.share_reg
    );
    oss << "   Balance status:  " << balance_status << "\n\n";
    
    // 5. Общая оценка качества
    oss << "5. OVERALL QUALITY ASSESSMENT:\n";
    oss << "   Quality Score: " << std::fixed << std::setprecision(2) 
        << result.quality_score << "/1.0\n";
    oss << "   Quality Level: ";
    
    switch (result.quality) {
        case SolutionQuality::EXCELLENT:
            oss << "EXCELLENT (>= 0.95)\n";
            break;
        case SolutionQuality::GOOD:
            oss << "GOOD (0.85 - 0.95)\n";
            break;
        case SolutionQuality::SATISFACTORY:
            oss << "SATISFACTORY (0.70 - 0.85)\n";
            break;
        case SolutionQuality::UNACCEPTABLE:
            oss << "UNACCEPTABLE (< 0.70)\n";
            break;
    }
    
    // 6. Рекомендации
    oss << "\nRECOMMENDATIONS:\n";
    for (const auto& rec : result.recommendations) {
        oss << "  - " << rec << "\n";
    }
    
    return oss.str();
}

std::string SolutionValidator::status_to_string(VerificationStatus status) {
    switch (status) {
        case VerificationStatus::VERIFICATION_OK:
            return "VERIFICATION_OK - All conditions satisfied";
        case VerificationStatus::VERIFICATION_WARNING:
            return "VERIFICATION_WARNING - Minor violations detected";
        case VerificationStatus::VERIFICATION_CRITICAL:
            return "VERIFICATION_CRITICAL - Correction required";
        case VerificationStatus::VERIFICATION_FAILED:
            return "VERIFICATION_FAILED - Restart optimization needed";
    }
    return "UNKNOWN";
}

std::string SolutionValidator::quality_to_string(SolutionQuality quality) {
    switch (quality) {
        case SolutionQuality::EXCELLENT:
            return "EXCELLENT - Ready for use";
        case SolutionQuality::GOOD:
            return "GOOD - Acceptable with visual check";
        case SolutionQuality::SATISFACTORY:
            return "SATISFACTORY - Parameter tuning recommended";
        case SolutionQuality::UNACCEPTABLE:
            return "UNACCEPTABLE - Correction required";
    }
    return "UNKNOWN";
}

// ============== Вспомогательные методы ==============

double SolutionValidator::compute_adaptive_interpolation_threshold(
    const OptimizationProblemData& data) const {
    return compute_adaptive_interpolation_threshold_impl(data, interp_tolerance);
}

double SolutionValidator::compute_adaptive_safety_threshold(
    const OptimizationProblemData& data) const {
    return compute_adaptive_safety_threshold_impl(data, epsilon_safe);
}

int SolutionValidator::find_worst_interpolation_node(const Polynomial& poly,
                                                    const OptimizationProblemData& data,
                                                    double& max_error) const {
    max_error = 0.0;
    int worst_idx = -1;
    
    for (size_t e = 0; e < data.interp_z.size(); ++e) {
        double z = data.interp_z[e];
        double f_z = data.interp_f[e];
        double F_z = poly.evaluate(z);
        double error = std::abs(F_z - f_z);
        
        if (error > max_error) {
            max_error = error;
            worst_idx = static_cast<int>(e);
        }
    }
    
    return worst_idx;
}

int SolutionValidator::find_most_dangerous_barrier(const Polynomial& poly,
                                                  const OptimizationProblemData& data,
                                                  double& min_distance) const {
    min_distance = std::numeric_limits<double>::infinity();
    int worst_idx = -1;
    
    for (size_t j = 0; j < data.repel_y.size(); ++j) {
        double y = data.repel_y[j];
        double y_forbidden = data.repel_forbidden[j];
        double F_y = poly.evaluate(y);
        double distance = std::abs(F_y - y_forbidden);
        
        if (distance < min_distance) {
            min_distance = distance;
            worst_idx = static_cast<int>(j);
        }
    }
    
    return worst_idx;
}

int SolutionValidator::count_extrema(const Polynomial& poly,
                                    const OptimizationProblemData& data) const {
    int N_grid = std::max(100, 10 * poly.degree());
    std::vector<double> roots = find_derivative_roots(
        poly, data.interval_a, data.interval_b, N_grid
    );
    return static_cast<int>(roots.size());
}

double SolutionValidator::compute_normalized_curvature(const Polynomial& poly,
                                                      const OptimizationProblemData& data) const {
    double a = data.interval_a;
    double b = data.interval_b;
    double range = b - a;
    
    if (range <= 0) return 0.0;
    
    // Вычисляем ||F''||^2 через интегрирование
    int n = 200;
    double h = range / n;
    double integral = 0.0;
    
    for (int i = 0; i <= n; ++i) {
        double x = a + i * h;
        double weight = (i == 0 || i == n) ? 0.5 : 1.0;
        double second_deriv = poly.second_derivative(x);
        integral += weight * second_deriv * second_deriv * h;
    }
    
    double expected_scale = compute_expected_curvature_scale(data);
    if (expected_scale > 0) {
        return integral / expected_scale;
    }
    return 0.0;
}

} // namespace mixed_approx
