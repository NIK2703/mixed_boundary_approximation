#include "mixed_approximation/solution_validator.h"
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/optimization_problem_data.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <algorithm>

namespace mixed_approx {

bool SolutionValidator::check_numerical_correctness(const Polynomial& poly, 
                                                    const OptimizationProblemData& data) const {
    // Проверка коэффициентов полинома
    for (const double& coeff : poly.coefficients()) {
        if (!std::isfinite(coeff)) {
            return false;
        }
    }
    
    // Проверка значений в аппроксимирующих точках
    for (double x : data.approx_x) {
        double value = poly.evaluate(x);
        if (!std::isfinite(value)) {
            return false;
        }
    }
    
    // Проверка значений в интерполяционных узлах
    for (double x : data.interp_z) {
        double value = poly.evaluate(x);
        if (!std::isfinite(value)) {
            return false;
        }
    }
    
    return true;
}

bool SolutionValidator::check_interpolation(const Polynomial& poly, 
                                            const OptimizationProblemData& data,
                                            double& max_error) const {
    max_error = 0.0;
    
    // Если нет интерполяционных узлов, считаем что условия выполнены
    if (data.interp_z.empty()) {
        return true;
    }
    
    for (size_t i = 0; i < data.interp_z.size(); ++i) {
        double x = data.interp_z[i];
        double target = data.interp_f[i];
        double computed = poly.evaluate(x);
        double error = std::abs(computed - target);
        max_error = std::max(max_error, error);
        
        if (error > interp_tolerance) {
            return false;
        }
    }
    
    return true;
}

bool SolutionValidator::check_barrier_safety(const Polynomial& poly, 
                                             const OptimizationProblemData& data,
                                             double& min_distance) const {
    min_distance = std::numeric_limits<double>::infinity();
    
    // Если нет отталкивающих точек, считаем что барьеры безопасны
    if (data.repel_y.empty()) {
        return true;
    }
    
    for (size_t i = 0; i < data.repel_y.size(); ++i) {
        double x = data.repel_y[i];
        double y_forbidden = data.repel_forbidden[i];
        double poly_value = poly.evaluate(x);
        double distance = std::abs(y_forbidden - poly_value);
        min_distance = std::min(min_distance, distance);
        
        if (distance < epsilon_safe) {
            return false;
        }
    }
    
    return true;
}

bool SolutionValidator::check_physical_plausibility(const Polynomial& poly, 
                                                   const OptimizationProblemData& data,
                                                   double& max_value) const {
    max_value = 0.0;
    
    // Оценка масштаба функции
    double scale = 1.0;
    if (!data.approx_f.empty()) {
        double max_f = *std::max_element(data.approx_f.begin(), data.approx_f.end());
        double min_f = *std::min_element(data.approx_f.begin(), data.approx_f.end());
        scale = std::max(std::abs(max_f), std::abs(min_f));
        if (scale < 1e-10) scale = 1.0;
    }
    
    double threshold = scale * max_value_factor;
    
    // Проверка значений в контрольных точках
    int num_points = std::min(num_check_points, 1000);
    double a = data.interval_a;
    double b = data.interval_b;
    
    for (int i = 0; i <= num_points; ++i) {
        double t = static_cast<double>(i) / num_points;
        double x = a + t * (b - a);
        double value = std::abs(poly.evaluate(x));
        max_value = std::max(max_value, value);
        
        if (value > threshold) {
            return false;
        }
    }
    
    return true;
}

bool SolutionValidator::apply_projection_correction(Polynomial& poly, 
                                                     const OptimizationProblemData& data) const {
    // Простая проекционная коррекция:adjust coefficients to satisfy interpolation
    // Note: This is a simplified implementation
    (void)poly;
    (void)data;
    return false;  // Простая реализация - коррекция не применяется
}

ValidationResult SolutionValidator::validate(const Polynomial& poly, 
                                            const OptimizationProblemData& data) const {
    ValidationResult result;
    result.is_valid = true;
    result.numerical_correct = true;
    result.interpolation_ok = true;
    result.barriers_safe = true;
    result.physically_plausible = true;
    result.message = "Validation passed";
    
    // 1. Проверка численной корректности (нет NaN/Inf)
    for (double x : data.approx_x) {
        double value = poly.evaluate(x);
        if (!std::isfinite(value)) {
            result.numerical_correct = false;
            result.is_valid = false;
            result.message = "Numerical anomaly detected: NaN or Inf in evaluation";
            result.warnings.push_back("Non-finite value at x = " + std::to_string(x));
        }
    }
    
    // 2. Проверка интерполяционных условий
    double max_interp_error = 0.0;
    for (size_t i = 0; i < data.interp_z.size(); ++i) {
        double x = data.interp_z[i];
        double target = data.interp_f[i];
        double computed = poly.evaluate(x);
        double error = std::abs(computed - target);
        max_interp_error = std::max(max_interp_error, error);
        
        if (error > 1e-10) {
            result.interpolation_ok = false;
            result.is_valid = false;
            result.warnings.push_back("Interpolation error at x = " + std::to_string(x) + 
                                     ": expected " + std::to_string(target) + 
                                     ", got " + std::to_string(computed));
        }
    }
    result.max_interpolation_error = max_interp_error;
    
    // 3. Проверка безопасности барьеров
    double eps_safe = 1e-8;
    for (size_t i = 0; i < data.repel_y.size(); ++i) {
        double x = data.repel_y[i];
        double y_forbidden = data.repel_forbidden[i];
        double poly_value = poly.evaluate(x);
        double distance = std::abs(y_forbidden - poly_value);
        
        if (distance < eps_safe) {
            result.barriers_safe = false;
            result.is_valid = false;
            result.warnings.push_back("Barrier too close at x = " + std::to_string(x) + 
                                     ": distance = " + std::to_string(distance));
        }
    }
    
    // 4. Проверка физической правдоподобности
    // Проверяем, что значения в аппроксимирующих точках не слишком отклоняются
    double mean_f = 0.0;
    for (double f : data.approx_f) {
        mean_f += f;
    }
    mean_f /= data.approx_f.size();
    
    double max_deviation = 0.0;
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        double x = data.approx_x[i];
        double f_target = data.approx_f[i];
        double poly_value = poly.evaluate(x);
        double deviation = std::abs(poly_value - f_target);
        max_deviation = std::max(max_deviation, deviation);
        
        // Проверяем, что отклонение не превышает 100*std(f)
        double std_f = 0.0;
        for (double f : data.approx_f) {
            std_f += (f - mean_f) * (f - mean_f);
        }
        std_f = std::sqrt(std_f / data.approx_f.size());
        
        if (std_f > 0 && deviation > 100 * std_f) {
            result.physically_plausible = false;
            result.is_valid = false;
            result.warnings.push_back("Large deviation at x = " + std::to_string(x) + 
                                     ": deviation = " + std::to_string(deviation) + 
                                     ", 100*std(f) = " + std::to_string(100 * std_f));
        }
    }
    
    // 5. Проверка числа обусловленности (через простую оценку)
    double condition = 1.0;
    if (data.num_approx_points() > 0) {
        // Простая оценка: отношение max/min значений в точках
        double max_val = std::numeric_limits<double>::lowest();
        double min_val = std::numeric_limits<double>::max();
        for (double x : data.approx_x) {
            double val = poly.evaluate(x);
            max_val = std::max(max_val, std::abs(val));
            min_val = std::min(min_val, std::abs(val));
        }
        if (min_val > 1e-15) {
            condition = max_val / min_val;
        }
    }
    result.condition_number = condition;
    result.numerically_stable = (condition < 1e12);
    
    if (!result.numerically_stable) {
        result.warnings.push_back("High condition number: " + std::to_string(condition));
    }
    
    return result;
}

std::string SolutionValidator::generate_report(const ValidationResult& result) const {
    std::ostringstream oss;
    oss << "=== VALIDATION REPORT ===\n\n";
    oss << "Overall: " << (result.is_valid ? "PASSED" : "FAILED") << "\n\n";
    oss << "Checks:\n";
    oss << "  Numerical correctness: " << (result.numerical_correct ? "PASS" : "FAIL") << "\n";
    oss << "  Interpolation: " << (result.interpolation_ok ? "PASS" : "FAIL") << "\n";
    oss << "  Barrier safety: " << (result.barriers_safe ? "PASS" : "FAIL") << "\n";
    oss << "  Physical plausibility: " << (result.physically_plausible ? "PASS" : "FAIL") << "\n";
    oss << "  Numerical stability: " << (result.numerically_stable ? "PASS" : "FAIL") << "\n";
    oss << "\nMetrics:\n";
    oss << "  Max interpolation error: " << result.max_interpolation_error << "\n";
    oss << "  Condition number: " << result.condition_number << "\n";
    
    if (!result.warnings.empty()) {
        oss << "\nWarnings:\n";
        for (const auto& w : result.warnings) {
            oss << "  - " << w << "\n";
        }
    }
    
    if (!result.message.empty()) {
        oss << "\nMessage: " << result.message << "\n";
    }
    
    return oss.str();
}

} // namespace mixed_approx
