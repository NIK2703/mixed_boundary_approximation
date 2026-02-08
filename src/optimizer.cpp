#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>
#include <algorithm>

namespace mixed_approx {

OptimizationResult GradientDescentOptimizer::optimize(
    const Functional& functional,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    
    Polynomial poly(result.coefficients);
    double prev_objective = functional.evaluate(poly);
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Вычисляем градиент
        std::vector<double> grad = functional.gradient(poly);
        
        // Вычисляем норму градиента
        double grad_norm = 0.0;
        for (double g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        // Критерий останова по градиенту
        if (grad_norm < tol_gradient_) {
            result.success = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter;
            return result;
        }
        
        // Шаг градиентного спуска (с постоянным шагом)
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= initial_step_ * grad[i];
        }
        
        // Создаем новый полином и вычисляем функционал
        Polynomial new_poly(new_coeffs);
        double new_objective = functional.evaluate(new_poly);
        
        // Проверяем сходимость по изменению функционала
        double objective_change = std::abs(prev_objective - new_objective);
        if (objective_change < tol_objective_ * std::max(1.0, std::abs(prev_objective))) {
            result.coefficients = new_coeffs;
            result.final_objective = new_objective;
            result.iterations = iter + 1;
            result.success = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        // Обновляем коэффициенты
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        poly = Polynomial(new_coeffs);
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

OptimizationResult AdaptiveGradientDescentOptimizer::optimize(
    const Functional& functional,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    
    Polynomial poly(result.coefficients);
    double prev_objective = functional.evaluate(poly);
    double step = initial_step_;
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        std::vector<double> grad = functional.gradient(poly);
        
        double grad_norm = 0.0;
        for (double g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (grad_norm < tol_gradient_) {
            result.success = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter;
            return result;
        }
        
        // Пробуем шаг
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= step * grad[i];
        }
        
        Polynomial new_poly(new_coeffs);
        double new_objective = functional.evaluate(new_poly);
        
        // Если функционал не уменьшился, уменьшаем шаг
        if (new_objective >= prev_objective) {
            step *= 0.5;
            if (step < 1e-12) {
                result.iterations = iter;
                result.message = "Stopped: step size too small";
                return result;
            }
            continue;
        }
        
        // Если улучшение есть, можно немного увеличить шаг
        if (new_objective < prev_objective - 1e-6) {
            step *= 1.1;
        }
        
        double objective_change = std::abs(prev_objective - new_objective);
        if (objective_change < tol_objective_ * std::max(1.0, std::abs(prev_objective))) {
            result.coefficients = new_coeffs;
            result.final_objective = new_objective;
            result.iterations = iter + 1;
            result.success = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        poly = Polynomial(new_coeffs);
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

} // namespace mixed_approx
