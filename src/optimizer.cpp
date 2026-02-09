#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mixed_approx {

OptimizationResult GradientDescentOptimizer::optimize(
    const ObjectiveFunctor& functor,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    result.converged = false;
    
    // Вычисляем начальное значение функционала
    double prev_objective = functor.value(initial_coeffs);
    result.final_objective = prev_objective;
    
    std::vector<double> grad;
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Вычисляем градиент
        functor.gradient(result.coefficients, grad);
        
        // Вычисляем норму градиента
        double grad_norm = 0.0;
        for (double g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        // Критерий останова по градиенту
        if (grad_norm < tol_gradient_) {
            result.success = true;
            result.converged = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter + 1;
            return result;
        }
        
        // Шаг градиентного спуска (с постоянным шагом)
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= initial_step_ * grad[i];
        }
        
        // Вычисляем новый функционал
        double new_objective = functor.value(new_coeffs);
        
        // Проверяем сходимость по изменению функционала
        double objective_change = std::abs(prev_objective - new_objective);
        if (objective_change < tol_objective_ * std::max(1.0, std::abs(prev_objective))) {
            result.coefficients = new_coeffs;
            result.final_objective = new_objective;
            result.iterations = iter + 1;
            result.success = true;
            result.converged = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        // Обновляем коэффициенты
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

// Оптимизация для Functional (для обратной совместимости)
OptimizationResult GradientDescentOptimizer::optimize(
    const Functional& functional,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    result.converged = false;
    
    // Начальный полином
    Polynomial poly(initial_coeffs);
    
    // Вычисляем начальное значение функционала
    double prev_objective = functional.evaluate(poly);
    result.final_objective = prev_objective;
    
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
            result.converged = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter + 1;
            return result;
        }
        
        // Шаг градиентного спуска (с постоянным шагом)
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= initial_step_ * grad[i];
        }
        
        // Новый полином
        Polynomial new_poly(new_coeffs);
        
        // Вычисляем новый функционал
        double new_objective = functional.evaluate(new_poly);
        
        // Проверяем сходимость по изменению функционала
        double objective_change = std::abs(prev_objective - new_objective);
        if (objective_change < tol_objective_ * std::max(1.0, std::abs(prev_objective))) {
            result.coefficients = new_coeffs;
            result.final_objective = new_objective;
            result.iterations = iter + 1;
            result.success = true;
            result.converged = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        // Обновляем полином и коэффициенты
        poly = new_poly;
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

OptimizationResult AdaptiveGradientDescentOptimizer::optimize(
    const ObjectiveFunctor& functor,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    result.converged = false;
    
    // Вычисляем начальное значение функционала
    double prev_objective = functor.value(initial_coeffs);
    result.final_objective = prev_objective;
    
    double step = initial_step_;
    std::vector<double> grad;
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Вычисляем градиент
        functor.gradient(result.coefficients, grad);
        
        double grad_norm = 0.0;
        for (double g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (grad_norm < tol_gradient_) {
            result.success = true;
            result.converged = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter + 1;
            return result;
        }
        
        // Пробуем шаг
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= step * grad[i];
        }
        
        // Вычисляем новый функционал
        double new_objective = functor.value(new_coeffs);
        
        // Если функционал не уменьшился, уменьшаем шаг
        if (new_objective >= prev_objective) {
            step *= 0.5;
            if (step < 1e-12) {
                result.iterations = iter + 1;
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
            result.converged = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

// Оптимизация для Functional (для обратной совместимости)
OptimizationResult AdaptiveGradientDescentOptimizer::optimize(
    const Functional& functional,
    const std::vector<double>& initial_coeffs) {
    
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    result.converged = false;
    
    // Начальный полином
    Polynomial poly(initial_coeffs);
    
    // Вычисляем начальное значение функционала
    double prev_objective = functional.evaluate(poly);
    result.final_objective = prev_objective;
    
    double step = initial_step_;
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Вычисляем градиент
        std::vector<double> grad = functional.gradient(poly);
        
        double grad_norm = 0.0;
        for (double g : grad) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (grad_norm < tol_gradient_) {
            result.success = true;
            result.converged = true;
            result.message = "Converged: gradient norm below tolerance";
            result.iterations = iter + 1;
            return result;
        }
        
        // Пробуем шаг
        std::vector<double> new_coeffs = result.coefficients;
        for (size_t i = 0; i < new_coeffs.size(); ++i) {
            new_coeffs[i] -= step * grad[i];
        }
        
        // Новый полином
        Polynomial new_poly(new_coeffs);
        
        // Вычисляем новый функционал
        double new_objective = functional.evaluate(new_poly);
        
        // Если функционал не уменьшился, уменьшаем шаг
        if (new_objective >= prev_objective) {
            step *= 0.5;
            if (step < 1e-12) {
                result.iterations = iter + 1;
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
            result.converged = true;
            result.message = "Converged: objective change below tolerance";
            return result;
        }
        
        // Обновляем полином и коэффициенты
        poly = new_poly;
        result.coefficients = new_coeffs;
        result.final_objective = new_objective;
        prev_objective = new_objective;
    }
    
    result.iterations = max_iterations_;
    result.message = "Max iterations reached";
    return result;
}

// Реализация базового optimize(Functional) в базовом классе Optimizer
OptimizationResult Optimizer::optimize(const Functional& functional,
                                      const std::vector<double>& initial_coeffs) {
    // По умолчанию используем адаптивный градиентный спуск
    AdaptiveGradientDescentOptimizer default_optimizer;
    default_optimizer.set_parameters(max_iterations_, tol_gradient_, tol_objective_, initial_step_);
    return default_optimizer.optimize(functional, initial_coeffs);
}

} // namespace mixed_approx
