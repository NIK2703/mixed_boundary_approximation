#ifndef MIXED_APPROXIMATION_OPTIMIZER_H
#define MIXED_APPROXIMATION_OPTIMIZER_H

#include "types.h"
#include "functional.h"
#include <vector>
#include <string>

namespace mixed_approx {

/**
 * @brief Абстрактный класс оптимизатора
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /**
     * @brief Оптимизация функционала
     * @param functional функционал для минимизации
     * @param initial_coeffs начальные коэффициенты полинома
     * @return результат оптимизации
     */
    virtual OptimizationResult optimize(const Functional& functional, 
                                        const std::vector<double>& initial_coeffs) = 0;
    
    /**
     * @brief Установка параметров оптимизатора
     * @param max_iterations максимальное число итераций
     * @param tol_gradient допуск по градиенту
     * @param tol_objective допуск по изменению функционала
     * @param initial_step начальный шаг (для некоторых методов)
     */
    virtual void set_parameters(int max_iterations, 
                                double tol_gradient, 
                                double tol_objective,
                                double initial_step = 0.01) {
        max_iterations_ = max_iterations;
        tol_gradient_ = tol_gradient;
        tol_objective_ = tol_objective;
        initial_step_ = initial_step;
    }
    
protected:
    int max_iterations_ = 1000;
    double tol_gradient_ = 1e-6;
    double tol_objective_ = 1e-8;
    double initial_step_ = 0.01;
};

/**
 * @brief Простой градиентный спуск с постоянным шагом
 */
class GradientDescentOptimizer : public Optimizer {
public:
    OptimizationResult optimize(const Functional& functional,
                                const std::vector<double>& initial_coeffs) override;
};

/**
 * @brief Градиентный спуск с адаптивным шагом (простая реализация)
 */
class AdaptiveGradientDescentOptimizer : public Optimizer {
public:
    OptimizationResult optimize(const Functional& functional,
                                const std::vector<double>& initial_coeffs) override;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OPTIMIZER_H
