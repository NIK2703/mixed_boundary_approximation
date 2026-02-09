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

/**
 * @brief L-BFGS-B оптимизатор с использованием библиотеки NLopt
 *
 * Рекомендуемый алгоритм для задач средней размерности (5 < n_free ≤ 50)
 * Поддерживает ограничения на переменные для стабилизации
 */
class LBFGSOptimizer : public Optimizer {
public:
    LBFGSOptimizer();
    ~LBFGSOptimizer();
    
    OptimizationResult optimize(const Functional& functional,
                                const std::vector<double>& initial_coeffs) override;
    
    /**
     * @brief Установка параметров L-BFGS-B (базовые параметры)
     * Использует базовый интерфейс Optimizer::set_parameters для общих параметров
     */
    using Optimizer::set_parameters;
    
    /**
     * @brief Установка параметров L-BFGS-B (включая специфичные)
     * @param max_iterations максимальное число итераций
     * @param tol_gradient допуск по градиенту
     * @param tol_objective допуск по изменению функционала
     * @param initial_step начальный шаг (не используется для L-BFGS-B)
     * @param lbfgs_memory размер памяти L-BFGS (количество сохраняемых коррекций)
     * @param max_line_search_iters максимальное число итераций линейного поиска
     */
    void set_lbfgs_parameters(int max_iterations,
                              double tol_gradient,
                              double tol_objective,
                              double initial_step = 0.01,
                              int lbfgs_memory = 10,
                              int max_line_search_iters = 20);
    
    /**
     * @brief Установка границ для параметров (для стабилизации)
     * @param lower_bound нижние границы (n размеров)
     * @param upper_bound верхние границы (n размеров)
     */
    void set_bounds(const std::vector<double>& lower_bound,
                    const std::vector<double>& upper_bound);

private:
    int lbfgs_memory_;
    int max_line_search_iters_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    bool bounds_set_;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OPTIMIZER_H
