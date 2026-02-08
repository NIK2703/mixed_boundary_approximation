#ifndef MIXED_APPROXIMATION_FUNCTIONAL_H
#define MIXED_APPROXIMATION_FUNCTIONAL_H

#include "types.h"
#include "polynomial.h"
#include <vector>

namespace mixed_approx {

/**
 * @brief Класс для вычисления функционала смешанной аппроксимации и его градиента
 * 
 * Функционал: J = Σ_i |f(x_i) - F(x_i)|^2 / σ_i + Σ_j B_j / |y_j^* - F(y_j)|^2 + γ ∫_a^b (F''(x))^2 dx
 */
class Functional {
private:
    ApproximationConfig config_;
    
public:
    /**
     * @brief Конструктор
     * @param config конфигурация метода
     */
    explicit Functional(const ApproximationConfig& config);
    
    /**
     * @brief Вычисление значения функционала
     * @param poly полином F(x)
     * @return значение J
     */
    double evaluate(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление градиента функционала по коэффициентам полинома
     * @param poly полином F(x)
     * @return градиент (вектор той же размерности, что и коэффициенты poly)
     */
    std::vector<double> gradient(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление компонент функционала для анализа
     * @param poly полином F(x)
     * @return кортеж: (J_approx, J_repel, J_reg, total)
     */
    struct Components {
        double approx_component;
        double repel_component;
        double reg_component;
        double total;
    };
    
    Components get_components(const Polynomial& poly) const;
    
private:
    /**
     * @brief Вычисление аппроксимирующего компонента
     * J_approx = Σ_i |f(x_i) - F(x_i)|^2 / σ_i
     */
    double compute_approx_component(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление отталкивающего компонента
     * J_repel = Σ_j B_j / |y_j^* - F(y_j)|^2
     * Использует защиту от деления на ноль через epsilon
     */
    double compute_repel_component(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление регуляризационного компонента
     * J_reg = γ ∫_a^b (F''(x))^2 dx
     */
    double compute_reg_component(const Polynomial& poly) const;
    
    /**
     * @brief Градиент аппроксимирующего компонента по коэффициентам полинома
     */
    std::vector<double> compute_approx_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Градиент отталкивающего компонента по коэффициентам полинома
     */
    std::vector<double> compute_repel_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Градиент регуляризационного компонента по коэффициентам полинома
     */
    std::vector<double> compute_reg_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление защищенного расстояния до отталкивающей точки
     * Использует epsilon для предотвращения деления на ноль
     */
    double safe_repel_distance(double poly_value, double target_value) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_FUNCTIONAL_H
