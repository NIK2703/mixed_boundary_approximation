#ifndef MIXED_APPROXIMATION_MIXED_APPROXIMATION_H
#define MIXED_APPROXIMATION_MIXED_APPROXIMATION_H

#include "types.h"
#include "polynomial.h"
#include "functional.h"
#include "validator.h"
#include "optimizer.h"
#include "decomposition.h"
#include <memory>

namespace mixed_approx {

/**
 * @brief Основной класс метода смешанной аппроксимации
 * 
 * Реализует метод смешанной аппроксимации для нахождения полинома F(x) степени n,
 * который минимизирует функционал:
 * J = Σ_i |f(x_i) - F(x_i)|^2 / σ_i + Σ_j B_j / |y_j^* - F(y_j)|^2 + γ ∫_a^b (F''(x))^2 dx
 * 
 * При этом F(z_e) = f(z_e) для всех интерполяционных узлов.
 */
class MixedApproximation {
public:
    /**
     * @brief Конструктор
     * @param config конфигурация метода
     */
    explicit MixedApproximation(const ApproximationConfig& config);
    
    /**
     * @brief Деструктор
     */
    ~MixedApproximation() = default;
    
    /**
     * @brief Выполнение аппроксимации
     * @param optimizer оптимизатор (если nullptr, используется градиентный спуск по умолчанию)
     * @return результат оптимизации
     */
    OptimizationResult solve(std::unique_ptr<Optimizer> optimizer = nullptr);
    
    /**
     * @brief Получение построенного полинома (полная форма)
     * @return полином F(x) в виде явных коэффициентов
     */
    Polynomial get_polynomial() const;
    
    /**
     * @brief Получение конфигурации
     * @return конфигурация
     */
    const ApproximationConfig& config() const { return config_; }
    
    /**
     * @brief Проверка корректности интерполяционных условий
     * @param tolerance допуск
     * @return true, если все условия выполняются с заданной точностью
     */
    bool check_interpolation_conditions(double tolerance = 1e-10) const;
    
    /**
     * @brief Вычисление расстояний до отталкивающих точек
     * @return вектор расстояний |F(y_j) - y_j^*|
     */
    std::vector<double> compute_repel_distances() const;
    
    /**
     * @brief Получение компонент функционала
     * @return структура с компонентами
     */
    Functional::Components get_functional_components() const;
    
private:
    ApproximationConfig config_;
    Functional functional_;
    Polynomial polynomial_;
    
    /**
     * @brief Построение начального приближения
     * @return полином, удовлетворяющий интерполяционным условиям
     */
    Polynomial build_initial_approximation();
    
    /**
     * @brief Применение интерполяционных ограничений к полиному
     * Корректирует коэффициенты полинома, чтобы выполнить F(z_e) = f(z_e)
     * @param poly исходный полином
     * @return скорректированный полином
     */
    Polynomial apply_interpolation_constraints(const Polynomial& poly) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_MIXED_APPROXIMATION_H
