#ifndef MIXED_APPROXIMATION_MIXED_APPROXIMATION_H
#define MIXED_APPROXIMATION_MIXED_APPROXIMATION_H

#include "types.h"
#include "polynomial.h"
#include "functional.h"
#include "validator.h"
#include "optimizer.h"
#include "decomposition.h"
#include "convergence_monitor.h"
#include "solution_validator.h"
#include "objective_functor.h"
#include <memory>

namespace mixed_approx {

// Forward declaration for Functional
class Functional;

/**
 * @brief Структура для хранения компонентов функционала
 */
struct FunctionalComponents {
    double approx_component;   // компонент аппроксимации
    double repel_component;    // компонент отталкивания
    double reg_component;      // компонент регуляризации
    double total;              // сумма компонентов
    
    FunctionalComponents()
        : approx_component(0.0), repel_component(0.0), reg_component(0.0), total(0.0) {}
};

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
     * @brief Конструктор по умолчанию
     */
    MixedApproximation();
    
    /**
     * @brief Конструктор из конфигурации (для обратной совместимости)
     * @param config конфигурация метода
     */
    explicit MixedApproximation(const ApproximationConfig& config);
    
    /**
     * @brief Деструктор
     */
    ~MixedApproximation() = default;
    
    /**
     * @brief Построение начального приближения
     * @param data данные задачи (аппроксимация, отталкивание, интерполяция)
     * @param n_free число свободных параметров (степень корректирующего полинома + 1)
     * @return результат инициализации
     */
    InitializationResult build_initial_approximation(const OptimizationProblemData& data, int n_free);
    
    /**
     * @brief Выполнение оптимизации
     * @param data данные задачи
     * @param n_free число свободных параметров
     * @param optimizer оптимизатор (если nullptr, используется L-BFGS-B)
     * @return результат оптимизации
     */
    OptimizationResult solve(const OptimizationProblemData& data, int n_free, std::unique_ptr<Optimizer> optimizer = nullptr);
    
    /**
     * @brief Выполнение оптимизации (для обратной совместимости)
     * Использует ранее установленную конфигурацию
     * @return результат оптимизации
     */
    OptimizationResult solve();
    
    /**
     * @brief Получение построенного полинома (полная форма)
     * @return полином F(x) в виде явных коэффициентов
     */
    Polynomial get_polynomial() const { return polynomial_; }
    
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
     * @brief Получение компонентов функционала (для обратной совместимости)
     * @return структура с компонентами функционала
     */
    FunctionalComponents get_functional_components() const;
    
private:
    Polynomial polynomial_;                    // построенный полином F(x)
    CompositePolynomial composite_poly_;       // параметризация F = P_int + Q·W
    OptimizationProblemData problem_data_;     // данные задачи
    bool parametrization_built_;               // флаг: построена ли параметризация
    ApproximationConfig config_;               // конфигурация (для обратной совместимости)
    bool has_config_;                          // флаг: установлена ли конфигурация
    
    /**
     * @brief Внутренняя инициализация из конфигурации
     */
    void initialize_from_config(const ApproximationConfig& config);
    
    /**
     * @brief Применение интерполяционных ограничений к полиному
     * @param poly исходный полином
     * @return скорректированный полином
     */
    Polynomial apply_interpolation_constraints(const Polynomial& poly) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_MIXED_APPROXIMATION_H
