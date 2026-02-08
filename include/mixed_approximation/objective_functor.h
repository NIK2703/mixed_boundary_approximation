#ifndef MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H
#define MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H

#include "types.h"
#include "composite_polynomial.h"
#include "correction_polynomial.h"
#include "optimization_problem_data.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <memory>

namespace mixed_approx {

// Forward declarations for classes used in post-processing
class ConvergenceMonitor;
class InitializationStrategySelector;
class OptimizationPostProcessor;

// ============== Шаг 2.1.9.1: Функтор стоимости для оптимизатора ==============

/**
 * @brief Функтор для вычисления функционала и его градиента
 * 
 * Архитектура:
 * - Хранит константную ссылку на CompositePolynomial (параметризацию)
 * - Хранит константную ссылку на OptimizationProblemData (данные задачи)
 * - Предоставляет три ключевых метода для оптимизатора:
 *   * value(q) — вычисление значения функционала
 *   * gradient(q, grad) — вычисление градиента
 *   * value_and_gradient(q, f, grad) — комбинированное вычисление
 */
class ObjectiveFunctor {
public:
    /**
     * @brief Конструктор
     * @param parametrisation композитный полином F(x) = P_int(x) + Q(x)·W(x)
     * @param data данные задачи оптимизации
     */
    ObjectiveFunctor(const CompositePolynomial& parametrisation,
                     const OptimizationProblemData& data);
    
    /**
     * @brief Вычисление значения функционала J(q)
     * @param q вектор коэффициентов корректирующего полинома Q(x)
     * @return значение функционала
     */
    double value(const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление градиента ∇J(q)
     * @param q вектор коэффициентов
     * @param grad вектор градиента (будет заполнен)
     */
    void gradient(const std::vector<double>& q, std::vector<double>& grad) const;
    
    /**
     * @brief Комбинированное вычисление значения и градиента
     * @param q вектор коэффициентов
     * @param f значение функционала (будет заполнено)
     * @param grad вектор градиента (будет заполнен)
     */
    void value_and_gradient(const std::vector<double>& q,
                            double& f,
                            std::vector<double>& grad) const;
    
    /**
     * @brief Вычисление компонент функционала для диагностики
     * @param q вектор коэффициентов
     * @return структура с компонентами
     */
    struct Components {
        double approx;
        double repel;
        double reg;
        double total;
    };
    
    Components compute_components(const std::vector<double>& q) const;
    
    /**
     * @brief Построение кэшей для ускорения вычислений
     */
    void build_caches();
    
    /**
     * @brief Обновление кэшей при изменении базиса
     * @param n_free число свободных параметров
     */
    void update_basis_cache(int n_free);
    
    /**
     * @brief Проверка валидности функтора
     */
    bool is_valid() const;
    
    /**
     * @brief Получение метаданных
     */
    int num_free_parameters() const { return param_.num_free_parameters(); }
    const CompositePolynomial& parameterization() const { return param_; }
    const OptimizationProblemData& data() const { return problem_data_; }

private:
    // ============== Компоненты функционала ==============
    
    /**
     * @brief Аппроксимирующий член: J_approx = Σ_i weight_i · (F(x_i) - target_i)²
     */
    double compute_approx_term(const std::vector<double>& q) const;
    
    /**
     * @brief Отталкивающий член: J_repel = Σ_j B_j / distance_j²
     */
    double compute_repel_term(const std::vector<double>& q) const;
    
    /**
     * @brief Регуляризационный член: J_reg = γ · ∫(F''(x))² dx
     */
    double compute_reg_term(const std::vector<double>& q) const;
    
    // ============== Градиенты компонентов ==============
    
    /**
     * @brief Градиент аппроксимирующего члена
     * ∂J_approx/∂q_k = 2 · Σ_i weight_i · residual_i · φ_k(x_i) · W_i
     */
    void compute_approx_gradient(const std::vector<double>& q,
                                std::vector<double>& grad) const;
    
    /**
     * @brief Градиент отталкивающего члена
     * ∂J_repel/∂q_k = -2 · Σ_j [B_j / distance_j³] · sign(...) · φ_k(y_j) · W_j
     */
    void compute_repel_gradient(const std::vector<double>& q,
                                std::vector<double>& grad) const;
    
    /**
     * @brief Градиент регуляризационного члена
     * ∂J_reg/∂q_k = 2γ · Σ_l K[k][l] · q_l
     */
    void compute_reg_gradient(const std::vector<double>& q,
                              std::vector<double>& grad) const;
    
    // ============== Защита от численных аномалий ==============
    
    /**
     * @brief Безопасное вычисление расстояния до барьера
     */
    double safe_barrier_distance(double poly_value, double forbidden_value) const;
    
    /**
     * @brief Проверка на численные аномалии
     */
    bool has_numerical_anomaly(double value) const;
    
    // Константные ссылки на внешние данные
    const CompositePolynomial& param_;
    const OptimizationProblemData& problem_data_;
    
    // Кэш для ускорения вычислений
    mutable OptimizationCache cache_;
    
    // Флаг готовности кэшей
    bool caches_built_;
};

// ============== Шаг 2.1.9.9: Интеграция с внешними библиотеками ==============

#ifdef USE_NLOPT

/**
 * @brief Адаптер для библиотеки NLopt
 */
class NloptAdapter {
public:
    /**
     * @brief Конструктор
     * @param functor функтор для вычисления функционала
     */
    explicit NloptAdapter(ObjectiveFunctor* functor);
    
    /**
     * @brief Функция обратного вызова для NLopt
     */
    static double nlopt_objective(unsigned n, const double* q, double* grad, void* data);
    
    /**
     * @brief Создание оптимизатора NLopt
     * @param algorithm алгоритм (NLOPT_LD_LBFGS и т.д.)
     * @return указатель на оптимизатор
     */
    void* create_optimizer(int algorithm);
    
    /**
     * @brief Установка параметров
     */
    void set_parameters(double ftol_rel, double xtol_rel, int max_eval);

};

#endif // USE_NLOPT

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H
