#ifndef MIXED_APPROXIMATION_CONSTRAINED_POLYNOMIAL_H
#define MIXED_APPROXIMATION_CONSTRAINED_POLYNOMIAL_H

#include "types.h"
#include "i_polynomial.h"
#include "interpolation_basis.h"
#include "weight_multiplier.h"
#include "correction_polynomial.h"
#include "lru_cache.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace mixed_approx {

/**
 * @brief Ограниченный полином с автоматическим выполнением интерполяционных условий
 *
 * Представление: F(x) = P_int(x) + Q(x) · W(x)
 * где:
 *   - P_int(x) — интерполяционный полином (барицентрический базис)
 *   - W(x) = ∏_{e=1..m} (x - z_e) — весовой множитель с корнями в узлах интерполяции
 *   - Q(x) = Σ_{k=0..deg_Q} q_k · φ_k(x) — корректирующий полином со свободными коэффициентами
 *
 * Ключевое свойство: F(z_e) = P_int(z_e) + Q(z_e)·W(z_e) = f(z_e) + 0 = f(z_e)
 *                   Интерполяционные условия выполняются автоматически!
 *
 * Оптимизация градиента по свободным параметрам q_k:
 *   ∂F(x)/∂q_k = φ_k(x) · W(x)
 */
class ConstrainedPolynomial : public IPolynomial {
private:
    // Основные компоненты
    InterpolationBasis p_int_;      // Интерполяционный базис P_int(x)
    WeightMultiplier W_;            // Весовой множитель W(x)
    CorrectionPolynomial Q_;       // Корректирующий полином Q(x)
    
    // Параметры интервала для нормализации
    double interval_a_;             // Начало интервала
    double interval_b_;            // Конец интервала
    double interval_center_;       // Центр интервала
    double interval_scale_;        // Масштаб интервала
    
    // Кэширование для повышения производительности
    mutable LRUCache<double, std::array<double, 3>> value_cache_;  // {F, F', F''} для точки x
    mutable LRUCache<double, double> W_cache_;                     // W(x)
    mutable LRUCache<double, double> Q_cache_;                     // Q(x)
    mutable LRUCache<double, double> basis_cache_;                 // φ_k(x) для последней точки
    mutable std::size_t last_evaluated_x_index_;                  // Индекс последней оценённой точки
    
    // Счётчики для отладки
    mutable std::size_t evaluation_count_;
    mutable std::size_t cache_hits_;
    
    // Флаг валидности
    mutable bool is_valid_;
    mutable std::string validation_message_;
    
    // Вспомогательные методы
    /**
     * @brief Вычисление W(x) с кэшированием
     */
    double compute_W(double x) const;
    
    /**
     * @brief Вычисление Q(x) с кэшированием
     */
    double compute_Q(double x) const;
    
    /**
     * @brief Вычисление W'(x)
     */
    double compute_W_derivative(double x, int order) const;
    
    /**
     * @brief Вычисление Q'(x) или Q''(x)
     */
    double compute_Q_derivative(double x, int order) const;
    
    /**
     * @brief Проверка близости к узлу интерполяции
     */
    bool is_near_node(double x, double& node_value) const;
    
public:
    /**
     * @brief Конструктор по умолчанию
     */
    ConstrainedPolynomial();
    
    /**
     * @brief Основной конструктор
     * @param nodes вектор узлов интерполяции {z_e}
     * @param values вектор значений функции в узлах {f(z_e)}
     * @param deg_Q степень корректирующего полинома Q(x)
     * @param interval_start начало интервала [a, b]
     * @param interval_end конец интервала
     */
    ConstrainedPolynomial(const std::vector<InterpolationNode>& nodes,
                         int deg_Q,
                         double interval_start = 0.0,
                         double interval_end = 1.0);
    
    /**
     * @brief Конструктор с указанием типа базиса
     */
    ConstrainedPolynomial(const std::vector<InterpolationNode>& nodes,
                         int deg_Q,
                         BasisType basis_type,
                         double interval_start = 0.0,
                         double interval_end = 1.0);
    
    /**
     * @brief Деструктор
     */
    ~ConstrainedPolynomial() override = default;
    
    // ============== Методы оценки функции ==============
    
    /**
     * @brief Вычисление значения F(x) в точке
     * @param x точка вычисления
     * @return значение F(x)
     */
    double evaluate(double x) const noexcept override;
    
    /**
     * @brief Безопасная версия evaluate с проверкой на NaN/Inf
     */
    bool evaluate_safe(double x, double& result) const noexcept override;
    
    /**
     * @brief Пакетная оценка значения и производных
     */
    void derivatives(double x, double& f, double& f1, double& f2) const noexcept override;
    
    /**
     * @brief Пакетная оценка в структурированном виде
     */
    EvaluationResult evaluate_with_derivatives(double x) const noexcept override;
    
    // ============== Методы вычисления производных ==============
    
    /**
     * @brief Первая производная в точке
     */
    double first_derivative(double x) const noexcept override;
    
    /**
     * @brief Вторая производная в точке
     */
    double second_derivative(double x) const noexcept override;
    
    // ============== Методы для оптимизации ==============
    
    /**
     * @brief Степень полного полинома F(x)
     * @return степень = max(deg(P_int), deg(Q) + deg(W))
     */
    std::size_t degree() const noexcept override;
    
    /**
     * @brief Число свободных параметров (коэффициентов Q)
     * @return deg(Q) + 1
     */
    std::size_t num_parameters() const noexcept override;
    
    /**
     * @brief Получение параметра по индексу
     */
    double parameter(std::size_t index) const override;
    
    /**
     * @brief Установка параметра по индексу
     */
    void set_parameter(std::size_t index, double value) override;
    
    /**
     * @brief Получение всех параметров
     */
    std::vector<double> parameters() const override;
    
    /**
     * @brief Установка всех параметров
     */
    void set_parameters(const std::vector<double>& params) override;
    
    // ============== Методы базисных функций ==============
    
    /**
     * @brief Оценка k-й базисной функции φₖ(x)
     * Для ConstrainedPolynomial: ∂F/∂q_k = φ_k(x) · W(x)
     */
    double basis_function(std::size_t k, double x) const override;
    
    /**
     * @brief Производная k-й базисной функции
     */
    double basis_derivative(std::size_t k, double x, int order) const override;
    
    /**
     * @brief Градиент по параметрам в точке x
     */
    std::vector<double> gradient(double x) const override;
    
    // ============== Вспомогательные методы ==============
    
    /**
     * @brief Текстовое представление для отладки
     */
    std::string to_string() const override;
    
    /**
     * @brief Внутренняя проверка корректности состояния
     */
    bool validate() const override;
    
    /**
     * @brief Сброс кэшей при изменении параметров
     */
    void reset_cache() override;
    
    /**
     * @brief Тип базиса
     */
    BasisType basis_type() const noexcept override;
    
    /**
     * @brief Границы интервала определения
     */
    std::array<double, 2> interval() const noexcept override;
    
    // ============== Специфичные методы ConstrainedPolynomial ==============
    
    /**
     * @brief Число узлов интерполяции
     */
    std::size_t interpolation_count() const { return p_int_.m_eff; }
    
    /**
     * @brief Проверка интерполяционных условий F(z_e) = f(z_e)
     */
    bool check_interpolation_conditions(double tolerance = 1e-10) const;
    
    /**
     * @brief Получение ссылки на интерполяционный базис
     */
    const InterpolationBasis& interpolation_basis() const { return p_int_; }
    
    /**
     * @brief Получение ссылки на весовой множитель
     */
    const WeightMultiplier& weight_multiplier() const { return W_; }
    
    /**
     * @brief Получение ссылки на корректирующий полином
     */
    const CorrectionPolynomial& correction_polynomial() const { return Q_; }
    
    /**
     * @brief Сброс счётчика оценок
     */
    void reset_evaluation_count() const {
        evaluation_count_ = 0;
        cache_hits_ = 0;
    }
    
    /**
     * @brief Получение количества оценок
     */
    std::size_t evaluation_count() const { return evaluation_count_; }
    
    /**
     * @brief Построение кэшей для набора точек
     */
    void build_caches(const std::vector<double>& points);
    
    /**
     * @brief Установка интервала
     */
    void set_interval(double a, double b) {
        interval_a_ = a;
        interval_b_ = b;
        interval_center_ = (a + b) / 2.0;
        interval_scale_ = (b - a) / 2.0;
        reset_cache();
    }
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_CONSTRAINED_POLYNOMIAL_H
