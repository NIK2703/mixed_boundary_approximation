#ifndef MIXED_APPROXIMATION_COMPOSITE_POLYNOMIAL_H
#define MIXED_APPROXIMATION_COMPOSITE_POLYNOMIAL_H

#include "types.h"
#include "interpolation_basis.h"
#include "weight_multiplier.h"
#include "correction_polynomial.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Стратегия вычисления значений композитного полинома
 */
enum class EvaluationStrategy {
    LAZY,       ///< Ленивая оценка: F(x) = P_int(x) + Q(x) * W(x) "на лету"
    ANALYTIC,   ///< Аналитическая сборка: вычисление явных коэффициентов F(x)
    HYBRID      ///< Гибридный: ленивая оценка для вычислений, аналитика для экспорта
};

/**
 * @brief Композитный полином F(x) = P_int(x) + Q(x)·W(x)
 *
 * Объединяет три компоненты параметризации:
 * - P_int(x): базисный интерполяционный полином
 * - W(x): весовой множитель с корнями в узлах интерполяции
 * - Q(x): корректирующий полином со свободными коэффициентами
 */
struct CompositePolynomial {
    // Компоненты разложения
    InterpolationBasis interpolation_basis;  ///< P_int(x)
    WeightMultiplier weight_multiplier;     ///< W(x)
    CorrectionPolynomial correction_poly;     ///< Q(x)
    
    // Стратегия вычислений
    EvaluationStrategy eval_strategy;        ///< Стратегия оценки значений
    bool analytic_coeffs_valid;             ///< Флаг: доступны ли аналитические коэффициенты
    
    // Аналитические коэффициенты F(x) (опционально)
    std::vector<double> analytic_coeffs;    ///< Коэффициенты F(x) в порядке убывания степеней [a_n, ..., a_0]
    
    // Кэшированные данные для ускорения вычислений
    struct EvaluationCache {
        std::vector<double> P_at_x;         ///< P_int(x_i) для аппроксимирующих точек
        std::vector<double> W_at_x;         ///< W(x_i) для аппроксимирующих точек
        std::vector<double> P_at_y;         ///< P_int(y_j) для отталкивающих точек
        std::vector<double> W_at_y;         ///< W(y_j) для отталкивающих точек
        
        std::vector<double> quad_points;     ///< Узлы квадратуры для регуляризации
        std::vector<double> W_at_quad;       ///< W(x) в узлах квадратуры
        std::vector<double> W1_at_quad;      ///< W'(x) в узлах квадратуры
        std::vector<double> W2_at_quad;     ///< W''(x) в узлах квадратуры
        
        std::vector<double> Q_at_quad;       ///< Q(x) в узлах квадратуры
        std::vector<double> Q1_at_quad;      ///< Q'(x) в узлах квадратуры
        std::vector<double> Q2_at_quad;      ///< Q''(x) в узлах квадратуры
        
        std::vector<double> P2_at_quad;      ///< P_int''(x) в узлах квадратуры
    } cache;
    
    // Флаг готовности кэшей
    bool caches_built;
    
    // Метаданные
    int total_degree;                       ///< n — степень итогового полинома F(x)
    int num_constraints;                    ///< m — число интерполяционных узлов
    int num_free_params;                     ///< n_free = n - m + 1
    double interval_a;                       ///< Левая граница интервала [a, b]
    double interval_b;                      ///< Правая граница интервала [a, b]
    
    // Сообщение о валидации
    std::string validation_message;
    
    CompositePolynomial()
        : eval_strategy(EvaluationStrategy::LAZY)
        , analytic_coeffs_valid(false)
        , caches_built(false)
        , total_degree(0)
        , num_constraints(0)
        , num_free_params(0)
        , interval_a(0.0)
        , interval_b(1.0)
        , validation_message("Not initialized") {}
    
    /**
     * @brief Построение композитного полинома из компонент
     * @param basis Базисный интерполяционный полином P_int(x)
     * @param W Весовой множитель W(x)
     * @param Q Корректирующий полином Q(x)
     * @param interval_start Левая граница интервала [a, b]
     * @param interval_end Правая граница интервала [a, b]
     * @param strategy Стратегия вычислений
     */
    void build(const InterpolationBasis& basis,
               const WeightMultiplier& W,
               const CorrectionPolynomial& Q,
               double interval_start = 0.0,
               double interval_end = 1.0,
               EvaluationStrategy strategy = EvaluationStrategy::LAZY);
    
    /**
     * @brief Вычисление значения F(x) в точке (ленивая оценка)
     * @param x Точка вычисления (в исходных координатах)
     * @return Значение F(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление производной F'(x) или F''(x) в точке
     * @param x Точка вычисления
     * @param order Порядок производной (1 или 2)
     * @return Значение производной
     */
    double evaluate_derivative(double x, int order = 1) const;
    
    /**
     * @brief Вычисление аналитических коэффициентов F(x)
     * @param max_degree_for_analytic Максимальная степень для аналитической сборки (по умолчанию 15)
     * @return true, если сборка выполнена успешно
     */
    bool build_analytic_coefficients(int max_degree_for_analytic = 15);
    
    /**
     * @brief Вычисление F(x) через аналитические коэффициенты (если доступны)
     * @param x Точка вычисления
     * @return Значение F(x) через схему Горнера
     */
    double evaluate_analytic(double x) const;
    
    /**
     * @brief Вычисление значения F(x) в нескольких точках (batch evaluation)
     * @param points Вектор точек
     * @param results Вектор результатов (будет заполнен)
     */
    void evaluate_batch(const std::vector<double>& points, std::vector<double>& results) const;
    
    /**
     * @brief Вычисление значения F(x) в нескольких точках через аналитические коэффициенты
     */
    void evaluate_batch_analytic(const std::vector<double>& points, std::vector<double>& results) const;
    
    /**
     * @brief Построение кэшей для ускорения вычислений
     * @param points_x Аппроксимирующие точки
     * @param points_y Отталкивающие точки
     * @param quad_points Узлы квадратуры (если пусто, будут сгенерированы автоматически)
     */
    void build_caches(const std::vector<double>& points_x,
                      const std::vector<double>& points_y,
                      const std::vector<double>& quad_points = {});
    
    /**
     * @brief Очистка кэшей
     */
    void clear_caches();
    
    /**
     * @brief Вычисление регуляризационного члена ∫(F''(x))²dx
     * @param gamma Коэффициент регуляризации
     * @return Значение интеграла
     */
    double compute_regularization_term(double gamma) const;
    
    /**
     * @brief Верификация корректности сборки
     * @param tolerance Допуск для проверки интерполяционных условий
     * @return true, если все проверки пройдены
     */
    bool verify_assembly(double tolerance = 1e-10);
    
    /**
     * @brief Проверка согласованности ленивой и аналитической оценок
     * @param num_test_points Число тестовых точек
     * @param relative_tolerance Относительный допуск
     * @return true, если оценки согласованы
     */
    bool verify_representations_consistency(int num_test_points = 10,
                                            double relative_tolerance = 1e-8) const;
    
    /**
     * @brief Получение диагностической информации
     * @return Строка с информацией о структуре
     */
    std::string get_diagnostic_info() const;
    
    /**
     * @brief Проверка валидности структуры
     * @return true, если структура корректна
     */
    bool is_valid() const;
    
    /**
     * @brief Получение степени полинома
     * @return Степень F(x)
     */
    int degree() const { return total_degree; }
    
    /**
     * @brief Получение числа свободных параметров
     * @return n_free = n - m + 1
     */
    int num_free_parameters() const { return num_free_params; }
    
private:
    // Вспомогательные методы для аналитической сборки
    
    /**
     * @brief Вычисление коэффициентов P_int(x) из барицентрического представления
     * @return Коэффициенты в порядке убывания степеней
     */
    std::vector<double> extract_p_int_coefficients() const;
    
    /**
     * @brief Свёртка коэффициентов Q(x) и W(x)
     * @param q_coeffs Коэффициенты Q(x) в порядке убывания степеней
     * @param w_coeffs Коэффициенты W(x) в порядке убывания степеней
     * @return Коэффициенты произведения
     */
    std::vector<double> convolve_coefficients(const std::vector<double>& q_coeffs,
                                              const std::vector<double>& w_coeffs) const;
    
    /**
     * @brief Генерация узлов квадратуры Гаусса-Лежандра
     * @param n Число узлов
     * @param points Вектор для узлов
     * @param weights Вектор для весов
     */
    void generate_gauss_legendre_quadrature(int n,
                                             std::vector<double>& points,
                                             std::vector<double>& weights) const;
    
    /**
     * @brief Преобразование узлов квадратуры из [-1, 1] в [a, b]
     */
    double transform_quadrature_node(double t) const;
};

// Вспомогательные функции для квадратуры

/**
 * @brief Вычисление интеграла от (F''(x))² на [a, b] через компоненты
 * @param gamma Коэффициент регуляризации
 * @return Значение интеграла
 */
double compute_regularization_via_components(const CompositePolynomial& F, double gamma);

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_COMPOSITE_POLYNOMIAL_H
