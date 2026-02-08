#ifndef MIXED_APPROXIMATION_DECOMPOSITION_H
#define MIXED_APPROXIMATION_DECOMPOSITION_H

#include "types.h"
#include "polynomial.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Перечисление методов интерполяции для построения P_int(x)
 */
enum class InterpolationMethod {
    LAGRANGE,       ///< Интерполяция по формуле Лагранжа
    BARYCENTRIC,    ///< Барицентрическая интерполяция
    NEWTON          ///< Интерполяция по формуле Ньютона
};

/**
 * @brief Метаданные разложения "базис + коррекция"
 */
struct DecompositionMetadata {
    int n_total;               // исходная степень полинома F(x)
    int m_constraints;         // число интерполяционных узлов
    int n_free;                // число свободных параметров (n - m + 1)
    bool is_valid;             // флаг корректности разложения
    std::string validation_message;  // диагностическое сообщение при ошибках
    double min_root_distance;  // минимальное расстояние между корнями W(x)
    bool requires_normalization; // флаг: требуется ли нормализация координат
    
    DecompositionMetadata()
        : n_total(0)
        , m_constraints(0)
        , n_free(0)
        , is_valid(false)
        , validation_message("Not initialized")
        , min_root_distance(0.0)
        , requires_normalization(false) {}
};

/**
 * @brief Весовой множитель W(x) = ∏_{e=1..m} (x - z_e)
 */
struct WeightMultiplier {
    std::vector<double> roots;          // отсортированные узлы z_e
    std::vector<double> coeffs;         // коэффициенты полинома W(x) (опционально)
    double min_root_distance;           // минимальное расстояние между корнями
    bool use_direct_evaluation;         // использовать прямое вычисление через корни
    
    WeightMultiplier()
        : min_root_distance(0.0)
        , use_direct_evaluation(true) {}
    
    /**
     * @brief Вычисление значения W(x) в точке
     * @param x точка вычисления
     * @return значение W(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Построение полинома W(x) по заданным корням
     * @param roots узлы интерполяции (не обязательно отсортированные)
     */
    void build_from_roots(const std::vector<double>& roots);
    
    /**
     * @brief Получение степени полинома
     * @return степень (равна числу корней)
     */
    int degree() const { return static_cast<int>(roots.size()); }
};

/**
 * @brief Базисный интерполяционный полином P_int(x)
 */
struct InterpolationBasis {
    std::vector<double> nodes;           // узлы интерполяции (копия из WeightMultiplier)
    std::vector<double> values;          // значения f(z_e) в узлах
    InterpolationMethod method;          // метод интерполяции
    std::vector<double> barycentric_weights;  // веса для барицентрического метода
    std::vector<double> divided_differences; // разделённые разности для метода Ньютона
    
    InterpolationBasis()
        : method(InterpolationMethod::BARYCENTRIC) {}
    
    /**
     * @brief Построение интерполяционного полинома
     * @param nodes узлы интерполяции
     * @param values значения функции в узлах
     * @param method метод интерполяции
     */
    void build(const std::vector<double>& nodes,
               const std::vector<double>& values,
               InterpolationMethod method = InterpolationMethod::BARYCENTRIC);
    
    /**
     * @brief Вычисление значения P_int(x) в точке
     * @param x точка вычисления
     * @return значение P_int(x)
     */
    double evaluate(double x) const;
    
private:
    // Вспомогательные методы для различных методов интерполяции
    double evaluate_barycentric(double x) const;
    double evaluate_newton(double x) const;
    double evaluate_lagrange(double x) const;
    
    void compute_barycentric_weights();
    void compute_divided_differences();
    
public:
    /**
     * @brief Проверка, что узлы уникальны (с заданной точностью)
     * @param tolerance допуск
     * @return true, если все узлы различны
     */
    static bool are_nodes_unique(const std::vector<double>& nodes, double tolerance);
    
    /**
     * @brief Сортировка узлов и соответствующих значений
     * @param nodes узлы (будут отсортированы)
     * @param values значения (будут переставлены соответственно)
     */
    static void sort_nodes_and_values(std::vector<double>& nodes, std::vector<double>& values);
};

/**
 * @brief Результат разложения полинома F(x) = P_int(x) + Q(x)·W(x)
 */
struct DecompositionResult {
    DecompositionMetadata metadata;      // метаданные разложения
    WeightMultiplier weight_multiplier;  // весовой множитель W(x)
    InterpolationBasis interpolation_basis;  // базисный полином P_int(x)
    Polynomial p_int_polynomial;        // P_int(x) в виде полинома (опционально, для аналитики)
    
    DecompositionResult()
        : metadata()
        , weight_multiplier()
        , interpolation_basis()
        , p_int_polynomial(Polynomial(0)) {}  // инициализируем нулевым полиномом
    
    /**
     * @brief Проверка корректности разложения
     * @return true, если разложение корректно
     */
    bool is_valid() const { return metadata.is_valid; }
    
    /**
     * @brief Получение диагностического сообщения
     * @return строка с сообщением
     */
    std::string message() const { return metadata.validation_message; }
    
    /**
     * @brief Построение полного полинома F(x) по коэффициентам Q(x)
     * @param q_coeffs коэффициенты корректирующего полинома Q(x) (размер n_free)
     * @return полином F(x) = P_int(x) + Q(x)·W(x)
     */
    Polynomial build_polynomial(const std::vector<double>& q_coeffs) const;
    
    /**
     * @brief Вычисление значения F(x) в точке без построения полного полинома
     * @param x точка вычисления
     * @param q_coeffs коэффициенты Q(x)
     * @return значение F(x)
     */
    double evaluate(double x, const std::vector<double>& q_coeffs) const;
};

/**
 * @brief Класс для разложения полинома по схеме "базис + коррекция"
 */
class Decomposer {
public:
    /**
     * @brief Параметры разложения
     */
    struct Parameters {
        int polynomial_degree;           // степень n
        double interval_start;           // a
        double interval_end;             // b
        std::vector<InterpolationNode> interp_nodes;  // интерполяционные узлы
        
        // Параметры численной устойчивости
        double epsilon_rank;             // порог для проверки ранга (1e-12)
        double epsilon_unique;           // порог для уникальности узлов (относительно длины интервала)
        double epsilon_bound;            // порог для проверки границ интервала (1e-9)
        
        Parameters()
            : polynomial_degree(0)
            , interval_start(0.0)
            , interval_end(1.0)
            , epsilon_rank(1e-12)
            , epsilon_unique(1e-12)
            , epsilon_bound(1e-9) {}
    };
    
    /**
     * @brief Выполнение разложения
     * @param params параметры разложения
     * @return результат разложения
     */
    static DecompositionResult decompose(const Parameters& params);
    
private:
    /**
     * @brief Проверка математической разрешимости задачи (ранг системы ограничений)
     * @param nodes узлы интерполяции
     * @param tolerance порог для проверки линейной независимости
     * @return true, если система разрешима
     * @param conflict_indices индексы линейно зависимых узлов (выходной параметр)
     */
    static bool check_rank_solvency(const std::vector<double>& nodes,
                                    double tolerance,
                                    std::vector<int>* conflict_indices = nullptr);
    
    /**
     * @brief Проверка условия n ≥ m - 1
     * @param n степень полинома
     * @param m число узлов
     * @return true, если условие выполняется
     */
    static bool check_degree_condition(int n, int m);
    
    /**
     * @brief Проверка уникальности узлов
     * @param nodes узлы
     * @param interval_length длина интервала [a, b]
     * @param tolerance относительный допуск
     * @param duplicate_pairs выход: пары индексов дублирующихся узлов
     * @return true, если все узлы уникальны
     */
    static bool check_unique_nodes(const std::vector<double>& nodes,
                                    double interval_length,
                                    double tolerance,
                                    std::vector<std::pair<int, int>>* duplicate_pairs = nullptr);
    
    /**
     * @brief Проверка расположения узлов внутри интервала
     * @param nodes узлы
     * @param a начало интервала
     * @param b конец интервала
     * @param tolerance допуск
     * @param out_of_bounds выход: индексы узлов вне интервала
     * @return true, если все узлы в интервале (с допуском)
     */
    static bool check_nodes_in_interval(const std::vector<double>& nodes,
                                         double a, double b,
                                         double tolerance,
                                         std::vector<int>* out_of_bounds = nullptr);
    
    /**
     * @brief Предварительные вычисления: сортировка узлов и значений
     */
    static void prepare_sorted_nodes_and_values(const std::vector<InterpolationNode>& input_nodes,
                                                 std::vector<double>* sorted_nodes,
                                                 std::vector<double>* sorted_values);
    
    /**
     * @brief Оценка масштаба весового множителя
     * @param roots корни W(x)
     * @return характерный масштаб
     */
    static double estimate_weight_multiplier_scale(const std::vector<double>& roots);
    
    /**
     * @brief Анализ диапазона значений интерполяции
     * @param values значения f(z_e)
     * @return разброс значений (max - min)
     */
    static double analyze_value_range(const std::vector<double>& values);
    
    /**
     * @brief Проверка на полиномиальную зависимость (для m ≤ 4)
     * @param nodes узлы
     * @param values значения
     * @return степень обнаруженного полинома (0 если не обнаружено)
     */
    static int detect_low_degree_polynomial(const std::vector<double>& nodes,
                                             const std::vector<double>& values);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_DECOMPOSITION_H
