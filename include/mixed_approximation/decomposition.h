#ifndef MIXED_APPROXIMATION_DECOMPOSITION_H
#define MIXED_APPROXIMATION_DECOMPOSITION_H

#include "types.h"
#include "polynomial.h"
#include "weight_multiplier.h"
#include "interpolation_basis.h"
#include "correction_polynomial.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Метаданные разложения "базис + коррекция"
 */
struct DecompositionMetadata {
    int n_total;               // исходная степень полинома F(x)
    int m_constraints;         // число интерполяционных узлов (исходное)
    int m_eff;                 // эффективное число узлов после объединения
    int n_free;                // число свободных параметров (n - m_eff + 1)
    bool is_valid;             // флаг корректности разложения
    std::string validation_message;  // диагностическое сообщение при ошибках
    double min_root_distance;  // минимальное расстояние между корнями W(x)
    bool requires_normalization; // флаг: требуется ли нормализация координат
    bool nodes_merged;          // были ли узлы объединены
    std::string interpolation_info; // информация о методе интерполяции
    
    DecompositionMetadata()
        : n_total(0)
        , m_constraints(0)
        , m_eff(0)
        , n_free(0)
        , is_valid(false)
        , validation_message("Not initialized")
        , min_root_distance(0.0)
        , requires_normalization(false)
        , nodes_merged(false)
        , interpolation_info("") {}
};

/**
 * @brief Результат разложения полинома F(x) = P_int(x) + Q(x)·W(x)
 */
struct DecompositionResult {
    DecompositionMetadata metadata;      // метаданные разложения
    WeightMultiplier weight_multiplier;  // весовой множитель W(x)
    InterpolationBasis interpolation_basis;  // базисный полином P_int(x)
    Polynomial p_int_polynomial;        // P_int(x) в виде полинома (опционально, для аналитики)
    
    // Кэшированные значения для ускорения (опционально)
    std::vector<double> cache_W_x;      // W(x_i) для всех аппроксимирующих точек
    std::vector<double> cache_W_y;      // W(y_j) для всех отталкивающих точек
    std::vector<double> cache_W1_x;     // W'(x_i)
    std::vector<double> cache_W1_y;     // W'(y_j)
    std::vector<double> cache_W2_x;     // W''(x_i)
    std::vector<double> cache_W2_y;     // W''(y_j)
    bool caches_built;                  // флаг: построены ли кэши
    
    // Корректирующий полином Q(x)
    CorrectionPolynomial correction_poly;  // параметризация корректирующей компоненты
    
    DecompositionResult()
        : metadata()
        , weight_multiplier()
        , interpolation_basis()
        , p_int_polynomial(Polynomial(0))
        , caches_built(false)
        , correction_poly() {}
    
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
     * @param x точка вычисления (в исходных координатах)
     * @param q_coeffs коэффициенты Q(x) в порядке убывания степеней [q_{deg_Q}, ..., q_0]
     * @return значение F(x)
     */
    double evaluate(double x, const std::vector<double>& q_coeffs) const;
    
    /**
     * @brief Построение кэшей значений W, W', W'' в точках данных
     * @param points_x аппроксимирующие точки {x_i}
     * @param points_y отталкивающие точки {y_j}
     */
    void build_caches(const std::vector<double>& points_x,
                       const std::vector<double>& points_y);
    
    /**
     * @brief Очистка кэшей
     */
    void clear_caches();
    
    /**
     * @brief Верификация разложения: проверка интерполяционных условий
     * @param tolerance допуск
     * @return true, если все условия F(z_e) = f(z_e) выполняются
     */
    bool verify_interpolation(double tolerance = 1e-10) const;
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
    
    /**
     * @brief Проверка математической разрешимости задачи (ранг системы ограничений)
     */
    static bool check_rank_solvency(const std::vector<double>& nodes,
                                    double tolerance,
                                    std::vector<int>* conflict_indices = nullptr);
    
    /**
     * @brief Проверка условия n ≥ m - 1
     */
    static bool check_degree_condition(int n, int m);
    
    /**
     * @brief Проверка уникальности узлов
     */
    static bool check_unique_nodes(const std::vector<double>& nodes,
                                  double interval_length,
                                  double tolerance,
                                  std::vector<std::pair<int, int>>* duplicate_pairs = nullptr);
    
    /**
     * @brief Проверка расположения узлов внутри интервала
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
     */
    static double estimate_weight_multiplier_scale(const std::vector<double>& roots);
    
    /**
     * @brief Анализ диапазона значений интерполяции
     */
    static double analyze_value_range(const std::vector<double>& values);
    
    /**
     * @brief Проверка на полиномиальную зависимость (для m ≤ 4)
     */
    static int detect_low_degree_polynomial(const std::vector<double>& nodes,
                                             const std::vector<double>& values);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_DECOMPOSITION_H
