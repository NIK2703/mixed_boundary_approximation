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
    int m_constraints;         // число интерполяционных узлов (исходное)
    int m_eff;                 // эффективное число узлов после объединения
    int n_free;                // число свободных параметров (n - m_eff + 1)
    bool is_valid;             // флаг корректности разложения
    std::string validation_message;  // диагностическое сообщение при ошибках
    double min_root_distance;  // минимальное расстояние между корнями W(x)
    bool requires_normalization; // флаг: требуется ли нормализация координат
    bool nodes_merged;         // были ли узлы объединены
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
 * @brief Весовой множитель W(x) = ∏_{e=1..m} (x - z_e)
 */
struct WeightMultiplier {
    std::vector<double> roots;          // отсортированные узлы z_e (в исходных координатах)
    std::vector<double> roots_norm;     // узлы в нормализованных координатах [-1, 1] (если нормализация включена)
    std::vector<double> coeffs;         // коэффициенты полинома W(x) в порядке убывания степеней [w_m=1, w_{m-1}, ..., w_0]
    double min_root_distance;           // минимальное расстояние между корнями
    bool use_direct_evaluation;         // использовать прямое вычисление через корни
    
    // Параметры нормализации координат
    double shift;                       // сдвиг для нормализации: x_norm = (x - shift) / scale
    double scale;                       // масштаб для нормализации
    bool is_normalized;                 // флаг: были ли корни нормализованы
    
    // Кэшированные значения (опционально, для ускорения)
    std::vector<double> cache_x_vals;   // закэшированные значения W(x) в точках {x_i}
    std::vector<double> cache_y_vals;   // закэшированные значения W(y) в точках {y_j}
    std::vector<double> cache_x_deriv1; // закэшированные значения W'(x) в точках {x_i}
    std::vector<double> cache_y_deriv1; // закэшированные значения W'(y) в точках {y_j}
    std::vector<double> cache_x_deriv2; // закэшированные значения W''(x) в точках {x_i}
    std::vector<double> cache_y_deriv2; // закэшированные значения W''(y) в точках {y_j}
    bool caches_ready;                  // флаг: готовы ли кэши
    
    WeightMultiplier()
        : min_root_distance(0.0)
        , use_direct_evaluation(true)
        , shift(0.0)
        , scale(1.0)
        , is_normalized(false)
        , caches_ready(false) {}
    
    /**
     * @brief Вычисление значения W(x) в точке
     * @param x точка вычисления (в исходных координатах)
     * @return значение W(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление первой производной W'(x) в точке
     * @param x точка вычисления (в исходных координатах)
     * @return значение W'(x)
     */
    double evaluate_derivative(double x, int order = 1) const;
    
    /**
     * @brief Построение полинома W(x) по заданным корням
     * @param roots узлы интерполяции (в исходных координатах, не обязательно отсортированные)
     * @param interval_start начало интервала [a, b] для оценки необходимости нормализации
     * @param interval_end конец интервала
     * @param enable_normalization включать ли нормализацию координат
     */
    void build_from_roots(const std::vector<double>& roots,
                          double interval_start = 0.0,
                          double interval_end = 1.0,
                          bool enable_normalization = true);
    
    /**
     * @brief Получение степени полинома
     * @return степень (равна числу корней)
     */
    int degree() const { return static_cast<int>(roots.size()); }
    
    /**
     * @brief Верификация корректности построения
     * Проверяет: W(z_e) ≈ 0, моничность, согласованность представлений
     * @param tolerance допуск для нулевых значений в корнях
     * @return true, если все проверки пройдены
     */
    bool verify_construction(double tolerance = 1e-10) const;
    
    /**
     * @brief Построение кэшей значений в заданных точках
     * @param points_x точки для кэширования W(x), W'(x), W''(x)
     * @param points_y точки для кэширования W(y), W'(y), W''(y)
     */
    void build_caches(const std::vector<double>& points_x,
                      const std::vector<double>& points_y);
    
    /**
     * @brief Очистка кэшей
     */
    void clear_caches();
    
    /**
     * @brief Умножение полинома W(x) на полином Q(x)
     * Выполняет свёртку коэффициентов
     * @param q_coeffs коэффициенты Q(x) в порядке убывания степеней [q_{deg_Q}, ..., q_0]
     * @return коэффициенты произведения Q(x)·W(x) в порядке убывания степеней
     */
    std::vector<double> multiply_by_Q(const std::vector<double>& q_coeffs) const;
    
    /**
     * @brief Вычисление значения Q(x)·W(x) без явного построения коэффициентов
     * @param x точка вычисления (в исходных координатах)
     * @param q_coeffs коэффициенты Q(x) в порядке убывания степеней
     * @return значение Q(x)·W(x)
     */
    double evaluate_product(double x, const std::vector<double>& q_coeffs) const;
};

/**
 * @brief Базисный интерполяционный полином P_int(x)
 */
struct InterpolationBasis {
    // Исходные данные (нормализованные)
    std::vector<double> nodes;           // узлы интерполяции (отсортированные, возможно нормализованные)
    std::vector<double> values;          // значения f(z_e) в узлах
    int m_eff;                           // эффективное число узлов после объединения близких
    
    // Параметры нормализации координат
    double x_center;                     // центр интервала [a, b] для нормализации
    double x_scale;                      // масштаб интервала
    bool is_normalized;                  // флаг: были ли узлы нормализованы
    
    // Метод интерполяции
    InterpolationMethod method;          // метод интерполяции
    
    // Барицентрические веса
    std::vector<double> barycentric_weights;  // веса для барицентрического метода
    double weight_scale;                 // масштабирующий коэффициент для весов
    
    // Для метода Ньютона
    std::vector<double> divided_differences; // разделённые разности
    
    // Кэшированные данные для ускорения
    std::vector<double> weighted_values; // w_e * f(z_e) для быстрого вычисления
    
    // Флаги состояния
    bool is_valid;                       // флаг корректности построения
    std::string error_message;           // сообщение об ошибке
    
    InterpolationBasis()
        : method(InterpolationMethod::BARYCENTRIC)
        , m_eff(0)
        , x_center(0.0)
        , x_scale(1.0)
        , is_normalized(false)
        , weight_scale(1.0)
        , is_valid(false)
        , error_message("Not built") {}
    
    /**
     * @brief Построение интерполяционного полинома с расширенными возможностями
     * @param nodes узлы интерполяции (в исходных координатах)
     * @param values значения функции в узлах
     * @param method метод интерполяции
     * @param interval_start начало интервала [a, b] для нормализации
     * @param interval_end конец интервала
     * @param enable_normalization включать ли нормализацию координат
     * @param enable_node_merging включать ли объединение близких узлов
     */
    void build(const std::vector<double>& nodes,
               const std::vector<double>& values,
               InterpolationMethod method = InterpolationMethod::BARYCENTRIC,
               double interval_start = 0.0,
               double interval_end = 1.0,
               bool enable_normalization = true,
               bool enable_node_merging = true);
    
    /**
     * @brief Вычисление значения P_int(x) в точке (в исходных координатах)
     * @param x точка вычисления
     * @return значение P_int(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление первой производной P_int'(x) в точке
     * @param x точка вычисления
     * @return значение производной
     */
    double evaluate_derivative(double x, int order = 1) const;
    
    /**
     * @brief Верификация точности интерполяции
     * Проверяет, что P_int(z_e) = f(z_e) с заданной точностью
     * @param tolerance допуск
     * @return true, если все условия выполняются
     */
    bool verify_interpolation(double tolerance = 1e-10) const;
    
    /**
     * @brief Получение информации о построении
     * @return строка с диагностической информацией
     */
    std::string get_info() const;
    
private:
    // Вспомогательные методы для различных методов интерполяции
    double evaluate_barycentric(double x) const;
    double evaluate_newton(double x) const;
    double evaluate_lagrange(double x) const;
    
    // Нормализация координат
    void normalize_nodes(double a, double b);
    void denormalize_point(double& x_norm, double& x_original) const;
    
    // Обработка близких узлов
    struct MergedNode {
        double x;           // нормализованная координата
        double value;       // усреднённое значение
        int count;          // сколько исходных узлов объединено
    };
    std::vector<MergedNode> merge_close_nodes(const std::vector<double>& nodes_norm,
                                              const std::vector<double>& values,
                                              double interval_length);
    
    // Вычисление барицентрических весов с логарифмической стабилизацией
    void compute_barycentric_weights();
    void compute_barycentric_weights_standard();
    void compute_barycentric_weights_logarithmic();
    
    // Кэширование
    void precompute_weighted_values();
    
    // Вычисление разделённых разностей (для метода Ньютона)
    void compute_divided_differences();
    
    // Вычисление производной для барицентрической формы
    double evaluate_barycentric_derivative(double x, int order) const;
    
    // Специальные случаи
    bool detect_equally_spaced_nodes(double tolerance = 1e-10) const;
    bool detect_chebyshev_nodes(double tolerance = 1e-10) const;
    void compute_chebyshev_weights();
    
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
    
    // Кэшированные значения для ускорения (опционально)
    std::vector<double> cache_W_x;      // W(x_i) для всех аппроксимирующих точек
    std::vector<double> cache_W_y;      // W(y_j) для всех отталкивающих точек
    std::vector<double> cache_W1_x;     // W'(x_i)
    std::vector<double> cache_W1_y;     // W'(y_j)
    std::vector<double> cache_W2_x;     // W''(x_i)
    std::vector<double> cache_W2_y;     // W''(y_j)
    bool caches_built;                  // флаг: построены ли кэши
    
    DecompositionResult()
        : metadata()
        , weight_multiplier()
        , interpolation_basis()
        , p_int_polynomial(Polynomial(0))
        , caches_built(false) {}
    
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
