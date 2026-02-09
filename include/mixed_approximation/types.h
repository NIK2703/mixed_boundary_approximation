#ifndef MIXED_APPROXIMATION_TYPES_H
#define MIXED_APPROXIMATION_TYPES_H

#include <vector>
#include <memory>
#include <string>
#include <utility>

namespace mixed_approx {

// Forward declaration
class Polynomial;

/**
 * @brief Структура для представления точки с весом (аппроксимирующие точки)
 */
struct WeightedPoint {
    double x;           // координата точки
    double value;       // значение функции в точке f(x)
    double weight;      // вес σ_i
    
    WeightedPoint(double x, double value, double weight)
        : x(x), value(value), weight(weight) {}
};

/**
 * @brief Структура для интерполяционного узла
 */
struct InterpolationNode {
    double x;           // координата узла
    double value;       // значение функции в узле f(z)
    
    InterpolationNode(double x, double value)
        : x(x), value(value) {}
};

/**
 * @brief Структура для точки отталкивания
 * 
 * Отталкивающая точка характеризуется:
 * - координатой x (где применяется отталкивание)
 * - запрещённым значением y_forbidden (какого значения F(x) следует избегать)
 * - весом B_j (интенсивность барьера)
 */
struct RepulsionPoint {
    double x;           // абсцисса точки (координата на оси X)
    double y_forbidden; // запрещённое значение функции (ордината на оси Y)
    double weight;      // вес отталкивания B_j
    
    RepulsionPoint(double x, double y_forbidden, double weight)
        : x(x), y_forbidden(y_forbidden), weight(weight) {}
    
    // Упрощённый конструктор: отталкивание от нуля по умолчанию
    explicit RepulsionPoint(double x, double weight)
        : x(x), y_forbidden(0.0), weight(weight) {}
    
    // Конструктор-адаптер из WeightedPoint (для обратной совместимости)
    // ВАЖНО: value интерпретируется как y_forbidden, а не как f(y_j)
    RepulsionPoint(const WeightedPoint& wp)
        : x(wp.x), y_forbidden(wp.value), weight(wp.weight) {}
    
    // Преобразование в WeightedPoint (для обратной совместимости)
    operator WeightedPoint() const {
        return WeightedPoint(x, y_forbidden, weight);
    }
};

/**
 * @brief Конфигурация метода смешанной аппроксимации
 */
struct ApproximationConfig {
    // Степень полинома
    int polynomial_degree;
    
    // Интервал определения [a, b]
    double interval_start;
    double interval_end;
    
    // Параметры
    double gamma;                    // коэффициент регуляризации (γ ≥ 0)
    std::vector<WeightedPoint> approx_points;   // аппроксимирующие точки {x_i}
    std::vector<RepulsionPoint> repel_points;   // отталкивающие точки {y_j} с запрещёнными значениями y_j^*
    std::vector<InterpolationNode> interp_nodes; // интерполяционные узлы {z_e}
    
    // Параметры численной устойчивости
    double epsilon;                  // минимальный порог для знаменателя (по умолчанию 1e-8)
    double interpolation_tolerance;  // допуск для интерполяционных условий (по умолчанию 1e-10)
    
    ApproximationConfig()
        : polynomial_degree(0)
        , interval_start(0.0)
        , interval_end(1.0)
        , gamma(0.0)
        , epsilon(1e-8)
        , interpolation_tolerance(1e-10) {}
};

/**
 * @brief Результат валидации решения
 */
struct ValidationResult {
    bool is_valid;                      // общий результат валидации
    bool numerical_correct;             // численная корректность (нет NaN/Inf)
    bool interpolation_ok;              // интерполяционные условия выполнены
    bool barriers_safe;                 // безопасность барьеров
    bool physically_plausible;          // физическая правдоподобность
    bool correction_applied;            // была ли применена коррекция
    std::string message;                // диагностическое сообщение
    std::vector<std::string> warnings;  // предупреждения (если есть)
    
    // Поля для параметризации (опционально, int вместо enum для избежания циклических зависимостей)
    int status;                         // 0=UNVALIDATED, 1=PASSED, 2=WARNING, 3=FAILED
    double max_interpolation_error;
    double condition_number;
    bool numerically_stable;
    
    ValidationResult()
        : is_valid(false)
        , numerical_correct(false)
        , interpolation_ok(false)
        , barriers_safe(false)
        , physically_plausible(false)
        , correction_applied(false)
        , status(0)
        , max_interpolation_error(0.0)
        , condition_number(0.0)
        , numerically_stable(false) {}
};

/**
 * @brief Стратегия инициализации
 */
enum class InitializationStrategy {
    ZERO = 0,              // нулевой полином Q(x) = 0
    INTERPOLATION,         // Q(x) построенный через интерполяцию невязки
    LEAST_SQUARES,         // Q(x) через метод наименьших квадратов
    HIERARCHICAL,          // иерархический подход с несколькими стратегиями
    MULTI_START,           // множественные запуски с разными стратегиями
    RANDOM,                // случайная инициализация
    BARRIER_PERTURBATION   // возмущение для избежания барьеров (шаг 1.2.6)
};

/**
 * @brief Метрики инициализации
 */
struct InitializationMetrics {
    double initial_objective;
    double objective_ratio;
    double min_barrier_distance;
    double rms_residual_norm;
    double condition_number;
    bool interpolation_ok;
    bool barriers_safe;
    
    InitializationMetrics()
        : initial_objective(0.0)
        , objective_ratio(0.0)
        , min_barrier_distance(0.0)
        , rms_residual_norm(0.0)
        , condition_number(0.0)
        , interpolation_ok(false)
        , barriers_safe(false) {}
};

/**
 * @brief Результат инициализации (построения начального приближения)
 */
struct InitializationResult {
    bool success;                       // флаг успешности
    std::string message;                // сообщение о результате
    std::vector<double> initial_coeffs; // начальные коэффициенты Q(x)
    InitializationStrategy strategy_used;  // использованная стратегия инициализации
    double elapsed_time;                // время построения (мс)
    double initial_objective;            // начальное значение функционала
    InitializationMetrics metrics;      // метрики инициализации
    std::vector<std::string> recommendations;  // рекомендации
    
    InitializationResult()
        : success(false)
        , strategy_used(InitializationStrategy::ZERO)
        , elapsed_time(0.0)
        , initial_objective(0.0) {}
};

/**
 * @brief Результат оптимизации
 */
struct OptimizationResult {
    std::vector<double> coefficients;      // коэффициенты Q(x) [q_{n_free-1}, ..., q_0]
    double final_objective;                // конечное значение функционала
    int iterations;                        // количество итераций
    bool success;                          // флаг успешности оптимизации
    bool converged;                        // флаг достижения сходимости
    std::string message;                   // сообщение о результате
    double elapsed_time;                   // время оптимизации (мс)
    std::shared_ptr<Polynomial> final_polynomial;  // построенный полином F(x)
    ValidationResult validation;           // результат валидации решения
    std::string diagnostic_report;         // детальный диагностический отчёт
    
    OptimizationResult()
        : final_objective(0.0)
        , iterations(0)
        , success(false)
        , converged(false)
        , elapsed_time(0.0)
        , final_polynomial(nullptr) {}
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_TYPES_H
