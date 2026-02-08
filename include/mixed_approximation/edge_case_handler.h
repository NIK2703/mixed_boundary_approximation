#ifndef MIXED_APPROXIMATION_EDGE_CASE_HANDLER_H
#define MIXED_APPROXIMATION_EDGE_CASE_HANDLER_H

#include "types.h"
#include "interpolation_basis.h"
#include "weight_multiplier.h"
#include "correction_polynomial.h"
#include "composite_polynomial.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <limits>
#include <sstream>
#include <algorithm>
#include <map>

namespace mixed_approx {

// ============== Шаг 2.1.8.1: Классификация крайних случаев ==============

/**
 * @brief Уровень критичности крайнего случая
 */
enum class EdgeCaseLevel {
    CRITICAL,     // Делает задачу математически неразрешимой - прервать алгоритм
    SPECIAL,      // Требует модификации параметризации - автоматическая адаптация
    WARNING,      // Ухудшает численную устойчивость - продолжить с предупреждениями
    RECOVERABLE   // Вызывает численные аномалии - динамическая коррекция с откатом
};

/**
 * @brief Тип крайнего случая
 */
enum class EdgeCaseType {
    NONE,                           // Нет крайнего случая
    ZERO_NODES,                     // m = 0: отсутствие интерполяционных узлов
    FULL_INTERPOLATION,             // m = n + 1: полная интерполяция
    OVERCONSTRAINED,                // m > n + 1: избыточные ограничения
    CLOSE_NODES,                    // Близкие интерполяционные узлы
    MULTIPLE_ROOTS,                 // Кратные корни в весовом множителе
    HIGH_DEGREE,                    // n > 30: высокая степень полинома
    DEGENERATE_DATA,                // Вырожденные данные
    CONSTANT_VALUES,                // Константные значения интерполяции
    LINEAR_DEPENDENCE,              // Линейная зависимость значений
    EMPTY_APPROX_POINTS,            // Пустое множество аппроксимирующих точек
    EMPTY_REPEL_POINTS,             // Пустое множество отталкивающих точек
    NUMERICAL_OVERFLOW,             // Переполнение при вычислениях
    GRADIENT_EXPLOSION,             // Взрывной рост градиента
    OSCILLATION                     // Осцилляции параметров
};

/**
 * @brief Структура для хранения информации о крайнем случае
 */
struct EdgeCaseInfo {
    EdgeCaseLevel level;
    EdgeCaseType type;
    std::string message;
    std::string recommendation;
    bool is_handled;
    std::map<std::string, std::string> details;
    
    EdgeCaseInfo()
        : level(EdgeCaseLevel::WARNING)
        , type(EdgeCaseType::NONE)
        , is_handled(false) {}
    
    EdgeCaseInfo(EdgeCaseLevel lvl, EdgeCaseType t, const std::string& msg, const std::string& rec)
        : level(lvl)
        , type(t)
        , message(msg)
        , recommendation(rec)
        , is_handled(false) {}
};

/**
 * @brief Статус обработки крайних случаев
 */
struct EdgeCaseHandlingResult {
    bool success;
    std::vector<EdgeCaseInfo> detected_cases;
    std::vector<EdgeCaseInfo> handled_cases;
    std::vector<std::string> info_messages;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    // Адаптированные данные
    int adapted_m;                  // Адаптированное число узлов
    int adapted_n;                  // Адаптированная степень полинома
    bool parameters_modified;       // Флаг модификации параметров
    
    EdgeCaseHandlingResult()
        : success(true)
        , adapted_m(0)
        , adapted_n(0)
        , parameters_modified(false) {}
    
    bool has_critical_errors() const;
    
    bool has_warnings() const { return !warnings.empty(); }
};

// ============== Шаг 2.1.8.2: Обработка случая m = 0 ==============

/**
 * @brief Результат адаптации для случая m = 0
 */
struct ZeroNodesResult {
    bool use_trivial_parameterization;  // Использовать упрощённую параметризацию
    int degree;                          // Степень Q(x) = n
    std::string info_message;
    
    ZeroNodesResult()
        : use_trivial_parameterization(true)
        , degree(0) {}
};

// ============== Шаг 2.1.8.3: Обработка случая m = n + 1 ==============

/**
 * @brief Результат адаптации для полной интерполяции
 */
struct FullInterpolationResult {
    bool is_degenerate;              // Вырожденный случай (Q(x) ≡ 0)
    int n_free;                      // Степень свободы (должна быть 0)
    std::string info_message;
    std::string recommendations;
    
    FullInterpolationResult()
        : is_degenerate(true)
        , n_free(0) {}
};

// ============== Шаг 2.1.8.4: Обработка избыточных ограничений ==============

/**
 * @brief Стратегия обработки избыточных ограничений
 */
enum class OverconstrainedStrategy {
    STRICT,          // Строгая: прервать алгоритм с ошибкой
    SOFT_SELECT,     // Мягкая: выбрать подмножество узлов
    MERGE_CLUSTERS   // Объединить близкие узлы
};

/**
 * @brief Результат адаптации для избыточных ограничений
 */
struct OverconstrainedResult {
    bool resolved;                   // Проблема решена
    OverconstrainedStrategy strategy; // Использованная стратегия
    int original_m;                  // Исходное число узлов
    int adapted_m;                   // Адаптированное число узлов
    std::vector<int> merged_indices; // Индексы объединённых узлов
    std::vector<int> selected_indices; // Индексы выбранных узлов
    std::string info_message;
    
    OverconstrainedResult()
        : resolved(false)
        , strategy(OverconstrainedStrategy::STRICT)
        , original_m(0)
        , adapted_m(0) {}
};

// ============== Шаг 2.1.8.5: Обработка кратных/близких корней ==============

/**
 * @brief Структура для описания кластера узлов
 */
struct NodeCluster {
    std::vector<int> indices;     // Индексы узлов в кластере
    double center;                // Центр кластера (координата)
    double value_center;          // Среднее значение f(z)
    double value_spread;          // Разброс значений
    bool should_merge;           // Флаг: нужно ли объединять
    
    NodeCluster()
        : center(0.0)
        , value_center(0.0)
        , value_spread(0.0)
        , should_merge(false) {}
};

/**
 * @brief Результат обработки близких узлов
 */
struct CloseNodesResult {
    bool has_close_nodes;         // Есть ли близкие узлы
    std::vector<NodeCluster> clusters; // Кластеры узлов
    int original_m;               // Исходное число узлов
    int effective_m;             // Эффективное число узлов после объединения
    double min_distance;         // Минимальное расстояние между узлами
    std::string info_message;
    
    CloseNodesResult()
        : has_close_nodes(false)
        , original_m(0)
        , effective_m(0)
        , min_distance(0.0) {}
};

// ============== Шаг 2.1.8.6: Обработка высокой степени ==============

/**
 * @brief Результат адаптации для высокой степени полинома
 */
struct HighDegreeResult {
    bool requires_adaptation;    // Требуется адаптация
    bool switch_to_chebyshev;    // Переключиться на базис Чебышёва
    bool use_long_double;         // Использовать long double
    bool suggest_splines;        // Рекомендовать сплайны
    int original_degree;          // Исходная степень
    int recommended_degree;      // Рекомендуемая степень
    std::vector<std::string> recommendations;
    
    HighDegreeResult()
        : requires_adaptation(false)
        , switch_to_chebyshev(false)
        , use_long_double(false)
        , suggest_splines(false)
        , original_degree(0)
        , recommended_degree(0) {}
};

// ============== Шаг 2.1.8.7: Обработка вырожденных данных ==============

/**
 * @brief Тип вырожденности данных
 */
enum class DegeneracyType {
    NONE,               // Нет вырожденности
    CONSTANT,           // Константные значения
    LINEAR,            // Линейная зависимость
    RANK_DEFICIENT     // Ранговый дефицит
};

/**
 * @brief Результат анализа вырожденных данных
 */
struct DegeneracyResult {
    bool is_degenerate;         // Есть ли вырожденность
    DegeneracyType type;        // Тип вырожденности
    int effective_degree;       // Эффективная степень
    double constant_value;      // Константное значение (если применимо)
    double rank_deficiency;     // Ранговый дефицит
    std::string info_message;
    std::string recommendations;
    
    DegeneracyResult()
        : is_degenerate(false)
        , type(DegeneracyType::NONE)
        , effective_degree(0)
        , constant_value(0.0)
        , rank_deficiency(0.0) {}
};

// ============== Шаг 2.1.8.8: Мониторинг численных аномалий ==============

/**
 * @brief Тип обнаруженной аномалии
 */
enum class AnomalyType {
    NONE,
    OVERFLOW,              // Переполнение/исчезновение значений
    GRADIENT_EXPLOSION,   // Взрывной рост градиента
    OSCILLATION,          // Осцилляции параметров
    NAN_DETECTED,         // Обнаружен NaN
    INF_DETECTED          // Обнаружен Inf
};

/**
 * @brief Запись о событии аномалии
 */
struct AnomalyEvent {
    AnomalyType type;
    int iteration;
    double value;
    std::string description;
    
    AnomalyEvent()
        : type(AnomalyType::NONE)
        , iteration(0)
        , value(0.0) {}
};

/**
 * @brief Результат мониторинга аномалий
 */
struct AnomalyMonitorResult {
    bool anomaly_detected;
    AnomalyType last_anomaly_type;
    std::vector<AnomalyEvent> events;
    int total_iterations;
    int anomaly_iterations;
    bool needs_recovery;
    bool needs_stop;
    
    AnomalyMonitorResult()
        : anomaly_detected(false)
        , last_anomaly_type(AnomalyType::NONE)
        , total_iterations(0)
        , anomaly_iterations(0)
        , needs_recovery(false)
        , needs_stop(false) {}
};

/**
 * @brief Класс для мониторинга численных аномалий во время оптимизации
 */
class NumericalAnomalyMonitor {
public:
    // Пороги обнаружения аномалий
    double gradient_threshold;       // Порог для градиента (по умолчанию 1e10)
    double oscillation_threshold;   // Порог для обнаружения осцилляций
    double step_reduction_factor;   // Коэффициент уменьшения шага (0.5)
    int max_consecutive_anomalies;  // Макс. число последовательных аномалий
    
    NumericalAnomalyMonitor();
    
    /**
     * @brief Сброс состояния монитора
     */
    void reset();
    
    /**
     * @brief Проверка на аномалии
     * @param iteration Номер итерации
     * @param current_value Текущее значение функционала
     * @param gradient_norm Норма градиента
     * @param params Текущие параметры
     * @param prev_params Предыдущие параметры
     * @return Результат проверки
     */
    AnomalyMonitorResult check_anomaly(int iteration,
                                       double current_value,
                                       double gradient_norm,
                                       const std::vector<double>& params,
                                       const std::vector<double>& prev_params);
    
    /**
     * @brief Проверка на переполнение
     */
    AnomalyType check_overflow(double value);
    
    /**
     * @brief Проверка на взрыв градиента
     */
    AnomalyType check_gradient_explosion(double gradient_norm, double initial_gradient);
    
    /**
     * @brief Проверка на осцилляции
     */
    AnomalyType check_oscillation(const std::vector<double>& params,
                                  const std::vector<double>& prev_params,
                                  const std::vector<double>& prev_prev_params);
    
    /**
     * @brief Получение истории аномалий
     */
    const std::vector<AnomalyEvent>& get_anomaly_history() const { return anomaly_history_; }
    
    /**
     * @brief Получение коэффициента для уменьшения шага
     */
    double get_step_reduction_factor() const { return step_reduction_factor; }
    
private:
    std::vector<AnomalyEvent> anomaly_history_;
    std::vector<double> value_history_;
    double initial_gradient_norm_;
    bool initial_gradient_set_;
    
    /**
     * @brief Добавление события аномалии
     */
    void add_anomaly_event(const AnomalyEvent& event);
};

// ============== Основной класс обработки крайних случаев ==============

/**
 * @brief Класс для обработки крайних случаев параметризации
 * 
 * Реализует многоуровневую стратегию защиты:
 * - Диагностика на этапе валидации
 * - Предварительная адаптация структур данных
 * - Верификация корректности адаптированной параметризации
 * - Динамический мониторинг во время оптимизации
 */
class EdgeCaseHandler {
public:
    // Пороги для различных проверок
    double epsilon_close_nodes;     // Порог близости узлов (по умолчанию 1e-8)
    double epsilon_value_spread;    // Порог разброса значений (1e-8)
    double epsilon_machine;         // Машинная точность (1e-12)
    double high_degree_threshold;   // Порог высокой степени (30)
    double condition_limit;         // Макс. допустимое число обусловленности
    double gradient_limit;          // Макс. допустимая норма градиента
    
    EdgeCaseHandler();
    
    // ============== Основные методы ==============
    
    /**
     * @brief Полный анализ и обработка всех крайних случаев
     * @param n Степень полинома
     * @param m Число интерполяционных узлов
     * @param interp_values Значения функции в узлах
     * @param config Конфигурация задачи
     * @return Результат обработки
     */
    EdgeCaseHandlingResult handle_all_cases(int n, int m,
                                            const std::vector<double>& interp_values,
                                            const ApproximationConfig& config);
    
    /**
     * @brief Анализ и обработка одного шага
     * @param step_name Имя шага
     * @param check_func Функция проверки
     * @param handle_func Функция обработки
     * @return Результат
     */
    template<typename CheckFunc, typename HandleFunc>
    EdgeCaseInfo analyze_and_handle(const std::string& step_name,
                                    CheckFunc check_func,
                                    HandleFunc handle_func);
    
    // ============== Специфические методы обработки ==============
    
    /**
     * @brief Обработка случая отсутствия интерполяционных узлов (m = 0)
     */
    ZeroNodesResult handle_zero_nodes(int n);
    
    /**
     * @brief Обработка случая полной интерполяции (m = n + 1)
     */
    FullInterpolationResult handle_full_interpolation(int n, int m);
    
    /**
     * @brief Обработка избыточных ограничений (m > n + 1)
     */
    OverconstrainedResult handle_overconstrained(int n, int m,
                                                  const std::vector<double>& interp_coords,
                                                  const std::vector<double>& interp_values,
                                                  OverconstrainedStrategy strategy);
    
    /**
     * @brief Обработка близких интерполяционных узлов
     */
    CloseNodesResult handle_close_nodes(const std::vector<double>& coords,
                                       const std::vector<double>& values,
                                       double interval_length);
    
    /**
     * @brief Обработка высокой степени полинома
     */
    HighDegreeResult handle_high_degree(int n);
    
    /**
     * @brief Анализ вырожденных данных
     */
    DegeneracyResult analyze_degeneracy(const std::vector<double>& values);
    
    /**
     * @brief Адаптация параметризации на основе результатов
     */
    bool adapt_parameterization(EdgeCaseHandlingResult& result,
                                ApproximationConfig& config,
                                InterpolationBasis& basis,
                                WeightMultiplier& weight,
                                CorrectionPolynomial& correction);
    
    // ============== Вспомогательные методы ==============
    
    /**
     * @brief Классификация крайнего случая
     */
    EdgeCaseLevel classify_case(EdgeCaseType type);
    
    /**
     * @brief Формирование диагностического сообщения
     */
    std::string format_diagnostic_message(const EdgeCaseInfo& info);
    
    /**
     * @brief Проверка на константные значения
     */
    bool is_constant(const std::vector<double>& values, double threshold = 1e-12);
    
    /**
     * @brief Вычисление разброса значений
     */
    double compute_value_spread(const std::vector<double>& values);
    
    /**
     * @brief Объединение близких узлов в кластеры
     */
    std::vector<NodeCluster> cluster_close_nodes(const std::vector<double>& coords,
                                                  const std::vector<double>& values,
                                                  double epsilon);
    
    /**
     * @brief Проверка, нужно ли объединять кластер
     */
    bool should_merge_cluster(const NodeCluster& cluster, double epsilon_value);
    
    /**
     * @brief Генерация рекомендаций для высокой степени
     */
    std::vector<std::string> generate_high_degree_recommendations(int n);
    
    /**
     * @brief Получение всех обнаруженных случаев
     */
    const std::vector<EdgeCaseInfo>& get_detected_cases() const { return detected_cases_; }
    
    /**
     * @brief Получение результата обработки
     */
    EdgeCaseHandlingResult get_result() const;
    
    /**
     * @brief Проверка наличия критических ошибок
     */
    bool has_critical_errors() const;
    
    /**
     * @brief Очистка истории
     */
    void clear();
    
private:
    std::vector<EdgeCaseInfo> detected_cases_;
    std::vector<EdgeCaseInfo> handled_cases_;
    std::vector<std::string> warnings_;
    std::vector<std::string> errors_;
    
    /**
     * @brief Добавление обнаруженного случая
     */
    void add_case(const EdgeCaseInfo& info);
    
    /**
     * @brief Добавление информационного сообщения
     */
    void add_info(const std::string& message);
    
    /**
     * @brief Добавление предупреждения
     */
    void add_warning(const std::string& message);
    
    /**
     * @brief Добавление ошибки
     */
    void add_error(const std::string& message);
};

// ============== Функции форматирования ==============

/**
 * @brief Форматирование результата обработки в строку
 */
std::string format_edge_case_result(const EdgeCaseHandlingResult& result);

/**
 * @brief Форматирование результата адаптации в строку
 */
std::string format_zero_nodes_result(const ZeroNodesResult& result);

/**
 * @brief Форматирование результата полной интерполяции
 */
std::string format_full_interpolation_result(const FullInterpolationResult& result);

/**
 * @brief Форматирование результата избыточных ограничений
 */
std::string format_overconstrained_result(const OverconstrainedResult& result);

/**
 * @brief Форматирование результата близких узлов
 */
std::string format_close_nodes_result(const CloseNodesResult& result);

/**
 * @brief Форматирование результата высокой степени
 */
std::string format_high_degree_result(const HighDegreeResult& result);

/**
 * @brief Форматирование результата вырожденности
 */
std::string format_degeneracy_result(const DegeneracyResult& result);

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_EDGE_CASE_HANDLER_H
