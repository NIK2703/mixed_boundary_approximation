#ifndef MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H
#define MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H

#include <vector>
#include <string>
#include <chrono>
#include "objective_functor.h"  // для Components

namespace mixed_approx {

/**
 * @brief Структура для журналирования итерации оптимизации
 */
struct OptimizationIterationLog {
    int iteration;
    double objective_value;
    double approx_term;
    double repel_term;
    double reg_term;
    double gradient_norm;
    double step_size;
    bool barrier_proximity;      // близость к барьеру отталкивания
    bool numerical_anomaly;      // численные аномалии (Inf/NaN)
    std::vector<double> coefficients_snapshot; // опционально для отладки
    
    OptimizationIterationLog()
        : iteration(0), objective_value(0.0), approx_term(0.0)
        , repel_term(0.0), reg_term(0.0), gradient_norm(0.0)
        , step_size(0.0), barrier_proximity(false), numerical_anomaly(false) {}
};

/**
 * @brief Причины остановки оптимизации
 */
enum class StopReason {
    NOT_CONVERGED = 0,           // не достигнуто условие остановки
    RELATIVE_OBJECTIVE_CHANGE,   // относительное изменение функционала
    ABSOLUTE_OBJECTIVE_CHANGE,   // абсолютное изменение функционала
    GRADIENT_NORM,              // норма градиента
    MAX_ITERATIONS,             // достигнут максимум итераций
    TIMEOUT,                    // истекло время
    OSCILLATIONS,               // осцилляции
    PLATEAU,                    // плато
    STAGNATION,                 // застывание
    DIVERGENCE,                 // расходимость
    VALIDATION_FAILED,          // валидация не пройдена
    NUMERICAL_ANOMALY,          // численная аномалия (NaN/Inf)
    NUMERICAL_ERROR,            // численная ошибка
    SUCCESS                     // успешное завершение (все критерии валидации пройдены)
};

/**
 * @brief Монитор сходимости для диагностики проблем и принятия решений об остановке
 * 
 * Реализует расширенные критерии останова согласно шагу 4.3:
 * - Относительное изменение функционала с медианной фильтрацией
 * - Норма градиента с компонентным анализом
 * - Адаптивный таймаут
 * - Диагностика осцилляций через автокорреляцию
 * - Обнаружение застывания через линейную регрессию
 * - Обнаружение расходимости
 */
class ConvergenceMonitor {
public:
    // Параметры мониторинга
    double tol_gradient;           // допуск по градиенту (ε_grad)
    double tol_objective;          // допуск по изменению функционала (ε_J)
    double tol_step;              // допуск по размеру шага
    double grad_scale;             // масштаб для нормировки градиента (обычно 1.0)
    double obj_scale;              // масштаб для нормировки функционала (J_scale)
    int max_iterations;            // максимальное число итераций
    double timeout_seconds;        // таймаут в секундах
    int window_size;               // размер окна для фильтрации (W)
    int plateau_patience;          // число итераций для подтверждения плато (N_plateau)
    double barrier_threshold;      // порог для доминирования компоненты (0.9)
    int max_oscillation_count;     // максимальное число осцилляций для остановки
    int max_plateau_count;         // максимальное число итераций на плато
    
    // Конструктор
    ConvergenceMonitor(double tol_grad = 1e-6, double tol_obj = 1e-8)
        : tol_gradient(tol_grad), tol_objective(tol_obj), tol_step(1e-12)
        , grad_scale(1.0), obj_scale(1.0)
        , max_iterations(1000), timeout_seconds(300.0), window_size(5)
        , plateau_patience(3), barrier_threshold(0.9)
        , max_oscillation_count(10), max_plateau_count(20)
        , current_iteration_(0), oscillation_count_(0)
        , plateau_count_(0), barrier_proximity_count_(0)
        , numerical_anomaly_count_(0), is_diverging_(false)
        , timer_active_(false), stop_reason_(StopReason::NOT_CONVERGED) {}
    
    // ============== Таймер ==============
    void start_timer() {
        start_time_ = std::chrono::high_resolution_clock::now();
        timer_active_ = true;
    }
    
    double elapsed_time() const {
        if (!timer_active_) return 0.0;
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - start_time_;
        return diff.count();
    }
    
    bool is_timeout_exceeded() const {
        return timer_active_ && elapsed_time() >= timeout_seconds;
    }
    
    // ============== Запись итерации ==============
    void record_iteration(double objective, double gradient_norm,
                          const ObjectiveFunctor::Components& components,
                          double step_size = 0.0) {
        objective_history_.push_back(objective);
        gradient_history_.push_back(gradient_norm);
        approx_history_.push_back(components.approx);
        repel_history_.push_back(components.repel);
        reg_history_.push_back(components.reg);
        step_size_history_.push_back(step_size);
        
        current_iteration_++;
    }
    
    // ============== Проверка критериев останова ==============
    /**
     * @brief Проверка всех критериев останова
     * @param gradient_norm текущая норма градиента
     * @param objective текущее значение функционала
     * @param components компоненты функционала
     * @param step_size размер шага
     * @param initial_objective начальное значение функционала (для нормировки)
     * @param initial_gradient_norm начальная норма градиента (для нормировки)
     * @param max_barrier_weight максимальный вес барьера (для адаптации)
     * @return StopReason - причина остановки (NOT_CONVERGED если критерии не выполнены)
     */
    StopReason check_stop_criteria(double gradient_norm, double objective,
                                   const ObjectiveFunctor::Components& components,
                                   double step_size, double initial_objective,
                                   double initial_gradient_norm, double max_barrier_weight = 0.0);
    
    // ============== Отдельные критерии ==============
    
    // 1. Относительное изменение функционала с медианной фильтрацией
    bool check_relative_objective_change_filtered();
    
    // 2. Норма градиента с компонентным анализом
    bool check_gradient_norm_with_balance(double gradient_norm, 
                                          const ObjectiveFunctor::Components& components,
                                          double initial_gradient_norm,
                                          double max_barrier_weight);
    
    // 3. Максимальное число итераций
    bool check_max_iterations() const { return current_iteration_ >= max_iterations; }
    
    // 4. Таймаут
    bool check_timeout() const { return is_timeout_exceeded(); }
    
    // 5. Диагностика осцилляций через автокорреляцию
    bool detect_oscillation_autocorrelation();
    
    // 6. Обнаружение застывания через линейную регрессию
    bool detect_stagnation_regression();
    
    // 7. Обнаружение расходимости
    bool detect_divergence_advanced(double current_objective, double initial_objective);
    
    // 8. Обнаружение численных аномалий
    bool detect_numerical_anomaly(double objective, const std::vector<double>& gradient);
    
    // ============== Методы для обратной совместимости ==============
    
    /**
     * @brief Простая проверка сходимости (для обратной совместимости)
     */
    bool is_converged(double gradient_norm, double objective_value,
                      double objective_change, double step_size);
    
    /**
     * @brief Обнаружение осцилляций по истории
     */
    bool detect_oscillation(const std::vector<double>& history);
    
    /**
     * @brief Обнаружение плато по текущему значению
     */
    bool detect_plateau(double current_objective);
    
    /**
     * @brief Обнаружение расходимости по текущему значению
     */
    bool detect_divergence(double current_objective);
    
    // ============== Вспомогательные методы ==============
    void update_barrier_proximity(bool active) {
        if (active) barrier_proximity_count_++;
        else barrier_proximity_count_ = 0;
    }
    
    void update_numerical_anomaly(bool active) {
        if (active) numerical_anomaly_count_++;
        else numerical_anomaly_count_ = 0;
    }
    
    // Сброс монитора
    void reset();
    
    // Получение диагностической информации
    std::string get_diagnostic_info() const;
    
    // Получение расширенного отчёта
    std::string generate_detailed_report(const OptimizationProblemData& data,
                                         const CompositePolynomial& poly) const;
    
    // Геттеры
    int iteration() const { return current_iteration_; }
    StopReason stop_reason() const { return stop_reason_; }
    const std::vector<double>& objective_history() const { return objective_history_; }
    const std::vector<double>& gradient_history() const { return gradient_history_; }
    const std::vector<double>& approx_history() const { return approx_history_; }
    const std::vector<double>& repel_history() const { return repel_history_; }
    const std::vector<double>& reg_history() const { return reg_history_; }
    const std::vector<double>& step_size_history() const { return step_size_history_; }
    double last_objective() const { return objective_history_.empty() ? 0.0 : objective_history_.back(); }
    double last_gradient_norm() const { return gradient_history_.empty() ? 0.0 : gradient_history_.back(); }
    
private:
    // Счётчики
    int current_iteration_;
    int oscillation_count_;
    int plateau_count_;
    int barrier_proximity_count_;
    int numerical_anomaly_count_;
    bool is_diverging_;
    
    // Таймер
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    bool timer_active_;
    
    // История
    std::vector<double> objective_history_;
    std::vector<double> gradient_history_;
    std::vector<double> approx_history_;
    std::vector<double> repel_history_;
    std::vector<double> reg_history_;
    std::vector<double> step_size_history_;
    
    // Причина остановки
    StopReason stop_reason_;
    
    // Вспомогательные функции для статистики
    static double median_of_window(const std::vector<double>& vec, int end_idx, int window_size);
    static double autocorrelation(const std::vector<double>& vec, int end_idx, int window_size);
    static void linear_regression(const std::vector<double>& vec, int end_idx, int window_size,
                                   double& slope, double& intercept, double& r_squared);
    
    // Преобразование причины остановки в строку
    static std::string stop_reason_to_string(StopReason reason);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H
