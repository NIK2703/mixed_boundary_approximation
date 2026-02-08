#ifndef MIXED_APPROXIMATION_NUMERICAL_ANOMALY_MONITOR_H
#define MIXED_APPROXIMATION_NUMERICAL_ANOMALY_MONITOR_H

#include "types.h"
#include <vector>
#include <string>
#include <functional>

namespace mixed_approx {

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

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_NUMERICAL_ANOMALY_MONITOR_H
