#ifndef MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H
#define MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H

#include <vector>
#include <string>

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
 * @brief Монитор сходимости для диагностики проблем
 */
class ConvergenceMonitor {
public:
    // Параметры мониторинга
    double tol_gradient;           // допуск по градиенту
    double tol_objective;          // допуск по изменению функционала
    double tol_step;               // минимальный шаг
    int max_oscillation_count;     // число итераций для определения осцилляций
    int max_plateau_count;         // число итераций для определения плато
    
    // Конструктор
    ConvergenceMonitor(double tol_grad = 1e-6, double tol_obj = 1e-8)
        : tol_gradient(tol_grad), tol_objective(tol_obj), tol_step(1e-12)
        , max_oscillation_count(5), max_plateau_count(20)
        , current_iteration_(0), oscillation_count_(0)
        , plateau_count_(0), barrier_proximity_count_(0)
        , numerical_anomaly_count_(0), is_diverging_(false) {}
    
    // Проверка критериев сходимости
    bool is_converged(double gradient_norm, double objective_value,
                      double objective_change, double step_size);
    
    // Проверка на осцилляции
    bool detect_oscillation(const std::vector<double>& history);
    
    // Проверка на плато
    bool detect_plateau(double current_objective);
    
    // Проверка на расходимость
    bool detect_divergence(double current_objective);
    
    // Обновление счётчиков проблем
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
    
    // Получение истории для анализа
    const std::vector<double>& objective_history() const { return objective_history_; }

private:
    int current_iteration_;
    int oscillation_count_;
    int plateau_count_;
    int barrier_proximity_count_;
    int numerical_anomaly_count_;
    bool is_diverging_;
    
    std::vector<double> objective_history_;
    std::vector<double> gradient_history_;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_CONVERGENCE_MONITOR_H
