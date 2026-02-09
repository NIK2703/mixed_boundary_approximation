#include "mixed_approximation/convergence_monitor.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/optimization_problem_data.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <iostream>

namespace mixed_approx {

// ============== Вспомогательные статистические функции ==============

double ConvergenceMonitor::median_of_window(const std::vector<double>& vec, int end_idx, int window_size) {
    if (end_idx < 0 || end_idx >= static_cast<int>(vec.size())) {
        return vec.empty() ? 0.0 : vec.back();
    }
    
    int start = std::max(0, end_idx - window_size + 1);
    std::vector<double> window;
    window.reserve(end_idx - start + 1);
    for (int i = start; i <= end_idx; ++i) {
        window.push_back(vec[i]);
    }
    
    std::sort(window.begin(), window.end());
    int mid = window.size() / 2;
    if (window.size() % 2 == 0) {
        return (window[mid-1] + window[mid]) / 2.0;
    } else {
        return window[mid];
    }
}

double ConvergenceMonitor::autocorrelation(const std::vector<double>& vec, int end_idx, int window_size) {
    if (end_idx < 1 || static_cast<int>(vec.size()) < window_size) {
        return 0.0;
    }
    
    int start = end_idx - window_size + 1;
    if (start < 0) start = 0;
    
    // Вычислить среднее по окну
    double sum = 0.0;
    int count = 0;
    for (int i = start; i <= end_idx; ++i) {
        sum += vec[i];
        count++;
    }
    double mean = sum / count;
    
    // Вычислить автокорреляцию первого порядка
    double num = 0.0, den1 = 0.0, den2 = 0.0;
    for (int i = start; i < end_idx; ++i) {
        double diff1 = vec[i] - mean;
        double diff2 = vec[i+1] - mean;
        num += diff1 * diff2;
        den1 += diff1 * diff1;
        den2 += diff2 * diff2;
    }
    
    if (den1 > 0 && den2 > 0) {
        return num / (std::sqrt(den1 * den2));
    }
    return 0.0;
}

void ConvergenceMonitor::linear_regression(const std::vector<double>& vec, int end_idx, int window_size,
                                            double& slope, double& intercept, double& r_squared) {
    if (end_idx < 1 || static_cast<int>(vec.size()) < window_size) {
        slope = intercept = r_squared = 0.0;
        return;
    }
    
    int start = end_idx - window_size + 1;
    if (start < 0) start = 0;
    int n = end_idx - start + 1;
    
    // x - индексы, y - значения
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = start + i;
        y[i] = vec[start + i];
    }
    
    // Вычисление средних
    double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    
    // Вычисление коэффициентов
    double S_xy = 0.0, S_xx = 0.0, S_yy = 0.0;
    for (int i = 0; i < n; ++i) {
        S_xy += (x[i] - x_mean) * (y[i] - y_mean);
        S_xx += (x[i] - x_mean) * (x[i] - x_mean);
        S_yy += (y[i] - y_mean) * (y[i] - y_mean);
    }
    
    if (S_xx > 0) {
        slope = S_xy / S_xx;
        intercept = y_mean - slope * x_mean;
        if (S_yy > 0) {
            r_squared = (S_xy * S_xy) / (S_xx * S_yy);
        } else {
            r_squared = 0.0;
        }
    } else {
        slope = intercept = r_squared = 0.0;
    }
}

// ============== Основные методы ==============

void ConvergenceMonitor::reset() {
    current_iteration_ = 0;
    oscillation_count_ = 0;
    plateau_count_ = 0;
    barrier_proximity_count_ = 0;
    numerical_anomaly_count_ = 0;
    is_diverging_ = false;
    timer_active_ = false;
    stop_reason_ = StopReason::NOT_CONVERGED;
    
    objective_history_.clear();
    gradient_history_.clear();
    approx_history_.clear();
    repel_history_.clear();
    reg_history_.clear();
    step_size_history_.clear();
}

std::string ConvergenceMonitor::get_diagnostic_info() const {
    char buffer[512];
    snprintf(buffer, sizeof(buffer),
             "Convergence Monitor Status:\n"
             "  Iteration: %d\n"
             "  Oscillation count: %d\n"
             "  Plateau count: %d\n"
             "  Barrier proximity count: %d\n"
             "  Numerical anomaly count: %d\n"
             "  Is diverging: %s\n"
             "  Elapsed time: %.2f s\n"
             "  Last objective: %.6e\n"
             "  Last gradient norm: %.6e",
             current_iteration_, oscillation_count_, plateau_count_,
             barrier_proximity_count_, numerical_anomaly_count_,
             is_diverging_ ? "yes" : "no",
             elapsed_time(),
             last_objective(),
             last_gradient_norm());
    return std::string(buffer);
}

std::string ConvergenceMonitor::generate_detailed_report(const OptimizationProblemData& data,
                                                         const CompositePolynomial& poly) const {
    std::string report;
    char buffer[1024];
    
    // Заголовок
    snprintf(buffer, sizeof(buffer),
             "=== ДЕТАЛЬНЫЙ ОТЧЁТ ОПТИМИЗАЦИИ ===\n\n");
    report += buffer;
    
    // Общая информация
    snprintf(buffer, sizeof(buffer),
             "Общая информация:\n"
             "  Итераций выполнено: %d\n"
             "  Время выполнения: %.3f с\n"
             "  Финальное значение функционала: %.6e\n"
             "  Финальная норма градиента: %.6e\n"
             "  Причина остановки: %d\n",
             current_iteration_, elapsed_time(), last_objective(), last_gradient_norm(),
             static_cast<int>(stop_reason_));
    report += buffer;
    
    // Критерии останова
    snprintf(buffer, sizeof(buffer),
             "\nКритерии останова:\n"
             "  Относительное изменение функционала (ε_J): %.2e\n"
             "  Норма градиента (ε_grad): %.2e\n"
             "  Макс. итерации: %d\n"
             "  Таймаут: %.1f с\n",
             tol_objective, tol_gradient, max_iterations, timeout_seconds);
    report += buffer;
    
    // Компоненты функционала
    if (!approx_history_.empty() && !repel_history_.empty() && !reg_history_.empty()) {
        double final_approx = approx_history_.back();
        double final_repel = repel_history_.back();
        double final_reg = reg_history_.back();
        double total = final_approx + final_repel + final_reg;
        
        snprintf(buffer, sizeof(buffer),
                 "\nКомпоненты функционала (финальные):\n"
                 "  Аппроксимация: %.6e (%.1f%%)\n"
                 "  Отталкивание: %.6e (%.1f%%)\n"
                 "  Регуляризация: %.6e (%.1f%%)\n"
                 "  Сумма: %.6e\n",
                 final_approx, 100.0 * final_approx / total,
                 final_repel, 100.0 * final_repel / total,
                 final_reg, 100.0 * final_reg / total,
                 total);
        report += buffer;
    }
    
    // Прогресс
    if (objective_history_.size() >= 2) {
        double improvement = (objective_history_.front() - objective_history_.back()) / objective_history_.front();
        snprintf(buffer, sizeof(buffer),
                 "\nПрогресс:\n"
                 "  Начальное J: %.6e\n"
                 "  Финальное J: %.6e\n"
                 "  Улучшение: %.2f%%\n",
                 objective_history_.front(), objective_history_.back(), 100.0 * improvement);
        report += buffer;
    }
    
    // Диагностика сходимости
    snprintf(buffer, sizeof(buffer),
             "\nДиагностика сходимости:\n");
    report += buffer;
    
    if (objective_history_.size() >= 10) {
        int last10 = std::min(10, static_cast<int>(objective_history_.size()));
        double last_val = objective_history_.back();
        double prev_val = objective_history_[objective_history_.size() - last10];
        double progress_last10 = (prev_val - last_val) / std::abs(prev_val) * 100.0;
        snprintf(buffer, sizeof(buffer),
                 "  Прогресс за последние %d итераций: %.3f%%\n",
                 last10, progress_last10);
        report += buffer;
    }
    
    // Осцилляции
    double autocorr = 0.0;
    if (objective_history_.size() >= 20) {
        autocorr = autocorrelation(objective_history_, objective_history_.size() - 1, 20);
    }
    snprintf(buffer, sizeof(buffer),
             "  Автокорреляция (последние 20): %.3f %s\n",
             autocorr, (autocorr < -0.7) ? "(осцилляции)" : "");
    report += buffer;
    
    // Застывание
    double slope = 0.0, intercept = 0.0, r2 = 0.0;
    if (objective_history_.size() >= 100) {
        linear_regression(objective_history_, objective_history_.size() - 1, 100, slope, intercept, r2);
        snprintf(buffer, sizeof(buffer),
                 "  Наклон тренда (последние 100): %.2e (R²=%.3f) %s\n",
                 slope, r2, (std::abs(slope) < 1e-12 * std::abs(last_objective()) / 100.0) ? "(застывание)" : "");
        report += buffer;
    }
    
    // Безопасность барьеров
    if (data.num_repel_points() > 0) {
        // Вычислить минимальное расстояние до барьеров
        // Это потребует вычисления полинома в точках y_j
        // Пока упрощённо: используем историю repel_term
        double min_repel = repel_history_.empty() ? 0.0 : *std::min_element(repel_history_.begin(), repel_history_.end());
        snprintf(buffer, sizeof(buffer),
                 "\nБезопасность барьеров:\n"
                 "  Число отталкивающих точек: %zu\n"
                 "  Минимальное расстояние (приблизительно): %.2e\n",
                 data.num_repel_points(), min_repel);
        report += buffer;
    }
    
    // Интерполяционные условия
    // Проверим, если полином и данные доступны
    snprintf(buffer, sizeof(buffer),
             "\nИнтерполяционные условия:\n"
             "  Число узлов интерполяции: %zu\n",
             data.num_interp_nodes());
    report += buffer;
    
    // Рекомендации
    snprintf(buffer, sizeof(buffer),
             "\nРекомендации:\n");
    report += buffer;
    
    if (stop_reason_ == StopReason::MAX_ITERATIONS) {
        report += "  • Увеличьте max_iter до 1500 для достижения полной сходимости\n";
    } else if (stop_reason_ == StopReason::TIMEOUT) {
        report += "  • Упростите задачу (уменьшите степень полинома) или увеличьте таймаут\n";
    } else if (stop_reason_ == StopReason::OSCILLATIONS) {
        report += "  • Уменьшите веса барьеров B_j или увеличьте параметр регуляризации γ\n";
    } else if (stop_reason_ == StopReason::STAGNATION) {
        report += "  • Попробуйте другие начальные приближения или увеличьте регуляризацию\n";
    } else if (stop_reason_ == StopReason::DIVERGENCE) {
        report += "  • Уменьшите начальный шаг или проверьте данные на аномалии\n";
    } else if (stop_reason_ == StopReason::RELATIVE_OBJECTIVE_CHANGE || 
               stop_reason_ == StopReason::GRADIENT_NORM) {
        report += "  • Оптимизация успешно завершена\n";
    }
    
    report += "\n=== Конец отчёта ===\n";
    return report;
}

// ============== Критерии останова ==============

StopReason ConvergenceMonitor::check_stop_criteria(double gradient_norm, double objective,
                                                   const ObjectiveFunctor::Components& components,
                                                   double step_size, double initial_objective,
                                                   double initial_gradient_norm, double max_barrier_weight) {
    // 1. Проверка на численные аномалии (высший приоритет)
    if (std::isnan(objective) || std::isinf(objective) || gradient_norm > 1e150) {
        stop_reason_ = StopReason::NUMERICAL_ANOMALY;
        return stop_reason_;
    }
    
    // 2. Проверка на расходимость
    if (detect_divergence_advanced(objective, initial_objective)) {
        stop_reason_ = StopReason::DIVERGENCE;
        return stop_reason_;
    }
    
    // 3. Ресурсные ограничения (максимум итераций, таймаут)
    if (check_max_iterations()) {
        stop_reason_ = StopReason::MAX_ITERATIONS;
        return stop_reason_;
    }
    if (check_timeout()) {
        stop_reason_ = StopReason::TIMEOUT;
        return stop_reason_;
    }
    
    // 4. Критерии сходимости (проверяем в порядке приоритета)
    
    // 4.1. Относительное изменение функционала с фильтрацией
    if (check_relative_objective_change_filtered()) {
        // Дополнительная проверка: требуется стабильность на N_plateau итераций
        // (уже внутри check_relative_objective_change_filtered)
        stop_reason_ = StopReason::RELATIVE_OBJECTIVE_CHANGE;
        return stop_reason_;
    }
    
    // 4.2. Норма градиента с компонентным анализом
    if (check_gradient_norm_with_balance(gradient_norm, components, initial_gradient_norm, max_barrier_weight)) {
        stop_reason_ = StopReason::GRADIENT_NORM;
        return stop_reason_;
    }
    
    // 5. Диагностические критерии (осцилляции, застывание) - останавливаем с предупреждением
    if (detect_oscillation_autocorrelation()) {
        oscillation_count_++;
        if (oscillation_count_ >= 50) {  // осцилляции сохраняются более 50 итераций
            stop_reason_ = StopReason::OSCILLATIONS;
            return stop_reason_;
        }
    } else {
        oscillation_count_ = 0;
    }
    
    if (detect_stagnation_regression()) {
        plateau_count_++;
        if (plateau_count_ >= 2) {  // застывание подтверждено
            stop_reason_ = StopReason::STAGNATION;
            return stop_reason_;
        }
    } else {
        plateau_count_ = 0;
    }
    
    // Не достигнуто условие остановки
    stop_reason_ = StopReason::NOT_CONVERGED;
    return stop_reason_;
}

bool ConvergenceMonitor::check_relative_objective_change_filtered() {
    if (objective_history_.size() < static_cast<size_t>(window_size * 2)) {
        return false;  // недостаточно данных
    }
    
    int idx = objective_history_.size() - 1;
    
    // Вычислить медианы двух последовательных окон
    double median_current = median_of_window(objective_history_, idx, window_size);
    double median_previous = median_of_window(objective_history_, idx - window_size, window_size);
    
    double abs_diff = std::abs(median_current - median_previous);
    double denominator = std::max(std::abs(median_current), obj_scale);
    
    // Проверить, что изменение меньше порога
    if (denominator < 1e-15) {
        return true;  // функционал близок к нулю, считаем сходящимся
    }
    
    double relative_change = abs_diff / denominator;
    
    // Защита от плато: требуется выполнение на plateau_patience последовательных итерациях
    // (простая реализация: проверяем, что последние plateau_patience итераций показывают уменьшение)
    if (relative_change < tol_objective) {
        // Проверим монотонность: последние N_plateau итераций должны показывать уменьшение или стабильность
        int N = plateau_patience;
        if (objective_history_.size() >= static_cast<size_t>(N + 1)) {
            bool monotonic = true;
            for (int i = objective_history_.size() - N; i < static_cast<int>(objective_history_.size()); ++i) {
                if (objective_history_[i] > objective_history_[i-1] + 1e-12 * std::abs(objective_history_[i])) {
                    monotonic = false;
                    break;
                }
            }
            return monotonic;
        }
        return true;
    }
    
    return false;
}

bool ConvergenceMonitor::check_gradient_norm_with_balance(double gradient_norm,
                                                          const ObjectiveFunctor::Components& components,
                                                          double initial_gradient_norm,
                                                          double max_barrier_weight) {
    // Основной критерий: норма градиента меньше порога
    double normalized_norm = gradient_norm / std::max(initial_gradient_norm, grad_scale);
    if (normalized_norm >= tol_gradient) {
        return false;
    }
    
    // Дополнительное условие баланса компонент (предотвращает "застывание" в неоптимальной точке)
    double total = components.approx + components.repel + components.reg;
    if (total <= 0) {
        return true;  // нет компонент, считаем что баланс не требуется
    }
    
    double r_approx = components.approx / total;
    double r_repel = components.repel / total;
    double r_reg = components.reg / total;
    
    // Если одна компонента доминирует (> barrier_threshold), и норма градиента уже мала,
    // это может указывать на застывание в седловой точке или плохо обусловленную задачу
    if ((r_repel > barrier_threshold || r_reg > barrier_threshold) && normalized_norm < 10.0 * tol_gradient) {
        return false;  // не сходимся, нужен дальнейший прогресс
    }
    
    // Адаптация к сильным барьерам
    if (max_barrier_weight > 1000.0) {
        double effective_tol = tol_gradient * (1.0 + 0.01 * std::log10(max_barrier_weight));
        if (normalized_norm < effective_tol) {
            return true;
        }
        return false;
    }
    
    return true;
}

bool ConvergenceMonitor::detect_oscillation_autocorrelation() {
    if (objective_history_.size() < static_cast<size_t>(window_size + 5)) {
        return false;
    }
    
    int idx = objective_history_.size() - 1;
    double rho = autocorrelation(objective_history_, idx, window_size);
    
    return (rho < -0.7);
}

bool ConvergenceMonitor::detect_stagnation_regression() {
    if (objective_history_.size() < 100) {
        return false;
    }
    
    int idx = objective_history_.size() - 1;
    double slope, intercept, r2;
    linear_regression(objective_history_, idx, 100, slope, intercept, r2);
    
    double current_obj = objective_history_.back();
    double threshold = 1e-12 * std::abs(current_obj) / 100.0;
    
    return (std::abs(slope) < threshold);
}

bool ConvergenceMonitor::detect_divergence_advanced(double current_objective, double initial_objective) {
    // Катастрофический рост
    if (current_objective > 100.0 * initial_objective) {
        is_diverging_ = true;
        return true;
    }
    
    // Постепенная расходимость: рост на 50% за 10 итераций
    if (objective_history_.size() >= 11) {
        int idx = objective_history_.size() - 1;
        double old_val = objective_history_[idx - 10];
        if (current_objective > 10.0 * old_val && 
            (current_objective - old_val) / (std::abs(old_val) + 1e-15) > 0.5) {
            return true;
        }
    }
    
    return false;
}

bool ConvergenceMonitor::detect_numerical_anomaly(double objective, const std::vector<double>& gradient) {
    if (!std::isfinite(objective) || std::abs(objective) > 1e150) {
        return true;
    }
    for (double g : gradient) {
        if (!std::isfinite(g) || std::abs(g) > 1e150) {
            return true;
        }
    }
    return false;
}

// ============== Остальные методы (сохранение обратной совместимости) ==============

bool ConvergenceMonitor::is_converged(double gradient_norm, double objective_value,
                                      double objective_change, double step_size) {
    // Простая проверка для обратной совместимости
    if (gradient_norm < tol_gradient * std::max(1.0, std::abs(objective_value))) {
        return true;
    }
    if (objective_change < tol_objective * std::max(1.0, std::abs(objective_value))) {
        return true;
    }
    if (step_size < tol_step) {
        return true;
    }
    return false;
}

bool ConvergenceMonitor::detect_oscillation(const std::vector<double>& history) {
    if (history.size() < 4) return false;
    
    size_t n = history.size();
    double diff1 = history[n-1] - history[n-3];
    double diff2 = history[n-2] - history[n-4];
    
    if (diff1 * diff2 < 0) {
        oscillation_count_++;
        return oscillation_count_ >= max_oscillation_count;
    }
    
    oscillation_count_ = 0;
    return false;
}

bool ConvergenceMonitor::detect_plateau(double current_objective) {
    objective_history_.push_back(current_objective);
    
    if (objective_history_.size() < static_cast<size_t>(max_plateau_count + 10)) {
        return false;
    }
    
    size_t start_idx = objective_history_.size() - max_plateau_count - 1;
    double old_value = objective_history_[start_idx];
    double relative_change = std::abs(current_objective - old_value) / std::max(1.0, std::abs(current_objective));
    
    if (relative_change < 1e-10) {
        plateau_count_++;
        return plateau_count_ >= 2;
    }
    
    plateau_count_ = 0;
    return false;
}

bool ConvergenceMonitor::detect_divergence(double current_objective) {
    if (objective_history_.empty()) {
        return false;
    }
    
    double old_value = objective_history_.back();
    if (current_objective > 10.0 * old_value && current_objective > old_value) {
        is_diverging_ = true;
        return true;
    }
    
    return false;
}

// ============== Вспомогательные функции ==============

std::string ConvergenceMonitor::stop_reason_to_string(StopReason reason) {
    switch (reason) {
        case StopReason::NOT_CONVERGED: return "NOT_CONVERGED";
        case StopReason::RELATIVE_OBJECTIVE_CHANGE: return "RELATIVE_OBJECTIVE_CHANGE";
        case StopReason::GRADIENT_NORM: return "GRADIENT_NORM";
        case StopReason::MAX_ITERATIONS: return "MAX_ITERATIONS";
        case StopReason::TIMEOUT: return "TIMEOUT";
        case StopReason::OSCILLATIONS: return "OSCILLATIONS";
        case StopReason::STAGNATION: return "STAGNATION";
        case StopReason::DIVERGENCE: return "DIVERGENCE";
        case StopReason::NUMERICAL_ANOMALY: return "NUMERICAL_ANOMALY";
        case StopReason::SUCCESS: return "SUCCESS";
        default: return "UNKNOWN";
    }
}

} // namespace mixed_approx
