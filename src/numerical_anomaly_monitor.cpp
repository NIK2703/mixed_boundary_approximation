#include "mixed_approximation/numerical_anomaly_monitor.h"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace mixed_approx {

NumericalAnomalyMonitor::NumericalAnomalyMonitor()
    : gradient_threshold(1e10)
    , oscillation_threshold(1e-6)
    , step_reduction_factor(0.5)
    , max_consecutive_anomalies(3)
    , initial_gradient_norm_(0.0)
    , initial_gradient_set_(false)
{
}

void NumericalAnomalyMonitor::reset() {
    anomaly_history_.clear();
    value_history_.clear();
    initial_gradient_set_ = false;
}

void NumericalAnomalyMonitor::add_anomaly_event(const AnomalyEvent& event) {
    anomaly_history_.push_back(event);
    if (event.type != AnomalyType::NONE) {
        value_history_.push_back(event.value);
    }
}

AnomalyMonitorResult NumericalAnomalyMonitor::check_anomaly(int iteration,
                                                           double current_value,
                                                           double gradient_norm,
                                                           const std::vector<double>& params,
                                                           const std::vector<double>& prev_params) {
    AnomalyMonitorResult result;
    result.total_iterations = iteration + 1;
    
    // Проверка на переполнение
    AnomalyType overflow_type = check_overflow(current_value);
    if (overflow_type != AnomalyType::NONE) {
        AnomalyEvent event;
        event.type = overflow_type;
        event.iteration = iteration;
        event.value = current_value;
        event.description = "Overflow or invalid value detected";
        add_anomaly_event(event);
        result.anomaly_detected = true;
        result.last_anomaly_type = overflow_type;
        result.needs_recovery = true;
        result.needs_stop = false;
        result.anomaly_iterations++;
    }
    
    // Проверка на взрыв градиента
    if (!initial_gradient_set_) {
        initial_gradient_norm_ = gradient_norm;
        initial_gradient_set_ = true;
    }
    
    AnomalyType gradient_type = check_gradient_explosion(gradient_norm, initial_gradient_norm_);
    if (gradient_type != AnomalyType::NONE) {
        AnomalyEvent event;
        event.type = gradient_type;
        event.iteration = iteration;
        event.value = gradient_norm;
        event.description = "Gradient explosion detected";
        add_anomaly_event(event);
        result.anomaly_detected = true;
        result.last_anomaly_type = gradient_type;
        result.needs_recovery = true;
        result.needs_stop = false;
        result.anomaly_iterations++;
    }
    
    // Проверка на осцилляции
    if (!prev_params.empty() && params.size() == prev_params.size()) {
        std::vector<double> prev_prev_params;
        if (value_history_.size() >= 2) {
            prev_prev_params = prev_params;
        }
        
        AnomalyType oscillation_type = check_oscillation(params, prev_params, prev_prev_params);
        if (oscillation_type != AnomalyType::NONE) {
            AnomalyEvent event;
            event.type = oscillation_type;
            event.iteration = iteration;
            event.value = std::abs(current_value - value_history_.back());
            event.description = "Parameter oscillation detected";
            add_anomaly_event(event);
            result.anomaly_detected = true;
            result.last_anomaly_type = oscillation_type;
            result.needs_recovery = true;
            result.needs_stop = false;
            result.anomaly_iterations++;
        }
    }
    
    // Проверка на необходимость остановки
    if (result.anomaly_iterations >= max_consecutive_anomalies) {
        result.needs_stop = true;
        result.needs_recovery = false;
    }
    
    result.events = anomaly_history_;
    return result;
}

AnomalyType NumericalAnomalyMonitor::check_overflow(double value) {
    if (std::isnan(value)) {
        return AnomalyType::NAN_DETECTED;
    }
    if (std::isinf(value)) {
        return AnomalyType::INF_DETECTED;
    }
    if (std::abs(value) > gradient_threshold) {
        return AnomalyType::OVERFLOW;
    }
    return AnomalyType::NONE;
}

AnomalyType NumericalAnomalyMonitor::check_gradient_explosion(double gradient_norm, double initial_gradient) {
    if (gradient_norm > gradient_threshold) {
        return AnomalyType::GRADIENT_EXPLOSION;
    }
    if (initial_gradient > 0 && gradient_norm > initial_gradient * 1e6) {
        return AnomalyType::GRADIENT_EXPLOSION;
    }
    return AnomalyType::NONE;
}

AnomalyType NumericalAnomalyMonitor::check_oscillation(const std::vector<double>& params,
                                                        const std::vector<double>& prev_params,
                                                        const std::vector<double>& /*prev_prev_params*/) {
    if (params.empty() || prev_params.empty()) {
        return AnomalyType::NONE;
    }
    
    double total_change = 0.0;
    double prev_total_change = 0.0;
    
    for (size_t i = 0; i < params.size(); ++i) {
        total_change += std::abs(params[i] - prev_params[i]);
    }
    
    if (total_change < oscillation_threshold) {
        return AnomalyType::OSCILLATION;
    }
    
    return AnomalyType::NONE;
}

} // namespace mixed_approx
