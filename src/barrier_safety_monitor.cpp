#include "mixed_approximation/barrier_safety_monitor.h"
#include "mixed_approximation/composite_polynomial.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>

namespace mixed_approx {

BarrierSafetyConfig BarrierSafetyConfig::create_adaptive(const ApproximationConfig& config) {
    BarrierSafetyConfig cfg;
    
    // Оценка характерного масштаба
    double max_func_value = 0.0;
    double max_forbidden = 0.0;
    
    for (const auto& pt : config.repel_points) {
        max_forbidden = std::max(max_forbidden, std::abs(pt.y_forbidden));
    }
    
    double scale_y = std::max(1.0, 0.1 * max_forbidden);
    
    // Адаптивные пороги
    cfg.epsilon_critical_base = std::max(1e-8, cfg.scale_factor_small * scale_y);
    cfg.epsilon_warning_base = std::max(1e-4, cfg.scale_factor_large * scale_y);
    
    return cfg;
}

BarrierSafetyMonitor::BarrierSafetyMonitor(const ApproximationConfig& config,
                                           const BarrierSafetyConfig& safety_config)
    : config_(config)
    , safety_config_(safety_config)
    , epsilon_critical_(safety_config.epsilon_critical_base)
    , epsilon_warning_(safety_config.epsilon_warning_base)
    , collapse_count_(0)
    , recovery_attempts_(0)
{
    // Инициализация отталкивающих точек из конфигурации
    for (const auto& pt : config.repel_points) {
        repel_points_.push_back(pt);
    }
    
    initialize_barrier_states();
}

BarrierSafetyMonitor::BarrierSafetyMonitor(const std::vector<RepulsionPoint>& repel_points,
                                           const BarrierSafetyConfig& safety_config)
    : config_(*static_cast<const ApproximationConfig*>(nullptr))
    , safety_config_(safety_config)
    , repel_points_(repel_points)
    , epsilon_critical_(safety_config.epsilon_critical_base)
    , epsilon_warning_(safety_config.epsilon_warning_base)
    , collapse_count_(0)
    , recovery_attempts_(0)
{
    initialize_barrier_states();
}

void BarrierSafetyMonitor::reset() {
    epsilon_critical_ = safety_config_.epsilon_critical_base;
    epsilon_warning_ = safety_config_.epsilon_warning_base;
    
    distance_history_.clear();
    functional_history_.clear();
    events_.clear();
    
    safe_parameters_.clear();
    
    collapse_count_ = 0;
    recovery_attempts_ = 0;
    
    initialize_barrier_states();
}

void BarrierSafetyMonitor::initialize_barrier_states() {
    barrier_states_.clear();
    barrier_states_.resize(repel_points_.size());
    
    for (size_t i = 0; i < repel_points_.size(); ++i) {
        barrier_states_[i].weight = repel_points_[i].weight;
        barrier_states_[i].zone = BarrierZone::SAFE;
        barrier_states_[i].iterations_in_zone = 0;
        barrier_states_[i].approaching = false;
        barrier_states_[i].current_distance = epsilon_warning_ * 10;
        barrier_states_[i].min_distance_10it = epsilon_warning_ * 10;
    }
}

double BarrierSafetyMonitor::compute_characteristic_scale() const {
    double max_abs = 0.0;
    
    for (const auto& pt : repel_points_) {
        max_abs = std::max(max_abs, std::abs(pt.y_forbidden));
    }
    
    return std::max(1.0, 0.1 * max_abs);
}

void BarrierSafetyMonitor::update_adaptive_thresholds() {
    if (distance_history_.empty()) return;
    
    // Находим минимум расстояний за последние history_size итераций
    double min_distance_overall = std::numeric_limits<double>::infinity();
    
    for (const auto& distances : distance_history_) {
        for (double d : distances) {
            if (d > 0) { // игнорируем нулевые расстояния
                min_distance_overall = std::min(min_distance_overall, d);
            }
        }
    }
    
    if (min_distance_overall < std::numeric_limits<double>::infinity()) {
        // Если приближаемся к барьерам, увеличиваем буфер безопасности
        epsilon_critical_ = std::max(epsilon_critical_, 2.0 * min_distance_overall);
        epsilon_warning_ = std::max(epsilon_warning_, 10.0 * min_distance_overall);
    }
}

BarrierZone BarrierSafetyMonitor::classify_zone(double distance) const {
    if (distance <= epsilon_critical_) {
        return BarrierZone::CRITICAL;
    } else if (distance <= epsilon_warning_) {
        return BarrierZone::WARNING;
    } else {
        return BarrierZone::SAFE;
    }
}

double BarrierSafetyMonitor::compute_danger_score(const std::vector<double>& distances,
                                                   const std::vector<double>& weights) const {
    if (distances.empty() || weights.empty()) return 0.0;
    
    double total_weight = 0.0;
    double danger_sum = 0.0;
    
    for (size_t i = 0; i < distances.size(); ++i) {
        double effective_dist = std::max(distances[i], epsilon_critical_);
        danger_sum += weights[i] / effective_dist;
        total_weight += weights[i];
    }
    
    if (total_weight == 0.0) return 0.0;
    
    return danger_sum / total_weight / epsilon_critical_;
}

double BarrierSafetyMonitor::compute_smoothed_barrier(double distance, double weight, BarrierZone& zone) {
    zone = classify_zone(distance);
    
    double result;
    double safe_distance = std::max(distance, epsilon_critical_);
    
    switch (zone) {
        case BarrierZone::CRITICAL:
            // Квадратично-сглаженный барьер для критической зоны
            if (safety_config_.use_quadratic_smoothing) {
                result = weight / (epsilon_critical_ * epsilon_critical_ + 
                                   safety_config_.smoothing_k * 
                                   std::pow(epsilon_critical_ - distance, 2));
            } else {
                // Логарифмический барьер как альтернатива
                result = -weight * std::log(safe_distance);
            }
            break;
            
        case BarrierZone::WARNING:
            // Интерполяция между сглаженным и стандартным барьером
            if (safety_config_.use_quadratic_smoothing) {
                double alpha = (distance - epsilon_critical_) / 
                              (epsilon_warning_ - epsilon_critical_);
                double smoothed_at_boundary = weight / 
                    (epsilon_critical_ * epsilon_critical_ * (1.0 + safety_config_.smoothing_k));
                
                result = weight * (alpha / (distance * distance) + 
                                   (1.0 - alpha) / smoothed_at_boundary);
            } else {
                result = weight / (distance * distance);
            }
            break;
            
        case BarrierZone::SAFE:
        default:
            // Стандартная формула
            result = weight / (safe_distance * safe_distance);
            break;
    }
    
    return result;
}

BarrierSafetyResult BarrierSafetyMonitor::check_safety(const std::vector<double>& distances,
                                                        const std::vector<double>& weights,
                                                        double functional_value) {
    BarrierSafetyResult result;
    
    if (distances.empty()) {
        return result;
    }
    
    result.barrier_states.resize(distances.size());
    
    // Обновление истории
    if (static_cast<int>(distance_history_.size()) >= safety_config_.history_size) {
        distance_history_.pop_front();
    }
    distance_history_.push_back(distances);
    
    // Обновление адаптивных порогов
    update_adaptive_thresholds();
    
    // Подсчёт барьеров по зонам
    int critical_count = 0;
    int warning_count = 0;
    double min_distance = std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < distances.size(); ++i) {
        BarrierZone zone = classify_zone(distances[i]);
        
        result.barrier_states[i].current_distance = distances[i];
        result.barrier_states[i].zone = zone;
        result.barrier_states[i].weight = (i < weights.size()) ? weights[i] : 0.0;
        
        // Отслеживание сближения
        if (!barrier_states_.empty() && i < barrier_states_.size()) {
            result.barrier_states[i].approaching = 
                distances[i] < barrier_states_[i].current_distance;
            result.barrier_states[i].iterations_in_zone = 
                (zone == barrier_states_[i].zone) ? 
                barrier_states_[i].iterations_in_zone + 1 : 0;
        }
        
        if (zone == BarrierZone::CRITICAL) {
            critical_count++;
        } else if (zone == BarrierZone::WARNING) {
            warning_count++;
        }
        
        min_distance = std::min(min_distance, distances[i]);
    }
    
    // Сохранение состояний
    barrier_states_ = result.barrier_states;
    
    result.critical_count = critical_count;
    result.warning_count = warning_count;
    
    // Вычисление оценки опасности
    result.danger_score = compute_danger_score(distances, weights);
    
    // Определение коэффициента демпфирования
    if (result.danger_score > safety_config_.danger_threshold_critical) {
        result.step_damping_factor = safety_config_.step_damping_critical;
        result.is_safe = false;
        result.step_rejection_recommended = (min_distance < 0.1 * epsilon_critical_);
    } else if (result.danger_score > safety_config_.danger_threshold_warning) {
        result.step_damping_factor = safety_config_.step_damping_warning;
        result.is_safe = true;
        result.step_rejection_recommended = false;
    } else {
        result.step_damping_factor = 1.0;
        result.is_safe = true;
        result.step_rejection_recommended = false;
    }
    
    // Проверка на необходимость восстановления
    result.recovery_required = 
        (critical_count > 0 && min_distance < 0.5 * epsilon_critical_);
    
    // Обновление истории функционала
    if (functional_history_.size() >= static_cast<size_t>(safety_config_.history_size)) {
        functional_history_.pop_front();
    }
    functional_history_.push_back(functional_value);
    
    return result;
}

std::vector<double> BarrierSafetyMonitor::protect_gradient(const std::vector<double>& raw_gradient,
                                                            const std::vector<double>& distances) {
    std::vector<double> protected_gradient = raw_gradient;
    
    if (raw_gradient.empty()) return protected_gradient;
    
    // Нормализация по компонентам
    double max_component = 0.0;
    for (double g : raw_gradient) {
        max_component = std::max(max_component, std::abs(g));
    }
    
    double max_allowed = safety_config_.gradient_max_per_component;
    
    if (max_component > max_allowed) {
        for (double& g : protected_gradient) {
            g = (g >= 0) ? std::min(g, max_allowed) : std::max(g, -max_allowed);
        }
    }
    
    // Нормализация вектора градиента
    double norm = 0.0;
    for (double g : protected_gradient) {
        norm += g * g;
    }
    norm = std::sqrt(norm);
    
    double max_norm = safety_config_.gradient_max_norm;
    
    if (norm > max_norm && norm > 0) {
        double scale = max_norm / norm;
        for (double& g : protected_gradient) {
            g *= scale;
        }
    }
    
    // Адаптивное ограничение на основе истории
    // Если норма слишком велика по сравнению с историей, применяем смягчение
    double max_history_norm = 0.0;
    // Здесь могла бы быть логика анализа истории градиентов
    
    if (max_history_norm > 0) {
        double adaptive_limit = std::max(1e4, safety_config_.gradient_adaptive_factor * max_history_norm);
        if (norm > adaptive_limit) {
            double damp_factor = adaptive_limit / norm;
            for (double& g : protected_gradient) {
                g *= damp_factor;
            }
        }
    }
    
    return protected_gradient;
}

double BarrierSafetyMonitor::get_step_damping_factor() const {
    return 1.0; // Базовое значение, будет обновлено в check_safety
}

bool BarrierSafetyMonitor::detect_collapse(double current_functional,
                                           double previous_functional,
                                           const std::vector<double>& distances) {
    // Признак 1: Резкий скачок функционала
    bool functional_collapse = false;
    if (previous_functional > 0 && current_functional / previous_functional >= 
        safety_config_.collapse_functional_ratio) {
        functional_collapse = true;
    }
    
    // Признак 2: NaN или Inf
    bool nan_inf = std::isnan(current_functional) || std::isinf(current_functional);
    
    // Признак 3: Расстояние до барьера слишком мало
    bool distance_collapse = false;
    for (double d : distances) {
        if (d < safety_config_.collapse_distance_ratio * epsilon_critical_) {
            distance_collapse = true;
            break;
        }
    }
    
    return functional_collapse || nan_inf || distance_collapse;
}

BarrierRecoveryResult BarrierSafetyMonitor::recover_from_collapse(
    const std::vector<double>& current_params,
    const std::vector<double>& safe_params,
    const std::vector<double>& distances,
    const std::vector<double>& weights)
{
    BarrierRecoveryResult result;
    
    if (recovery_attempts_ >= safety_config_.max_recovery_attempts) {
        result.message = "Достигнут максимум попыток восстановления";
        result.success = false;
        return result;
    }
    
    recovery_attempts_++;
    collapse_count_++;
    
    // Логирование события
    BarrierEvent event;
    event.distance = distances.empty() ? 0.0 : *std::min_element(distances.begin(), distances.end());
    event.functional_value = current_params.empty() ? 0.0 : current_params[0];
    log_event(event);
    
    // Шаг 1: Откат к безопасным параметрам, если они есть
    if (!safe_params.empty()) {
        result.rollback_performed = true;
        result.corrected_params = safe_params;
    } else {
        // Шаг 2: Вычисляем направление "от барьера"
        std::vector<double> escape_dir = compute_escape_direction(distances, weights);
        
        if (!escape_dir.empty() && !current_params.empty()) {
            double min_distance = epsilon_warning_;
            for (double d : distances) {
                min_distance = std::min(min_distance, d);
            }
            
            double escape_step = 0.01 * std::min(epsilon_warning_, 
                (current_params.empty() ? 1.0 : std::abs(current_params[0])));
            
            result.corrected_params.resize(current_params.size());
            for (size_t i = 0; i < current_params.size(); ++i) {
                if (escape_dir.empty()) {
                    result.corrected_params[i] = current_params[i];
                } else {
                    result.corrected_params[i] = current_params[i] + 
                        escape_step * escape_dir[i];
                }
            }
            
            // Нормализация направления
            double norm = 0.0;
            for (double v : escape_dir) {
                norm += v * v;
            }
            norm = std::sqrt(norm);
            
            if (norm > 0) {
                for (double& v : result.corrected_params) {
                    v = std::max(-1e6, std::min(1e6, v)); // Ограничение значений
                }
            }
            
            result.escaped_distance = min_distance;
        } else {
            result.corrected_params = current_params;
        }
    }
    
    result.success = true;
    result.message = "Восстановление выполнено успешно";
    
    return result;
}

std::vector<double> BarrierSafetyMonitor::compute_escape_direction(
    const std::vector<double>& distances,
    const std::vector<double>& weights)
{
    std::vector<double> direction;
    
    if (barrier_states_.empty() || repel_points_.empty()) {
        return direction;
    }
    
    // Направление от ближайших барьеров
    double sum_weights = 0.0;
    direction.assign(repel_points_.size(), 0.0);
    
    for (size_t i = 0; i < distances.size(); ++i) {
        if (distances[i] < epsilon_warning_) {
            // sign(δ) показывает направление от барьера
            // Мы используем вес барьера как интенсивность
            direction[i] = (distances[i] >= 0 ? 1.0 : -1.0) * 
                          ((i < weights.size()) ? weights[i] : 1.0);
            sum_weights += std::abs(direction[i]);
        }
    }
    
    return direction;
}

bool BarrierSafetyMonitor::apply_preventive_correction(std::vector<double>& distances,
                                                        const std::vector<double>& weights,
                                                        double scale_y) {
    bool corrected = false;
    
    double safety_threshold = 5.0 * epsilon_critical_;
    
    for (size_t i = 0; i < distances.size(); ++i) {
        if (distances[i] < safety_threshold) {
            // Увеличиваем расстояние до безопасного уровня
            double correction = safety_threshold - distances[i];
            distances[i] = safety_threshold;
            corrected = true;
        }
    }
    
    return corrected;
}

void BarrierSafetyMonitor::update_history(const std::vector<double>& distances) {
    if (static_cast<int>(distance_history_.size()) >= safety_config_.history_size) {
        distance_history_.pop_front();
    }
    distance_history_.push_back(distances);
    
    // Обновление минимумов для каждого барьера
    for (size_t i = 0; i < barrier_states_.size() && i < distances.size(); ++i) {
        double min_d = distances[i];
        for (const auto& history : distance_history_) {
            if (i < history.size()) {
                min_d = std::min(min_d, history[i]);
            }
        }
        barrier_states_[i].min_distance_10it = min_d;
    }
}

std::pair<double, double> BarrierSafetyMonitor::get_adaptive_thresholds() const {
    return {epsilon_critical_, epsilon_warning_};
}

void BarrierSafetyMonitor::log_event(const BarrierEvent& event) {
    if (!safety_config_.enable_logging) return;
    
    BarrierEvent logged_event = event;
    logged_event.protection = BarrierProtectionType::STEP_DAMPING;
    logged_event.damping_factor = get_step_damping_factor();
    
    events_.push_back(logged_event);
}

std::string BarrierSafetyMonitor::generate_report() const {
    std::ostringstream oss;
    
    oss << "=== Barrier Safety Monitor Report ===\n";
    oss << "Thresholds: critical=" << std::scientific << epsilon_critical_
        << ", warning=" << epsilon_warning_ << "\n";
    oss << "Total barriers: " << repel_points_.size() << "\n";
    int critical_count = barrier_states_.empty() ? 0 : 
        static_cast<int>(std::count_if(barrier_states_.begin(), barrier_states_.end(),
                      [](const BarrierState& s) { return s.zone == BarrierZone::CRITICAL; }));
    int warning_count = barrier_states_.empty() ? 0 : 
        static_cast<int>(std::count_if(barrier_states_.begin(), barrier_states_.end(),
                      [](const BarrierState& s) { return s.zone == BarrierZone::WARNING; }));
    
    oss << "Critical barriers: " << critical_count << "\n";
    oss << "Warning barriers: " << warning_count << "\n";
    oss << "Collapse count: " << collapse_count_ << "\n";
    oss << "Recovery attempts: " << recovery_attempts_ << "\n";
    oss << "Total events logged: " << events_.size() << "\n";
    
    if (!events_.empty()) {
        oss << "\nLast 5 events:\n";
        int count = std::min(5, static_cast<int>(events_.size()));
        for (int i = events_.size() - count; i < static_cast<int>(events_.size()); ++i) {
            const auto& e = events_[i];
            oss << "  Iter " << e.iteration << ": dist=" << std::scientific << e.distance
                << ", zone=" << static_cast<int>(e.zone) << "\n";
        }
    }
    
    return oss.str();
}

bool BarrierSafetyMonitor::has_critical_warning() const {
    // Проверяем, есть ли устойчивые проблемы (>3 коллапса за 100 итераций)
    int recent_collapses = 0;
    int window = std::min(100, static_cast<int>(events_.size()));
    
    for (int i = events_.size() - window; i < static_cast<int>(events_.size()); ++i) {
        if (events_[i].zone == BarrierZone::CRITICAL) {
            recent_collapses++;
        }
    }
    
    return recent_collapses > 3;
}

std::vector<std::string> BarrierSafetyMonitor::get_recommendations() const {
    std::vector<std::string> recs;
    
    if (has_critical_warning()) {
        recs.push_back("Рекомендация: Уменьшить вес B_j проблемных барьеров");
        recs.push_back("Рекомендация: Увеличить минимальное расстояние epsilon_critical");
        recs.push_back("Рекомендация: Рассмотреть альтернативную постановку задачи");
    }
    
    if (collapse_count_ > 0) {
        recs.push_back("Обнаружены коллапсы: рекомендуется проверить данные");
    }
    
    return recs;
}

void BarrierSafetyMonitor::set_safe_parameters(const std::vector<double>& params) {
    safe_parameters_ = params;
}

} // namespace mixed_approx
