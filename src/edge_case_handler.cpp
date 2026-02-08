#include "mixed_approximation/edge_case_handler.h"
#include "mixed_approximation/functional.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <numeric>
#include <cmath>

namespace mixed_approx {

// ============== EdgeCaseHandlingResult ==============

bool EdgeCaseHandlingResult::has_critical_errors() const {
    for (const auto& info : detected_cases) {
        if (info.level == EdgeCaseLevel::CRITICAL) {
            return true;
        }
    }
    return !errors.empty();
}

// ============== NumericalAnomalyMonitor ==============

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

EdgeCaseHandlingResult EdgeCaseHandler::get_result() const {
    EdgeCaseHandlingResult result;
    result.success = errors_.empty();
    result.detected_cases = detected_cases_;
    result.handled_cases = handled_cases_;
    result.warnings = warnings_;
    result.errors = errors_;
    return result;
}

bool EdgeCaseHandler::has_critical_errors() const {
    for (const auto& info : detected_cases_) {
        if (info.level == EdgeCaseLevel::CRITICAL) {
            return true;
        }
    }
    return !errors_.empty();
}

// ============== EdgeCaseHandler ==============

EdgeCaseHandler::EdgeCaseHandler()
    : epsilon_close_nodes(1e-8)
    , epsilon_value_spread(1e-8)
    , epsilon_machine(1e-12)
    , high_degree_threshold(30)
    , condition_limit(1e8)
    , gradient_limit(1e10)
{
}

void EdgeCaseHandler::clear() {
    detected_cases_.clear();
    handled_cases_.clear();
    warnings_.clear();
    errors_.clear();
}

void EdgeCaseHandler::add_case(const EdgeCaseInfo& info) {
    detected_cases_.push_back(info);
}

void EdgeCaseHandler::add_info(const std::string& message) {
    EdgeCaseInfo info;
    info.level = EdgeCaseLevel::WARNING;
    info.type = EdgeCaseType::NONE;
    info.message = message;
    info.is_handled = true;
    handled_cases_.push_back(info);
}

void EdgeCaseHandler::add_warning(const std::string& message) {
    EdgeCaseInfo info;
    info.level = EdgeCaseLevel::WARNING;
    info.type = EdgeCaseType::NONE;
    info.message = message;
    info.is_handled = true;
    detected_cases_.push_back(info);
    warnings_.push_back(message);
}

void EdgeCaseHandler::add_error(const std::string& message) {
    EdgeCaseInfo info;
    info.level = EdgeCaseLevel::CRITICAL;
    info.type = EdgeCaseType::NONE;
    info.message = message;
    info.is_handled = false;
    detected_cases_.push_back(info);
    errors_.push_back(message);
}

EdgeCaseLevel EdgeCaseHandler::classify_case(EdgeCaseType type) {
    switch (type) {
        case EdgeCaseType::OVERCONSTRAINED:
            return EdgeCaseLevel::CRITICAL;
        case EdgeCaseType::ZERO_NODES:
        case EdgeCaseType::FULL_INTERPOLATION:
        case EdgeCaseType::MULTIPLE_ROOTS:
            return EdgeCaseLevel::SPECIAL;
        case EdgeCaseType::CLOSE_NODES:
        case EdgeCaseType::HIGH_DEGREE:
            return EdgeCaseLevel::WARNING;
        case EdgeCaseType::DEGENERATE_DATA:
        case EdgeCaseType::CONSTANT_VALUES:
        case EdgeCaseType::LINEAR_DEPENDENCE:
            return EdgeCaseLevel::WARNING;
        case EdgeCaseType::NUMERICAL_OVERFLOW:
        case EdgeCaseType::GRADIENT_EXPLOSION:
        case EdgeCaseType::OSCILLATION:
            return EdgeCaseLevel::RECOVERABLE;
        default:
            return EdgeCaseLevel::WARNING;
    }
}

// ============== Шаг 2.1.8.2: Обработка случая m = 0 ==============

ZeroNodesResult EdgeCaseHandler::handle_zero_nodes(int n) {
    ZeroNodesResult result;
    
    result.use_trivial_parameterization = true;
    result.degree = n;
    
    std::ostringstream oss;
    oss << "INFO: No interpolation nodes (m = 0).\n";
    oss << "      Task reduces to regularized approximation without constraints.\n";
    oss << "      Parameterization simplified: F(x) = Q(x) with degree " << n << ".\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.3: Обработка случая m = n + 1 ==============

FullInterpolationResult EdgeCaseHandler::handle_full_interpolation(int /*n*/, int m) {
    FullInterpolationResult result;
    
    result.is_degenerate = true;
    result.n_free = 0;
    
    std::ostringstream oss;
    oss << "WARNING: Full interpolation (m = n + 1 = " << m << ").\n";
    oss << "         Solution is uniquely defined by interpolation conditions.\n";
    oss << "         Approximation and repulsion criteria are inactive.\n";
    oss << "         Correction polynomial degenerates: Q(x) = 0.\n";
    result.info_message = oss.str();
    
    std::ostringstream rec_oss;
    rec_oss << "Recommendations:\n";
    rec_oss << "  1. To include approximation data, increase polynomial degree to n >= " << m << "\n";
    rec_oss << "  2. To include repulsion, increase polynomial degree to n >= " << (m + 1) << "\n";
    rec_oss << "  3. Current solution: pure Lagrange interpolation of degree " << (m - 1) << "\n";
    result.recommendations = rec_oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.4: Обработка избыточных ограничений ==============

OverconstrainedResult EdgeCaseHandler::handle_overconstrained(int n, int m,
                                                              const std::vector<double>& interp_coords,
                                                              const std::vector<double>& /*interp_values*/,
                                                              OverconstrainedStrategy strategy) {
    OverconstrainedResult result;
    
    result.original_m = m;
    result.strategy = strategy;
    
    if (strategy == OverconstrainedStrategy::STRICT) {
        result.resolved = false;
        std::ostringstream oss;
        oss << "ERROR: Overconstrained system (m = " << m << ", n = " << n << ").\n";
        oss << "       Maximum allowed nodes for degree " << n << " is n+1 = " << (n + 1) << ".\n";
        oss << "       Recommended actions:\n";
        oss << "         1. Reduce number of nodes to " << (n + 1) << " or fewer\n";
        oss << "         2. Increase polynomial degree to " << (m - 1) << " or more\n";
        oss << "         3. Convert some nodes to approximation points with high weights\n";
        result.info_message = oss.str();
        return result;
    }
    
    // Мягкая стратегия: выбор подмножества узлов
    int max_nodes = n + 1;
    result.adapted_m = std::min(m, max_nodes);
    
    if (m <= max_nodes) {
        result.resolved = true;
        result.info_message = "No overconstraint detected";
        return result;
    }
    
    // Выбор подмножества узлов для точной интерполяции
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return interp_coords[a] < interp_coords[b];
    });
    
    std::vector<int> selected;
    double step = static_cast<double>(m - 1) / max_nodes;
    for (int i = 0; i < max_nodes; ++i) {
        int idx = static_cast<int>(i * step + 0.5);
        selected.push_back(indices[std::min(idx, m - 1)]);
    }
    
    bool has_left = false, has_right = false;
    for (int idx : selected) {
        if (idx == 0) has_left = true;
        if (idx == m - 1) has_right = true;
    }
    if (!has_left && m > 0) selected.push_back(0);
    if (!has_right && m > 0 && selected.size() < static_cast<size_t>(max_nodes)) {
        selected.push_back(m - 1);
    }
    
    if (selected.size() > static_cast<size_t>(max_nodes)) {
        selected.resize(max_nodes);
    }
    
    result.selected_indices = selected;
    result.adapted_m = static_cast<int>(selected.size());
    result.resolved = true;
    
    std::ostringstream oss;
    oss << "INFO: Overconstrained system adapted (m = " << m << " -> " << result.adapted_m << ").\n";
    oss << "       Selected " << result.adapted_m << " nodes for exact interpolation.\n";
    oss << "       Remaining " << (m - result.adapted_m) << " nodes converted to approximation points.\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.5: Обработка близких узлов ==============

std::vector<NodeCluster> EdgeCaseHandler::cluster_close_nodes(const std::vector<double>& coords,
                                                              const std::vector<double>& values,
                                                              double epsilon) {
    std::vector<NodeCluster> clusters;
    
    if (coords.empty()) return clusters;
    
    int n = static_cast<int>(coords.size());
    std::vector<bool> visited(n, false);
    
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        
        NodeCluster cluster;
        cluster.indices.push_back(i);
        visited[i] = true;
        
        double sum_x = coords[i];
        double sum_val = values[i];
        
        for (int j = i + 1; j < n; ++j) {
            if (visited[j]) continue;
            
            double dist = std::abs(coords[j] - coords[i]);
            if (dist < epsilon) {
                cluster.indices.push_back(j);
                visited[j] = true;
                sum_x += coords[j];
                sum_val += values[j];
            }
        }
        
        cluster.center = sum_x / cluster.indices.size();
        cluster.value_center = sum_val / cluster.indices.size();
        
        double max_diff = 0.0;
        for (int idx : cluster.indices) {
            double diff = std::abs(values[idx] - cluster.value_center);
            max_diff = std::max(max_diff, diff);
        }
        cluster.value_spread = max_diff;
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

bool EdgeCaseHandler::should_merge_cluster(const NodeCluster& cluster, double epsilon_value) {
    return cluster.value_spread < epsilon_value * std::max(1.0, std::abs(cluster.value_center));
}

CloseNodesResult EdgeCaseHandler::handle_close_nodes(const std::vector<double>& coords,
                                                     const std::vector<double>& values,
                                                     double interval_length) {
    CloseNodesResult result;
    
    result.original_m = static_cast<int>(coords.size());
    
    if (coords.empty()) {
        result.has_close_nodes = false;
        result.effective_m = 0;
        result.info_message = "No nodes to process";
        return result;
    }
    
    // Вычисляем минимальное расстояние между узлами
    result.min_distance = std::numeric_limits<double>::max();
    for (int i = 1; i < result.original_m; ++i) {
        double dist = std::abs(coords[i] - coords[i-1]);
        result.min_distance = std::min(result.min_distance, dist);
    }
    
    // Определяем порог близости
    double close_threshold = epsilon_close_nodes * interval_length;
    
    if (result.min_distance >= close_threshold) {
        result.has_close_nodes = false;
        result.effective_m = result.original_m;
        result.info_message = "No close nodes detected";
        return result;
    }
    
    result.has_close_nodes = true;
    
    // Объединяем близкие узлы в кластеры
    result.clusters = cluster_close_nodes(coords, values, close_threshold);
    
    double value_threshold = epsilon_value_spread * std::max(1.0, 
        *std::max_element(values.begin(), values.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }));
    
    int merged_count = 0;
    for (auto& cluster : result.clusters) {
        if (should_merge_cluster(cluster, value_threshold)) {
            cluster.should_merge = true;
            merged_count++;
        }
    }
    
    result.effective_m = result.original_m;
    for (const auto& cluster : result.clusters) {
        if (cluster.should_merge) {
            result.effective_m -= (static_cast<int>(cluster.indices.size()) - 1);
        }
    }
    
    std::ostringstream oss;
    oss << "INFO: Close nodes detected (min distance = " << std::scientific 
        << std::setprecision(2) << (result.min_distance / interval_length) << " of interval).\n";
    oss << "      " << merged_count << " cluster(s) with value spread below threshold.\n";
    oss << "      Effective node count reduced from " << result.original_m 
        << " to " << result.effective_m << ".\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.6: Обработка высокой степени ==============

std::vector<std::string> EdgeCaseHandler::generate_high_degree_recommendations(int n) {
    std::vector<std::string> recs;
    
    std::ostringstream oss;
    oss << "WARNING: High polynomial degree (n = " << n << ").\n";
    oss << "         Risk of numerical instability and Runge oscillations.\n";
    recs.push_back(oss.str());
    
    recs.push_back("Recommended alternatives:");
    recs.push_back("  - Use cubic splines (recommended)");
    recs.push_back("  - Use B-splines of degree 3-5");
    recs.push_back("  - Split interval into segments and use local polynomials");
    
    return recs;
}

HighDegreeResult EdgeCaseHandler::handle_high_degree(int n) {
    HighDegreeResult result;
    
    result.original_degree = n;
    
    if (n <= high_degree_threshold) {
        result.requires_adaptation = false;
        result.switch_to_chebyshev = false;
        result.use_long_double = false;
        result.suggest_splines = false;
        result.recommended_degree = n;
        return result;
    }
    
    result.requires_adaptation = true;
    
    if (n > 15) {
        result.switch_to_chebyshev = true;
    }
    
    if (n > 40) {
        result.suggest_splines = true;
    }
    
    result.recommended_degree = std::min(n, 30);
    result.recommendations = generate_high_degree_recommendations(n);
    
    return result;
}

// ============== Шаг 2.1.8.7: Анализ вырожденных данных ==============

double EdgeCaseHandler::compute_value_spread(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    auto min_max = std::minmax_element(values.begin(), values.end());
    return *(min_max.second) - *(min_max.first);
}

bool EdgeCaseHandler::is_constant(const std::vector<double>& values, double threshold) {
    if (values.empty()) return true;
    
    double spread = compute_value_spread(values);
    double max_abs = *std::max_element(values.begin(), values.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    
    double reference = std::max(max_abs, 1.0);
    return spread < threshold * reference;
}

DegeneracyResult EdgeCaseHandler::analyze_degeneracy(const std::vector<double>& values) {
    DegeneracyResult result;
    
    if (values.empty()) {
        result.is_degenerate = false;
        result.type = DegeneracyType::NONE;
        result.info_message = "Empty values - no degeneracy";
        return result;
    }
    
    // Проверка на константные значения
    if (is_constant(values, epsilon_machine)) {
        result.is_degenerate = true;
        result.type = DegeneracyType::CONSTANT;
        result.constant_value = values[0];
        result.effective_degree = 0;
        
        std::ostringstream oss;
        oss << "INFO: Constant interpolation values detected (value = " << values[0] << ").\n";
        oss << "      Basis polynomial simplifies to constant.\n";
        oss << "      First and second derivatives are zero.\n";
        result.info_message = oss.str();
        return result;
    }
    
    // Проверка на линейную зависимость
    if (values.size() >= 3) {
        double slope1 = (values[1] - values[0]);
        double slope2 = (values[2] - values[1]);
        
        if (std::abs(slope1 - slope2) < 1e-6) {
            result.is_degenerate = true;
            result.type = DegeneracyType::LINEAR;
            result.effective_degree = 1;
            
            std::ostringstream oss;
            oss << "INFO: Linear dependence detected in interpolation values.\n";
            oss << "      Effective degree reduced to 1.\n";
            result.info_message = oss.str();
            return result;
        }
    }
    
    result.is_degenerate = false;
    result.type = DegeneracyType::NONE;
    result.info_message = "No degeneracy detected";
    return result;
}

// ============== Основной метод обработки всех случаев ==============

EdgeCaseHandlingResult EdgeCaseHandler::handle_all_cases(int n, int m,
                                                         const std::vector<double>& interp_values,
                                                         const ApproximationConfig& config) {
    EdgeCaseHandlingResult result;
    result.success = true;
    
    int max_nodes = n + 1;
    double interval_length = config.interval_end - config.interval_start;
    
    // Случай 1: m = 0 (отсутствие интерполяционных узлов)
    if (m == 0) {
        ZeroNodesResult zero_result = handle_zero_nodes(n);
        add_info(zero_result.info_message);
        
        EdgeCaseInfo info;
        info.level = EdgeCaseLevel::SPECIAL;
        info.type = EdgeCaseType::ZERO_NODES;
        info.message = "No interpolation nodes";
        info.recommendation = "Using simplified parameterization F(x) = Q(x)";
        info.is_handled = true;
        add_case(info);
        
        result.adapted_m = 0;
        result.adapted_n = n;
        result.parameters_modified = true;
    }
    
    // Случай 2: m = n + 1 (полная интерполяция)
    if (m == max_nodes) {
        FullInterpolationResult full_result = handle_full_interpolation(n, m);
        add_warning(full_result.info_message);
        
        EdgeCaseInfo info;
        info.level = EdgeCaseLevel::SPECIAL;
        info.type = EdgeCaseType::FULL_INTERPOLATION;
        info.message = "Full interpolation case";
        info.recommendation = full_result.recommendations;
        info.is_handled = true;
        add_case(info);
        
        result.adapted_m = m;
        result.adapted_n = n;
    }
    
    // Случай 3: m > n + 1 (избыточные ограничения) - используем STRICT стратегию
    if (m > max_nodes) {
        std::vector<double> interp_coords;
        for (const auto& node : config.interp_nodes) {
            interp_coords.push_back(node.x);
        }
        
        OverconstrainedResult over_result = handle_overconstrained(
            n, m, interp_coords, interp_values,
            OverconstrainedStrategy::STRICT);
        
        if (!over_result.resolved) {
            add_error(over_result.info_message);
            result.success = false;
        } else {
            add_warning(over_result.info_message);
            EdgeCaseInfo info;
            info.level = EdgeCaseLevel::SPECIAL;
            info.type = EdgeCaseType::OVERCONSTRAINED;
            info.message = "Overconstrained system adapted";
            info.recommendation = "Selected subset of nodes";
            info.is_handled = true;
            add_case(info);
            
            result.adapted_m = over_result.adapted_m;
            result.adapted_n = n;
            result.parameters_modified = true;
        }
    }
    
    // Случай 4: Близкие узлы
    if (!config.interp_nodes.empty()) {
        std::vector<double> coords, values;
        for (const auto& node : config.interp_nodes) {
            coords.push_back(node.x);
            values.push_back(node.value);
        }
        
        CloseNodesResult close_result = handle_close_nodes(coords, values, interval_length);
        
        if (close_result.has_close_nodes) {
            add_warning(close_result.info_message);
            EdgeCaseInfo info;
            info.level = EdgeCaseLevel::WARNING;
            info.type = EdgeCaseType::CLOSE_NODES;
            info.message = "Close interpolation nodes detected";
            info.recommendation = "Clusters merged based on value spread";
            info.is_handled = true;
            add_case(info);
            
            if (result.adapted_m == 0) {
                result.adapted_m = close_result.effective_m;
            }
        }
    }
    
    // Случай 5: Высокая степень полинома
    if (n > high_degree_threshold) {
        HighDegreeResult high_result = handle_high_degree(n);
        
        for (const auto& rec : high_result.recommendations) {
            add_warning(rec);
        }
        
        EdgeCaseInfo info;
        info.level = EdgeCaseLevel::WARNING;
        info.type = EdgeCaseType::HIGH_DEGREE;
        info.message = "High polynomial degree";
        info.recommendation = "Consider using Chebyshev basis or splines";
        info.is_handled = true;
        add_case(info);
    }
    
    // Случай 6: Вырожденные данные
    if (!interp_values.empty()) {
        DegeneracyResult deg_result = analyze_degeneracy(interp_values);
        
        if (deg_result.is_degenerate) {
            add_info(deg_result.info_message);
            
            EdgeCaseType edge_type = (deg_result.type == DegeneracyType::CONSTANT) ?
                EdgeCaseType::CONSTANT_VALUES : EdgeCaseType::LINEAR_DEPENDENCE;
            
            EdgeCaseInfo info;
            info.level = EdgeCaseLevel::WARNING;
            info.type = edge_type;
            info.message = (deg_result.type == DegeneracyType::CONSTANT) ? 
                           "Constant values" : "Linear dependence";
            info.recommendation = deg_result.recommendations;
            info.is_handled = true;
            add_case(info);
        }
    }
    
    // Заполняем warnings и errors в результате
    result.warnings = warnings_;
    result.errors = errors_;
    
    // Проверка на критические ошибки
    if (result.has_critical_errors()) {
        result.success = false;
    }
    
    return result;
}

// ============== Адаптация параметризации ==============

bool EdgeCaseHandler::adapt_parameterization(EdgeCaseHandlingResult& /*result*/,
                                             ApproximationConfig& /*config*/,
                                             InterpolationBasis& /*basis*/,
                                             WeightMultiplier& /*weight*/,
                                             CorrectionPolynomial& correction) {
    bool success = true;
    
    // Адаптация для случая высокой степени
    if (correction.degree > 15 && correction.basis_type == BasisType::MONOMIAL) {
        add_warning("Switching to Chebyshev basis for numerical stability");
        correction.basis_type = BasisType::CHEBYSHEV;
    }
    
    return success;
}

// ============== Вспомогательные функции форматирования ==============

std::string format_edge_case_result(const EdgeCaseHandlingResult& result) {
    std::ostringstream oss;
    
    oss << "=================================================\n";
    oss << "Edge Case Handling Report\n";
    oss << "=================================================\n\n";
    
    oss << "Status: " << (result.success ? "SUCCESS" : "FAILURE") << "\n\n";
    
    if (!result.detected_cases.empty()) {
        oss << "Detected Cases:\n";
        for (size_t i = 0; i < result.detected_cases.size(); ++i) {
            const auto& info = result.detected_cases[i];
            oss << "  " << (i + 1) << ". [" << static_cast<int>(info.level) << "] "
                << info.message << "\n";
            if (!info.recommendation.empty()) {
                oss << "     Rec: " << info.recommendation << "\n";
            }
        }
        oss << "\n";
    }
    
    if (!result.warnings.empty()) {
        oss << "Warnings:\n";
        for (const auto& w : result.warnings) {
            oss << "  - " << w << "\n";
        }
        oss << "\n";
    }
    
    if (!result.errors.empty()) {
        oss << "Errors:\n";
        for (const auto& e : result.errors) {
            oss << "  - " << e << "\n";
        }
        oss << "\n";
    }
    
    if (result.parameters_modified) {
        oss << "Parameters Modified: YES\n";
        oss << "  - Original m: " << (result.adapted_m > 0 ? "varies" : "N/A") << "\n";
        oss << "  - Adapted m: " << result.adapted_m << "\n";
    }
    
    oss << "=================================================\n";
    
    return oss.str();
}

std::string format_zero_nodes_result(const ZeroNodesResult& result) {
    return result.info_message;
}

std::string format_full_interpolation_result(const FullInterpolationResult& result) {
    std::ostringstream oss;
    oss << result.info_message << "\n\n" << result.recommendations;
    return oss.str();
}

std::string format_overconstrained_result(const OverconstrainedResult& result) {
    std::ostringstream oss;
    oss << result.info_message;
    if (result.resolved && !result.selected_indices.empty()) {
        oss << "\nSelected node indices: ";
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), result.selected_indices.size()); ++i) {
            oss << result.selected_indices[i] << " ";
        }
        if (result.selected_indices.size() > 5) {
            oss << "... (total " << result.selected_indices.size() << ")";
        }
    }
    return oss.str();
}

std::string format_close_nodes_result(const CloseNodesResult& result) {
    return result.info_message;
}

std::string format_high_degree_result(const HighDegreeResult& result) {
    std::ostringstream oss;
    oss << "High Degree Analysis:\n";
    oss << "  - Original degree: " << result.original_degree << "\n";
    oss << "  - Requires adaptation: " << (result.requires_adaptation ? "YES" : "NO") << "\n";
    oss << "  - Switch to Chebyshev: " << (result.switch_to_chebyshev ? "YES" : "NO") << "\n";
    oss << "  - Suggest splines: " << (result.suggest_splines ? "YES" : "NO") << "\n";
    oss << "  - Recommended degree: " << result.recommended_degree << "\n";
    
    if (!result.recommendations.empty()) {
        oss << "\nRecommendations:\n";
        for (const auto& rec : result.recommendations) {
            oss << rec << "\n";
        }
    }
    
    return oss.str();
}

std::string format_degeneracy_result(const DegeneracyResult& result) {
    std::ostringstream oss;
    oss << "Degeneracy Analysis:\n";
    oss << "  - Is degenerate: " << (result.is_degenerate ? "YES" : "NO") << "\n";
    oss << "  - Type: ";
    switch (result.type) {
        case DegeneracyType::NONE: oss << "None"; break;
        case DegeneracyType::CONSTANT: oss << "Constant values"; break;
        case DegeneracyType::LINEAR: oss << "Linear dependence"; break;
        case DegeneracyType::RANK_DEFICIENT: oss << "Rank deficient"; break;
    }
    oss << "\n";
    
    if (result.is_degenerate) {
        oss << "  - Info: " << result.info_message << "\n";
    }
    
    return oss.str();
}

} // namespace mixed_approx
