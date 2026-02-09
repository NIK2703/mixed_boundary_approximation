#include "mixed_approximation/parameterization_data.h"
#include "mixed_approximation/functional.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <numeric>

namespace mixed_approx {

// ============== Реализация InterpolationNodeSet ==============

void InterpolationNodeSet::build(const std::vector<double>& x,
                                  const std::vector<double>& y,
                                  double interval_start,
                                  double interval_end,
                                  double merge_threshold) {
    // Проверка входных данных
    if (x.size() != y.size()) {
        is_valid = false;
        validation_error = "Number of x coordinates does not match number of y values";
        return;
    }
    
    if (x.empty()) {
        is_valid = false;
        validation_error = "Empty node set";
        return;
    }
    
    // Проверка на дубликаты
    std::vector<double> x_sorted = x;
    std::vector<double> y_sorted = y;
    std::vector<size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return x[a] < x[b];
    });
    
    for (size_t i = 0; i < indices.size(); ++i) {
        x_sorted[i] = x[indices[i]];
        y_sorted[i] = y[indices[i]];
    }
    
    // Проверка дубликатов с учётом допуска
    double tol = 1e-10 * (interval_end - interval_start);
    for (size_t i = 1; i < x_sorted.size(); ++i) {
        if (std::abs(x_sorted[i] - x_sorted[i-1]) < tol) {
            std::ostringstream oss;
            oss << "Duplicate node detected at x = " << x_sorted[i];
            add_warning(oss.str());
            // Объединяем дубликаты с усреднением значений
            y_sorted[i-1] = (y_sorted[i-1] + y_sorted[i]) / 2.0;
            x_sorted.erase(x_sorted.begin() + i);
            y_sorted.erase(y_sorted.begin() + i);
            --i;
        }
    }
    
    // Сохранение данных
    x_coords = x_sorted;
    y_values = y_sorted;
    count = static_cast<int>(x_coords.size());
    
    // Вычисление параметров нормализации
    norm_center = (interval_start + interval_end) / 2.0;
    norm_scale = (interval_end - interval_start) / 2.0;
    
    // Проверка масштаба
    if (norm_scale <= 0) {
        is_valid = false;
        validation_error = "Invalid interval: interval_end must be greater than interval_start";
        return;
    }
    
    // Нормализация координат в [-1, 1]
    x_norm.resize(count);
    for (int i = 0; i < count; ++i) {
        x_norm[i] = (x_coords[i] - norm_center) / norm_scale;
    }
    
    // Вычисление минимального расстояния между узлами
    min_distance = std::numeric_limits<double>::max();
    for (int i = 1; i < count; ++i) {
        double dist = std::abs(x_coords[i] - x_coords[i-1]);
        if (dist < min_distance) {
            min_distance = dist;
        }
    }
    if (count == 1) {
        min_distance = interval_end - interval_start;
    }
    
    // Вычисление диапазона значений
    double min_val = *std::min_element(y_values.begin(), y_values.end());
    double max_val = *std::max_element(y_values.begin(), y_values.end());
    value_range = max_val - min_val;
    
    // Проверка наличия близких узлов
    double close_threshold = merge_threshold * (interval_end - interval_start);
    has_close_nodes = min_distance < close_threshold;
    
    // Объединение близких узлов при необходимости
    if (has_close_nodes) {
        detect_and_merge_close_nodes(close_threshold);
    }
    
    is_valid = true;
    validation_error = "";
}

void InterpolationNodeSet::detect_and_merge_close_nodes(double epsilon) {
    if (x_coords.size() < 2) return;
    
    std::vector<double> new_x;
    std::vector<double> new_y;
    
    int i = 0;
    while (i < count) {
        double cluster_sum = x_coords[i];
        double value_sum = y_values[i];
        int cluster_count = 1;
        
        // Объединяем все близкие узлы
        while (i + 1 < count && 
               std::abs(x_coords[i + 1] - x_coords[i]) < epsilon) {
            cluster_sum += x_coords[i + 1];
            value_sum += y_values[i + 1];
            ++cluster_count;
            ++i;
        }
        
        new_x.push_back(cluster_sum / cluster_count);
        new_y.push_back(value_sum / cluster_count);
        ++i;
    }
    
    if (new_x.size() < x_coords.size()) {
        x_coords = std::move(new_x);
        y_values = std::move(new_y);
        count = static_cast<int>(x_coords.size());
        
        // Пересчёт нормализованных координат
        x_norm.resize(count);
        for (int i = 0; i < count; ++i) {
            x_norm[i] = (x_coords[i] - norm_center) / norm_scale;
        }
        
        // Пересчёт минимального расстояния
        min_distance = std::numeric_limits<double>::max();
        for (int i = 1; i < count; ++i) {
            double dist = std::abs(x_coords[i] - x_coords[i-1]);
            min_distance = std::min(min_distance, dist);
        }
        
        std::ostringstream oss;
        oss << "Merged " << (x_coords.size() - new_x.size()) 
            << " close nodes, effective node count: " << count;
        add_warning(oss.str());
    }
}

std::string InterpolationNodeSet::get_info() const {
    std::ostringstream oss;
    oss << "InterpolationNodeSet:\n";
    oss << "  - Count: " << count << "\n";
    oss << "  - Valid: " << (is_valid ? "yes" : "no") << "\n";
    oss << "  - Min distance: " << std::scientific << std::setprecision(3) << min_distance << "\n";
    oss << "  - Value range: " << value_range << "\n";
    oss << "  - Has close nodes: " << (has_close_nodes ? "yes" : "no") << "\n";
    oss << "  - Normalization: center=" << norm_center << ", scale=" << norm_scale << "\n";
    
    if (!validation_error.empty()) {
        oss << "  - Error: " << validation_error << "\n";
    }
    
    return oss.str();
}

// ============== Реализация ParameterizationBuilder ==============

void ParameterizationBuilder::log(const std::string& message, bool is_warning) {
    std::ostringstream oss;
    oss << "[" << (is_warning ? "WARN" : "INFO") << "] " << message;
    build_log_.push_back(oss.str());
}

bool ParameterizationBuilder::validate_nodes(const ApproximationConfig& config) {
    log("Validating interpolation nodes...");
    
    // Извлечение узлов из конфигурации
    std::vector<double> x_nodes;
    std::vector<double> y_nodes;
    for (const auto& node : config.interp_nodes) {
        x_nodes.push_back(node.x);
        y_nodes.push_back(node.value);
    }
    
    // Построение набора узлов
    nodes_.build(x_nodes, y_nodes, config.interval_start, config.interval_end);
    
    if (!nodes_.is_valid) {
        add_error("Node validation failed: " + nodes_.validation_error);
        return false;
    }
    
    // Проверка достаточного числа узлов
    if (nodes_.count < 1) {
        add_error("At least one interpolation node is required");
        return false;
    }
    
    nodes_validated_ = true;
    log("Node validation passed. Effective node count: " + std::to_string(nodes_.count));
    return true;
}

bool ParameterizationBuilder::correct_formulation(ApproximationConfig& config) {
    log("Checking formulation for corrections...");
    
    int n = config.polynomial_degree;
    int m = nodes_.count;
    
    // Вызов обработчика крайних случаев
    EdgeCaseHandlingResult edge_result = edge_case_handler_.handle_all_cases(n, m, nodes_.y_values, config);
    
    // Логирование обнаруженных случаев
    for (const auto& case_info : edge_result.detected_cases) {
        std::ostringstream case_msg;
        case_msg << "Edge case detected: " << case_info.message << " (Level: ";
        switch (case_info.level) {
            case EdgeCaseLevel::CRITICAL: case_msg << "CRITICAL"; break;
            case EdgeCaseLevel::SPECIAL: case_msg << "SPECIAL"; break;
            case EdgeCaseLevel::WARNING: case_msg << "WARNING"; break;
            case EdgeCaseLevel::RECOVERABLE: case_msg << "RECOVERABLE"; break;
        }
        case_msg << ")";
        if (case_info.level == EdgeCaseLevel::CRITICAL || 
            case_info.level == EdgeCaseLevel::SPECIAL) {
            add_warning(case_msg.str());
        } else {
            log(case_msg.str());
        }
    }
    
    // Проверка критических ошибок
    if (!edge_result.success) {
        add_error("Edge case handling failed");
        return false;
    }
    
    // Проверка: n >= m - 1 (для deg_Q >= 0)
    if (n < m - 1) {
        std::ostringstream oss;
        oss << "Polynomial degree n=" << n << " is too low for m=" << m 
            << " interpolation nodes. Minimum degree is " << (m - 1);
        add_error(oss.str());
        return false;
    }
    
    // Логирование предупреждений от edge case handler
    for (const auto& warning : edge_result.warnings) {
        add_warning(warning);
    }
    
    // Логирование ошибок от edge case handler
    for (const auto& error : edge_result.errors) {
        add_error(error);
    }
    
    // Случай m = 0 (отсутствие интерполяционных узлов)
    if (m == 0) {
        log("No interpolation nodes - using simplified parameterization F(x) = Q(x)");
        add_warning("No interpolation constraints - approximation may not pass through any points");
    }
    
    // Случай m = n + 1 (полная интерполяция)
    if (n == m - 1) {
        log("Full interpolation case: deg_Q = -1, F(x) = P_int(x)");
        add_warning("Full interpolation - approximation and repulsion criteria are inactive");
    }
    
    // Инициализация корректирующего полинома
    correction_.initialize(n - m, BasisType::MONOMIAL, 
                          config.interval_start, 
                          config.interval_end);
    
    // Выбор базиса на основе обнаруженных случаев
    for (const auto& case_info : edge_result.detected_cases) {
        if (case_info.type == EdgeCaseType::HIGH_DEGREE && 
            correction_.degree > 10) {
            correction_.basis_type = BasisType::CHEBYSHEV;
            log("Using Chebyshev basis for high degree polynomial");
            break;
        }
    }
    
    return true;
}

bool ParameterizationBuilder::build_basis() {
    log("Building interpolation basis...");
    
    // Проверка предварительных условий
    if (!nodes_validated_) {
        add_error("Cannot build basis: nodes not validated");
        return false;
    }
    
    // Извлечение нормализованных координат
    std::vector<double> nodes_norm = nodes_.x_norm;
    std::vector<double> values = nodes_.y_values;
    
    // Построение базисного полинома
    basis_.build(nodes_norm, values, 
                  InterpolationMethod::BARYCENTRIC,
                  nodes_.norm_center - nodes_.norm_scale,
                  nodes_.norm_center + nodes_.norm_scale,
                  false,  // узлы уже нормализованы
                  true);
    
    if (!basis_.is_valid) {
        add_error("Basis construction failed: " + basis_.error_message);
        return false;
    }
    
    basis_built_ = true;
    std::ostringstream oss;
    oss << "Interpolation basis built successfully. Method: ";
    if (basis_.method == InterpolationMethod::BARYCENTRIC) {
        oss << "Barycentric";
    } else {
        oss << "Other";
    }
    log(oss.str());
    return true;
}

bool ParameterizationBuilder::build_weight_multiplier() {
    log("Building weight multiplier W(x) = Π(x - z_e)...");
    
    if (!basis_built_) {
        add_error("Cannot build weight multiplier: basis not built");
        return false;
    }
    
    // Построение весового множителя из узлов
    weight_.build_from_roots(basis_.nodes,
                            basis_.x_center - basis_.x_scale,
                            basis_.x_center + basis_.x_scale,
                            true);
    
    // Верификация построения
    if (!weight_.verify_construction()) {
        add_error("Weight multiplier verification failed");
        return false;
    }
    
    weight_built_ = true;
    log("Weight multiplier built. Degree: " + std::to_string(weight_.degree()));
    return true;
}

bool ParameterizationBuilder::build_correction_poly() {
    log("Building correction polynomial Q(x)...");
    
    if (!weight_built_) {
        add_error("Cannot build correction polynomial: weight multiplier not built");
        return false;
    }
    
    // Инициализация коэффициентов
    correction_.initialize_coefficients(InitializationMethod::ZERO,
                                       {}, {}, 
                                       basis_, weight_,
                                       0.0, 1.0);
    
    if (!correction_.is_initialized) {
        add_error("Correction polynomial initialization failed");
        return false;
    }
    
    correction_built_ = true;
    log("Correction polynomial built. Degree: " + std::to_string(correction_.degree));
    return true;
}

bool ParameterizationBuilder::assemble_composite() {
    log("Assembling composite polynomial F(x) = P_int(x) + Q(x)·W(x)...");
    
    if (!correction_built_) {
        add_error("Cannot assemble composite: correction polynomial not built");
        return false;
    }
    
    // Сборка составного полинома
    composite_.build(basis_, weight_, correction_,
                     basis_.x_center - basis_.x_scale,
                     basis_.x_center + basis_.x_scale,
                     EvaluationStrategy::LAZY);
    
    if (!composite_.is_valid()) {
        add_error("Composite polynomial assembly failed");
        return false;
    }
    
    composite_assembled_ = true;
    log("Composite polynomial assembled. Degree: " + std::to_string(composite_.degree()));
    return true;
}

bool ParameterizationBuilder::verify_parameterization() {
    log("Verifying parameterization...");
    
    if (!composite_assembled_) {
        add_error("Cannot verify: composite not assembled");
        return false;
    }
    
    // Верификация сборки
    if (!composite_.verify_assembly(1e-8)) {
        add_error("Parameterization verification failed");
        return false;
    }
    
    // Проверка согласованности представлений
    if (!composite_.verify_representations_consistency()) {
        add_warning("Lazy and analytic representations show discrepancy");
    }
    
    verified_ = true;
    log("Parameterization verified successfully");
    return true;
}

void ParameterizationBuilder::reset() {
    nodes_validated_ = false;
    basis_built_ = false;
    weight_built_ = false;
    correction_built_ = false;
    composite_assembled_ = false;
    verified_ = false;
    
    build_log_.clear();
    warnings_.clear();
    errors_.clear();
    
    log("Builder reset");
}

// ============== Реализация ParameterizationWorkspace ==============

void ParameterizationWorkspace::compute_batch_product(const double* a, 
                                                      const double* b, 
                                                      double* result, 
                                                      size_t size) {
    // Пакетное умножение векторов
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

} // namespace mixed_approx
