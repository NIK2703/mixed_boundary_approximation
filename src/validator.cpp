#include "mixed_approximation/validator.h"

namespace mixed_approx {

// ==================== Основные методы валидации ====================

std::string Validator::validate(const ApproximationConfig& config, bool strict_mode) {
    ValidationReport report = validate_full(config, strict_mode);
    
    if (report.has_errors()) {
        return report.format(false);  // без рекомендаций для краткости
    }
    
    return "";
}

ValidationReport Validator::validate_full(const ApproximationConfig& config, bool strict_mode) {
    ValidationReport report;
    
    // Заполняем базовую статистику
    report.approx_points_count = config.approx_points.size();
    report.repel_points_count = config.repel_points.size();
    report.interp_nodes_count = config.interp_nodes.size();
    report.polynomial_degree = config.polynomial_degree;
    report.free_parameters = config.polynomial_degree - config.interp_nodes.size();
    
    // Статистика по весам
    if (!config.approx_points.empty()) {
        report.min_approx_weight = config.approx_points[0].weight;
        report.max_approx_weight = config.approx_points[0].weight;
        for (const auto& p : config.approx_points) {
            report.min_approx_weight = std::min(report.min_approx_weight, p.weight);
            report.max_approx_weight = std::max(report.max_approx_weight, p.weight);
        }
    }
    if (!config.repel_points.empty()) {
        report.min_repel_weight = config.repel_points[0].weight;
        report.max_repel_weight = config.repel_points[0].weight;
        for (const auto& p : config.repel_points) {
            report.min_repel_weight = std::min(report.min_repel_weight, p.weight);
            report.max_repel_weight = std::max(report.max_repel_weight, p.weight);
        }
    }
    
    // Выполняем все проверки
    std::string interval_check = check_interval(config);
    if (!interval_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, interval_check, 
            "Check that interval_start and interval_end are finite numbers and interval_start < interval_end.");
    }
    
    std::string points_interval_check = check_points_in_interval(config);
    if (!points_interval_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, points_interval_check,
            "Move all points inside the interval [a, b] or correct the interval boundaries.");
    }
    
    std::string disjoint_check = check_disjoint_sets(config);
    if (!disjoint_check.empty()) {
        // Анализируем сообщение, чтобы определить уровень
        if (disjoint_check.find("fatal") != std::string::npos || 
            disjoint_check.find("FATAL") != std::string::npos) {
            report.errors.emplace_back(ValidationLevel::Error, disjoint_check,
                "Remove or adjust conflicting points. Approximation and interpolation points must not coincide.");
        } else {
            report.warnings.emplace_back(ValidationLevel::Warning, disjoint_check,
                "Consider adjusting weights or positions to resolve the conflict between approximation and repulsion points.");
        }
    }
    
    std::string weights_check = check_positive_weights(config);
    if (!weights_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, weights_check,
            "All weights must be positive. Check the values of sigma_i and B_j.");
    }
    
    std::string count_check = check_interpolation_nodes_count(config);
    if (!count_check.empty()) {
        if (count_check.find("exceeds") != std::string::npos) {
            report.errors.emplace_back(ValidationLevel::Error, count_check,
                "Increase polynomial degree or reduce the number of interpolation nodes.");
        } else {
            report.warnings.emplace_back(ValidationLevel::Warning, count_check,
                "Consider increasing polynomial degree for more flexibility or removing some interpolation nodes.");
        }
    }
    
    std::string unique_check = check_unique_interpolation_nodes(config);
    if (!unique_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, unique_check,
            "Each interpolation node must have a unique x-coordinate. Remove duplicate nodes.");
    }
    
    std::string empty_check = check_nonempty_sets(config);
    if (!empty_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, empty_check,
            "At least one set of points (approximation or interpolation) must be non-empty.");
    }
    
    std::string anomalies_check = check_numerical_anomalies(config);
    if (!anomalies_check.empty()) {
        report.warnings.emplace_back(ValidationLevel::Warning, anomalies_check,
            "Consider normalizing weights or rescaling the problem to improve numerical stability.");
    }
    
    std::string value_conflict_check = check_repel_interp_value_conflict(config);
    if (!value_conflict_check.empty()) {
        report.errors.emplace_back(ValidationLevel::Error, value_conflict_check,
            "Adjust the forbidden y-value of the repulsion point or remove the conflicting interpolation node.");
    }
    
    // В strict_mode предупреждения становятся ошибками
    if (strict_mode && report.has_warnings()) {
        for (const auto& w : report.warnings) {
            report.errors.emplace_back(ValidationLevel::Error, w.message, w.recommendation);
        }
        report.warnings.clear();
    }
    
    return report;
}

} // namespace mixed_approx
