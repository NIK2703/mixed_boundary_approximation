#include "mixed_approximation/edge_case_handler.h"
#include "mixed_approximation/functional.h"

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

} // namespace mixed_approx
