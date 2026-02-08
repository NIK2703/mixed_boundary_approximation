#include "mixed_approximation/edge_case_result_formatter.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace mixed_approx {

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
