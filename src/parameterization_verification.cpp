#include "mixed_approximation/parameterization_verification.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <random>

namespace mixed_approx {

std::string ParameterizationVerification::format(bool detailed) const {
    std::ostringstream oss;
    
    oss << "=================================================\n";
    oss << "Parameterization Verification Report\n";
    oss << "=================================================\n\n";
    
    oss << "Overall Status: ";
    switch (overall_status) {
        case VerificationStatus::PASSED:
            oss << "PASSED";
            break;
        case VerificationStatus::WARNING:
            oss << "WARNING";
            break;
        case VerificationStatus::FAILED:
            oss << "FAILED";
            break;
    }
    oss << "\n\n";
    
    oss << "Parameters:\n";
    oss << "  - Polynomial degree (n): " << polynomial_degree << "\n";
    oss << "  - Interpolation nodes (m): " << num_constraints << "\n";
    oss << "  - Free parameters (n_free): " << num_free_params << "\n";
    oss << "  - Interval: [" << interval_a << ", " << interval_b << "]\n\n";
    
    oss << "Interpolation Test:\n";
    oss << "  - Status: " << (interpolation_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Nodes checked: " << interpolation_test.total_nodes << "/"
        << interpolation_test.total_nodes << "\n";
    oss << "  - Failed nodes: " << interpolation_test.failed_nodes << "\n";
    oss << "  - Max absolute error: " << std::scientific << std::setprecision(3)
        << interpolation_test.max_absolute_error << "\n";
    oss << "  - Max relative error: " << interpolation_test.max_relative_error << "\n";
    oss << "  - Tolerance: " << interpolation_test.tolerance << "\n";
    if (detailed && !interpolation_test.node_errors.empty()) {
        oss << "  - Node errors:\n";
        for (const auto& err : interpolation_test.node_errors) {
            if (err.absolute_error > interpolation_test.tolerance) {
                oss << "    Node " << err.node_index << " (x=" << err.coordinate << "):\n";
                oss << "      Target: " << err.target_value << ", Computed: " << err.computed_value << "\n";
                oss << "      Abs error: " << err.absolute_error << ", W(x): " << err.W_value << "\n";
            }
        }
    }
    oss << "\n";
    
    oss << "Completeness Test:\n";
    oss << "  - Status: " << (completeness_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Expected rank: " << completeness_test.expected_rank << "\n";
    oss << "  - Actual rank: " << completeness_test.actual_rank << "\n";
    oss << "  - Condition number: " << std::scientific << std::setprecision(3)
        << completeness_test.condition_number << "\n";
    oss << "  - Min singular value: " << completeness_test.min_singular_value << "\n";
    oss << "  - Relative min SV: " << completeness_test.relative_min_sv << "\n\n";
    
    oss << "Stability Test:\n";
    oss << "  - Status: " << (stability_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Perturbation sensitivity: " << std::scientific << std::setprecision(3)
        << stability_test.perturbation_sensitivity << "\n";
    oss << "  - Scale balance ratio: " << stability_test.scale_balance_ratio << "\n";
    oss << "  - Gradient condition number: " << stability_test.gradient_condition_number << "\n\n";
    
    if (!recommendations.empty()) {
        oss << "Recommendations:\n";
        for (size_t i = 0; i < recommendations.size(); ++i) {
            const auto& rec = recommendations[i];
            oss << "  " << (i + 1) << ". [" << static_cast<int>(rec.type) << "] " << rec.message << "\n";
            if (!rec.rationale.empty()) {
                oss << "     Rationale: " << rec.rationale << "\n";
            }
        }
        oss << "\n";
    }
    
    if (!warnings.empty()) {
        oss << "Warnings:\n";
        for (const auto& w : warnings) {
            oss << "  - " << w << "\n";
        }
        oss << "\n";
    }
    
    if (!errors.empty()) {
        oss << "Errors:\n";
        for (const auto& e : errors) {
            oss << "  - " << e << "\n";
        }
        oss << "\n";
    }
    
    oss << "=================================================\n";
    
    return oss.str();
}

ParameterizationVerifier::ParameterizationVerifier(
    double interp_tolerance,
    double svd_tolerance,
    double condition_limit,
    double perturbation_scale)
    : interp_tolerance_(interp_tolerance)
    , svd_tolerance_(svd_tolerance)
    , condition_limit_(condition_limit)
    , perturbation_scale_(perturbation_scale)
{
}

} // namespace mixed_approx
