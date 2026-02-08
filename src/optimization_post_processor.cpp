#include "mixed_approximation/optimization_post_processor.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/objective_functor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace mixed_approx {

OptimizationPostProcessor::OptimizationPostProcessor(
    const CompositePolynomial& param,
    const OptimizationProblemData& data)
    : param_(param)
    , data_(data)
{
}

PostOptimizationReport OptimizationPostProcessor::generate_report(
    const std::vector<double>& final_coeffs,
    double final_objective)
{
    PostOptimizationReport report;
    
    // Check interpolation conditions
    report.max_interpolation_error = 0.0;
    for (size_t e = 0; e < data_.num_interp_nodes(); ++e) {
        double F_z = param_.evaluate(data_.interp_z[e]);
        double error = std::abs(F_z - data_.interp_f[e]);
        report.max_interpolation_error = std::max(report.max_interpolation_error, error);
    }
    report.interpolation_satisfied = report.max_interpolation_error < 1e-10;
    
    // Check barrier safety
    report.min_barrier_distance = std::numeric_limits<double>::max();
    for (size_t j = 0; j < data_.num_repel_points(); ++j) {
        double F_y = param_.evaluate(data_.repel_y[j]);
        double distance = std::abs(data_.repel_forbidden[j] - F_y);
        report.min_barrier_distance = std::min(report.min_barrier_distance, distance);
    }
    report.barrier_constraints_satisfied = report.min_barrier_distance > data_.epsilon * 10;
    
    // Compute components
    ObjectiveFunctor functor(param_, data_);
    functor.build_caches();
    auto comps = functor.compute_components(final_coeffs);
    
    report.approx_percentage = comps.total > 0 ? 100.0 * comps.approx / comps.total : 0.0;
    report.repel_percentage = comps.total > 0 ? 100.0 * comps.repel / comps.total : 0.0;
    report.reg_percentage = comps.total > 0 ? 100.0 * comps.reg / comps.total : 0.0;
    
    // Generate recommendations
    if (report.repel_percentage > 90.0) {
        report.recommendations.push_back(
            "Strong barriers (J_repel >> J_approx). "
            "Consider decreasing B_j by 10x and re-running optimization.");
    }
    if (report.approx_percentage > 90.0 && !report.barrier_constraints_satisfied) {
        report.recommendations.push_back(
            "Barriers ineffective, violations detected. "
            "Consider increasing B_j by 10x.");
    }
    if (report.reg_percentage > 90.0) {
        report.recommendations.push_back(
            "Excessive regularization (J_reg >> J_approx + J_repel). "
            "Consider decreasing gamma by 5x.");
    }
    if (report.interpolation_satisfied && report.barrier_constraints_satisfied) {
        report.recommendations.push_back(
            "Good component balance, parameters are optimal.");
    }
    
    return report;
}

std::string OptimizationPostProcessor::generate_text_report(
    const PostOptimizationReport& report)
{
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
             "OPTIMIZATION REPORT\n"
             "==================\n"
             "Solution Quality:\n"
             "  Max interpolation error: %.2e %s\n"
             "  Min barrier distance: %.2e %s\n"
             "\n"
             "Component Balance:\n"
             "  Approximation: %.1f%%\n"
             "  Repulsion: %.1f%%\n"
             "  Regularization: %.1f%%\n",
             report.max_interpolation_error,
             report.interpolation_satisfied ? "[OK]" : "[FAIL]",
             report.min_barrier_distance,
             report.barrier_constraints_satisfied ? "[OK]" : "[FAIL]",
             report.approx_percentage,
             report.repel_percentage,
             report.reg_percentage);
    
    std::string result(buffer);
    
    if (!report.recommendations.empty()) {
        result += "\nRecommendations:\n";
        for (const auto& rec : report.recommendations) {
            result += "  - " + rec + "\n";
        }
    }
    
    return result;
}

void OptimizationPostProcessor::suggest_parameter_corrections(
    const PostOptimizationReport& report,
    ApproximationConfig& config)
{
    if (report.repel_percentage > 100.0 && config.repel_points.size() > 0) {
        for (auto& p : config.repel_points) {
            p.weight /= 10.0;
        }
    }
    
    if (report.approx_percentage > 100.0 && !report.barrier_constraints_satisfied) {
        for (auto& p : config.repel_points) {
            p.weight *= 10.0;
        }
    }
    
    if (report.reg_percentage > 10.0 * (report.approx_percentage + report.repel_percentage)) {
        config.gamma /= 5.0;
    }
}

} // namespace mixed_approx
