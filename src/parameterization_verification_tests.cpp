#include "mixed_approximation/parameterization_verification.h"
#include "mixed_approximation/composite_polynomial.h"
#include <cmath>
#include <limits>
#include <random>

namespace mixed_approx {

ParameterizationVerification ParameterizationVerifier::verify(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes) {
    ParameterizationVerification result;
    
    result.polynomial_degree = composite.total_degree;
    result.num_constraints = composite.num_constraints;
    result.num_free_params = composite.num_free_params;
    result.interval_a = composite.interval_a;
    result.interval_b = composite.interval_b;
    
    result.interpolation_test = test_interpolation(composite, interp_nodes);
    result.completeness_test = test_completeness(
        composite.correction_poly,
        composite.weight_multiplier,
        composite.interval_a,
        composite.interval_b);
    result.stability_test = test_stability(composite, interp_nodes);
    
    if (!result.interpolation_test.passed) {
        result.overall_status = VerificationStatus::FAILED;
        result.errors.push_back("Interpolation conditions are not satisfied");
    } else if (!result.completeness_test.passed) {
        result.overall_status = VerificationStatus::FAILED;
        result.errors.push_back("Solution space is not complete (rank deficient)");
    } else if (!result.stability_test.passed) {
        result.overall_status = VerificationStatus::WARNING;
        result.warnings.push_back("Numerical stability issues detected");
    } else {
        result.overall_status = VerificationStatus::PASSED;
    }
    
    for (const auto& err : result.interpolation_test.node_errors) {
        if (err.absolute_error > result.interpolation_test.tolerance) {
            result.recommendations.push_back(
                diagnose_interpolation_error(err, 1e-12));
        }
    }
    
    if (!result.completeness_test.passed) {
        result.recommendations.push_back(
            diagnose_condition_issue(result.completeness_test,
                                   composite.correction_poly.basis_type));
    }
    
    return result;
}

ParameterizationVerification ParameterizationVerifier::verify_components(
    const InterpolationBasis& basis,
    const WeightMultiplier& W,
    const CorrectionPolynomial& Q,
    const std::vector<InterpolationNode>& interp_nodes) {
    CompositePolynomial composite;
    composite.interpolation_basis = basis;
    composite.weight_multiplier = W;
    composite.correction_poly = Q;
    
    composite.total_degree = W.degree() + Q.degree;
    composite.num_constraints = static_cast<int>(basis.m_eff);
    composite.num_free_params = Q.n_free;
    
    if (basis.is_normalized) {
        composite.interval_a = basis.x_center - basis.x_scale;
        composite.interval_b = basis.x_center + basis.x_scale;
    } else {
        composite.interval_a = 0.0;
        composite.interval_b = 1.0;
    }
    
    return verify(composite, interp_nodes);
}

InterpolationTestResult ParameterizationVerifier::test_interpolation(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes) {
    InterpolationTestResult result;
    result.tolerance = interp_tolerance_;
    result.total_nodes = static_cast<int>(interp_nodes.size());
    
    const double W_tolerance = 1e-10;
    
    for (int i = 0; i < result.total_nodes; ++i) {
        const auto& node = interp_nodes[i];
        
        double F_val = composite.evaluate(node.x);
        double W_val = composite.weight_multiplier.evaluate(node.x);
        
        double abs_err = std::abs(F_val - node.value);
        double rel_err = (std::abs(node.value) > 1e-12) ?
                         abs_err / std::abs(node.value) : abs_err;
        
        bool W_ok = std::abs(W_val) < W_tolerance;
        bool interp_ok = abs_err < interp_tolerance_;
        
        NodeError node_err;
        node_err.node_index = i;
        node_err.coordinate = node.x;
        node_err.target_value = node.value;
        node_err.computed_value = F_val;
        node_err.absolute_error = abs_err;
        node_err.relative_error = rel_err;
        node_err.W_value = W_val;
        node_err.W_acceptable = W_ok;
        
        result.node_errors.push_back(node_err);
        
        if (!interp_ok) {
            result.failed_nodes++;
        }
        
        result.max_absolute_error = std::max(result.max_absolute_error, abs_err);
        result.max_relative_error = std::max(result.max_relative_error, rel_err);
    }
    
    result.passed = (result.failed_nodes == 0);
    
    return result;
}

CompletenessTestResult ParameterizationVerifier::test_completeness(
    const CorrectionPolynomial& Q,
    const WeightMultiplier& W,
    double interval_a,
    double interval_b) {
    CompletenessTestResult result;
    
    int n_free = Q.n_free;
    result.expected_rank = n_free;
    
    if (n_free <= 0) {
        result.passed = true;
        result.actual_rank = 0;
        result.info_messages.push_back("No free parameters (m = n + 1 or m > n)");
        return result;
    }
    
    std::vector<double> test_points = chebyshev_nodes(n_free, interval_a, interval_b);
    
    std::vector<std::vector<double>> G(n_free, std::vector<double>(n_free, 0.0));
    
    for (int i = 0; i < n_free; ++i) {
        double x = test_points[i];
        double W_val = W.evaluate(x);
        double x_work = (Q.basis_type == BasisType::CHEBYSHEV) ?
                        (x - Q.x_center) / Q.x_scale : x;
        
        for (int k = 0; k < n_free; ++k) {
            double phi_k = compute_basis_function_public(Q, x_work, k);
            G[i][k] = phi_k * W_val;
        }
    }
    
    double condition_number;
    std::vector<double> singular_values = compute_singular_values(G, condition_number);
    
    result.condition_number = condition_number;
    result.singular_values = singular_values;
    
    if (!singular_values.empty()) {
        double max_sv = singular_values[0];
        result.min_singular_value = singular_values.back();
        result.relative_min_sv = (max_sv > 0) ?
                                  result.min_singular_value / max_sv : 0.0;
    }
    
    double sv_threshold = svd_tolerance_ * (singular_values.empty() ? 1.0 : singular_values[0]);
    result.actual_rank = 0;
    for (double sv : singular_values) {
        if (sv > sv_threshold) {
            result.actual_rank++;
        }
    }
    
    bool rank_ok = (result.actual_rank == result.expected_rank);
    bool condition_ok = (condition_number < condition_limit_);
    
    result.passed = rank_ok && condition_ok;
    
    if (!rank_ok) {
        result.warnings.push_back(
            "Matrix rank is deficient: expected " + std::to_string(result.expected_rank) +
            ", got " + std::to_string(result.actual_rank));
    }
    
    if (!condition_ok) {
        result.warnings.push_back(
            "Poorly conditioned basis matrix: cond = " +
            std::to_string(condition_number));
    }
    
    return result;
}

StabilityTestResult ParameterizationVerifier::test_stability(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes) {
    StabilityTestResult result;
    
    if (composite.num_free_params <= 0 || interp_nodes.empty()) {
        result.passed = true;
        result.info_messages.push_back("Skipped: insufficient data for stability test");
        return result;
    }
    
    double a = composite.interval_a;
    double b = composite.interval_b;
    double mid_point = (a + b) / 2.0;
    
    double max_P_int = 0.0;
    double max_Q = 0.0;
    double max_W = 0.0;
    
    std::vector<double> test_points;
    int num_test = std::min(20, composite.num_free_params + 5);
    for (int i = 0; i < num_test; ++i) {
        double t = -1.0 + 2.0 * (i + 0.5) / num_test;
        test_points.push_back(0.5 * (b - a) * t + 0.5 * (a + b));
    }
    
    for (double x : test_points) {
        double P_val = std::abs(composite.interpolation_basis.evaluate(x));
        double W_val = std::abs(composite.weight_multiplier.evaluate(x));
        double Q_val = std::abs(composite.correction_poly.evaluate_Q(x));
        
        max_P_int = std::max(max_P_int, P_val);
        max_W = std::max(max_W, W_val);
        max_Q = std::max(max_Q, Q_val);
    }
    
    double QW_scale = max_Q * max_W;
    result.max_component_scale = std::max(max_P_int, QW_scale);
    result.min_component_scale = std::min(max_P_int, QW_scale);
    
    if (result.min_component_scale > 0) {
        result.scale_balance_ratio = result.max_component_scale / result.min_component_scale;
    } else {
        result.scale_balance_ratio = std::numeric_limits<double>::infinity();
    }
    
    if (composite.num_constraints > 0) {
        std::vector<InterpolationNode> perturbed_nodes = interp_nodes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-perturbation_scale_, perturbation_scale_);
        
        double perturb_factor = perturbation_scale_ * (b - a);
        for (auto& node : perturbed_nodes) {
            node.x += perturb_factor * dist(gen);
        }
        
        double F_original = composite.evaluate(mid_point);
        
        CompositePolynomial perturbed_composite;
        perturbed_composite.interpolation_basis = composite.interpolation_basis;
        perturbed_composite.weight_multiplier = composite.weight_multiplier;
        perturbed_composite.correction_poly = composite.correction_poly;
        perturbed_composite.total_degree = composite.total_degree;
        perturbed_composite.num_constraints = perturbed_nodes.size();
        perturbed_composite.num_free_params = composite.num_free_params;
        perturbed_composite.interval_a = a;
        perturbed_composite.interval_b = b;
        
        double F_perturbed = perturbed_composite.evaluate(mid_point);
        double delta_F = std::abs(F_perturbed - F_original);
        
        if (std::abs(F_original) > 1e-12) {
            result.perturbation_sensitivity = delta_F / std::abs(F_original);
        } else {
            result.perturbation_sensitivity = delta_F;
        }
    }
    
    double max_grad_component = 0.0;
    double min_grad_component = std::numeric_limits<double>::infinity();
    
    for (const auto& node : interp_nodes) {
        double x = node.x;
        double W_val = composite.weight_multiplier.evaluate(x);
        
        for (int k = 0; k < composite.num_free_params; ++k) {
            double x_work = (composite.correction_poly.basis_type == BasisType::CHEBYSHEV) ?
                           (x - composite.correction_poly.x_center) / composite.correction_poly.x_scale : x;
            double phi_k = compute_basis_function_public(composite.correction_poly, x_work, k);
            double grad_component = std::abs(phi_k * W_val);
            max_grad_component = std::max(max_grad_component, grad_component);
            min_grad_component = std::min(min_grad_component, grad_component);
        }
    }
    
    if (min_grad_component > 0) {
        result.gradient_condition_number = max_grad_component / min_grad_component;
    } else {
        result.gradient_condition_number = std::numeric_limits<double>::infinity();
    }
    
    bool scale_ok = (result.scale_balance_ratio < 1e6);
    bool gradient_ok = (result.gradient_condition_number < 1e6);
    bool perturb_ok = (result.perturbation_sensitivity < 1e-4);
    
    result.passed = scale_ok && gradient_ok && perturb_ok;
    
    if (!scale_ok) {
        result.warnings.push_back(
            "Poor scale balance: ratio = " + std::to_string(result.scale_balance_ratio));
    }
    
    if (!gradient_ok) {
        result.warnings.push_back(
            "Poor gradient conditioning: ratio = " +
            std::to_string(result.gradient_condition_number));
    }
    
    return result;
}

void ParameterizationVerifier::set_parameters(
    double interp_tolerance,
    double svd_tolerance,
    double condition_limit,
    double perturbation_scale) {
    interp_tolerance_ = interp_tolerance;
    svd_tolerance_ = svd_tolerance;
    condition_limit_ = condition_limit;
    perturbation_scale_ = perturbation_scale;
}

} // namespace mixed_approx
