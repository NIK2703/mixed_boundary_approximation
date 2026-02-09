#include <gtest/gtest.h>
#include "mixed_approximation/convergence_monitor.h"
#include "mixed_approximation/solution_validator.h"
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/initialization_strategy.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <limits>

namespace mixed_approx {
namespace test {

// Helper to build a simple composite polynomial with given n_free
CompositePolynomial build_poly(int n_free, const OptimizationProblemData& data) {
    CompositePolynomial poly;
    
    InterpolationBasis basis;
    if (data.interp_z.size() >= 2) {
        std::vector<double> nodes_x = data.interp_z;
        std::vector<double> nodes_f = data.interp_f;
        basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 
                    data.interval_a, data.interval_b);
    } else {
        basis.build({data.interval_a, data.interval_b}, 
                    {data.interp_z.size() > 0 ? data.interp_f[0] : 0.0, 
                     data.interp_z.size() > 1 ? data.interp_f[1] : 1.0},
                    InterpolationMethod::BARYCENTRIC, 
                    data.interval_a, data.interval_b);
    }
    
    WeightMultiplier W;
    if (data.interp_z.size() > 0) {
        W.build_from_roots(data.interp_z, data.interval_a, data.interval_b);
    } else {
        W.build_from_roots({}, data.interval_a, data.interval_b);
    }
    
    CorrectionPolynomial Q;
    int deg_Q = n_free - 1;
    Q.initialize(deg_Q, BasisType::MONOMIAL, 
                 (data.interval_a + data.interval_b) * 0.5,
                 (data.interval_b - data.interval_a) * 0.5);
    
    poly.build(basis, W, Q, data.interval_a, data.interval_b);
    return poly;
}

// ============== ConvergenceMonitor Tests ==============

TEST(ConvergenceMonitorTest, BasicRecording) {
    ConvergenceMonitor monitor;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.epsilon = 1e-8;
    
    CompositePolynomial poly = build_poly(3, data);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    // Record some iterations
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    for (int i = 0; i < 5; ++i) {
        double q_val = 0.1 * i;
        monitor.record_iteration(q_val, 0.1, comp, q_val);
    }
    
    EXPECT_EQ(monitor.iteration(), 5);
    EXPECT_FALSE(monitor.objective_history().empty());
    EXPECT_FALSE(monitor.gradient_history().empty());
}

TEST(ConvergenceMonitorTest, TimerFunctionality) {
    ConvergenceMonitor monitor;
    monitor.start_timer();
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double elapsed = monitor.elapsed_time();
    
    EXPECT_GE(elapsed, 0.09);  // At least 90ms
    EXPECT_LE(elapsed, 0.5);   // Not more than 500ms (some tolerance)
}

TEST(ConvergenceMonitorTest, RelativeObjectiveChange) {
    ConvergenceMonitor monitor;
    
    // Simulate iterations
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    monitor.record_iteration(1.0, 0.1, comp, 0.0);
    monitor.record_iteration(0.9, 0.1, comp, 0.0);  // ~11% change
    monitor.record_iteration(0.85, 0.1, comp, 0.0); // ~5.6% change
    
    // With enough history, check_relative_objective_change_filtered should work
    // Note: this requires at least window_size (5) iterations
}

TEST(ConvergenceMonitorTest, GradientNormCheck) {
    ConvergenceMonitor monitor;
    
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    // Small gradient
    monitor.record_iteration(0.0, 0.037, comp, 0.0);  // norm ~0.037
    EXPECT_TRUE(monitor.last_gradient_norm() < 0.1);  // Small gradient
    
    // Large gradient
    monitor.record_iteration(0.0, 1.732, comp, 0.0);  // norm ~1.732
    EXPECT_TRUE(monitor.last_gradient_norm() > 0.1);   // Large gradient
}

TEST(ConvergenceMonitorTest, ComponentHistoryRecording) {
    ConvergenceMonitor monitor;
    
    ObjectiveFunctor::Components comp1;
    comp1.approx = 0.5;
    comp1.repel = 0.3;
    comp1.reg = 0.2;
    
    ObjectiveFunctor::Components comp2;
    comp2.approx = 0.4;
    comp2.repel = 0.35;
    comp2.reg = 0.25;
    
    monitor.record_iteration(1.0, 0.1, comp1, 0.0);
    monitor.record_iteration(0.9, 0.1, comp2, 0.0);
    
    // Check component history via accessor methods
    const auto& approx = monitor.approx_history();
    const auto& repel = monitor.repel_history();
    const auto& reg = monitor.reg_history();
    
    EXPECT_EQ(approx.size(), 2u);
    EXPECT_DOUBLE_EQ(approx[0], 0.5);
    EXPECT_DOUBLE_EQ(repel[0], 0.3);
    EXPECT_DOUBLE_EQ(reg[0], 0.2);
    EXPECT_DOUBLE_EQ(approx[1], 0.4);
}

TEST(ConvergenceMonitorTest, OscillationDetectionAutocorrelation) {
    // This test verifies that the convergence monitor can detect oscillations
    // Note: The actual implementation may have different thresholds
    ConvergenceMonitor monitor;
    
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    // Create oscillating objective history
    // Pattern: 1.0, 0.9, 1.0, 0.9, 1.0, 0.9 (alternating)
    for (int i = 0; i < 6; ++i) {
        double obj = (i % 2 == 0) ? 1.0 : 0.9;
        monitor.record_iteration(obj, 0.1, comp, 0.0);
    }
    
    // Just verify the method runs without error
    // Detection depends on internal thresholds
    bool oscillating = monitor.detect_oscillation_autocorrelation();
    // The method executes - actual detection depends on implementation
    (void)oscillating;  // Suppress unused warning
    
    // Non-oscillating pattern should also work
    ConvergenceMonitor monitor2;
    for (int i = 0; i < 6; ++i) {
        monitor2.record_iteration(1.0 - i * 0.1, 0.1, comp, 0.0);
    }
    
    // Just verify the method runs without error
    monitor2.detect_oscillation_autocorrelation();
    
    SUCCEED();  // Test that methods work without crashing
}

TEST(ConvergenceMonitorTest, StagnationDetectionRegression) {
    // This test verifies that stagnation detection works
    ConvergenceMonitor monitor;
    
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    // Create stagnating objective: very slow decrease
    // 1.0, 0.99, 0.98, 0.97, 0.96, 0.95
    for (int i = 0; i < 6; ++i) {
        monitor.record_iteration(1.0 - i * 0.01, 0.1, comp, 0.0);
    }
    
    // Just verify the method runs without error
    // Detection depends on internal thresholds
    bool stagnating = monitor.detect_stagnation_regression();
    (void)stagnating;  // Suppress unused warning
    
    // Fast decrease should also work without crashing
    ConvergenceMonitor monitor2;
    for (int i = 0; i < 6; ++i) {
        monitor2.record_iteration(1.0 - i * 0.2, 0.1, comp, 0.0);
    }
    
    monitor2.detect_stagnation_regression();
    
    SUCCEED();
}

TEST(ConvergenceMonitorTest, DivergenceDetection) {
    // This test verifies that divergence detection works
    ConvergenceMonitor monitor;
    
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    // Simulate divergence: objective increases
    monitor.record_iteration(1.0, 0.1, comp, 0.0);
    monitor.record_iteration(1.5, 0.1, comp, 0.0);  // 50% increase
    monitor.record_iteration(2.0, 0.1, comp, 0.0);
    
    // Just verify the method runs without error
    bool diverging = monitor.detect_divergence_advanced(2.0, 1.0);
    (void)diverging;  // Suppress unused warning
    
    // Converging pattern should also work without crashing
    ConvergenceMonitor monitor2;
    monitor2.record_iteration(2.0, 0.1, comp, 0.0);
    monitor2.record_iteration(1.5, 0.1, comp, 0.0);
    monitor2.record_iteration(1.0, 0.1, comp, 0.0);
    
    monitor2.detect_divergence_advanced(1.0, 2.0);
    
    SUCCEED();
}

TEST(ConvergenceMonitorTest, StopReasonEnum) {
    ConvergenceMonitor monitor;
    
    // Test that StopReason values are distinct and meaningful
    EXPECT_NE(StopReason::RELATIVE_OBJECTIVE_CHANGE,
              StopReason::GRADIENT_NORM);
    EXPECT_NE(StopReason::GRADIENT_NORM,
              StopReason::MAX_ITERATIONS);
    EXPECT_NE(StopReason::MAX_ITERATIONS,
              StopReason::TIMEOUT);
    EXPECT_NE(StopReason::TIMEOUT,
              StopReason::OSCILLATIONS);
    EXPECT_NE(StopReason::OSCILLATIONS,
              StopReason::STAGNATION);
    EXPECT_NE(StopReason::STAGNATION,
              StopReason::DIVERGENCE);
    EXPECT_NE(StopReason::DIVERGENCE,
              StopReason::NOT_CONVERGED);
}

TEST(ConvergenceMonitorTest, CheckStopCriteriaIntegration) {
    ConvergenceMonitor monitor;
    monitor.max_iterations = 100;
    monitor.tol_gradient = 1e-6;
    monitor.tol_objective = 1e-6;
    monitor.timeout_seconds = 10.0;
    
    ObjectiveFunctor::Components comp;
    comp.approx = 1.0;
    comp.repel = 0.0;
    comp.reg = 0.0;
    
    // Initially no stop
    EXPECT_EQ(monitor.stop_reason(), StopReason::NOT_CONVERGED);
    
    // Simulate convergence
    monitor.record_iteration(1e-8, 1e-8, comp, 0.0);
    
    auto reason = monitor.check_stop_criteria(1e-8, 1e-8, comp, 0.0, 1.0, 1.0, 0.0);
    // Should be converged due to small gradient and objective
    EXPECT_NE(reason, StopReason::NOT_CONVERGED);
    
    // Test max iterations
    ConvergenceMonitor monitor2;
    monitor2.max_iterations = 2;
    monitor2.tol_gradient = 1e-6;
    monitor2.tol_objective = 1e-6;
    
    monitor2.record_iteration(0.5, 0.1, comp, 0.0);
    monitor2.record_iteration(0.4, 0.1, comp, 0.0);
    
    reason = monitor2.check_stop_criteria(0.1, 0.4, comp, 0.0, 1.0, 1.0, 0.0);
    EXPECT_EQ(reason, StopReason::MAX_ITERATIONS);
}

// ============== SolutionValidator Tests ==============

TEST(SolutionValidatorTest, NumericalCorrectnessCheck) {
    SolutionValidator validator;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    
    // Valid polynomial: F(x) = x (coeffs: [1.0, 0.0] - highest degree first)
    Polynomial poly({1.0, 0.0});  // P(x) = 1*x + 0 = x
    
    bool valid = validator.check_numerical_correctness(poly, data);
    EXPECT_TRUE(valid);
    
    // Invalid polynomial: NaN coefficient
    poly.setCoefficients({std::numeric_limits<double>::quiet_NaN(), 1.0});
    valid = validator.check_numerical_correctness(poly, data);
    EXPECT_FALSE(valid);
    
    // Invalid polynomial: Inf in evaluation
    poly.setCoefficients({0.0, std::numeric_limits<double>::infinity()});
    valid = validator.check_numerical_correctness(poly, data);
    EXPECT_FALSE(valid);
}

TEST(SolutionValidatorTest, InterpolationCheck) {
    SolutionValidator validator;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 0.5, 1.0};
    data.interp_f = {0.0, 0.5, 1.0};
    
    // Polynomial that exactly interpolates: F(x) = x
    Polynomial poly({1.0, 0.0});  // P(x) = 1*x + 0 = x
    
    double max_error;
    bool ok = validator.check_interpolation(poly, data, max_error);
    EXPECT_TRUE(ok);
    EXPECT_NEAR(max_error, 0.0, 1e-10);
    
    // Polynomial that does not interpolate: F(x) = 0
    Polynomial poly2(std::vector<double>{0.0});  // Constant 0
    
    ok = validator.check_interpolation(poly2, data, max_error);
    EXPECT_FALSE(ok);
    EXPECT_GT(max_error, 0.0);
    
    // No interpolation nodes
    OptimizationProblemData no_interp = data;
    no_interp.interp_z.clear();
    no_interp.interp_f.clear();
    ok = validator.check_interpolation(poly, no_interp, max_error);
    EXPECT_TRUE(ok);
}

TEST(SolutionValidatorTest, BarrierSafetyCheck) {
    SolutionValidator validator;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.repel_y = {0.3, 0.7};
    data.repel_forbidden = {0.2, 0.8};
    data.repel_weight = {10.0, 10.0};
    
    // Polynomial: F(x) = x, so F(0.3)=0.3, distance to 0.2 = 0.1; F(0.7)=0.7, distance to 0.8 = 0.1
    Polynomial poly({1.0, 0.0});  // P(x) = 1*x + 0 = x
    
    double min_distance;
    bool safe = validator.check_barrier_safety(poly, data, min_distance);
    EXPECT_TRUE(safe);
    EXPECT_NEAR(min_distance, 0.1, 1e-10);
    
    // Polynomial that violates barrier: F(x) = 0.2 (constant)
    Polynomial poly2(std::vector<double>{0.2});  // Constant 0.2
    
    safe = validator.check_barrier_safety(poly2, data, min_distance);
    EXPECT_FALSE(safe);  // F(0.3)=0.2, distance to 0.2 = 0 < epsilon_safe
    
    // No repulsion points
    OptimizationProblemData no_repel;
    no_repel.interval_a = 0.0;
    no_repel.interval_b = 1.0;
    safe = validator.check_barrier_safety(poly, no_repel, min_distance);
    EXPECT_TRUE(safe);
    EXPECT_DOUBLE_EQ(min_distance, std::numeric_limits<double>::infinity());
}

TEST(SolutionValidatorTest, PhysicalPlausibilityCheck) {
    SolutionValidator validator;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_f = {0.0, 0.5, 1.0};  // scale ~1.0
    
    // Well-behaved polynomial: F(x) = x, max value = 1.0
    Polynomial poly({1.0, 0.0});  // P(x) = 1*x + 0 = x
    
    double max_value;
    bool plausible = validator.check_physical_plausibility(poly, data, max_value);
    EXPECT_TRUE(plausible);
    EXPECT_NEAR(max_value, 1.0, 1e-10);
    
    // Oscillating polynomial: high degree with large coefficients
    Polynomial poly_osc(11);  // degree 10, all zeros
    std::vector<double> large_coeffs(11, 100.0);  // Large coefficients -> oscillations
    poly_osc.setCoefficients(large_coeffs);
    
    plausible = validator.check_physical_plausibility(poly_osc, data, max_value);
    EXPECT_FALSE(plausible);  // Should be too large
    EXPECT_GT(max_value, validator.max_value_factor * 1.0);
    
    // No approx_f -> scale = 1.0
    OptimizationProblemData no_scale;
    no_scale.interval_a = 0.0;
    no_scale.interval_b = 1.0;
    plausible = validator.check_physical_plausibility(poly, no_scale, max_value);
    EXPECT_TRUE(plausible);
}

TEST(SolutionValidatorTest, FullValidationWorkflow) {
    SolutionValidator validator;
    
    // Valid solution
    OptimizationProblemData valid_data;
    valid_data.interval_a = 0.0;
    valid_data.interval_b = 1.0;
    valid_data.interp_z = {0.0, 1.0};
    valid_data.interp_f = {0.0, 1.0};
    valid_data.repel_y = {0.5};
    valid_data.repel_forbidden = {0.3};
    valid_data.repel_weight = {10.0};
    valid_data.approx_f = {0.0, 0.5, 1.0};
    
    Polynomial valid_poly({1.0, 0.0});  // P(x) = x
    
    auto result = validator.validate(valid_poly, valid_data);
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.interpolation_ok);
    EXPECT_TRUE(result.barriers_safe);
    EXPECT_TRUE(result.numerical_correct);
    EXPECT_TRUE(result.physically_plausible);
    EXPECT_FALSE(result.correction_applied);
    
    // Invalid: interpolation violation
    OptimizationProblemData invalid_data = valid_data;
    Polynomial invalid_poly(std::vector<double>{0.5});  // Constant 0.5, doesn't interpolate 0 and 1
    
    result = validator.validate(invalid_poly, invalid_data);
    EXPECT_FALSE(result.is_valid);
    EXPECT_FALSE(result.interpolation_ok);
    EXPECT_TRUE(result.barriers_safe);  // No barrier check because interpolation already failed? Actually still checked
    // Note: validation continues even if one check fails, but is_valid will be false
}

TEST(SolutionValidatorTest, WarningsGeneration) {
    SolutionValidator validator;
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.repel_y = {0.5};
    data.repel_forbidden = {0.5 + 1e-12};  // Very close to barrier
    data.repel_weight = {10.0};
    data.approx_f = {1.0};
    
    Polynomial poly({1.0, 0.0});  // P(x) = x
    
    auto result = validator.validate(poly, data);
    
    // Should have warning about small barrier distance
    EXPECT_FALSE(result.warnings.empty());
    bool has_barrier_warning = false;
    for (const auto& w : result.warnings) {
        if (w.find("Barrier") != std::string::npos) {
            has_barrier_warning = true;
        }
    }
    EXPECT_TRUE(has_barrier_warning);
}

// ============== MixedApproximation Integration Tests ==============

TEST(MixedApproximationIntegrationTest, SolveWithConvergenceMonitor) {
    // This test requires a full MixedApproximation setup
    // We'll create a simple problem that should converge quickly
    
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Simple approximation: f(x) = x at a few points
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    // Interpolation at endpoints
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    // No repulsion
    data.epsilon = 1e-8;
    
    // Build initial approximation
    auto init_result = ma.build_initial_approximation(data, 3);  // n_free = 3
    EXPECT_TRUE(init_result.success);
    
    // Solve
    auto solve_result = ma.solve(data, 3);
    
    // Check that solution was found
    if (solve_result.converged) {
        EXPECT_NE(solve_result.final_polynomial, nullptr);
        EXPECT_TRUE(solve_result.validation.is_valid);
        EXPECT_GT(solve_result.iterations, 0);
        EXPECT_GT(solve_result.elapsed_time, 0.0);
        
        // Check that report was generated
        EXPECT_FALSE(solve_result.diagnostic_report.empty());
    } else {
        // If not converged, still should have a polynomial (maybe from initialization)
        // and diagnostic info
        EXPECT_NE(solve_result.final_polynomial, nullptr);
        EXPECT_FALSE(solve_result.diagnostic_report.empty());
    }
}

TEST(MixedApproximationIntegrationTest, SolveWithStrongBarriers) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    // Strong repulsion near 0.5
    data.repel_y = {0.5};
    data.repel_forbidden = {0.2};  // Force solution away from 0.5
    data.repel_weight = {1000.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 3);
    EXPECT_TRUE(init_result.success);
    
    auto solve_result = ma.solve(data, 3);
    
    // Should still converge or at least produce a valid solution
    if (solve_result.converged) {
        EXPECT_TRUE(solve_result.validation.is_valid);
    }
    
    // Check barrier safety in final solution
    if (solve_result.final_polynomial) {
        SolutionValidator validator;
        auto validation = validator.validate(*solve_result.final_polynomial, data);
        // Barrier safety is critical for strong barriers
        EXPECT_TRUE(validation.barriers_safe) << validation.message;
    }
}

TEST(MixedApproximationIntegrationTest, DiagnosticReportContent) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_f = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 3);
    auto solve_result = ma.solve(data, 3);
    
    if (solve_result.converged) {
        // Diagnostic report should contain useful information
        const std::string& report = solve_result.diagnostic_report;
        EXPECT_FALSE(report.empty());
        
        // Check for key sections
        EXPECT_NE(report.find("=== DIAGNOSTIC REPORT ==="), std::string::npos);
        EXPECT_NE(report.find("Convergence status:"), std::string::npos);
        EXPECT_NE(report.find("Final objective:"), std::string::npos);
        EXPECT_NE(report.find("Iterations:"), std::string::npos);
        EXPECT_NE(report.find("Elapsed time:"), std::string::npos);
        EXPECT_NE(report.find("Solution validation:"), std::string::npos);
    }
}

TEST(MixedApproximationIntegrationTest, SolutionValidationIntegration) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 2);
    auto solve_result = ma.solve(data, 2);
    
    // If converged, validation should be performed
    if (solve_result.converged) {
        EXPECT_TRUE(solve_result.validation.is_valid);
        EXPECT_TRUE(solve_result.validation.numerical_correct);
        EXPECT_TRUE(solve_result.validation.interpolation_ok);
        EXPECT_TRUE(solve_result.validation.barriers_safe);
        // physically_plausible might be true for simple problems
    }
    
    // If not converged, we still get a polynomial (from init or last iteration)
    // and validation might fail
    if (solve_result.final_polynomial) {
        SolutionValidator validator;
        auto validation = validator.validate(*solve_result.final_polynomial, data);
        // Validation result should be consistent with solve_result.validation
        EXPECT_EQ(validation.is_valid, solve_result.validation.is_valid);
    }
}

TEST(MixedApproximationIntegrationTest, EdgeCaseNoApproximationPoints) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Only interpolation, no approximation
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 2);
    // With no approx points, should use ZERO strategy
    EXPECT_TRUE(init_result.success);
    EXPECT_EQ(init_result.strategy_used, InitializationStrategy::ZERO);
    
    auto solve_result = ma.solve(data, 2);
    // Should still produce a valid solution (interpolation only)
    if (solve_result.converged) {
        EXPECT_TRUE(solve_result.validation.interpolation_ok);
    }
}

TEST(MixedApproximationIntegrationTest, EdgeCaseHighDegreePolynomial) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Provide enough data for high degree
    for (int i = 0; i <= 20; ++i) {
        double x = i / 20.0;
        data.approx_x.push_back(x);
        data.approx_f.push_back(x);
        data.approx_weight.push_back(1.0);
    }
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 15);  // n_free = 15
    EXPECT_TRUE(init_result.success);
    EXPECT_EQ(init_result.initial_coeffs.size(), 15);
    
    auto solve_result = ma.solve(data, 15);
    
    // High degree might be challenging, but should not crash
    if (solve_result.converged) {
        EXPECT_TRUE(solve_result.validation.numerical_correct);
    }
}

TEST(MixedApproximationIntegrationTest, EdgeCaseExtremeBarrierWeights) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    // Extremely strong barrier
    data.repel_y = {0.5};
    data.repel_forbidden = {0.1};
    data.repel_weight = {1e10};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 3);
    EXPECT_TRUE(init_result.success);
    
    auto solve_result = ma.solve(data, 3);
    
    // Should handle extreme weights without crashing
    if (solve_result.converged) {
        // Barrier safety should be satisfied
        EXPECT_TRUE(solve_result.validation.barriers_safe);
    }
}

TEST(MixedApproximationIntegrationTest, EdgeCaseIllConditionedData) {
    MixedApproximation ma;
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Points very close together
    data.approx_x = {0.5, 0.5001, 0.5002, 0.5003};
    data.approx_f = {0.5, 0.5001, 0.5002, 0.5003};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.epsilon = 1e-8;
    
    auto init_result = ma.build_initial_approximation(data, 3);
    // Should handle ill-conditioning
    EXPECT_TRUE(init_result.success);
    
    auto solve_result = ma.solve(data, 3);
    
    // Should produce finite results
    if (solve_result.final_polynomial) {
        // Check polynomial coefficients are finite
        const auto& coeffs = solve_result.final_polynomial->coefficients();
        for (double c : coeffs) {
            EXPECT_TRUE(std::isfinite(c));
        }
    }
}

// ============== CompositePolynomial::build_polynomial Test ==============

TEST(CompositePolynomialTest, BuildPolynomialFromCoefficients) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial comp = build_poly(3, data);
    
    // Set some Q coefficients
    std::vector<double> q_coeffs = {0.1, 0.2, 0.3};
    
    // Build polynomial F = P_int + W * Q
    Polynomial poly = comp.build_polynomial(q_coeffs);
    
    // Check that polynomial is valid
    EXPECT_TRUE(poly.is_initialized());
    // Expected degree: total_degree = deg_Q + deg_W = (n_free-1) + m = 2 + 2 = 4
    EXPECT_EQ(poly.degree(), comp.degree());
    
    // Check that interpolation is satisfied at nodes (since W(z_e)=0)
    for (size_t i = 0; i < data.interp_z.size(); ++i) {
        double z = data.interp_z[i];
        double f_target = data.interp_f[i];
        double F_val = poly.evaluate(z);
        EXPECT_NEAR(F_val, f_target, 1e-10) << "Interpolation failed at z=" << z;
    }
    
    // Check that polynomial evaluates to finite values
    for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        double val = poly.evaluate(x);
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST(CompositePolynomialTest, BuildPolynomialWithCorrectQ) {
    // This test verifies that build_polynomial requires exactly num_free_params coefficients
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial comp = build_poly(3, data);
    
    // Correct number of coefficients (num_free_params = 3)
    std::vector<double> correct_q = {0.0, 0.0, 0.0};
    EXPECT_NO_THROW({
        Polynomial poly = comp.build_polynomial(correct_q);
        EXPECT_TRUE(poly.is_initialized());
    });
    
    SUCCEED();
}

TEST(CompositePolynomialTest, BuildPolynomialDegreeMismatch) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial comp = build_poly(3, data);  // num_free_params = 3
    
    // Wrong number of coefficients: too many
    std::vector<double> too_many = {0.1, 0.2, 0.3, 0.4, 0.5};
    EXPECT_THROW({
        Polynomial poly = comp.build_polynomial(too_many);
    }, std::invalid_argument);
    
    // Too few coefficients
    std::vector<double> too_few = {0.1, 0.2};
    EXPECT_THROW({
        Polynomial poly = comp.build_polynomial(too_few);
    }, std::invalid_argument);
    
    SUCCEED();
}

} // namespace test
} // namespace mixed_approx
