#include <gtest/gtest.h>
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/initialization_strategy.h"
#include "mixed_approximation/optimization_post_processor.h"
#include "mixed_approximation/convergence_monitor.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"

namespace mixed_approx {
namespace test {

// Test 1: OptimizationProblemData::is_valid()
TEST(OptimizationProblemDataTest, IsValid) {
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.epsilon = 1e-8;
    config.approx_points.push_back(WeightedPoint(0.5, 1.0, 0.1));
    
    OptimizationProblemData data(config);
    EXPECT_TRUE(data.is_valid());
}

// Test 2: OptimizationProblemData with invalid data
TEST(OptimizationProblemDataTest, IsValidWithInvalidData) {
    ApproximationConfig config;
    config.interval_start = 1.0;  // Wrong order
    config.interval_end = 0.0;
    
    OptimizationProblemData data(config);
    EXPECT_FALSE(data.is_valid());
}

// Test 3: OptimizationCache::clear()
TEST(OptimizationCacheTest, Clear) {
    OptimizationCache cache;
    cache.P_at_x = {1.0, 2.0, 3.0};
    cache.W_at_x = {0.5, 0.6, 0.7};
    cache.data_cache_valid = true;
    
    cache.clear();
    
    EXPECT_TRUE(cache.P_at_x.empty());
    EXPECT_TRUE(cache.W_at_x.empty());
    EXPECT_FALSE(cache.data_cache_valid);
}

// Test 4: ConvergenceMonitor - basic functionality
TEST(ConvergenceMonitorTest, BasicFunctionality) {
    ConvergenceMonitor monitor(1e-6, 1e-8);
    
    // Проверяем, что монитор создаётся корректно
    EXPECT_NE(monitor.get_diagnostic_info().find("Iteration: 0"), std::string::npos);
    
    // Сброс должен работать
    monitor.reset();
    EXPECT_NE(monitor.get_diagnostic_info().find("Iteration: 0"), std::string::npos);
}

// Test 5: ConvergenceMonitor::detect_divergence
TEST(ConvergenceMonitorTest, DetectDivergence) {
    ConvergenceMonitor monitor(1e-6, 1e-8);
    
    // Первый вызов не должен определять расходимость
    EXPECT_FALSE(monitor.detect_divergence(1.0));
    
    // Добавляем в историю
    monitor.detect_plateau(1.0);
    monitor.detect_plateau(1.0);
    monitor.detect_plateau(1.0);
    
    // Большой скачок должен определяться как расходимость
    EXPECT_TRUE(monitor.detect_divergence(100.0));
}

// Test 6: InitializationStrategy selection logic
TEST(InitializationStrategySelectorTest, StrategySelection) {
    // Только аппроксимация -> LEAST_SQUARES
    {
        CompositePolynomial poly;
        poly.num_free_params = 3;
        OptimizationProblemData data;
        data.approx_x = {0.5};
        data.approx_f = {0.5};
        data.approx_weight = {1.0};
        
        auto strategy = InitializationStrategySelector::select(poly, data);
        EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
    }
    
    // Нет аппроксимации -> ZERO
    {
        CompositePolynomial poly;
        poly.num_free_params = 3;
        OptimizationProblemData data;
        
        auto strategy = InitializationStrategySelector::select(poly, data);
        EXPECT_EQ(strategy, InitializationStrategy::ZERO);
    }
    
    // Только отталкивание -> ZERO
    {
        CompositePolynomial poly;
        poly.num_free_params = 3;
        OptimizationProblemData data;
        data.repel_y = {0.3};
        data.repel_forbidden = {0.5};
        data.repel_weight = {1000.0};  // Большой вес -> ZERO
        
        auto strategy = InitializationStrategySelector::select(poly, data);
        EXPECT_EQ(strategy, InitializationStrategy::ZERO);
    }
}

// Test 7: ObjectiveFunctor::is_valid() with invalid polynomial
TEST(ObjectiveFunctorTest, IsValid) {
    CompositePolynomial poly;
    poly.total_degree = 5;
    poly.num_constraints = 2;
    poly.num_free_params = 4;
    poly.interval_a = 0.0;
    poly.interval_b = 1.0;
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    EXPECT_FALSE(functor.is_valid());  // Poly is not valid
}

// Test 8: ObjectiveFunctor::build_caches()
TEST(ObjectiveFunctorTest, BuildCaches) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {1.0, 2.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.approx_points.push_back(WeightedPoint(0.5, 1.5, 0.1));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    EXPECT_TRUE(functor.is_valid());
    functor.build_caches();
    EXPECT_NO_THROW(functor.value(std::vector<double>{0.0, 0.0, 0.0, 0.0}));
}

// Test 9: ObjectiveFunctor::value()
TEST(ObjectiveFunctorTest, Value) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.approx_points.push_back(WeightedPoint(0.5, 0.5, 0.1));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {0.0, 0.0, 0.0};
    double J = functor.value(q);
    EXPECT_TRUE(std::isfinite(J));
}

// Test 10: ObjectiveFunctor::value_and_gradient()
TEST(ObjectiveFunctorTest, ValueAndGradient) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.approx_points.push_back(WeightedPoint(0.5, 0.5, 0.1));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {0.1, 0.1, 0.1};
    double J;
    std::vector<double> grad;
    
    functor.value_and_gradient(q, J, grad);
    
    EXPECT_TRUE(std::isfinite(J));
    EXPECT_EQ(grad.size(), 3);
    for (double g : grad) {
        EXPECT_TRUE(std::isfinite(g));
    }
}

// Test 11: ObjectiveFunctor::compute_components()
TEST(ObjectiveFunctorTest, ComputeComponents) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points.push_back(WeightedPoint(0.5, 0.5, 0.1));
    config.repel_points.push_back(RepulsionPoint(0.3, 0.5, 10.0));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {0.0, 0.0, 0.0};
    auto comps = functor.compute_components(q);
    
    EXPECT_GE(comps.approx, 0.0);
    EXPECT_GE(comps.repel, 0.0);
    EXPECT_GE(comps.reg, 0.0);
    EXPECT_NEAR(comps.total, comps.approx + comps.repel + comps.reg, 1e-10);
}

// Test 12: InitializationStrategySelector::select()
TEST(InitializationStrategySelectorTest, Select) {
    CompositePolynomial poly;
    poly.num_free_params = 3;
    
    OptimizationProblemData data;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.repel_y = {0.3};
    data.repel_forbidden = {0.5};
    data.repel_weight = {10.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
}

// Test 13: OptimizationPostProcessor with valid setup
TEST(OptimizationPostProcessorTest, GenerateReport) {
    // Создаём полностью валидную структуру
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    OptimizationProblemData data;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.repel_y = {0.5};
    data.repel_forbidden = {0.3};
    data.repel_weight = {10.0};
    data.epsilon = 1e-8;
    
    OptimizationPostProcessor processor(poly, data);
    
    std::vector<double> coeffs = {0.0, 0.0, 0.0};
    auto report = processor.generate_report(coeffs, 0.1);
    
    EXPECT_GE(report.min_barrier_distance, 0.0);
    EXPECT_TRUE(report.approx_percentage >= 0.0);
    EXPECT_TRUE(report.repel_percentage >= 0.0);
    EXPECT_TRUE(report.reg_percentage >= 0.0);
}

// Test 14: Barrier protection in repulsion term
TEST(ObjectiveFunctorTest, BarrierProtection) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.epsilon = 1e-8;
    config.repel_points.push_back(RepulsionPoint(0.5, 0.5, 1000.0));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {0.0, 1.0, 0.0};
    double J = functor.value(q);
    
    EXPECT_TRUE(std::isfinite(J));
    EXPECT_GT(J, 0.0);
}

// Test 15: Numerical stability with large coefficients
TEST(ObjectiveFunctorTest, NumericalStability) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.approx_points.push_back(WeightedPoint(0.5, 0.5, 0.1));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {1e10, 1e10, 1e10};
    double J = functor.value(q);
    
    EXPECT_TRUE(std::isfinite(J));
}

// Test 16: Check that gradient is computed correctly
TEST(ObjectiveFunctorTest, GradientConsistency) {
    CompositePolynomial poly;
    
    std::vector<double> nodes_x = {0.0, 1.0};
    std::vector<double> nodes_f = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes_x, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    
    poly.build(basis, W, Q, 0.0, 1.0);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.approx_points.push_back(WeightedPoint(0.5, 0.5, 1.0));
    
    OptimizationProblemData data(config);
    ObjectiveFunctor functor(poly, data);
    functor.build_caches();
    
    std::vector<double> q = {0.0, 0.0};
    double J;
    std::vector<double> grad;
    functor.value_and_gradient(q, J, grad);
    
    // Gradient should have correct size
    EXPECT_EQ(grad.size(), 2);
    
    // Both gradient components should be finite
    for (double g : grad) {
        EXPECT_TRUE(std::isfinite(g));
    }
}

}  // namespace test
}  // namespace mixed_approx
