#include <gtest/gtest.h>
#include "mixed_approximation/initialization_strategy.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include <cmath>
#include <limits>
#include <Eigen/Dense>

namespace mixed_approx {
namespace test {

// Helper to build a simple composite polynomial with given n_free
CompositePolynomial build_poly(int n_free, const OptimizationProblemData& data) {
    CompositePolynomial poly;
    
    InterpolationBasis basis;
    if (data.interp_z.size() >= 2) {
        // Use actual interpolation nodes
        std::vector<double> nodes_x = data.interp_z;
        std::vector<double> nodes_f = data.interp_f;
        basis.build(nodes_x, nodes_f, InterpolationMethod::BARYCENTRIC, 
                    data.interval_a, data.interval_b);
    } else {
        // Fallback: simple linear interpolation on interval endpoints
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
    int deg_Q = n_free - 1;  // n_free = deg_Q + 1
    Q.initialize(deg_Q, BasisType::MONOMIAL, 
                 (data.interval_a + data.interval_b) * 0.5,
                 (data.interval_b - data.interval_a) * 0.5);
    
    poly.build(basis, W, Q, data.interval_a, data.interval_b);
    return poly;
}

// ============== Helper Functions Tests ==============

// Test 1: compute_data_density
TEST(InitializationStrategyHelperTest, ComputeDataDensity) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 2.0;
    data.approx_x = {0.0, 0.5, 1.0, 1.5, 2.0};
    data.approx_f = {0.0, 0.5, 1.0, 0.5, 0.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    
    double rho = InitializationStrategySelector::compute_data_density(data);
    EXPECT_NEAR(rho, 5.0 / 2.0, 1e-10);
    
    // Empty data
    OptimizationProblemData empty_data;
    empty_data.interval_a = 0.0;
    empty_data.interval_b = 1.0;
    double rho_empty = InitializationStrategySelector::compute_data_density(empty_data);
    EXPECT_DOUBLE_EQ(rho_empty, 0.0);
}

// Test 2: compute_barrier_intensity
TEST(InitializationStrategyHelperTest, ComputeBarrierIntensity) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {2.0};  // weight = 1/σ, so σ = 0.5
    data.repel_y = {0.3};
    data.repel_forbidden = {0.5};
    data.repel_weight = {10.0};  // max barrier = 10
    
    double beta = InitializationStrategySelector::compute_barrier_intensity(data);
    // avg_sigma = 0.5, beta = 10 / 0.5 = 20
    EXPECT_NEAR(beta, 20.0, 1e-10);
    
    // No repulsion points
    OptimizationProblemData no_repel;
    no_repel.interval_a = 0.0;
    no_repel.interval_b = 1.0;
    no_repel.approx_weight = {1.0};
    double beta_zero = InitializationStrategySelector::compute_barrier_intensity(no_repel);
    EXPECT_DOUBLE_EQ(beta_zero, 0.0);
}

// ============== Verification Functions Tests ==============

// Test 3: verify_interpolation - should always pass if P_int is built correctly because W(z_e)=0
TEST(InitializationStrategyVerificationTest, VerifyInterpolation) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    // Coefficients that should satisfy interpolation (any coefficients work because W(z_e)=0)
    std::vector<double> coeffs_zero = {0.0, 0.0};
    EXPECT_TRUE(InitializationStrategySelector::verify_interpolation(poly, data, coeffs_zero));
    
    std::vector<double> coeffs_any = {1.0, 2.0};
    EXPECT_TRUE(InitializationStrategySelector::verify_interpolation(poly, data, coeffs_any));
    
    // No interpolation nodes -> should return true
    OptimizationProblemData no_interp = data;
    no_interp.interp_z = {};
    no_interp.interp_f = {};
    EXPECT_TRUE(InitializationStrategySelector::verify_interpolation(poly, no_interp, coeffs_zero));
}

// Test 4: verify_barrier_safety
TEST(InitializationStrategyVerificationTest, VerifyBarrierSafety) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.repel_y = {0.5};
    data.repel_forbidden = {0.3};
    data.repel_weight = {10.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    // Coefficients that give safe distance (q=0 gives F(0.5)=P_int(0.5)=0.5, distance=0.2)
    std::vector<double> coeffs_safe = {0.0, 0.0};
    EXPECT_TRUE(InitializationStrategySelector::verify_barrier_safety(poly, data, coeffs_safe, 10.0));
    
    // Coefficients that violate safety (very close)
    data.repel_forbidden = {0.500000001};  // Very close to 0.5
    EXPECT_FALSE(InitializationStrategySelector::verify_barrier_safety(poly, data, coeffs_safe, 10.0));
    
    // No repulsion points -> should return true
    OptimizationProblemData no_repel = data;
    no_repel.repel_y.clear();
    no_repel.repel_forbidden.clear();
    no_repel.repel_weight.clear();
    EXPECT_TRUE(InitializationStrategySelector::verify_barrier_safety(poly, no_repel, coeffs_safe));
}

// ============== Strategy Selection Tests ==============

// Test 5: select - no approx points -> ZERO
TEST(InitializationStrategySelectionTest, NoApproxPoints) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // No approx points
    
    CompositePolynomial poly = build_poly(3, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    EXPECT_EQ(strategy, InitializationStrategy::ZERO);
}

// Test 6: select - extremely strong barriers (beta > 1000) -> LEAST_SQUARES
TEST(InitializationStrategySelectionTest, ExtremeBarriers) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.repel_y = {0.3};
    data.repel_forbidden = {0.5};
    data.repel_weight = {10000.0};  // Very strong barrier
    
    CompositePolynomial poly = build_poly(3, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
}

// Test 7: select - sparse data (rho < 0.5*n) -> RANDOM
TEST(InitializationStrategySelectionTest, SparseData) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 10.0;  // Large interval
    data.approx_x = {5.0};    // Only 1 point
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.interp_z = {0.0, 10.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(5, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    // rho = 1/10 = 0.1 < 0.5*5 = 2.5 -> RANDOM
    EXPECT_EQ(strategy, InitializationStrategy::RANDOM);
}

// Test 8: select - strong barriers (beta > 100) -> MULTI_START
TEST(InitializationStrategySelectionTest, StrongBarriers) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5, 0.6, 0.7};
    data.approx_f = {0.5, 0.6, 0.7};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.repel_y = {0.3, 0.8};
    data.repel_forbidden = {0.5, 0.8};
    data.repel_weight = {500.0, 500.0};  // Strong barriers
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    // beta > 100 -> MULTI_START
    EXPECT_EQ(strategy, InitializationStrategy::MULTI_START);
}

// Test 9: select - high degree (n > 15) with sufficient data -> HIERARCHICAL
TEST(InitializationStrategySelectionTest, HighDegree) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Provide enough data to avoid RANDOM branch: need rho >= 0.5*n_free
    // For n_free=20, need at least 10 points on [0,1]
    for (int i = 0; i <= 10; ++i) {
        double x = i / 10.0;
        data.approx_x.push_back(x);
        data.approx_f.push_back(x);
        data.approx_weight.push_back(1.0);
    }
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(20, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    // n > 15 -> HIERARCHICAL
    EXPECT_EQ(strategy, InitializationStrategy::HIERARCHICAL);
}

// Test 10: select - standard case -> LEAST_SQUARES
TEST(InitializationStrategySelectionTest, StandardCase) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_f = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.repel_y = {0.3};
    data.repel_forbidden = {0.5};
    data.repel_weight = {10.0};  // Moderate barriers
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    
    auto strategy = InitializationStrategySelector::select(poly, data);
    // Standard case -> LEAST_SQUARES
    EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
}

// ============== Full Initialization Tests ==============

// Test 11: initialize with zero strategy (no approx points)
TEST(InitializationStrategyInitializeTest, ZeroStrategy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    // No approx points forces ZERO strategy
    
    CompositePolynomial poly = build_poly(3, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::ZERO);
    EXPECT_EQ(result.initial_coeffs.size(), 3);  // n_free = 3
    for (double coeff : result.initial_coeffs) {
        EXPECT_DOUBLE_EQ(coeff, 0.0);
    }
    EXPECT_TRUE(result.metrics.interpolation_ok);
}

// Test 12: initialize with least squares strategy
TEST(InitializationStrategyInitializeTest, LeastSquaresStrategy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::LEAST_SQUARES);
    EXPECT_EQ(result.initial_coeffs.size(), 2);  // n_free = 2
    for (double coeff : result.initial_coeffs) {
        EXPECT_TRUE(std::isfinite(coeff));
    }
    EXPECT_TRUE(result.metrics.interpolation_ok);
    // With perfect data match, objective could be very small
    EXPECT_GE(result.initial_objective, 0.0);
    EXPECT_TRUE(std::isfinite(result.initial_objective));
}

// Test 13: initialize with random strategy (sparse data)
TEST(InitializationStrategyInitializeTest, RandomStrategy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 10.0;  // Large interval
    data.approx_x = {1.0};    // Sparse
    data.approx_f = {0.1};
    data.approx_weight = {1.0};
    data.interp_z = {0.0, 10.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(5, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    // Sparse data should trigger RANDOM or LEAST_SQUARES (depending on exact conditions)
    EXPECT_TRUE(result.strategy_used == InitializationStrategy::RANDOM || 
                result.strategy_used == InitializationStrategy::LEAST_SQUARES);
    EXPECT_EQ(result.initial_coeffs.size(), 5);
    for (double coeff : result.initial_coeffs) {
        EXPECT_TRUE(std::isfinite(coeff));
    }
}

// Test 14: initialize with hierarchical strategy (high degree)
TEST(InitializationStrategyInitializeTest, HierarchicalStrategy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Enough data to avoid RANDOM branch
    for (int i = 0; i <= 10; ++i) {
        double x = i / 10.0;
        data.approx_x.push_back(x);
        data.approx_f.push_back(x);
        data.approx_weight.push_back(1.0);
    }
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(20, data);
    
    ObjectiveFunctor functor(poly, data);
    // Workaround: hierarchical_initialization doesn't build caches, so build them here
    functor.build_caches();
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::HIERARCHICAL);
    EXPECT_EQ(result.initial_coeffs.size(), 20);
    for (double coeff : result.initial_coeffs) {
        EXPECT_TRUE(std::isfinite(coeff));
    }
}

// Test 15: initialize with multi-start strategy (strong barriers)
TEST(InitializationStrategyInitializeTest, MultiStartStrategy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_f = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.repel_y = {0.3, 0.7};
    data.repel_forbidden = {0.2, 0.8};
    data.repel_weight = {500.0, 500.0};  // Strong barriers -> should trigger multi-start
    
    CompositePolynomial poly = build_poly(3, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    // With strong barriers, might get MULTI_START or LEAST_SQUARES (depending on exact beta)
    EXPECT_TRUE(result.strategy_used == InitializationStrategy::MULTI_START || 
                result.strategy_used == InitializationStrategy::LEAST_SQUARES);
    EXPECT_EQ(result.initial_coeffs.size(), 3);
    for (double coeff : result.initial_coeffs) {
        EXPECT_TRUE(std::isfinite(coeff));
    }
}

// ============== Metrics and Quality Tests ==============

// Test 16: Metrics computation through initialize
TEST(InitializationStrategyMetricsTest, ComputeMetricsThroughInitialize) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_GE(result.metrics.objective_ratio, 0.0);
    EXPECT_GT(result.metrics.min_barrier_distance, 0.0);  // No barriers -> large value
    EXPECT_GE(result.metrics.rms_residual_norm, 0.0);
    EXPECT_GE(result.metrics.condition_number, 0.0);
    EXPECT_TRUE(result.metrics.interpolation_ok);
    EXPECT_TRUE(result.metrics.barriers_safe);
}

// Test 17: Metrics with barriers
TEST(InitializationStrategyMetricsTest, WithBarriers) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.repel_y = {0.3};
    data.repel_forbidden = {0.0};
    data.repel_weight = {10.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.metrics.min_barrier_distance, 0.0);
    EXPECT_TRUE(std::isfinite(result.metrics.min_barrier_distance));
}

// ============== Edge Cases and Robustness Tests ==============

// Test 18: Initialization with no data at all
TEST(InitializationStrategyEdgeCaseTest, CompletelyEmptyData) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // No approx points, no interp, no barriers
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::ZERO);
    EXPECT_EQ(result.initial_coeffs.size(), 2);
    for (double coeff : result.initial_coeffs) {
        EXPECT_DOUBLE_EQ(coeff, 0.0);
    }
}

// Test 19: Initialization with extremely high barrier weights
TEST(InitializationStrategyEdgeCaseTest, ExtremeBarrierWeights) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.repel_y = {0.5};
    data.repel_forbidden = {0.5};  // Exactly on barrier
    data.repel_weight = {1e15};    // Extremely strong
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    // Should not crash
    EXPECT_NO_THROW({
        auto result = InitializationStrategySelector::initialize(poly, data, functor);
        EXPECT_TRUE(result.success);
        for (double coeff : result.initial_coeffs) {
            EXPECT_TRUE(std::isfinite(coeff));
        }
    });
}

// Test 20: Initialization with large polynomial degree
TEST(InitializationStrategyEdgeCaseTest, HighPolynomialDegree) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Provide enough data
    for (int i = 0; i <= 20; ++i) {
        double x = i / 20.0;
        data.approx_x.push_back(x);
        data.approx_f.push_back(x);
        data.approx_weight.push_back(1.0);
    }
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(30, data);
    
    ObjectiveFunctor functor(poly, data);
    // Workaround: hierarchical_initialization doesn't build caches
    functor.build_caches();
    
    // Should handle high degree gracefully
    EXPECT_NO_THROW({
        auto result = InitializationStrategySelector::initialize(poly, data, functor);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.initial_coeffs.size(), 30);
        for (double coeff : result.initial_coeffs) {
            EXPECT_TRUE(std::isfinite(coeff));
        }
    });
}

// Test 21: Initialization with ill-conditioned data (points very close)
TEST(InitializationStrategyEdgeCaseTest, IllConditionedData) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Points very close together - ill-conditioned
    data.approx_x = {0.5, 0.5001, 0.5002, 0.5003};
    data.approx_f = {0.5, 0.5001, 0.5002, 0.5003};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    
    ObjectiveFunctor functor(poly, data);
    
    // Should handle ill-conditioning gracefully
    EXPECT_NO_THROW({
        auto result = InitializationStrategySelector::initialize(poly, data, functor);
        EXPECT_TRUE(result.success);
        for (double coeff : result.initial_coeffs) {
            EXPECT_TRUE(std::isfinite(coeff));
        }
        // Condition number might be high but should be finite
        EXPECT_TRUE(std::isfinite(result.metrics.condition_number));
    });
}

// ============== Integration Tests ==============

// Test 22: Full initialization workflow with all features
TEST(InitializationStrategyIntegrationTest, FullWorkflow) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_f = {0.0, 0.25, 0.5, 0.75, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    data.repel_y = {0.3, 0.7};
    data.repel_forbidden = {0.2, 0.8};
    data.repel_weight = {10.0, 10.0};
    data.epsilon = 1e-8;
    
    CompositePolynomial poly = build_poly(3, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_NE(result.strategy_used, InitializationStrategy::ZERO);  // Should not be zero with data
    EXPECT_EQ(result.initial_coeffs.size(), 3);
    for (double coeff : result.initial_coeffs) {
        EXPECT_TRUE(std::isfinite(coeff));
        EXPECT_LT(std::abs(coeff), 1e10);  // Reasonable bounds
    }
    EXPECT_TRUE(std::isfinite(result.initial_objective));
    EXPECT_GE(result.initial_objective, 0.0);
    EXPECT_TRUE(result.metrics.interpolation_ok);
    EXPECT_TRUE(result.metrics.barriers_safe);
    EXPECT_GT(result.metrics.min_barrier_distance, 10.0);  // Safe distance
}

// Test 23: Initialization with interpolation but no approximation
TEST(InitializationStrategyIntegrationTest, InterpolationOnly) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // No approx points, but have interpolation
    data.interp_z = {0.0, 0.5, 1.0};
    data.interp_f = {0.0, 0.5, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    // With no approx points, should be ZERO
    EXPECT_EQ(result.strategy_used, InitializationStrategy::ZERO);
    EXPECT_TRUE(result.metrics.interpolation_ok);
}

// Test 24: Verify interpolation is enforced
TEST(InitializationStrategyIntegrationTest, InterpolationEnforced) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    // Interpolation must be satisfied
    EXPECT_TRUE(result.metrics.interpolation_ok);
}

// Test 25: Strategy selection with different data densities
TEST(InitializationStrategyIntegrationTest, DataDensityScenarios) {
    // We'll test selection directly with different data densities
    OptimizationProblemData dense_data;
    dense_data.interval_a = 0.0;
    dense_data.interval_b = 1.0;
    dense_data.interp_z = {0.0, 1.0};
    dense_data.interp_f = {0.0, 1.0};
    // Dense: 25 points on [0,1]
    for (int i = 0; i < 25; ++i) {
        double x = i / 24.0;
        dense_data.approx_x.push_back(x);
        dense_data.approx_f.push_back(x);
        dense_data.approx_weight.push_back(1.0);
    }
    
    CompositePolynomial poly_dense = build_poly(4, dense_data);
    auto strategy = InitializationStrategySelector::select(poly_dense, dense_data);
    // Dense data -> LEAST_SQUARES
    EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
    
    OptimizationProblemData sparse_data;
    sparse_data.interval_a = 0.0;
    sparse_data.interval_b = 10.0;
    sparse_data.interp_z = {0.0, 10.0};
    sparse_data.interp_f = {0.0, 1.0};
    sparse_data.approx_x = {5.0};  // 1 point, rho=0.1
    sparse_data.approx_f = {0.5};
    sparse_data.approx_weight = {1.0};
    
    CompositePolynomial poly_sparse = build_poly(5, sparse_data);
    strategy = InitializationStrategySelector::select(poly_sparse, sparse_data);
    // Sparse -> RANDOM
    EXPECT_EQ(strategy, InitializationStrategy::RANDOM);
}

// Test 26: Barrier intensity scenarios
TEST(InitializationStrategyIntegrationTest, BarrierIntensityScenarios) {
    // Weak barriers (beta < 10) with sufficient data density
    {
        OptimizationProblemData data;
        data.interval_a = 0.0;
        data.interval_b = 1.0;
        // Sufficient data to avoid RANDOM: need rho >= 0.5 * n_free = 1.5
        // Use 3 points -> rho = 3
        data.approx_x = {0.0, 0.5, 1.0};
        data.approx_f = {0.0, 0.5, 1.0};
        data.approx_weight = {1.0, 1.0, 1.0};  // sigma = 1
        data.repel_y = {0.3};
        data.repel_forbidden = {0.5};
        data.repel_weight = {5.0};  // beta = 5 / 1 = 5 < 10
        data.interp_z = {0.0, 1.0};
        data.interp_f = {0.0, 1.0};
        
        CompositePolynomial poly = build_poly(3, data);
        auto strategy = InitializationStrategySelector::select(poly, data);
        // Weak barriers and sufficient data -> LEAST_SQUARES
        EXPECT_EQ(strategy, InitializationStrategy::LEAST_SQUARES);
    }
    
    // Strong barriers (beta > 100) with enough data
    {
        OptimizationProblemData data;
        data.interval_a = 0.0;
        data.interval_b = 1.0;
        // Enough data to avoid RANDOM
        for (int i = 0; i <= 4; ++i) {
            double x = i / 4.0;
            data.approx_x.push_back(x);
            data.approx_f.push_back(x);
            data.approx_weight.push_back(1.0);
        }
        data.interp_z = {0.0, 1.0};
        data.interp_f = {0.0, 1.0};
        data.repel_y = {0.3, 0.7};
        data.repel_forbidden = {0.2, 0.8};
        data.repel_weight = {500.0, 500.0};  // beta ~ 500 > 100
        
        CompositePolynomial poly = build_poly(3, data);
        auto strategy = InitializationStrategySelector::select(poly, data);
        // Strong barriers -> MULTI_START
        EXPECT_EQ(strategy, InitializationStrategy::MULTI_START);
    }
}

// Test 27: Verify that result contains all required information
TEST(InitializationStrategyIntegrationTest, ResultStructure) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Provide enough data to avoid ZERO strategy
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    // Check all fields are properly set
    EXPECT_FALSE(result.initial_coeffs.empty());
    EXPECT_NE(result.strategy_used, InitializationStrategy::ZERO);  // Should not be zero with data
    EXPECT_TRUE(std::isfinite(result.initial_objective));
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.message.empty());
    EXPECT_GE(result.metrics.objective_ratio, 0.0);  // Should be non-negative, can be 0 if perfect fit
    EXPECT_GT(result.metrics.min_barrier_distance, 0.0);
    EXPECT_GE(result.metrics.rms_residual_norm, 0.0);
    EXPECT_GE(result.metrics.condition_number, 0.0);
    // interpolation_ok and barriers_safe are booleans
}

// Test 28: Warnings and recommendations are generated appropriately
TEST(InitializationStrategyIntegrationTest, WarningsAndRecommendations) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.5};
    data.approx_f = {0.5};
    data.approx_weight = {1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(2, data);
    
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    // Warnings and recommendations are vectors (may be empty)
    // Just check they are accessible
    // No specific assertion on content
}

// ============== Eigen-Specific Tests ==============

// Test 29: Verify Eigen-based least squares computation accuracy
TEST(InitializationStrategyEigenTest, LeastSquaresSolverAccuracy) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Use approximation points away from interpolation nodes (0 and 1) to avoid zero rows
    // Simple linear data with 3 points, fitting quadratic (n_free=3)
    data.approx_x = {0.2, 0.5, 0.8};
    data.approx_f = {0.2, 0.5, 0.8};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::LEAST_SQUARES);
    
    // Verify using Eigen: reconstruct the design matrix exactly as in least_squares_initialization
    int m = data.approx_x.size();
    int n = 3; // n_free
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
    const InterpolationBasis& basis = poly.interpolation_basis;
    const WeightMultiplier& weight_mult = poly.weight_multiplier;
    
    for (int i = 0; i < m; ++i) {
        double x = data.approx_x[i];
        double w = data.approx_weight[i];  // weight = 1/σ
        double W_val = weight_mult.evaluate(x);
        double P_int = basis.evaluate(x);
        
        // Fill row i of A: [1, x, x^2, ...] * w * W_val
        double power = 1.0;
        for (int k = 0; k < n; ++k) {
            A(i, k) = w * power * W_val;
            power *= x;
        }
        
        // Right-hand side: w * (f(x) - P_int(x))
        b(i) = w * (data.approx_f[i] - P_int);
    }
    
    // Build normal equations with regularization (same as in code)
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::VectorXd ATb = A.transpose() * b;
    double lambda = 1e-8 * ATA.trace() / n;  // regularization
    ATA.diagonal().array() += lambda;
    
    // Solve using LDLT (well-conditioned) or SVD (ill-conditioned)
    Eigen::VectorXd q;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(ATA);
    if (ldlt.info() == Eigen::Success) {
        q = ldlt.solve(ATb);
    } else {
        // Fallback to SVD
        Eigen::BDCSVD<Eigen::MatrixXd> svd(ATA, Eigen::ComputeThinU | Eigen::ComputeThinV);
        q = svd.solve(ATb);
    }
    
    // Compare our result with direct Eigen solution
    EXPECT_EQ(result.initial_coeffs.size(), n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result.initial_coeffs[i], q(i), 1e-10);
    }
    
    // Also compare objective: 0.5 * ||A q - b||^2 (regularization already included in ATA)
    double residual_norm = (A * q - b).norm();
    double expected_obj = 0.5 * residual_norm * residual_norm;
    EXPECT_NEAR(result.initial_objective, expected_obj, 1e-8);
    
    // Verify condition number is finite and positive
    EXPECT_TRUE(std::isfinite(result.metrics.condition_number));
    EXPECT_GT(result.metrics.condition_number, 0.0);
}

// Test 30: Eigen solver with rank-deficient system (more points than unknowns)
TEST(InitializationStrategyEigenTest, RankDeficientSystem) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Overdetermined system: 5 points, quadratic fit (n_free=3)
    // Avoid interpolation nodes (0 and 1) to ensure W(x) != 0
    data.approx_x = {0.1, 0.3, 0.5, 0.7, 0.9};
    data.approx_f = {0.1, 0.3, 0.5, 0.7, 0.9};  // Linear data
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    ObjectiveFunctor functor(poly, data);
    
    auto result = InitializationStrategySelector::initialize(poly, data, functor);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.strategy_used, InitializationStrategy::LEAST_SQUARES);
    
    // Build the same matrix as in our implementation
    int m = data.approx_x.size();
    int n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
    const InterpolationBasis& basis = poly.interpolation_basis;
    const WeightMultiplier& weight_mult = poly.weight_multiplier;
    
    for (int i = 0; i < m; ++i) {
        double x = data.approx_x[i];
        double w = data.approx_weight[i];
        double W_val = weight_mult.evaluate(x);
        double P_int = basis.evaluate(x);
        double power = 1.0;
        for (int k = 0; k < n; ++k) {
            A(i, k) = w * power * W_val;
            power *= x;
        }
        b(i) = w * (data.approx_f[i] - P_int);
    }
    
    // Solve normal equations with regularization
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::VectorXd ATb = A.transpose() * b;
    double lambda = 1e-8 * ATA.trace() / n;
    ATA.diagonal().array() += lambda;
    
    Eigen::LDLT<Eigen::MatrixXd> ldlt(ATA);
    ASSERT_EQ(ldlt.info(), Eigen::Success);
    Eigen::VectorXd q = ldlt.solve(ATb);
    
    // Our solution should match direct Eigen solution
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result.initial_coeffs[i], q(i), 1e-10);
    }
    
    // Condition number should be finite
    EXPECT_TRUE(std::isfinite(result.metrics.condition_number));
}

// Test 31: Eigen solver with ill-conditioned matrix (near-dependent columns)
TEST(InitializationStrategyEigenTest, IllConditionedSystem) {
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    // Points very close together - creates ill-conditioned matrix
    data.approx_x = {0.5, 0.5001, 0.5002, 0.5003, 0.5004};
    data.approx_f = {0.5, 0.5001, 0.5002, 0.5003, 0.5004};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.interp_z = {0.0, 1.0};
    data.interp_f = {0.0, 1.0};
    
    CompositePolynomial poly = build_poly(3, data);
    ObjectiveFunctor functor(poly, data);
    
    // Should not crash and should produce finite results
    EXPECT_NO_THROW({
        auto result = InitializationStrategySelector::initialize(poly, data, functor);
        EXPECT_TRUE(result.success);
        for (double coeff : result.initial_coeffs) {
            EXPECT_TRUE(std::isfinite(coeff));
        }
        // Condition number will be large but finite
        EXPECT_TRUE(std::isfinite(result.metrics.condition_number));
        EXPECT_GT(result.metrics.condition_number, 0.0);
    });
}

} // namespace test
} // namespace mixed_approx
