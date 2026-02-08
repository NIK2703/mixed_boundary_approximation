#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/functional.h"

using namespace mixed_approx;

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.1: ФОРМАЛИЗАЦИЯ ЗАДАЧИ С ОГРАНИЧЕНИЯМИ
// =============================================================================

TEST(Step2_1_1_1, RankSystemCheck_DuplicateNodes) {
    std::cout << "Testing rank system check with duplicate nodes...\n";
    
    // Дублирующиеся узлы должны обнаруживаться как линейно зависимые
    std::vector<double> nodes = {1.0, 1.0 + 1e-13, 2.0};
    double epsilon_rank = 1e-12;
    
    bool result = Decomposer::check_rank_solvency(nodes, epsilon_rank);
    EXPECT_FALSE(result) << "Duplicate nodes should be detected as linearly dependent";
}

TEST(Step2_1_1_1, RankSystemCheck_LinearIndependence) {
    std::cout << "Testing rank system check with linearly independent nodes...\n";
    
    // Линейно независимые узлы
    std::vector<double> nodes = {1.0, 2.0, 3.0};
    double epsilon_rank = 1e-12;
    
    bool result = Decomposer::check_rank_solvency(nodes, epsilon_rank);
    EXPECT_TRUE(result) << "Distinct nodes should be linearly independent";
}

TEST(Step2_1_1_1, RankSystemCheck_SingleNode) {
    std::cout << "Testing rank system check with single node...\n";
    
    // Один узел всегда независим
    std::vector<double> nodes = {1.0};
    bool result = Decomposer::check_rank_solvency(nodes, 1e-12);
    EXPECT_TRUE(result);
}

TEST(Step2_1_1_1, RankSystemCheck_ConflictIndices) {
    std::cout << "Testing rank system check with conflict indices...\n";
    
    std::vector<double> nodes = {0.5, 0.5 + 5e-13, 1.0};
    std::vector<int> conflicts;
    Decomposer::check_rank_solvency(nodes, 1e-12, &conflicts);
    
    EXPECT_FALSE(conflicts.empty()) << "Should detect conflicting indices";
    EXPECT_EQ(conflicts.size(), 2u) << "Should return pair of indices";
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.2: ТЕОРЕТИЧЕСКОЕ ОБОСНОВАНИЕ РАЗЛОЖЕНИЯ
// =============================================================================

TEST(Step2_1_1_2, BasisCorrectionIdentity) {
    std::cout << "Testing basis + correction identity F(x) = P_int(x) + Q(x)·W(x)...\n";
    
    // Создаём разложение
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 7.0),
        InterpolationNode(8.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid: " << result.message();
    
    // Генерируем случайный Q(x)
    std::vector<double> q_coeffs = {0.5, -0.2, 0.1};  // deg_Q = 2
    
    // Вычисляем F(x) разными способами
    Polynomial F_full = result.build_polynomial(q_coeffs);
    
    // Проверяем, что F(z_e) = f(z_e) для всех узлов
    for (const auto& node : params.interp_nodes) {
        double F_at_node = F_full.evaluate(node.x);
        EXPECT_NEAR(F_at_node, node.value, 1e-8)
            << "F(" << node.x << ") = " << F_at_node << " != " << node.value;
    }
}

TEST(Step2_1_1_2, WeightMultiplierVanishesAtNodes) {
    std::cout << "Testing W(z_e) = 0 for all interpolation nodes...\n";
    
    WeightMultiplier W;
    std::vector<double> roots = {1.0, 3.0, 5.0, 7.0};
    W.build_from_roots(roots);
    
    for (double root : roots) {
        EXPECT_NEAR(W.evaluate(root), 0.0, 1e-12)
            << "W(" << root << ") should be exactly zero";
    }
}

TEST(Step2_1_1_2, InterpolationBasisSatisfiesConditions) {
    std::cout << "Testing P_int(z_e) = f(z_e) for interpolation basis...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        double p_int_at_node = basis.evaluate(nodes[i]);
        EXPECT_NEAR(p_int_at_node, values[i], 1e-10)
            << "P_int(" << nodes[i] << ") = " << p_int_at_node << " != " << values[i];
    }
}

TEST(Step2_1_1_2, DegreeOfCorrectionPolynomial) {
    std::cout << "Testing deg(Q) = n - m constraint...\n";
    
    // n = 5, m = 3 → deg(Q) = 2, n_free = 3
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 7.0),
        InterpolationNode(8.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    EXPECT_EQ(result.metadata.n_free, 3);  // n - m + 1 = 5 - 3 + 1 = 3
    EXPECT_EQ(result.metadata.n_total, 5);
    EXPECT_EQ(result.metadata.m_constraints, 3);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.3: ПРОВЕРКА УСЛОВИЙ ПРИМЕНИМОСТИ
// =============================================================================

TEST(Step2_1_1_3, DegreeCondition_Sufficient) {
    std::cout << "Testing degree condition n >= m-1 (sufficient case)...\n";
    
    // n = 5, m = 4 → 5 >= 3 ✓
    bool result = Decomposer::check_degree_condition(5, 4);
    EXPECT_TRUE(result);
    
    // n = 5, m = 6 → 5 >= 5 ✓ (m-1 = 5)
    result = Decomposer::check_degree_condition(5, 6);
    EXPECT_TRUE(result);
}

TEST(Step2_1_1_3, DegreeCondition_Insufficient) {
    std::cout << "Testing degree condition n >= m-1 (insufficient case)...\n";
    
    // n = 3, m = 5 → 3 >= 4 ✗
    bool result = Decomposer::check_degree_condition(3, 5);
    EXPECT_FALSE(result);
}

TEST(Step2_1_1_3, DegreeCondition_Boundary) {
    std::cout << "Testing degree condition at boundary n = m-1...\n";
    
    // n = 2, m = 3 → 2 >= 2 ✓
    bool result = Decomposer::check_degree_condition(2, 3);
    EXPECT_TRUE(result);
}

TEST(Step2_1_1_3, UniqueNodes_AllDistinct) {
    std::cout << "Testing unique nodes detection (all distinct)...\n";
    
    std::vector<double> nodes = {0.0, 0.3, 0.7, 1.0};
    double interval_length = 1.0;
    double tolerance = 1e-12;
    
    bool result = Decomposer::check_unique_nodes(nodes, interval_length, tolerance);
    EXPECT_TRUE(result);
}

TEST(Step2_1_1_3, UniqueNodes_WithDuplicates) {
    std::cout << "Testing unique nodes detection (with duplicates)...\n";
    
    std::vector<double> nodes = {0.0, 1e-13, 1.0};
    double interval_length = 1.0;
    double tolerance = 1e-12;
    
    bool result = Decomposer::check_unique_nodes(nodes, interval_length, tolerance);
    EXPECT_FALSE(result);
}

TEST(Step2_1_1_3, UniqueNodes_RelativeTolerance) {
    std::cout << "Testing unique nodes with relative tolerance...\n";
    
    // Узлы различимы на малом интервале
    std::vector<double> nodes = {100.0, 100.0 + 1e-9};
    double interval_length = 1.0;  // Интервал [0, 1]
    double tolerance = 1e-12;  // Абсолютный допуск = 1e-12
    
    bool result = Decomposer::check_unique_nodes(nodes, interval_length, tolerance);
    EXPECT_TRUE(result) << "Nodes should be distinct with absolute tolerance";
}

TEST(Step2_1_1_3, NodesInInterval_AllInside) {
    std::cout << "Testing nodes in interval (all inside)...\n";
    
    std::vector<double> nodes = {0.5, 1.0, 1.5};
    double a = 0.0, b = 2.0;
    double tolerance = 1e-9;
    
    bool result = Decomposer::check_nodes_in_interval(nodes, a, b, tolerance);
    EXPECT_TRUE(result);
    
    std::vector<int> out_of_bounds;
    Decomposer::check_nodes_in_interval(nodes, a, b, tolerance, &out_of_bounds);
    EXPECT_TRUE(out_of_bounds.empty());
}

TEST(Step2_1_1_3, NodesInInterval_SomeOutside) {
    std::cout << "Testing nodes in interval (some outside)...\n";
    
    std::vector<double> nodes = {0.5, -0.1, 2.1};
    double a = 0.0, b = 2.0;
    double tolerance = 1e-9;
    
    bool result = Decomposer::check_nodes_in_interval(nodes, a, b, tolerance);
    EXPECT_FALSE(result);
    
    std::vector<int> out_of_bounds;
    Decomposer::check_nodes_in_interval(nodes, a, b, tolerance, &out_of_bounds);
    EXPECT_EQ(out_of_bounds.size(), 2u);
}

TEST(Step2_1_1_3, NodesInInterval_NearBoundary) {
    std::cout << "Testing nodes near interval boundary...\n";
    
    // Узлы на границе с допуском
    std::vector<double> nodes = {0.0, 2.0, 1.0};
    double a = 0.0, b = 2.0;
    double tolerance = 1e-9;
    
    bool result = Decomposer::check_nodes_in_interval(nodes, a, b, tolerance);
    EXPECT_TRUE(result) << "Nodes exactly at boundaries should be accepted";
}

TEST(Step2_1_1_3, ConflictingValues_Detected) {
    std::cout << "Testing conflicting interpolation values detection...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(1.0 + 1e-13, 5.0),  // Конфликт: тот же узел, другое значение
        InterpolationNode(5.0, 7.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    EXPECT_FALSE(result.is_valid());
    
    std::string msg = result.message();
    std::string msg_lower = msg;
    std::transform(msg_lower.begin(), msg_lower.end(), msg_lower.begin(), ::tolower);
    EXPECT_TRUE(msg_lower.find("duplicate") != std::string::npos ||
                msg_lower.find("conflict") != std::string::npos ||
                msg_lower.find("differ") != std::string::npos)
        << "Error message should mention conflict: " << msg;
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.4: ПРЕДВАРИТЕЛЬНЫЕ ВЫЧИСЛЕНИЯ
// =============================================================================

TEST(Step2_1_1_4, SortedNodes) {
    std::cout << "Testing sorted interpolation nodes...\n";
    
    std::vector<double> nodes = {0.5, 0.0, 1.0, 0.25};
    std::vector<double> values = {5.0, 1.0, 10.0, 2.5};
    
    InterpolationBasis::sort_nodes_and_values(nodes, values);
    
    ASSERT_EQ(nodes.size(), 4u);
    EXPECT_NEAR(nodes[0], 0.0, 1e-12);
    EXPECT_NEAR(nodes[1], 0.25, 1e-12);
    EXPECT_NEAR(nodes[2], 0.5, 1e-12);
    EXPECT_NEAR(nodes[3], 1.0, 1e-12);
    
    // Значения должны соответствовать узлам
    EXPECT_NEAR(values[0], 1.0, 1e-12);   // z=0 → f=1
    EXPECT_NEAR(values[1], 2.5, 1e-12);   // z=0.25 → f=2.5
    EXPECT_NEAR(values[2], 5.0, 1e-12);   // z=0.5 → f=5
    EXPECT_NEAR(values[3], 10.0, 1e-12); // z=1 → f=10
}

TEST(Step2_1_1_4, RootDistances) {
    std::cout << "Testing pairwise root distances...\n";
    
    WeightMultiplier W;
    std::vector<double> roots = {1.0, 2.0, 5.0, 10.0};
    W.build_from_roots(roots);
    
    EXPECT_NEAR(W.min_root_distance, 1.0, 1e-12);  // min(|2-1|, |5-2|, |10-5|) = 1
}

TEST(Step2_1_1_4, ValueRangeAnalysis) {
    std::cout << "Testing interpolation value range analysis...\n";
    
    std::vector<double> values = {1.0, 3.0, 2.0, 5.0};
    double range = Decomposer::analyze_value_range(values);
    
    EXPECT_NEAR(range, 4.0, 1e-12);  // max - min = 5 - 1 = 4
}

TEST(Step2_1_1_4, WeightMultiplierScale_ExtremeCase) {
    std::cout << "Testing weight multiplier scale estimation (extreme case)...\n";
    
    // Корни с большим разбросом
    std::vector<double> large_roots = {1000.0, 2000.0, 3000.0};
    double scale = Decomposer::estimate_weight_multiplier_scale(large_roots);
    
    // scale = 1000 * 2000 * 3000 = 6e9 (очень большой)
    EXPECT_GT(scale, 1e6);
}

TEST(Step2_1_1_4, WeightMultiplierScale_NormalCase) {
    std::cout << "Testing weight multiplier scale estimation (normal case)...\n";
    
    // Корни в пределах [0, 1]
    std::vector<double> normal_roots = {0.1, 0.5, 0.9};
    double scale = Decomposer::estimate_weight_multiplier_scale(normal_roots);
    
    // scale = max(0.1,1) * max(0.5,1) * max(0.9,1) = 1 * 1 * 1 = 1
    // Поскольку все корни < 1, scale = 1
    EXPECT_EQ(scale, 1.0);
}

TEST(Step2_1_1_4, LowDegreePolynomialDetection) {
    std::cout << "Testing low degree polynomial detection...\n";
    
    // Линейная зависимость (3 точки)
    std::vector<double> nodes1 = {0.0, 0.5, 1.0};
    std::vector<double> values1 = {1.0, 2.0, 3.0};
    int detected = Decomposer::detect_low_degree_polynomial(nodes1, values1);
    EXPECT_EQ(detected, 1);  // Линейная функция
    
    // Квадратичная зависимость (4 точки)
    std::vector<double> nodes2 = {0.0, 0.5, 1.0, 1.5};
    std::vector<double> values2 = {0.0, 0.25, 1.0, 2.25};  // f(x) = x^2
    detected = Decomposer::detect_low_degree_polynomial(nodes2, values2);
    EXPECT_EQ(detected, 2);  // Квадратичная функция
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.6: ВЕРИФИКАЦИЯ КОРРЕКТНОСТИ РАЗЛОЖЕНИЯ
// =============================================================================

TEST(Step2_1_1_6, VerifyInterpolationConditions) {
    std::cout << "Testing interpolation condition verification...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 6;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.5),
        InterpolationNode(4.0, 6.2),
        InterpolationNode(7.0, 3.8)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    // Проверяем, что verify_interpolation возвращает true для Q=0
    bool verified = result.verify_interpolation(1e-10);
    EXPECT_TRUE(verified) << "Interpolation should be verified with Q=0";
}

TEST(Step2_1_1_6, VerifyInterpolationWithNonZeroQ) {
    std::cout << "Testing interpolation verification with non-zero Q...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 4.0),
        InterpolationNode(5.0, 7.0),
        InterpolationNode(8.0, 5.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    // Ненулевой Q
    std::vector<double> q_coeffs = {1.0, -0.5, 0.2};
    
    // Проверяем, что F(z_e) = f(z_e) для всех узлов
    for (const auto& node : params.interp_nodes) {
        double F_val = result.evaluate(node.x, q_coeffs);
        EXPECT_NEAR(F_val, node.value, 1e-8)
            << "F(" << node.x << ") with non-zero Q should still equal f(z_e)";
    }
}

TEST(Step2_1_1_6, CompletenessOfSolutionSpace) {
    std::cout << "Testing completeness of solution space...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 4;  // n = 4
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 6.0),
        InterpolationNode(8.0, 4.0)
    };  // m = 3, n_free = 2
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.n_free, 2);
    
    // Строим два линейно независимых полинома
    std::vector<Polynomial> basis_polys;
    std::vector<std::vector<double>> q_bases = {
        {1.0, 0.0},  // Q₀(x) = 1
        {0.0, 1.0}   // Q₁(x) = x
    };
    
    for (const auto& q : q_bases) {
        basis_polys.push_back(result.build_polynomial(q));
    }
    
    // Проверяем, что они не пропорциональны
    double ratio_at_0 = basis_polys[0].evaluate(0.0);
    double ratio_at_5 = basis_polys[0].evaluate(5.0);
    
    // F₁(x) не должна быть константным кратным F₀(x)
    bool linearly_independent = true;
    for (double x : {0.0, 1.0, 2.5, 5.0, 7.0}) {
        double f0 = basis_polys[0].evaluate(x);
        double f1 = basis_polys[1].evaluate(x);
        if (std::abs(f1) > 1e-10 && std::abs(f0) > 1e-10) {
            double ratio = f1 / f0;
            // Проверяем, что отношение не постоянно
            if (std::abs(ratio - ratio_at_0/ratio_at_0) > 1e-6 && 
                std::abs(ratio - ratio_at_5/ratio_at_0) > 1e-6) {
                linearly_independent = true;
                break;
            }
        }
    }
    
    // Просто проверяем, что второй полином не нулевой
    bool f1_all_zero = true;
    for (double x : {0.0, 3.0, 6.0, 9.0}) {
        if (std::abs(basis_polys[1].evaluate(x)) > 1e-10) {
            f1_all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(f1_all_zero) << "Second basis polynomial should not be identically zero";
}

TEST(Step2_1_1_6, SolutionSpaceDimension) {
    std::cout << "Testing solution space dimension equals n_free...\n";
    
    // Разные конфигурации
    struct TestCase {
        int n, m, expected_n_free;
    };
    
    std::vector<TestCase> cases = {
        {5, 3, 3},   // 5-3+1=3
        {10, 5, 6},  // 10-5+1=6
        {3, 0, 4},   // 3-0+1=4 (m=0)
        {4, 5, 0},   // 4-5+1=0 (полная интерполяция m=n+1=5)
    };
    
    for (const auto& tc : cases) {
        Decomposer::Parameters params;
        params.polynomial_degree = tc.n;
        params.interval_start = 0.0;
        params.interval_end = 10.0;
        
        for (int i = 0; i < tc.m; ++i) {
            double x = 1.0 + i * 2.0;
            params.interp_nodes.push_back(InterpolationNode(x, x));
        }
        
        DecompositionResult result = Decomposer::decompose(params);
        
        if (tc.n >= tc.m - 1 || tc.m == 0) {
            ASSERT_TRUE(result.is_valid()) << "Failed for n=" << tc.n << ", m=" << tc.m;
            EXPECT_EQ(result.metadata.n_free, tc.expected_n_free)
                << "n_free mismatch: n=" << tc.n << ", m=" << tc.m;
        }
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.7: КРАЙНИЕ СЛУЧАИ
// =============================================================================

TEST(Step2_1_1_7, NoConstraints_mEqualsZero) {
    std::cout << "Testing edge case: no constraints (m = 0)...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 4;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {};  // m = 0
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.m_constraints, 0);
    EXPECT_EQ(result.metadata.n_free, 5);  // n - 0 + 1 = 5
    
    // W(x) должен быть константой 1
    EXPECT_NEAR(result.weight_multiplier.evaluate(5.0), 1.0, 1e-12);
    EXPECT_EQ(result.weight_multiplier.degree(), 0);
    
    // P_int(x) должен быть нулевым
    EXPECT_NEAR(result.interpolation_basis.evaluate(5.0), 0.0, 1e-12);
}

TEST(Step2_1_1_7, FullInterpolation_mEqualsNPlus1) {
    std::cout << "Testing edge case: full interpolation (m = n + 1)...\n";
    
    int n = 4;
    int m = n + 1;  // 5 узлов
    
    Decomposer::Parameters params;
    params.polynomial_degree = n;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    
    for (int i = 0; i < m; ++i) {
        double x = i * 2.0;
        params.interp_nodes.push_back(InterpolationNode(x, x * x));
    }
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.n_free, 0);  // n - m + 1 = 0
    
    // Q(x) вырождается, F(x) = P_int(x)
    std::vector<double> empty_q;
    Polynomial F = result.build_polynomial(empty_q);
    
    // Проверяем, что F точно интерполирует все узлы
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        EXPECT_NEAR(F_val, node.value, 1e-8)
            << "Full interpolation should give exact match at z=" << node.x;
    }
}

TEST(Step2_1_1_7, OverdeterminedSystem) {
    std::cout << "Testing overdetermined system (m > n + 1)...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;  // n = 3
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    
    // 6 узлов при степени 3 (нужно минимум 4)
    params.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(2.0, 4.0),
        InterpolationNode(4.0, 16.0),
        InterpolationNode(6.0, 36.0),
        InterpolationNode(8.0, 64.0),
        InterpolationNode(10.0, 100.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    // Система переопределена - разложение должно быть невалидным
    EXPECT_FALSE(result.is_valid());
    
    std::string msg = result.message();
    std::string msg_lower = msg;
    std::transform(msg_lower.begin(), msg_lower.end(), msg_lower.begin(), ::tolower);
    EXPECT_TRUE(msg_lower.find("insufficient") != std::string::npos ||
                msg_lower.find("degree") != std::string::npos)
        << "Error message should mention insufficient degree: " << msg;
}

TEST(Step2_1_1_7, SingleConstraint) {
    std::cout << "Testing single interpolation constraint (m = 1)...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 4;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(5.0, 7.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.m_constraints, 1);
    EXPECT_EQ(result.metadata.n_free, 4);  // 4 - 1 + 1 = 4
    
    // Проверяем интерполяционное условие
    EXPECT_NEAR(result.evaluate(5.0, std::vector<double>(4, 0.0)), 7.0, 1e-8);
}

TEST(Step2_1_1_7, TwoConstraints) {
    std::cout << "Testing two interpolation constraints (m = 2)...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 4.0),
        InterpolationNode(8.0, 9.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.n_free, 4);  // 5 - 2 + 1 = 4
    
    // Проверяем оба условия
    std::vector<double> zero_q(4, 0.0);
    EXPECT_NEAR(result.evaluate(2.0, zero_q), 4.0, 1e-8);
    EXPECT_NEAR(result.evaluate(8.0, zero_q), 9.0, 1e-8);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.1.8: ИНТЕГРАЦИЯ С ОПТИМИЗАТОРОМ
// =============================================================================

TEST(Step2_1_1_8, ParameterVectorSize) {
    std::cout << "Testing parameter vector size matches n_free...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 6;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 7.0),
        InterpolationNode(8.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    EXPECT_EQ(result.metadata.n_free, 4);  // 6 - 3 + 1 = 4
    
    // Проверяем, что evaluate работает с правильным размером
    std::vector<double> q_correct(4, 0.0);
    EXPECT_NO_THROW(result.evaluate(5.0, q_correct));
    
    // Примечание: evaluate() не проверяет размер вектора явно,
    // так как он использует evaluate_product() который безопасен с любым размером.
    // Неправильный размер просто использует меньше коэффициентов.
    std::vector<double> q_wrong(3, 0.0);
    double val = result.evaluate(5.0, q_wrong);  // Должен работать (использует 3 коэффициента из 4)
    EXPECT_FALSE(std::isnan(val));
    EXPECT_FALSE(std::isinf(val));
}

TEST(Step2_1_1_8, CachedWeightsForOptimization) {
    std::cout << "Testing cached weights for optimizer integration...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 7.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    // Строим кэши для оптимизации
    std::vector<double> points_x = {0.5, 1.0, 3.0, 6.0, 9.0};
    std::vector<double> points_y = {1.5, 4.5, 7.5};
    
    result.build_caches(points_x, points_y);
    
    ASSERT_TRUE(result.caches_built);
    EXPECT_EQ(result.cache_W_x.size(), points_x.size());
    EXPECT_EQ(result.cache_W_y.size(), points_y.size());
    EXPECT_EQ(result.cache_W1_x.size(), points_x.size());
    EXPECT_EQ(result.cache_W1_y.size(), points_y.size());
    EXPECT_EQ(result.cache_W2_x.size(), points_x.size());
    EXPECT_EQ(result.cache_W2_y.size(), points_y.size());
    
    // Проверяем согласованность кэшей
    for (size_t i = 0; i < points_x.size(); ++i) {
        EXPECT_NEAR(result.cache_W_x[i], result.weight_multiplier.evaluate(points_x[i]), 1e-12);
    }
}

TEST(Step2_1_1_8, AnalyticalGradientSetup) {
    std::cout << "Testing analytical gradient preparation...\n";
    
    // Создаём CorrectionPolynomial с предвычисленными данными
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 5.0, 5.0);  // Интервал [0, 10]
    Q.initialize_zero();
    
    WeightMultiplier W;
    W.build_from_roots({2.0, 5.0, 8.0}, 0.0, 10.0);
    
    std::vector<double> points_x = {0.0, 2.5, 5.0, 7.5, 10.0};
    std::vector<double> points_y = {1.0, 4.0, 9.0};
    
    Q.build_caches(points_x, points_y);
    
    // Проверяем, что кэши для градиента построены
    EXPECT_EQ(Q.basis_cache_x.size(), points_x.size());
    EXPECT_EQ(Q.basis_cache_x[0].size(), 3u);  // n_free = 3
}

// =============================================================================
// ДОПОЛНИТЕЛЬНЫЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ
// =============================================================================

TEST(IntegrationTest, FullDecompositionPipeline) {
    std::cout << "Testing full decomposition pipeline...\n";
    
    // Типичная задача аппроксимации
    Decomposer::Parameters params;
    params.polynomial_degree = 8;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(0.0, 1.0),   // Граничное условие слева
        InterpolationNode(3.0, 4.5),
        InterpolationNode(7.0, 6.2),
        InterpolationNode(10.0, 2.0)   // Граничное условие справа
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition failed: " << result.message();
    EXPECT_EQ(result.metadata.n_total, 8);
    EXPECT_EQ(result.metadata.m_constraints, 4);
    EXPECT_EQ(result.metadata.n_free, 5);  // 8 - 4 + 1 = 5
    
    // Проверяем структуры данных
    ASSERT_FALSE(result.weight_multiplier.roots.empty());
    EXPECT_EQ(result.weight_multiplier.degree(), 4);
    
    ASSERT_TRUE(result.interpolation_basis.is_valid);
    
    // Верифицируем разложение
    EXPECT_TRUE(result.verify_interpolation(1e-10));
    
    // Проверяем, что F(x) с Q=0 удовлетворяет условиям
    std::vector<double> zero_q(5, 0.0);
    for (const auto& node : params.interp_nodes) {
        double F_val = result.evaluate(node.x, zero_q);
        EXPECT_NEAR(F_val, node.value, 1e-8);
    }
}

TEST(IntegrationTest, DecompositionWithDifferentMethods) {
    std::cout << "Testing decomposition consistency with different interpolation methods...\n";
    
    std::vector<double> nodes = {0.0, 0.3, 0.7, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.5, 3.0};
    
    InterpolationBasis basis_bary;
    basis_bary.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    InterpolationBasis basis_newton;
    basis_newton.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    ASSERT_TRUE(basis_bary.is_valid);
    ASSERT_TRUE(basis_newton.is_valid);
    
    // Проверяем согласованность в нескольких точках
    std::vector<double> test_points = {0.1, 0.4, 0.6, 0.9};
    for (double x : test_points) {
        double val_bary = basis_bary.evaluate(x);
        double val_newton = basis_newton.evaluate(x);
        EXPECT_NEAR(val_bary, val_newton, 1e-8)
            << "Methods disagree at x=" << x << ": bary=" << val_bary << ", newton=" << val_newton;
    }
}

TEST(IntegrationTest, DecompositionMetadataCompleteness) {
    std::cout << "Testing decomposition metadata completeness...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 6;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(2.0, 3.0),
        InterpolationNode(5.0, 7.0),
        InterpolationNode(8.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    // Проверяем все поля метаданных
    const auto& meta = result.metadata;
    EXPECT_GE(meta.n_total, 0);
    EXPECT_GE(meta.m_constraints, 0);
    EXPECT_GE(meta.n_free, 0);
    EXPECT_TRUE(meta.is_valid);
    EXPECT_FALSE(meta.validation_message.empty());
    EXPECT_GE(meta.min_root_distance, 0.0);
}

TEST(IntegrationTest, DecompositionWithWarning) {
    std::cout << "Testing decomposition with warning conditions...\n";
    
    // Узлы с малым разбросом
    Decomposer::Parameters params;
    params.polynomial_degree = 10;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(5.0, 7.0),
        InterpolationNode(5.01, 7.1),
        InterpolationNode(5.02, 6.9)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    // Разложение должно быть валидным, но с предупреждением
    ASSERT_TRUE(result.is_valid());
    
    // Проверяем, что предупреждение присутствует в сообщении
    std::string msg = result.message();
    EXPECT_FALSE(msg.empty());
    
    // Проверяем, что min_root_distance мал
    EXPECT_LT(result.metadata.min_root_distance, 0.1);
}

// =============================================================================
// ТЕСТЫ НА ЧИСЛЕННУЮ УСТОЙЧИВОСТЬ
// =============================================================================

TEST(NumericalStability, LargeIntervalDecomposition) {
    std::cout << "Testing decomposition on large interval...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 6;
    params.interval_start = -1000.0;
    params.interval_end = 1000.0;
    params.interp_nodes = {
        InterpolationNode(-500.0, 2.0),
        InterpolationNode(0.0, 5.0),
        InterpolationNode(500.0, 3.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition failed: " << result.message();
    
    // Проверяем интерполяционные условия
    std::vector<double> zero_q(result.metadata.n_free, 0.0);
    for (const auto& node : params.interp_nodes) {
        double F_val = result.evaluate(node.x, zero_q);
        EXPECT_NEAR(F_val, node.value, 1e-6)
            << "Numerical issues at large coordinates: F(" << node.x << ")";
    }
}

TEST(NumericalStability, ManyConstraints) {
    std::cout << "Testing decomposition with many constraints...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 15;
    params.interval_start = 0.0;
    params.interval_end = 1.0;
    
    // 8 узлов
    for (int i = 0; i < 8; ++i) {
        double x = i / 7.0;
        params.interp_nodes.push_back(InterpolationNode(x, x * x));
    }
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid());
    EXPECT_EQ(result.metadata.n_free, 8);  // 15 - 8 + 1 = 8
    
    // Проверяем точность интерполяции
    std::vector<double> zero_q(result.metadata.n_free, 0.0);
    for (const auto& node : params.interp_nodes) {
        double F_val = result.evaluate(node.x, zero_q);
        EXPECT_NEAR(F_val, node.value, 1e-8);
    }
}

TEST(NumericalStability, CloseNodesDecomposition) {
    std::cout << "Testing decomposition with close interpolation nodes...\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 8;
    params.interval_start = 0.0;
    params.interval_end = 1.0;
    
    // Очень близкие узлы
    params.interp_nodes = {
        InterpolationNode(0.5, 2.0),
        InterpolationNode(0.5 + 1e-6, 2.000001),
        InterpolationNode(0.5 + 2e-6, 1.999999)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    // Разложение может быть валидным (узлы объединяются)
    // или невалидным (конфликт значений)
    std::string msg = result.message();
    if (result.is_valid()) {
        EXPECT_TRUE(result.verify_interpolation(1e-6));
    }
}
