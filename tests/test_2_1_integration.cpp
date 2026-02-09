/**
 * @file test_step_2_1_integration.cpp
 * @brief Интеграционные тесты для шага 2.1: Параметризация полинома с учётом интерполяционных ограничений
 * 
 * Этот файл содержит интеграционные тесты, которые проверяют полный конвейер
 * параметризации "базис + коррекция": F(x) = P_int(x) + Q(x)·W(x)
 * 
 * Тесты охватывают:
 * - Полный цикл разложения от входных данных до верификации
 * - Взаимодействие Decomposer, InterpolationBasis, WeightMultiplier, CorrectionPolynomial
 * - Проверку интерполяционных условий после оптимизации
 * - Крайние случаи (m=0, m=n+1)
 * - Интеграцию с оптимизатором
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/config_reader.h"
#include "mixed_approximation/functional.h"

using namespace mixed_approx;

// =============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// =============================================================================

/**
 * @brief Проверяет, что все интерполяционные условия выполняются
 */
bool verify_all_interpolation_conditions(
    const DecompositionResult& result,
    const std::vector<InterpolationNode>& nodes,
    double tolerance = 1e-8
) {
    std::vector<double> zero_q(result.metadata.n_free, 0.0);
    for (const auto& node : nodes) {
        double F_val = result.evaluate(node.x, zero_q);
        if (std::abs(F_val - node.value) > tolerance) {
            std::cerr << "Interpolation failed at x=" << node.x 
                      << ": F(x)=" << F_val << ", expected=" << node.value 
                      << ", diff=" << std::abs(F_val - node.value) << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Создаёт тестовую конфигурацию MixedApproximation
 */
ApproximationConfig create_test_config(
    int degree,
    double a, double b,
    const std::vector<InterpolationNode>& interp_nodes,
    const std::vector<WeightedPoint>& approx_points = {},
    const std::vector<RepulsionPoint>& repel_points = {},
    double gamma = 0.1
) {
    ApproximationConfig config;
    config.polynomial_degree = degree;
    config.interval_start = a;
    config.interval_end = b;
    config.gamma = gamma;
    config.interp_nodes = interp_nodes;
    config.approx_points = approx_points;
    config.repel_points = repel_points;
    return config;
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: ПОЛНЫЙ КОНВЕЙЕР РАЗЛОЖЕНИЯ
// =============================================================================

TEST(Step2_1_Integration, FullDecompositionPipeline) {
    std::cout << "Testing full decomposition pipeline...\n";
    
    // Типичная задача аппроксимации с интерполяционными условиями
    Decomposer::Parameters params;
    params.polynomial_degree = 8;  // n = 8
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(0.0, 1.0),   // Граничное условие слева
        InterpolationNode(3.0, 4.5),
        InterpolationNode(7.0, 6.2),
        InterpolationNode(10.0, 2.0)   // Граничное условие справа
    };
    
    // Выполняем разложение
    DecompositionResult result = Decomposer::decompose(params);
    
    // Проверяем результат разложения
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
    EXPECT_TRUE(verify_all_interpolation_conditions(result, params.interp_nodes, 1e-8));
    
    std::cout << "  Decomposition pipeline succeeded\n";
}

TEST(Step2_1_Integration, DecompositionWithNonZeroQ) {
    std::cout << "Testing decomposition with non-zero correction polynomial...\n";
    
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
    
    // Ненулевой Q(x) - добавляем возмущение
    std::vector<double> q_coeffs = {0.5, -0.2, 0.1};  // deg_Q = 2
    
    // Проверяем, что F(z_e) = f(z_e) для всех узлов НЕЗАВИСИМО от Q
    for (const auto& node : params.interp_nodes) {
        double F_val = result.evaluate(node.x, q_coeffs);
        EXPECT_NEAR(F_val, node.value, 1e-8)
            << "F(" << node.x << ") = " << F_val << " != " << node.value;
    }
    
    std::cout << "  Non-zero Q test passed\n";
}

TEST(Step2_1_Integration, WeightMultiplierVanishesAtNodes) {
    std::cout << "Testing W(z_e) = 0 property...\n";
    
    WeightMultiplier W;
    std::vector<double> roots = {1.0, 3.0, 5.0, 7.0};
    W.build_from_roots(roots, 0.0, 10.0);
    
    for (double root : roots) {
        double W_val = W.evaluate(root);
        EXPECT_NEAR(W_val, 0.0, 1e-12)
            << "W(" << root << ") should be exactly zero, got " << W_val;
    }
    
    // Проверяем, что W(x) ≠ 0 вне узлов
    EXPECT_GT(std::abs(W.evaluate(0.0)), 1e-6);
    EXPECT_GT(std::abs(W.evaluate(2.0)), 1e-6);
    EXPECT_GT(std::abs(W.evaluate(10.0)), 1e-6);
    
    std::cout << "  Weight multiplier vanishes at nodes test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: КРАЙНИЕ СЛУЧАИ
// =============================================================================

TEST(Step2_1_Integration, NoConstraintsCase) {
    std::cout << "Testing edge case: no interpolation constraints (m = 0)...\n";
    
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
    
    // F(x) = Q(x) при нулевых коэффициентах Q
    std::vector<double> zero_q(5, 0.0);
    EXPECT_NEAR(result.evaluate(0.0, zero_q), 0.0, 1e-12);
    EXPECT_NEAR(result.evaluate(5.0, zero_q), 0.0, 1e-12);
    EXPECT_NEAR(result.evaluate(10.0, zero_q), 0.0, 1e-12);
    
    std::cout << "  No constraints case test passed\n";
}

TEST(Step2_1_Integration, FullInterpolationCase) {
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
    
    // Q(x) вырожден, F(x) = P_int(x)
    std::vector<double> empty_q;
    Polynomial F = result.build_polynomial(empty_q);
    
    // Проверяем, что F точно интерполирует все узлы
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        EXPECT_NEAR(F_val, node.value, 1e-8)
            << "Full interpolation should give exact match at z=" << node.x;
    }
    
    std::cout << "  Full interpolation case test passed\n";
}

TEST(Step2_1_Integration, SingleConstraintCase) {
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
    std::vector<double> zero_q(4, 0.0);
    EXPECT_NEAR(result.evaluate(5.0, zero_q), 7.0, 1e-8);
    
    std::cout << "  Single constraint case test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: ИНТЕГРАЦИЯ С OPTIMIZER
// =============================================================================

TEST(Step2_1_Integration, OptimizerIntegration) {
    std::cout << "Testing integration with optimizer...\n";
    
    ApproximationConfig config = create_test_config(
        3,  // degree
        0.0, 1.0,  // interval
        {
            InterpolationNode(0.0, 0.0),
            InterpolationNode(1.0, 1.0)
        },  // interp_nodes
        {WeightedPoint(0.5, 1.0, 0.5)},  // approx_points
        {},  // repel_points
        0.1  // gamma
    );
    
    // Создаём MixedApproximation - это запускает оптимизацию
    MixedApproximation method(config);
    
    // Проверяем, что оптимизация завершилась
    Polynomial poly = method.get_polynomial();
    ASSERT_FALSE(poly.coefficients().empty());
    
    // Проверяем интерполяционные условия
    EXPECT_TRUE(method.check_interpolation_conditions(1e-6));
    
    // Проверяем, что функционал конечен
    Functional functional(config);
    double J = functional.evaluate(poly);
    EXPECT_FALSE(std::isnan(J));
    EXPECT_FALSE(std::isinf(J));
    
    std::cout << "  Optimizer integration test passed\n";
}

TEST(Step2_1_Integration, FunctionalWithDecomposition) {
    std::cout << "Testing functional evaluation with decomposition...\n";
    
    ApproximationConfig config = create_test_config(
        3, 0.0, 1.0,
        {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 0.0)},
        {WeightedPoint(0.5, 1.0, 1.0)},
        {},
        0.1
    );
    
    // Создаём функционал и проверяем его работу
    Functional functional(config);
    
    // Получаем начальное приближение (P_int)
    MixedApproximation method(config);
    Polynomial poly = method.get_polynomial();
    
    // Вычисляем функционал
    double J = functional.evaluate(poly);
    
    EXPECT_FALSE(std::isnan(J));
    EXPECT_FALSE(std::isinf(J));
    EXPECT_GE(J, 0.0);
    
    // Проверяем компоненты функционала
    auto components = functional.get_components(poly);
    
    std::cout << "  Approx component: " << components.approx_component << "\n";
    std::cout << "  Reg component: " << components.reg_component << "\n";
    std::cout << "  Total: " << components.total << "\n";
    
    // Компонент аппроксимации должен быть положительным (F(0.5) ≈ 0, значение = 1)
    EXPECT_GE(components.approx_component, 0.0);
    
    std::cout << "  Functional evaluation test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: ЧИСЛЕННАЯ УСТОЙЧИВОСТЬ
// =============================================================================

TEST(Step2_1_Integration, LargeIntervalNumericalStability) {
    std::cout << "Testing numerical stability on large interval...\n";
    
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
    
    std::cout << "  Large interval numerical stability test passed\n";
}

TEST(Step2_1_Integration, ManyConstraintsNumericalStability) {
    std::cout << "Testing numerical stability with many constraints...\n";
    
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
    
    std::cout << "  Many constraints numerical stability test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: COMPOSITE POLYNOMIAL
// =============================================================================

TEST(Step2_1_Integration, CompositePolynomialFullPipeline) {
    std::cout << "Testing CompositePolynomial full pipeline...\n";
    
    // Создаём компоненты
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.4, 0.7, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);  // n = 6, m = 4
    Q.initialize_zero();
    
    // Создаём композитный полином
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0, EvaluationStrategy::HYBRID);
    
    ASSERT_TRUE(F.is_valid());
    
    // Проверяем интерполяционные условия - ЭТО КЛЮЧЕВОЕ ТРЕБОВАНИЕ
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "Interpolation must hold at all constraint nodes";
    }
    
    std::cout << "  CompositePolynomial full pipeline test passed\n";
}

TEST(Step2_1_Integration, CompositePolynomialWithNonZeroQ) {
    std::cout << "Testing CompositePolynomial with non-zero Q...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.5, 1.0};  // Q(x) = 0.5 + x
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Интерполяционные условия должны выполняться НЕЗАВИСИМО от Q
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "Interpolation condition must hold at x = " << nodes[i];
    }
    
    std::cout << "  CompositePolynomial with non-zero Q test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: ПРОВЕРКА ПОЛНОГО ЖИЗНЕННОГО ЦИКЛА
// =============================================================================

TEST(Step2_1_Integration, FullLifecycleTest) {
    std::cout << "Testing full lifecycle: create -> decompose -> optimize -> verify...\n";
    
    // 1. Создаём конфигурацию
    ApproximationConfig config = create_test_config(
        4,  // degree
        0.0, 10.0,  // interval
        {
            InterpolationNode(0.0, 1.0),
            InterpolationNode(5.0, 5.0),
            InterpolationNode(10.0, 2.0)
        },  // interp_nodes
        {
            WeightedPoint(2.0, 1.0, 2.0),
            WeightedPoint(8.0, 1.0, 3.0)
        },  // approx_points
        {},  // repel_points
        0.01  // gamma
    );
    
    // 2. Выполняем разложение
    Decomposer::Parameters params;
    params.polynomial_degree = config.polynomial_degree;
    params.interval_start = config.interval_start;
    params.interval_end = config.interval_end;
    params.interp_nodes = config.interp_nodes;
    
    DecompositionResult decomposition = Decomposer::decompose(params);
    ASSERT_TRUE(decomposition.is_valid()) << "Decomposition failed: " << decomposition.message();
    
    // 3. Создаём MixedApproximation (оптимизация)
    MixedApproximation method(config);
    Polynomial result_poly = method.get_polynomial();
    
    // 4. Проверяем результат
    EXPECT_TRUE(method.check_interpolation_conditions(1e-6));
    
    // 5. Вычисляем функционал
    Functional functional(config);
    double J = functional.evaluate(result_poly);
    EXPECT_FALSE(std::isnan(J));
    EXPECT_FALSE(std::isinf(J));
    
    std::cout << "  Final objective value: " << J << "\n";
    std::cout << "  Full lifecycle test passed\n";
}

TEST(Step2_1_Integration, RandomPolynomialReconstruction) {
    std::cout << "Testing random polynomial reconstruction...\n";
    
    // Генерируем случайный полином степени 4
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coeff_dist(-10.0, 10.0);
    
    std::vector<double> true_coeffs(5);
    for (double& c : true_coeffs) c = coeff_dist(rng);
    
    // Создаём интерполяционные узлы на основе значений этого полинома
    std::vector<InterpolationNode> interp_nodes;
    std::vector<double> test_x = {0.0, 0.33, 0.66, 1.0};
    
    Polynomial true_poly(4);
    true_poly.setCoefficients(true_coeffs);
    
    for (double x : test_x) {
        interp_nodes.push_back(InterpolationNode(x, true_poly.evaluate(x)));
    }
    
    // Добавляем ещё один узел чтобы m = n + 1 = 5
    interp_nodes.push_back(InterpolationNode(0.5, true_poly.evaluate(0.5)));
    
    // Создаём конфигурацию для аппроксимации
    ApproximationConfig config = create_test_config(
        4,  // degree
        0.0, 1.0,  // interval
        interp_nodes,
        {},  // approx_points (только интерполяция)
        {},  // repel_points
        0.0  // gamma
    );
    
    // Разложение
    Decomposer::Parameters params;
    params.polynomial_degree = 4;
    params.interval_start = 0.0;
    params.interval_end = 1.0;
    params.interp_nodes = interp_nodes;
    
    DecompositionResult result = Decomposer::decompose(params);
    ASSERT_TRUE(result.is_valid());
    
    // При m = n + 1 = 5, Q вырожден, F(x) = P_int(x)
    // Проверяем, что мы восстановили исходный полином
    std::vector<double> zero_q(result.metadata.n_free, 0.0);  // n_free = 0
    Polynomial F = result.build_polynomial(zero_q);
    
    // Проверяем совпадение в нескольких точках
    std::vector<double> check_x = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (double x : check_x) {
        double true_val = true_poly.evaluate(x);
        double approx_val = F.evaluate(x);
        EXPECT_NEAR(approx_val, true_val, 1e-6)
            << "Reconstructed polynomial differs from original at x=" << x;
    }
    
    std::cout << "  Random polynomial reconstruction test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: ПРОИЗВОДНЫЕ И РЕГУЛЯРИЗАЦИЯ
// =============================================================================

TEST(Step2_1_Integration, SecondDerivativeConsistency) {
    std::cout << "Testing second derivative consistency...\n";
    
    // F(x) = x^2 на [0, 1] с интерполяцией в узлах
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};  // x^2
    // Используем Ньютоновскую форму для более стабильных производных
    basis.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = 0
    Q.coeffs = {0.0};  // Q(x) = 0
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Проверяем вторую производную через численное дифференцирование
    double h = 1e-6;
    std::vector<double> test_points = {0.2, 0.5, 0.8};
    
    for (double x : test_points) {
        // Численная вторая производная
        double f_pp = F.evaluate(x + h);
        double f_pm = F.evaluate(x);
        double f_mm = F.evaluate(x - h);
        double numerical_second = (f_pp - 2.0 * f_pm + f_mm) / (h * h);
        
        // Аналитическая вторая производная
        double analytical_second = F.evaluate_derivative(x, 2);
        
        // Ожидаем F''(x) = 2 для F(x) = x^2
        // Увеличиваем допуск для численной проверки
        double rel_diff = std::abs(numerical_second - analytical_second) / 
                         (std::abs(numerical_second) + 1e-10);
        
        std::cout << "  x = " << x << ": numerical = " << numerical_second 
                  << ", analytical = " << analytical_second << "\n";
        
        // Проверяем, что вторая производная близка к 2
        EXPECT_NEAR(analytical_second, 2.0, 1e-3)
            << "Analytical second derivative should be 2.0 at x = " << x;
        EXPECT_NEAR(numerical_second, 2.0, 1e-3)
            << "Numerical second derivative should be 2.0 at x = " << x;
    }
    
    std::cout << "  Second derivative consistency test passed\n";
}

TEST(Step2_1_Integration, RegularizationTerm) {
    std::cout << "Testing regularization term computation...\n";
    
    // F(x) = x^2 на [0, 1], F''(x) = 2, ∫₀¹ 4 dx = 4
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.0};  // Q(x) = 0, F(x) = P_int(x) = x^2
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Проверяем регуляризационный член
    double gamma = 1.0;
    double reg = F.compute_regularization_term(gamma);
    
    // Для F(x) = x^2 на [0,1]: ∫₀¹ 4 dx = 4
    EXPECT_NEAR(reg, 4.0, 1e-2);
    
    // Проверяем линейность по gamma
    double gamma2 = 2.0;
    double reg2 = F.compute_regularization_term(gamma2);
    EXPECT_NEAR(reg2 / reg, 2.0, 1e-6);
    
    std::cout << "  Regularization term test passed\n";
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ: КЭШИРОВАНИЕ
// =============================================================================

TEST(Step2_1_Integration, CacheBuilding) {
    std::cout << "Testing cache building for optimization...\n";
    
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
    
    std::cout << "  Cache building test passed\n";
}

TEST(Step2_1_Integration, BatchEvaluation) {
    std::cout << "Testing batch evaluation performance...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.1};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<double> points = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    
    // Пакетная оценка
    std::vector<double> batch_results;
    F.evaluate_batch(points, batch_results);
    
    // Поточечная оценка
    std::vector<double> single_results;
    for (double x : points) {
        single_results.push_back(F.evaluate(x));
    }
    
    ASSERT_EQ(batch_results.size(), single_results.size());
    
    for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_NEAR(batch_results[i], single_results[i], 1e-12)
            << "Batch and single evaluations should match at index " << i;
    }
    
    std::cout << "  Batch evaluation test passed\n";
}

// =============================================================================
// КОНЕЦ ФАЙЛА
// =============================================================================
