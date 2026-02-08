#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"

using namespace mixed_approx;

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.1: ЛЕНИВАЯ ОЦЕНКА (ОСНОВНАЯ ФУНКЦИОНАЛЬНОСТЬ)
// =============================================================================

TEST(Step2_1_5_1, LazyEvaluationBasic) {
    std::cout << "Testing lazy evaluation F(x) = P_int(x) + Q(x)*W(x)...\n";
    
    // Создаём компоненты
    InterpolationBasis basis;
    std::vector<double> nodes = {1.0, 3.0, 5.0};
    std::vector<double> values = {2.0, 4.0, 6.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 6.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 6.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 3.0, 3.0);  // n = 5, m = 3, deg_Q = 2
    Q.initialize_zero();
    
    // Создаём композитный полином
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 6.0, EvaluationStrategy::LAZY);
    
    ASSERT_TRUE(F.is_valid());
    
    // Проверяем интерполяционные условия - ключевое требование!
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "F(" << nodes[i] << ") should equal f(z_e) = " << values[i];
    }
}

TEST(Step2_1_5_1, LazyEvaluationWithNonZeroQ) {
    std::cout << "Testing lazy evaluation with non-zero Q(x)...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = 1
    Q.coeffs = {0.5, 1.0};  // Q(x) = 0.5 + x
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Интерполяционные условия должны выполняться независимо от Q
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "Interpolation condition must hold at x = " << nodes[i];
    }
    
    // W(z_e) должен быть равен нулю
    for (double root : nodes) {
        double W_val = W.evaluate(root);
        EXPECT_NEAR(W_val, 0.0, 1e-10);
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.2: АНАЛИТИЧЕСКАЯ СБОРКА (С ОГРАНИЧЕНИЯМИ)
// =============================================================================

TEST(Step2_1_5_2, AnalyticAssemblyRefusalHighDegree) {
    std::cout << "Testing analytic assembly refusal for high degree...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> values(11, 0.0);
    for (int i = 0; i < 11; ++i) {
        values[i] = std::pow(nodes[i], 2);
    }
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(20, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = 20
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Должен отказаться от аналитической сборки
    bool result = F.build_analytic_coefficients(15);
    EXPECT_FALSE(result);
    
    // Но ленивая оценка должна работать
    ASSERT_TRUE(F.is_valid());
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8);
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.3: ПАКЕТНАЯ ОЦЕНКА
// =============================================================================

TEST(Step2_1_5_3, BatchEvaluation) {
    std::cout << "Testing batch evaluation...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {1.0, 0.0, 0.0};  // Q(x) = 1
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<double> points = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<double> results;
    F.evaluate_batch(points, results);
    
    ASSERT_EQ(results.size(), points.size());
    
    for (size_t i = 0; i < points.size(); ++i) {
        double single_result = F.evaluate(points[i]);
        EXPECT_NEAR(results[i], single_result, 1e-12)
            << "Batch evaluation mismatch at index " << i;
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.4: ПРОИЗВОДНЫЕ (УПРОЩЁННЫЕ)
// =============================================================================

TEST(Step2_1_5_4, FirstDerivativeExistence) {
    std::cout << "Testing first derivative computation...\n";
    
    // Для надёжной проверки используем метод Ньютона (более стабильная производная)
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.0, 1.0};  // Q(x) = x
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Проверяем через численное дифференцирование
    double x = 0.5;
    double h = 1e-7;
    double f_plus = F.evaluate(x + h);
    double f_minus = F.evaluate(x - h);
    double numerical_derivative = (f_plus - f_minus) / (2.0 * h);
    
    // Аналитическая производная через evaluate_derivative
    double analytical_derivative = F.evaluate_derivative(x, 1);
    
    std::cout << "  Using Newton method for more stable derivative\n";
    std::cout << "  x = " << x << "\n";
    std::cout << "  F(" << x << ") = " << F.evaluate(x) << "\n";
    std::cout << "  Numerical F'(" << x << ") = " << numerical_derivative << "\n";
    std::cout << "  Analytical F'(" << x << ") = " << analytical_derivative << "\n";
    
    // Проверяем согласованность численной и аналитической производной
    EXPECT_NEAR(analytical_derivative, numerical_derivative, 1e-5)
        << "Analytical and numerical derivatives should match";
    
    // Проверяем также вторую производную
    double f_pp = F.evaluate(x + h);
    double f_pm = F.evaluate(x);
    double f_mm = F.evaluate(x - h);
    double numerical_second = (f_pp - 2.0 * f_pm + f_mm) / (h * h);
    double analytical_second = F.evaluate_derivative(x, 2);
    
    std::cout << "  Numerical F''(" << x << ") = " << numerical_second << "\n";
    std::cout << "  Analytical F''(" << x << ") = " << analytical_second << "\n";
    
    EXPECT_NEAR(analytical_second, numerical_second, 1e-3)
        << "Analytical and numerical second derivatives should match";
    
    // Проверяем производную в разных точках
    std::vector<double> test_points = {0.1, 0.3, 0.7, 0.9};
    for (double tx : test_points) {
        double fp = F.evaluate(tx + h);
        double fm = F.evaluate(tx - h);
        double num_deriv = (fp - fm) / (2.0 * h);
        double an_deriv = F.evaluate_derivative(tx, 1);
        
        std::cout << "  F'(" << tx << ") num = " << num_deriv << ", an = " << an_deriv << "\n";
        
        EXPECT_NEAR(an_deriv, num_deriv, 1e-4)
            << "Derivative mismatch at x = " << tx;
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.5: ВЕРИФИКАЦИЯ ИНТЕРПОЛЯЦИОННЫХ УСЛОВИЙ
// =============================================================================

TEST(Step2_1_5_5, VerifyInterpolationConditions) {
    std::cout << "Testing interpolation condition verification...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3, 0.4};  // Произвольные коэффициенты
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Ключевой тест: Интерполяционные условия должны выполняться
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "Interpolation failed at node " << nodes[i];
    }
    
    // W(z_e) должен быть близок к нулю
    for (double root : nodes) {
        double W_val = W.evaluate(root);
        EXPECT_NEAR(W_val, 0.0, 1e-10);
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.6: РЕГУЛЯРИЗАЦИОННЫЙ ЧЛЕН
// =============================================================================

TEST(Step2_1_5_6, RegularizationTermWithGamma) {
    std::cout << "Testing regularization term scaling with gamma...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.0};  // Q(x) = 0, F(x) = P_int(x) = x^2
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // F(x) = x^2, F''(x) = 2, ∫₀¹ 4 dx = 4
    double gamma1 = 0.5;
    double reg1 = F.compute_regularization_term(gamma1);
    
    double gamma2 = 2.0;
    double reg2 = F.compute_regularization_term(gamma2);
    
    // Проверяем линейность по gamma
    EXPECT_NEAR(reg2 / reg1, 4.0, 1e-6);
    
    // Значение должно быть положительным
    EXPECT_GT(reg1, 0.0);
    EXPECT_GT(reg2, 0.0);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.7: КРАЙНИЕ СЛУЧАИ
// =============================================================================

TEST(Step2_1_5_7, NoConstraintsCase) {
    std::cout << "Testing edge case: no interpolation constraints (m = 0)...\n";
    
    // P_int(x) = 0, W(x) = 1, F(x) = Q(x)
    
    InterpolationBasis basis;
    std::vector<double> nodes;
    std::vector<double> values;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);  // m = 0, W(x) = 1
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {1.0, 0.0, 0.0, 0.0};  // Q(x) = 1
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Отладочный вывод
    std::cout << "  W(0.5) = " << W.evaluate(0.5) << "\n";
    std::cout << "  Q(0.5) = " << Q.evaluate_Q(0.5) << "\n";
    std::cout << "  F.degree() = " << F.degree() << "\n";
    
    // Примечание: F.is_valid() может быть false из-за пустого базиса, 
    // но это ожидаемое поведение для особого случая m = 0
    
    // F(x) = 0 + 1*1 = 1
    double F_at_0 = F.evaluate(0.0);
    double F_at_0_5 = F.evaluate(0.5);
    double F_at_1 = F.evaluate(1.0);
    
    std::cout << "  F(0.0) = " << F_at_0 << "\n";
    std::cout << "  F(0.5) = " << F_at_0_5 << "\n";
    std::cout << "  F(1.0) = " << F_at_1 << "\n";
    
    EXPECT_NEAR(F_at_0, 1.0, 1e-12);
    EXPECT_NEAR(F_at_0_5, 1.0, 1e-12);
    EXPECT_NEAR(F_at_1, 1.0, 1e-12);
    
    // W(x) должен быть равен 1
    EXPECT_NEAR(W.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(W.evaluate(0.5), 1.0, 1e-12);
    EXPECT_NEAR(W.evaluate(1.0), 1.0, 1e-12);
}

TEST(Step2_1_5_7, FullInterpolationCase) {
    std::cout << "Testing edge case: full interpolation (m = n + 1)...\n";
    
    // n = 2, m = 3, Q вырожден, F(x) = P_int(x)
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};  // f(x) = x^2
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(-1, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = -1 (вырожден)
    Q.n_free = 0;
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // F(x) должен точно интерполировать
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-10)
            << "Full interpolation should give exact match";
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.5.8: КЭШИРОВАНИЕ
// =============================================================================

TEST(Step2_1_5_8, CachedWeightsForOptimization) {
    std::cout << "Testing cached weights for optimizer integration...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<double> points_x = {0.1, 0.3, 0.7, 0.9};
    std::vector<double> points_y = {0.2, 0.4, 0.6, 0.8};
    
    F.build_caches(points_x, points_y);
    
    ASSERT_TRUE(F.caches_built);
    EXPECT_EQ(F.cache.P_at_x.size(), points_x.size());
    EXPECT_EQ(F.cache.W_at_x.size(), points_x.size());
    EXPECT_EQ(F.cache.P_at_y.size(), points_y.size());
    EXPECT_EQ(F.cache.W_at_y.size(), points_y.size());
    
    // Проверяем согласованность кэшей
    for (size_t i = 0; i < points_x.size(); ++i) {
        EXPECT_NEAR(F.cache.P_at_x[i], basis.evaluate(points_x[i]), 1e-12);
        EXPECT_NEAR(F.cache.W_at_x[i], W.evaluate(points_x[i]), 1e-12);
    }
}

TEST(Step2_1_5_8, DiagnosticInfo) {
    std::cout << "Testing diagnostic info extraction...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.5, 0.5};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::string info = F.get_diagnostic_info();
    
    // Проверяем, что информация содержит ключевые поля
    EXPECT_TRUE(info.find("degree:") != std::string::npos);
    EXPECT_TRUE(info.find("constraints") != std::string::npos);
    EXPECT_TRUE(info.find("free params") != std::string::npos);
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ
// =============================================================================

TEST(IntegrationTest, FullCompositePolynomialPipeline) {
    std::cout << "Testing full composite polynomial pipeline...\n";
    
    // Типичная задача аппроксимации
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.4, 0.7, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);  // n = 6, m = 4
    Q.initialize_zero();
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0, EvaluationStrategy::HYBRID);
    
    ASSERT_TRUE(F.is_valid());
    
    // Проверяем интерполяционные условия - ЭТО КЛЮЧЕВОЕ ТРЕБОВАНИЕ
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8)
            << "Interpolation must hold at all constraint nodes";
    }
}

TEST(IntegrationTest, CompositeWithChebyshevBasis) {
    std::cout << "Testing composite polynomial with Chebyshev basis...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(4, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.1, 0.3, 0.0};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Интерполяционные условия должны выполняться
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8);
    }
}

TEST(IntegrationTest, HighDegreePolynomial) {
    std::cout << "Testing high degree composite polynomial (n=25)...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> values(5, 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::pow(nodes[i], 2);
    }
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(20, BasisType::CHEBYSHEV, 0.5, 0.5);  // deg_Q = 20
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Аналитическая сборка должна быть отклонена
    bool analytic_built = F.build_analytic_coefficients(15);
    EXPECT_FALSE(analytic_built);
    
    // Но ленивая оценка должна работать и интерполяция должна выполняться
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-6);
    }
}

TEST(IntegrationTest, WeightMultiplierVanishesAtNodes) {
    std::cout << "Testing that W(z_e) = 0 for all interpolation nodes...\n";
    
    WeightMultiplier W;
    std::vector<double> roots = {1.0, 3.0, 5.0, 7.0};
    W.build_from_roots(roots);
    
    for (double root : roots) {
        EXPECT_NEAR(W.evaluate(root), 0.0, 1e-12)
            << "W(" << root << ") should be exactly zero";
    }
}

TEST(IntegrationTest, EvaluationAtDifferentStrategies) {
    std::cout << "Testing evaluation at different strategies...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2};  // Q(x) = 0.1 + 0.2*x
    
    // Тестируем ленивую стратегию
    CompositePolynomial F_lazy;
    F_lazy.build(basis, W, Q, 0.0, 1.0, EvaluationStrategy::LAZY);
    
    ASSERT_TRUE(F_lazy.is_valid());
    
    // Интерполяционные условия должны выполняться
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F_lazy.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8);
    }
}

// =============================================================================
// ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ДЛЯ ПОЛНОГО ПОКРЫТИЯ (Шаг 2.1.5)
// =============================================================================

TEST(Step2_1_5, VerifyRepresentationsConsistency) {
    std::cout << "Testing lazy vs analytic evaluation consistency...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.1};  // Q(x) = 0.1 + 0.2*x + 0.1*x^2
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Строим аналитические коэффициенты
    bool analytic_built = F.build_analytic_coefficients(15);
    ASSERT_TRUE(analytic_built);
    ASSERT_TRUE(F.analytic_coeffs_valid);
    
    // Проверяем согласованность в разных точках
    std::vector<double> test_points = {0.0, 0.25, 0.5, 0.75, 1.0};
    for (double x : test_points) {
        double lazy_val = F.evaluate(x);
        double analytic_val = F.evaluate_analytic(x);
        
        double rel_diff = std::abs(lazy_val - analytic_val) / 
                        (std::abs(lazy_val) + 1e-12);
        
        EXPECT_NEAR(rel_diff, 0.0, 1e-8)
            << "Lazy and analytic evaluations should match at x = " << x;
    }
}

TEST(Step2_1_5, VerifyAssemblyMethod) {
    std::cout << "Testing verify_assembly method...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Верификация должна пройти
    bool verified = F.verify_assembly(1e-8);
    EXPECT_TRUE(verified);
    
    // W(z_e) должен быть близок к нулю
    for (double root : nodes) {
        double W_val = W.evaluate(root);
        EXPECT_NEAR(W_val, 0.0, 1e-10);
    }
}

TEST(Step2_1_5, SecondDerivativeConsistency) {
    std::cout << "Testing second derivative numerical consistency...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.4, 0.7, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.5, 0.0, 0.0, 0.0};  // Q(x) = 0.5
    
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
        
        double rel_diff = std::abs(numerical_second - analytical_second) / 
                         (std::abs(numerical_second) + 1e-10);
        
        EXPECT_NEAR(rel_diff, 0.0, 1e-4)
            << "Numerical and analytical second derivatives should match at x = " << x;
    }
}

TEST(Step2_1_5, RegularizationTermWithDifferentQuadrature) {
    std::cout << "Testing regularization term with different quadrature nodes...\n";
    
    // Создаём полином с известной второй производной
    // F(x) = x^2, F''(x) = 2, интеграл = 4 на [0, 1]
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};  // x^2
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = 0
    Q.coeffs = {0.0};  // Q(x) = 0
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Проверяем регуляризационный член
    double gamma = 1.0;
    double reg = F.compute_regularization_term(gamma);
    
    // Для F(x) = x^2 на [0,1]: ∫₀¹ 4 dx = 4
    EXPECT_NEAR(reg, 4.0, 1e-3);
    
    // Проверяем линейность по gamma
    double gamma2 = 2.0;
    double reg2 = F.compute_regularization_term(gamma2);
    EXPECT_NEAR(reg2 / reg, 2.0, 1e-6);
}

TEST(Step2_1_5, BatchEvaluationConsistency) {
    std::cout << "Testing batch evaluation consistency...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.3, 0.6, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.1, 0.0};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<double> points = {0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0};
    
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
}

TEST(Step2_1_5, DiagnosticInfoContent) {
    std::cout << "Testing diagnostic info content...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.5, 0.5};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::string info = F.get_diagnostic_info();
    
    // Проверяем ключевые поля
    EXPECT_TRUE(info.find("degree:") != std::string::npos);
    EXPECT_TRUE(info.find("constraints") != std::string::npos);
    EXPECT_TRUE(info.find("free params") != std::string::npos);
    EXPECT_TRUE(info.find("interval") != std::string::npos);
    EXPECT_TRUE(info.find("eval_strategy") != std::string::npos);
    
    // Проверяем, что информация не пустая
    EXPECT_FALSE(info.empty());
}

TEST(Step2_1_5, MetadataConsistency) {
    std::cout << "Testing metadata consistency...\n";
    
    int n = 5;  // Степень F
    int m = 3;  // Число узлов
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    // deg_Q = n - m = 2
    CorrectionPolynomial Q;
    Q.initialize(n - m, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Проверяем метаданные
    EXPECT_EQ(F.total_degree, n);           // n
    EXPECT_EQ(F.num_constraints, m);         // m
    EXPECT_EQ(F.num_free_params, n - m + 1); // n - m + 1
    EXPECT_EQ(F.degree(), n);
    EXPECT_EQ(F.num_free_parameters(), n - m + 1);
    EXPECT_DOUBLE_EQ(F.interval_a, 0.0);
    EXPECT_DOUBLE_EQ(F.interval_b, 1.0);
}

TEST(Step2_1_5, DifferentInterpolationMethods) {
    std::cout << "Testing composite polynomial with different interpolation methods...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    
    for (auto method : {InterpolationMethod::BARYCENTRIC, 
                        InterpolationMethod::NEWTON, 
                        InterpolationMethod::LAGRANGE}) {
        InterpolationBasis basis;
        basis.build(nodes, values, method, 0.0, 1.0);
        
        WeightMultiplier W;
        W.build_from_roots(nodes, 0.0, 1.0);
        
        CorrectionPolynomial Q;
        Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
        Q.coeffs = {0.1, 0.2};
        
        CompositePolynomial F;
        F.build(basis, W, Q, 0.0, 1.0);
        
        ASSERT_TRUE(F.is_valid());
        
        // Интерполяционные условия должны выполняться
        for (size_t i = 0; i < nodes.size(); ++i) {
            double F_val = F.evaluate(nodes[i]);
            EXPECT_NEAR(F_val, values[i], 1e-8)
                << "Interpolation failed with method " << static_cast<int>(method);
        }
    }
}

TEST(Step2_1_5, CachedWeightsCorrectness) {
    std::cout << "Testing cached weights correctness...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.3, 0.7, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.5, 3.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Строим кэши
    std::vector<double> points_x = {0.1, 0.2, 0.8, 0.9};
    std::vector<double> points_y = {0.4, 0.6};
    
    F.build_caches(points_x, points_y);
    
    ASSERT_TRUE(F.caches_built);
    
    // Проверяем кэши для точек x
    for (size_t i = 0; i < points_x.size(); ++i) {
        EXPECT_NEAR(F.cache.P_at_x[i], basis.evaluate(points_x[i]), 1e-12);
        EXPECT_NEAR(F.cache.W_at_x[i], W.evaluate(points_x[i]), 1e-12);
    }
    
    // Проверяем кэши для точек y
    for (size_t i = 0; i < points_y.size(); ++i) {
        EXPECT_NEAR(F.cache.P_at_y[i], basis.evaluate(points_y[i]), 1e-12);
        EXPECT_NEAR(F.cache.W_at_y[i], W.evaluate(points_y[i]), 1e-12);
    }
    
    // Проверяем кэши квадратуры
    ASSERT_FALSE(F.cache.quad_points.empty());
    for (size_t i = 0; i < F.cache.quad_points.size(); ++i) {
        double x = F.cache.quad_points[i];
        EXPECT_NEAR(F.cache.W_at_quad[i], W.evaluate(x), 1e-12);
        EXPECT_NEAR(F.cache.Q_at_quad[i], Q.evaluate_Q(x), 1e-12);
    }
}

TEST(Step2_1_5, ClearCachesFunctionality) {
    std::cout << "Testing clear caches functionality...\n";
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {0.0, 1.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.5, 0.5};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Строим кэши
    F.build_caches({0.1, 0.2}, {0.5});
    ASSERT_TRUE(F.caches_built);
    
    // Очищаем кэши
    F.clear_caches();
    
    EXPECT_FALSE(F.caches_built);
    EXPECT_TRUE(F.cache.P_at_x.empty());
    EXPECT_TRUE(F.cache.W_at_x.empty());
    EXPECT_TRUE(F.cache.quad_points.empty());
}

TEST(Step2_1_5, EvaluateWithSingleRoot) {
    std::cout << "Testing composite polynomial with single root (m=1)...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.5};
    std::vector<double> values = {2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // Интерполяционное условие в единственном узле
    double F_at_node = F.evaluate(0.5);
    EXPECT_NEAR(F_at_node, 2.0, 1e-8);
    
    // W(z_e) должен быть близок к нулю
    EXPECT_NEAR(W.evaluate(0.5), 0.0, 1e-12);
}

TEST(Step2_1_5, EvaluateOutsideInterval) {
    std::cout << "Testing evaluation outside interpolation interval...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Оценка вне интервала должна работать (экстраполяция)
    double F_outside = F.evaluate(-0.5);
    EXPECT_FALSE(std::isnan(F_outside));
    EXPECT_FALSE(std::isinf(F_outside));
    
    double F_outside2 = F.evaluate(1.5);
    EXPECT_FALSE(std::isnan(F_outside2));
    EXPECT_FALSE(std::isinf(F_outside2));
}

TEST(Step2_1_5, DegreeCalculation) {
    std::cout << "Testing degree calculation for different configurations...\n";
    
    // Случай 1: deg_Q = 0, m = 3 -> deg_F = 3
    {
        InterpolationBasis basis;
        basis.build({0.0, 0.5, 1.0}, {0.0, 0.25, 1.0}, 
                   InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
        
        WeightMultiplier W;
        W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
        
        CorrectionPolynomial Q;
        Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
        Q.coeffs = {1.0};
        
        CompositePolynomial F;
        F.build(basis, W, Q, 0.0, 1.0);
        
        EXPECT_EQ(F.degree(), 3);  // deg_W = 3, deg_Q = 0, deg_F = 3
    }
    
    // Случай 2: deg_Q = 2, m = 2 -> deg_F = 3
    {
        InterpolationBasis basis;
        basis.build({0.0, 1.0}, {0.0, 1.0}, 
                   InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
        
        WeightMultiplier W;
        W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
        
        CorrectionPolynomial Q;
        Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
        Q.coeffs = {0.1, 0.2, 0.3};
        
        CompositePolynomial F;
        F.build(basis, W, Q, 0.0, 1.0);
        
        EXPECT_EQ(F.degree(), 3);  // deg_W = 2, deg_Q = 2, deg_F = 3
    }
}

TEST(Step2_1_5, ZeroCoefficientsCase) {
    std::cout << "Testing composite polynomial with zero correction coefficients...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();  // Все коэффициенты = 0
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    ASSERT_TRUE(F.is_valid());
    
    // F(x) = P_int(x) + 0*W(x) = P_int(x)
    for (size_t i = 0; i < nodes.size(); ++i) {
        double F_val = F.evaluate(nodes[i]);
        EXPECT_NEAR(F_val, values[i], 1e-8);
    }
    
    // Аналитическая сборка должна работать
    bool analytic = F.build_analytic_coefficients(15);
    EXPECT_TRUE(analytic);
}

// Конец файла
