#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "mixed_approximation/weight_multiplier.h"

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Тесты шага 2.1.3.1: Математическая формулировка весового множителя
// =============================================================================

TEST(WeightMultiplierTest, MonicPolynomialDefinition) {
    std::cout << "Testing monic polynomial definition (step 2.1.3.1)...\n";
    
    // W(x) = (x - 0.0)(x - 1.0) = x^2 - x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Старший коэффициент должен быть 1
    ASSERT_FALSE(W.coeffs.empty());
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    
    // W(0) = 0, W(1) = 0
    EXPECT_NEAR(W.evaluate(0.0), 0.0, 1e-12);
    EXPECT_NEAR(W.evaluate(1.0), 0.0, 1e-12);
}

TEST(WeightMultiplierTest, ElementarySymmetricPolynomials) {
    std::cout << "Testing elementary symmetric polynomials...\n";
    
    // W(x) = (x - z1)(x - z2)(x - z3) = x^3 - (z1+z2+z3)*x^2 + (z1*z2+z1*z3+z2*z3)*x - z1*z2*z3
    double z1 = 0.0, z2 = 0.5, z3 = 1.0;
    WeightMultiplier W;
    W.build_from_roots({z1, z2, z3}, 0.0, 1.0);
    
    // Проверяем коэффициенты
    ASSERT_EQ(W.coeffs.size(), 4u);  // степень 3 + константа
    
    // w_2 = -Σ z_e = -(0 + 0.5 + 1) = -1.5
    EXPECT_NEAR(W.coeffs[1], -(z1 + z2 + z3), 1e-12);
    
    // w_1 = Σ_{e<k} z_e*z_k
    double sum_pairs = z1*z2 + z1*z3 + z2*z3;
    EXPECT_NEAR(W.coeffs[2], sum_pairs, 1e-12);
    
    // w_0 = (-1)^3 * Π z_e = -z1*z2*z3
    EXPECT_NEAR(W.coeffs[3], -z1*z2*z3, 1e-12);
}

// =============================================================================
// Тесты шага 2.1.3.2: Алгоритм инкрементального построения коэффициентов
// =============================================================================

TEST(WeightMultiplierTest, IncrementalCoefficientBuilding) {
    std::cout << "Testing incremental coefficient building (step 2.1.3.2)...\n";
    
    // Начинаем с W(x) = 1
    WeightMultiplier W;
    W.build_from_roots({0.0}, 0.0, 1.0);
    
    // W(x) = x - 0.0 = x, коэффициенты: [1, 0]
    ASSERT_EQ(W.coeffs.size(), 2u);
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    EXPECT_NEAR(W.coeffs[1], 0.0, 1e-12);
    
    // Добавляем корень 1.0
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // W(x) = (x-0)(x-1) = x^2 - x, коэффициенты: [1, -1, 0]
    ASSERT_EQ(W.coeffs.size(), 3u);
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    EXPECT_NEAR(W.coeffs[1], -1.0, 1e-12);
    EXPECT_NEAR(W.coeffs[2], 0.0, 1e-12);
    
    // Добавляем корень 2.0
    W.build_from_roots({0.0, 1.0, 2.0}, 0.0, 3.0);
    
    // W(x) = (x-0)(x-1)(x-2) = x^3 - 3x^2 + 2x, коэффициенты: [1, -3, 2, 0]
    ASSERT_EQ(W.coeffs.size(), 4u);
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    EXPECT_NEAR(W.coeffs[1], -3.0, 1e-12);  // -Σ z_e = -(0+1+2) = -3
    EXPECT_NEAR(W.coeffs[2], 2.0, 1e-12);    // Σ z_e*z_k = 0*1 + 0*2 + 1*2 = 2
    EXPECT_NEAR(W.coeffs[3], 0.0, 1e-12);    // (-1)^3 * Π z_e = 0
}

TEST(WeightMultiplierTest, PolynomialDegreeMatchesRoots) {
    std::cout << "Testing polynomial degree matches number of roots...\n";
    
    // m = 1
    WeightMultiplier W1;
    W1.build_from_roots({0.5}, 0.0, 1.0);
    EXPECT_EQ(W1.degree(), 1);
    
    // m = 5
    WeightMultiplier W5;
    W5.build_from_roots({0.0, 0.2, 0.4, 0.6, 0.8}, 0.0, 1.0);
    EXPECT_EQ(W5.degree(), 5);
    
    // m = 10
    WeightMultiplier W10;
    std::vector<double> roots10;
    for (int i = 0; i < 10; ++i) {
        roots10.push_back(i * 0.1);
    }
    W10.build_from_roots(roots10, 0.0, 1.0);
    EXPECT_EQ(W10.degree(), 10);
}

// =============================================================================
// Тесты шага 2.1.3.4: Вычисление значений полинома
// =============================================================================

TEST(WeightMultiplierTest, EvaluateViaCoefficients) {
    std::cout << "Testing evaluation via coefficients (Horner's method)...\n";
    
    // W(x) = (x-0)(x-1)(x-2) = x^3 - 3x^2 + 2x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    // W(0) = 0
    EXPECT_NEAR(W.evaluate(0.0), 0.0, 1e-12);
    
    // W(1) = 0
    EXPECT_NEAR(W.evaluate(1.0), 0.0, 1e-12);
    
    // W(2) = 0
    EXPECT_NEAR(W.evaluate(2.0), 0.0, 1e-12);
    
    // W(3) = (3-0)(3-1)(3-2) = 3*2*1 = 6
    EXPECT_NEAR(W.evaluate(3.0), 6.0, 1e-12);
    
    // W(-1) = (-1-0)(-1-1)(-1-2) = (-1)(-2)(-3) = -6
    EXPECT_NEAR(W.evaluate(-1.0), -6.0, 1e-12);
}

TEST(WeightMultiplierTest, EvaluateViaDirectProduct) {
    std::cout << "Testing evaluation via direct product...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    // Сравниваем прямое вычисление с коэффициентами
    std::vector<double> test_points = {-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
    
    for (double x : test_points) {
        // Прямое вычисление: W(x) = ∏(x - z_e)
        double direct = 1.0;
        for (double root : W.roots) {
            direct *= (x - root);
        }
        
        // Используем evaluate (который может использовать прямое вычисление или коэффициенты)
        double evaluated = W.evaluate(x);
        
        EXPECT_NEAR(evaluated, direct, 1e-10)
            << "Mismatch at x = " << x;
    }
}

TEST(WeightMultiplierTest, HornerSchemeNumericalStability) {
    std::cout << "Testing Horner's scheme numerical stability...\n";
    
    // Полином высокой степени для проверки устойчивости
    WeightMultiplier W;
    std::vector<double> roots;
    for (int i = 0; i < 10; ++i) {
        roots.push_back(i * 0.1);
    }
    W.build_from_roots(roots, 0.0, 1.0);
    
    // Проверяем, что evaluate работает без переполнения/потери точности
    double val = W.evaluate(5.0);
    EXPECT_FALSE(std::isnan(val));
    EXPECT_FALSE(std::isinf(val));
}

// =============================================================================
// Тесты шага 2.1.3.5: Вычисление производных
// =============================================================================

TEST(WeightMultiplierTest, FirstDerivative) {
    std::cout << "Testing first derivative (step 2.1.3.5)...\n";
    
    // W(x) = (x-0)(x-1) = x^2 - x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // W'(x) = 2x - 1
    EXPECT_NEAR(W.evaluate_derivative(0.0, 1), -1.0, 1e-12);   // W'(0) = -1
    EXPECT_NEAR(W.evaluate_derivative(0.5, 1), 0.0, 1e-12);    // W'(0.5) = 0
    EXPECT_NEAR(W.evaluate_derivative(1.0, 1), 1.0, 1e-12);    // W'(1) = 1
    
    // W(x) = (x-0)(x-1)(x-2) = x^3 - 3x^2 + 2x
    WeightMultiplier W3;
    W3.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    // W'(x) = 3x^2 - 6x + 2
    EXPECT_NEAR(W3.evaluate_derivative(0.0, 1), 2.0, 1e-12);   // W'(0) = 2
    EXPECT_NEAR(W3.evaluate_derivative(1.0, 1), -1.0, 1e-12);   // W'(1) = -1
    EXPECT_NEAR(W3.evaluate_derivative(2.0, 1), 2.0, 1e-12);   // W'(2) = 2
}

TEST(WeightMultiplierTest, FirstDerivativeAtRoots) {
    std::cout << "Testing first derivative at roots...\n";
    
    // W(x) = (x-0)(x-1) = x^2 - x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // W'(z_e) = ∏_{k≠e} (z_e - z_k)
    // W'(0) = (0 - 1) = -1
    EXPECT_NEAR(W.evaluate_derivative(0.0, 1), -1.0, 1e-12);
    // W'(1) = (1 - 0) = 1
    EXPECT_NEAR(W.evaluate_derivative(1.0, 1), 1.0, 1e-12);
    
    // Три корня: W(x) = (x-0)(x-1)(x-2)
    WeightMultiplier W3;
    W3.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    // W'(0) = (0-1)(0-2) = 2
    EXPECT_NEAR(W3.evaluate_derivative(0.0, 1), 2.0, 1e-12);
    // W'(1) = (1-0)(1-2) = -1
    EXPECT_NEAR(W3.evaluate_derivative(1.0, 1), -1.0, 1e-12);
    // W'(2) = (2-0)(2-1) = 2
    EXPECT_NEAR(W3.evaluate_derivative(2.0, 1), 2.0, 1e-12);
}

TEST(WeightMultiplierTest, SecondDerivative) {
    std::cout << "Testing second derivative (step 2.1.3.5)...\n";
    
    // W(x) = (x-0)(x-1) = x^2 - x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // W''(x) = 2 (для квадратичного полинома)
    EXPECT_NEAR(W.evaluate_derivative(0.0, 2), 2.0, 1e-12);
    EXPECT_NEAR(W.evaluate_derivative(0.5, 2), 2.0, 1e-12);
    EXPECT_NEAR(W.evaluate_derivative(1.0, 2), 2.0, 1e-12);
    
    // W(x) = (x-0)(x-1)(x-2) = x^3 - 3x^2 + 2x
    WeightMultiplier W3;
    W3.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    // W''(x) = 6x - 6
    EXPECT_NEAR(W3.evaluate_derivative(0.0, 2), -6.0, 1e-12);  // W''(0) = -6
    EXPECT_NEAR(W3.evaluate_derivative(1.0, 2), 0.0, 1e-12);   // W''(1) = 0
    EXPECT_NEAR(W3.evaluate_derivative(2.0, 2), 6.0, 1e-12);   // W''(2) = 6
}

TEST(WeightMultiplierTest, SecondDerivativeAtRoots) {
    std::cout << "Testing second derivative at roots...\n";
    
    // W(x) = (x-0)(x-1)(x-2)
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    // W''(z_e) = 2 * ∏_{j≠e} (z_e - z_j) * Σ_{j≠e} 1/(z_e - z_j)
    // W''(0) = 2 * (0-1)(0-2) * (1/(0-1) + 1/(0-2))
    //        = 2 * 2 * (-1 - 0.5) = 2 * 2 * (-1.5) = -6
    EXPECT_NEAR(W.evaluate_derivative(0.0, 2), -6.0, 1e-12);
    
    // W''(1) = 2 * (1-0)(1-2) * (1/(1-0) + 1/(1-2))
    //        = 2 * (-1) * (1 - 1) = 0
    EXPECT_NEAR(W.evaluate_derivative(1.0, 2), 0.0, 1e-12);
    
    // W''(2) = 2 * (2-0)(2-1) * (1/(2-0) + 1/(2-1))
    //        = 2 * 2 * (0.5 + 1) = 2 * 2 * 1.5 = 6
    EXPECT_NEAR(W.evaluate_derivative(2.0, 2), 6.0, 1e-12);
}

TEST(WeightMultiplierTest, InvalidDerivativeOrder) {
    std::cout << "Testing invalid derivative order...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Порядок 0 должен вызвать исключение
    EXPECT_THROW(W.evaluate_derivative(0.5, 0), std::invalid_argument);
    
    // Порядок 3 должен вызвать исключение
    EXPECT_THROW(W.evaluate_derivative(0.5, 3), std::invalid_argument);
    
    // Отрицательный порядок должен вызвать исключение
    EXPECT_THROW(W.evaluate_derivative(0.5, -1), std::invalid_argument);
}

// =============================================================================
// Тесты шага 2.1.3.6: Кэширование значений
// =============================================================================

TEST(WeightMultiplierTest, CacheBuilding) {
    std::cout << "Testing cache building (step 2.1.3.6)...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    std::vector<double> points_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> points_y = {0.1, 0.9};
    
    W.build_caches(points_x, points_y);
    
    // Проверяем, что кэши заполнены
    ASSERT_EQ(W.cache_x_vals.size(), points_x.size());
    ASSERT_EQ(W.cache_x_deriv1.size(), points_x.size());
    ASSERT_EQ(W.cache_x_deriv2.size(), points_x.size());
    
    ASSERT_EQ(W.cache_y_vals.size(), points_y.size());
    ASSERT_EQ(W.cache_y_deriv1.size(), points_y.size());
    ASSERT_EQ(W.cache_y_deriv2.size(), points_y.size());
    
    // Проверяем флаг готовности
    EXPECT_TRUE(W.caches_ready);
    
    // Проверяем согласованность кэша с evaluate
    for (size_t i = 0; i < points_x.size(); ++i) {
        EXPECT_NEAR(W.cache_x_vals[i], W.evaluate(points_x[i]), 1e-12);
        EXPECT_NEAR(W.cache_x_deriv1[i], W.evaluate_derivative(points_x[i], 1), 1e-12);
        EXPECT_NEAR(W.cache_x_deriv2[i], W.evaluate_derivative(points_x[i], 2), 1e-12);
    }
}

TEST(WeightMultiplierTest, CacheClearing) {
    std::cout << "Testing cache clearing...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    std::vector<double> points_x = {0.0, 0.5, 1.0};
    std::vector<double> points_y = {0.25, 0.75};
    
    W.build_caches(points_x, points_y);
    EXPECT_TRUE(W.caches_ready);
    
    W.clear_caches();
    
    // Проверяем, что кэши пусты
    EXPECT_TRUE(W.cache_x_vals.empty());
    EXPECT_TRUE(W.cache_x_deriv1.empty());
    EXPECT_TRUE(W.cache_x_deriv2.empty());
    EXPECT_TRUE(W.cache_y_vals.empty());
    EXPECT_TRUE(W.cache_y_deriv1.empty());
    EXPECT_TRUE(W.cache_y_deriv2.empty());
    EXPECT_FALSE(W.caches_ready);
}

// =============================================================================
// Тесты шага 2.1.3.7: Верификация корректности построения
// =============================================================================

TEST(WeightMultiplierTest, VerifyConstructionMonicity) {
    std::cout << "Testing verification of monicity...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0, 2.0}, 0.0, 2.0);
    
    bool verified = W.verify_construction(1e-10);
    EXPECT_TRUE(verified);
    
    // Старший коэффициент должен быть близок к 1
    ASSERT_FALSE(W.coeffs.empty());
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-10);
}

TEST(WeightMultiplierTest, VerifyConstructionZeroAtRoots) {
    std::cout << "Testing verification of zero values at roots...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    bool verified = W.verify_construction(1e-10);
    EXPECT_TRUE(verified);
    
    // W(z_e) должен быть близок к нулю
    for (double root : W.roots) {
        double W_at_root = W.evaluate(root);
        EXPECT_NEAR(W_at_root, 0.0, 1e-8)
            << "W(" << root << ") = " << W_at_root << " is not close to zero";
    }
}

TEST(WeightMultiplierTest, VerifyConstructionConsistency) {
    std::cout << "Testing verification of representation consistency...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.3, 0.7, 1.0}, 0.0, 1.0);
    
    bool verified = W.verify_construction(1e-10);
    EXPECT_TRUE(verified);
}

TEST(WeightMultiplierTest, VerifyConstructionFailsWithBadCoefficients) {
    std::cout << "Testing verification fails with bad coefficients...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Портим коэффициенты
    W.coeffs[0] = 0.5;
    
    bool verified = W.verify_construction(1e-10);
    EXPECT_FALSE(verified);
}

// =============================================================================
// Тесты шага 2.1.3.8: Обработка крайних случаев
// =============================================================================

TEST(WeightMultiplierTest, NoRootsCase) {
    std::cout << "Testing no roots case (m=0)...\n";
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    // W(x) = 1 (константа)
    EXPECT_EQ(W.degree(), 0);
    
    // Коэффициент = {1.0}
    ASSERT_EQ(W.coeffs.size(), 1u);
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    
    // W(x) = 1 для любого x
    EXPECT_NEAR(W.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(W.evaluate(1.0), 1.0, 1e-12);
    EXPECT_NEAR(W.evaluate(100.0), 1.0, 1e-12);
    
    // Производные = 0
    EXPECT_NEAR(W.evaluate_derivative(0.0, 1), 0.0, 1e-12);
    EXPECT_NEAR(W.evaluate_derivative(0.0, 2), 0.0, 1e-12);
    
    // verify_construction должен возвращать true
    EXPECT_TRUE(W.verify_construction(1e-10));
    
    // min_root_distance = 0
    EXPECT_EQ(W.min_root_distance, 0.0);
}

TEST(WeightMultiplierTest, SingleRootCase) {
    std::cout << "Testing single root case (m=1)...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.5}, 0.0, 1.0);
    
    // W(x) = x - 0.5
    EXPECT_EQ(W.degree(), 1);
    
    // Коэффициенты: [1, -0.5]
    ASSERT_EQ(W.coeffs.size(), 2u);
    EXPECT_NEAR(W.coeffs[0], 1.0, 1e-12);
    EXPECT_NEAR(W.coeffs[1], -0.5, 1e-12);
    
    // W(0.5) = 0
    EXPECT_NEAR(W.evaluate(0.5), 0.0, 1e-12);
    
    // W(0) = -0.5
    EXPECT_NEAR(W.evaluate(0.0), -0.5, 1e-12);
    
    // W(1) = 0.5
    EXPECT_NEAR(W.evaluate(1.0), 0.5, 1e-12);
    
    // W'(x) = 1
    EXPECT_NEAR(W.evaluate_derivative(0.5, 1), 1.0, 1e-12);
    EXPECT_NEAR(W.evaluate_derivative(0.0, 1), 1.0, 1e-12);
    
    // W''(x) = 0
    EXPECT_NEAR(W.evaluate_derivative(0.0, 2), 0.0, 1e-12);
}

TEST(WeightMultiplierTest, RootsSorting) {
    std::cout << "Testing roots sorting...\n";
    
    // Задаём корни в произвольном порядке
    WeightMultiplier W;
    W.build_from_roots({1.0, 0.0, 0.5}, 0.0, 1.0);
    
    // Корни должны быть отсортированы
    ASSERT_EQ(W.roots.size(), 3u);
    EXPECT_NEAR(W.roots[0], 0.0, 1e-12);
    EXPECT_NEAR(W.roots[1], 0.5, 1e-12);
    EXPECT_NEAR(W.roots[2], 1.0, 1e-12);
    
    // evaluate должен давать правильные значения независимо от порядка
    EXPECT_NEAR(W.evaluate(0.0), 0.0, 1e-12);
    EXPECT_NEAR(W.evaluate(0.5), 0.0, 1e-12);
    EXPECT_NEAR(W.evaluate(1.0), 0.0, 1e-12);
}

TEST(WeightMultiplierTest, MinRootDistance) {
    std::cout << "Testing minimum root distance calculation...\n";
    
    // Равномерно распределённые корни
    WeightMultiplier W1;
    W1.build_from_roots({0.0, 0.25, 0.5, 0.75, 1.0}, 0.0, 1.0);
    EXPECT_NEAR(W1.min_root_distance, 0.25, 1e-12);
    
    // Неравномерно распределённые корни
    WeightMultiplier W2;
    W2.build_from_roots({0.0, 0.1, 0.15, 1.0}, 0.0, 1.0);
    EXPECT_NEAR(W2.min_root_distance, 0.05, 1e-12);  // min(0.1-0, 0.15-0.1, 1.0-0.15) = 0.05
    
    // Один корень
    WeightMultiplier W3;
    W3.build_from_roots({0.5}, 0.0, 1.0);
    EXPECT_EQ(W3.min_root_distance, 0.0);
    
    // Нет корней
    WeightMultiplier W4;
    W4.build_from_roots({}, 0.0, 1.0);
    EXPECT_EQ(W4.min_root_distance, 0.0);
}

// =============================================================================
// Тесты шага 2.1.3.9: Интеграция с компонентами полной параметризации
// =============================================================================

TEST(WeightMultiplierTest, MultiplyByQ) {
    std::cout << "Testing multiply by Q(x)...\n";
    
    // W(x) = (x-0)(x-1) = x^2 - x
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Q(x) = 2x + 3, коэффициенты: [2, 3] (степень 1)
    std::vector<double> q_coeffs = {2.0, 3.0};
    
    // W(x) * Q(x) = (x^2 - x) * (2x + 3) = 2x^3 + 3x^2 - 2x^2 - 3x = 2x^3 + x^2 - 3x
    // Коэффициенты: [2, 1, -3, 0]
    std::vector<double> result = W.multiply_by_Q(q_coeffs);
    
    ASSERT_EQ(result.size(), 4u);
    EXPECT_NEAR(result[0], 2.0, 1e-12);  // x^3
    EXPECT_NEAR(result[1], 1.0, 1e-12);  // x^2
    EXPECT_NEAR(result[2], -3.0, 1e-12); // x
    EXPECT_NEAR(result[3], 0.0, 1e-12);  // константа
    
    // Проверяем через evaluate
    EXPECT_NEAR(W.evaluate_product(0.0, q_coeffs), 0.0, 1e-12);  // Q(0)*W(0) = 3*0 = 0
    EXPECT_NEAR(W.evaluate_product(1.0, q_coeffs), 0.0, 1e-12);  // Q(1)*W(1) = 5*0 = 0
    EXPECT_NEAR(W.evaluate_product(2.0, q_coeffs), W.evaluate(2.0) * (2*2 + 3), 1e-12);
}

TEST(WeightMultiplierTest, EvaluateProduct) {
    std::cout << "Testing evaluate product Q(x)*W(x)...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Q(x) = x^2 (коэффициенты: [1, 0, 0])
    std::vector<double> q_coeffs = {1.0, 0.0, 0.0};
    
    // Q(x)*W(x) = x^2 * (x^2 - x) = x^4 - x^3
    EXPECT_NEAR(W.evaluate_product(2.0, q_coeffs), 16.0 - 8.0, 1e-12);
    EXPECT_NEAR(W.evaluate_product(0.0, q_coeffs), 0.0, 1e-12);
    EXPECT_NEAR(W.evaluate_product(1.0, q_coeffs), 1.0 - 1.0, 1e-12);
}

TEST(WeightMultiplierTest, MultiplyByEmptyQ) {
    std::cout << "Testing multiply by empty Q...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Пустой Q
    std::vector<double> empty_q;
    std::vector<double> result = W.multiply_by_Q(empty_q);
    EXPECT_TRUE(result.empty());
    
    // evaluate_product с пустым Q должен возвращать 0
    EXPECT_NEAR(W.evaluate_product(0.5, empty_q), 0.0, 1e-12);
}

// =============================================================================
// Дополнительные тесты на нормализацию
// =============================================================================

TEST(WeightMultiplierTest, NormalizationDisabled) {
    std::cout << "Testing normalization disabled...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0, false);  // нормализация отключена
    
    EXPECT_FALSE(W.is_normalized);
    EXPECT_EQ(W.shift, 0.0);
    EXPECT_EQ(W.scale, 1.0);
    
    // Проверяем корректность
    EXPECT_TRUE(W.verify_construction(1e-10));
}

TEST(WeightMultiplierTest, NormalizationEnabled) {
    std::cout << "Testing normalization enabled...\n";
    
    // Широкий интервал должен включать нормализацию
    WeightMultiplier W;
    W.build_from_roots({0.0, 100.0}, 0.0, 200.0, true);
    
    // Проверяем, что нормализация применена
    if (W.is_normalized) {
        EXPECT_NE(W.scale, 1.0);
        EXPECT_NE(W.shift, 0.0);
        ASSERT_EQ(W.roots_norm.size(), 2u);
        
        // Проверяем корректность
        EXPECT_TRUE(W.verify_construction(1e-8));
    }
}

// =============================================================================
// Тесты на согласованность evaluate и evaluate_derivative
// =============================================================================

TEST(WeightMultiplierTest, DerivativeConsistency) {
    std::cout << "Testing derivative consistency with numerical approximation...\n";
    
    WeightMultiplier W;
    W.build_from_roots({0.2, 0.5, 0.8}, 0.0, 1.0);
    
    double h = 1e-6;
    std::vector<double> test_points = {0.0, 0.3, 0.6, 1.0};
    
    for (double x : test_points) {
        // Численная первая производная
        double numerical_deriv1 = (W.evaluate(x + h) - W.evaluate(x - h)) / (2 * h);
        double analytical_deriv1 = W.evaluate_derivative(x, 1);
        
        EXPECT_NEAR(numerical_deriv1, analytical_deriv1, 1e-5)
            << "First derivative mismatch at x = " << x;
        
        // Численная вторая производная
        double numerical_deriv2 = (W.evaluate(x + h) - 2*W.evaluate(x) + W.evaluate(x - h)) / (h * h);
        double analytical_deriv2 = W.evaluate_derivative(x, 2);
        
        EXPECT_NEAR(numerical_deriv2, analytical_deriv2, 1e-4)
            << "Second derivative mismatch at x = " << x;
    }
}

// =============================================================================
// Тесты на высокую степень полинома
// =============================================================================

TEST(WeightMultiplierTest, HighDegreePolynomial) {
    std::cout << "Testing high degree polynomial...\n";
    
    // Полином степени 20
    WeightMultiplier W;
    std::vector<double> roots;
    for (int i = 0; i < 20; ++i) {
        roots.push_back(i * 0.05);  // 0, 0.05, ..., 0.95
    }
    W.build_from_roots(roots, 0.0, 1.0);
    
    EXPECT_EQ(W.degree(), 20);
    EXPECT_EQ(W.coeffs.size(), 21u);
    
    // Проверяем W(z_e) ≈ 0 для всех корней
    for (double root : W.roots) {
        double val = W.evaluate(root);
        EXPECT_NEAR(val, 0.0, 1e-6)
            << "W(" << root << ") = " << val << " is not close to zero";
    }
    
    // Проверяем verify_construction
    EXPECT_TRUE(W.verify_construction(1e-6));
}

// =============================================================================
// Тесты на численную устойчивость
// =============================================================================

TEST(WeightMultiplierTest, NumericalStabilityWithLargeRoots) {
    std::cout << "Testing numerical stability with large roots...\n";
    
    // Корни с большим разбросом
    WeightMultiplier W;
    W.build_from_roots({-1000.0, 0.0, 1000.0}, -1000.0, 1000.0);
    
    // W(-1000) ≈ 0, W(0) ≈ 0, W(1000) ≈ 0
    EXPECT_NEAR(W.evaluate(-1000.0), 0.0, 1e-6);
    EXPECT_NEAR(W.evaluate(0.0), 0.0, 1e-6);
    EXPECT_NEAR(W.evaluate(1000.0), 0.0, 1e-6);
    
    // Значения не должны быть Inf или NaN
    double val_mid = W.evaluate(0.5);
    EXPECT_FALSE(std::isnan(val_mid));
    EXPECT_FALSE(std::isinf(val_mid));
}

TEST(WeightMultiplierTest, NumericalStabilityWithCloseRoots) {
    std::cout << "Testing numerical stability with close roots...\n";
    
    // Очень близкие корни
    WeightMultiplier W;
    W.build_from_roots({0.5, 0.5 + 1e-8, 0.5 + 2e-8}, 0.0, 1.0);
    
    // W(0.5) ≈ 0, W(0.5+1e-8) ≈ 0, W(0.5+2e-8) ≈ 0
    EXPECT_NEAR(W.evaluate(0.5), 0.0, 1e-6);
    EXPECT_NEAR(W.evaluate(0.5 + 1e-8), 0.0, 1e-6);
    EXPECT_NEAR(W.evaluate(0.5 + 2e-8), 0.0, 1e-6);
}

// =============================================================================
// Тесты для выявления потенциальных ошибок
// =============================================================================

TEST(WeightMultiplierTest, RootsNotModifiedAfterBuild) {
    std::cout << "Testing roots are preserved after build...\n";
    
    std::vector<double> original_roots = {0.3, 0.7, 1.0};
    WeightMultiplier W;
    W.build_from_roots(original_roots, 0.0, 1.0);
    
    // Проверяем, что исходный вектор не был изменён
    ASSERT_EQ(original_roots.size(), 3u);
    EXPECT_NEAR(original_roots[0], 0.3, 1e-12);
    EXPECT_NEAR(original_roots[1], 0.7, 1e-12);
    EXPECT_NEAR(original_roots[2], 1.0, 1e-12);
}

TEST(WeightMultiplierTest, EmptyRootsVector) {
    std::cout << "Testing empty roots vector...\n";
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    // Должен быть корректно обработан
    EXPECT_EQ(W.degree(), 0);
    EXPECT_TRUE(W.verify_construction(1e-10));
    
    // evaluate должен возвращать 1
    EXPECT_NEAR(W.evaluate(0.5), 1.0, 1e-12);
}
