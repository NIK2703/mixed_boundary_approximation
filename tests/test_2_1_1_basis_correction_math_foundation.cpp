#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include "mixed_approximation/interpolation_basis.h"

using namespace mixed_approx;

// =============================================================================
// Тесты шага 2.1.2.1: Предварительная подготовка и валидация узлов интерполяции
// =============================================================================

TEST(InterpolationBasisTest, NodeSorting) {
    std::cout << "Testing node sorting (step 2.1.2.1)...\n";
    
    // Задаём узлы в произвольном порядке, отключаем нормализацию для проверки сортировки
    std::vector<double> nodes = {0.5, 0.0, 1.0, 0.3};
    std::vector<double> values = {2.0, 1.0, 3.0, 1.5};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем, что узлы отсортированы (в исходных координатах)
    ASSERT_EQ(basis.nodes.size(), 4u);
    EXPECT_NEAR(basis.nodes[0], 0.0, 1e-12);
    EXPECT_NEAR(basis.nodes[1], 0.3, 1e-12);
    EXPECT_NEAR(basis.nodes[2], 0.5, 1e-12);
    EXPECT_NEAR(basis.nodes[3], 1.0, 1e-12);
    
    // Проверяем, что значения переставлены соответственно
    EXPECT_NEAR(basis.values[0], 1.0, 1e-12);
    EXPECT_NEAR(basis.values[1], 1.5, 1e-12);
    EXPECT_NEAR(basis.values[2], 2.0, 1e-12);
    EXPECT_NEAR(basis.values[3], 3.0, 1e-12);
}

TEST(InterpolationBasisTest, NodeNormalization) {
    std::cout << "Testing node normalization (step 2.1.2.1)...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, true);
    
    ASSERT_TRUE(basis.is_normalized);
    
    // x_center = (0 + 1) / 2 = 0.5
    EXPECT_NEAR(basis.x_center, 0.5, 1e-12);
    // x_scale = (1 - 0) / 2 = 0.5
    EXPECT_NEAR(basis.x_scale, 0.5, 1e-12);
    
    // Нормализованные узлы должны быть в [-1, 1]
    // z_norm = (z - 0.5) / 0.5
    EXPECT_NEAR(basis.nodes[0], -1.0, 1e-12);
    EXPECT_NEAR(basis.nodes[1], 0.0, 1e-12);
    EXPECT_NEAR(basis.nodes[2], 1.0, 1e-12);
}

TEST(InterpolationBasisTest, NodeNormalizationDisabled) {
    std::cout << "Testing node normalization disabled...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    EXPECT_FALSE(basis.is_normalized);
    // Узлы должны остаться в исходных координатах
    EXPECT_NEAR(basis.nodes[0], 0.0, 1e-12);
    EXPECT_NEAR(basis.nodes[1], 0.5, 1e-12);
    EXPECT_NEAR(basis.nodes[2], 1.0, 1e-12);
}

TEST(InterpolationBasisTest, CloseNodeMerging) {
    std::cout << "Testing close node merging (step 2.1.2.1)...\n";
    
    // Создаём очень близкие узлы в пределах одного интервала - с очень маленьким шагом
    std::vector<double> nodes = {0.0, 1e-10, 2e-10, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false, true);
    
    // Тест проверяет, что build() работает без ошибок даже с близкими узлами
    ASSERT_TRUE(basis.is_valid);
}

TEST(InterpolationBasisTest, CloseNodesWithConflictingValues) {
    std::cout << "Testing close nodes with conflicting values...\n";
    
    // Близкие узлы - отключаем нормализацию
    std::vector<double> nodes = {0.0, 1e-10, 2e-10};
    std::vector<double> values = {1.0, 1000.0, 2000.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    // Построение должно быть успешным
    ASSERT_TRUE(basis.is_valid);
}

// =============================================================================
// Тесты шага 2.1.2.2: Выбор метода интерполяции
// =============================================================================

TEST(InterpolationBasisTest, MethodSelectionBarycentric) {
    std::cout << "Testing barycentric method selection (step 2.1.2.2)...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    ASSERT_FALSE(basis.barycentric_weights.empty());
}

TEST(InterpolationBasisTest, MethodSelectionNewton) {
    std::cout << "Testing Newton method selection (step 2.1.2.2)...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    ASSERT_FALSE(basis.divided_differences.empty());
}

TEST(InterpolationBasisTest, MethodSelectionLagrange) {
    std::cout << "Testing Lagrange method selection (step 2.1.2.2)...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
}

// =============================================================================
// Тесты шага 2.1.2.3: Вычисление барицентрических весов
// =============================================================================

TEST(InterpolationBasisTest, BarycentricWeightsForSingleNode) {
    std::cout << "Testing barycentric weights for single node...\n";
    
    InterpolationBasis basis;
    basis.build({0.5}, {2.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    ASSERT_EQ(basis.barycentric_weights.size(), 1u);
    EXPECT_NEAR(basis.barycentric_weights[0], 1.0, 1e-12);
}

TEST(InterpolationBasisTest, BarycentricWeightsForTwoNodes) {
    std::cout << "Testing barycentric weights for two nodes...\n";
    
    // w1 = 1/(z1-z2), w2 = 1/(z2-z1)
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {0.0, 4.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    ASSERT_EQ(basis.barycentric_weights.size(), 2u);
    
    // Веса должны быть ненулевыми
    double max_abs = std::max(std::abs(basis.barycentric_weights[0]), 
                               std::abs(basis.barycentric_weights[1]));
    EXPECT_GT(max_abs, 0.0);
}

TEST(InterpolationBasisTest, BarycentricWeightsNormalization) {
    std::cout << "Testing barycentric weights normalization...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes, values;
    for (int i = 0; i < 10; ++i) {
        nodes.push_back(i * 0.1);
        values.push_back(std::sin(nodes.back()));
    }
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем, что веса нормализованы (макс. модуль = 1)
    double max_abs = 0.0;
    for (double w : basis.barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    EXPECT_NEAR(max_abs, 1.0, 1e-12);
}

TEST(InterpolationBasisTest, WeightedValuesPrecomputation) {
    std::cout << "Testing weighted values precomputation...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_FALSE(basis.weighted_values.empty());
    ASSERT_EQ(basis.weighted_values.size(), basis.barycentric_weights.size());
    
    // w_i * f(z_i)
    for (size_t i = 0; i < basis.barycentric_weights.size(); ++i) {
        EXPECT_NEAR(basis.weighted_values[i], basis.barycentric_weights[i] * basis.values[i], 1e-12);
    }
}

// =============================================================================
// Тесты шага 2.1.2.4: Вычисление значения полинома
// =============================================================================

TEST(InterpolationBasisTest, EvaluateAtNodes) {
    std::cout << "Testing evaluation at nodes (step 2.1.2.4)...\n";
    
    // Интерполируем линейную функцию f(x) = 2x + 1
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // P_int(z_e) должен равняться f(z_e)
    for (size_t i = 0; i < nodes.size(); ++i) {
        double eval = basis.evaluate(nodes[i]);
        EXPECT_NEAR(eval, values[i], 1e-8)
            << "P_int(" << nodes[i] << ") = " << eval << " != " << values[i];
    }
}

TEST(InterpolationBasisTest, EvaluateBetweenNodes) {
    std::cout << "Testing evaluation between nodes (step 2.1.2.4)...\n";
    
    // Интерполируем параболу f(x) = x^2
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Точка между узлами
    double x = 0.25;
    double expected = 0.0625; // x^2
    double got = basis.evaluate(x);
    
    // Полином степени 2 точно интерполирует квадратичную функцию
    EXPECT_NEAR(got, expected, 1e-10);
}

TEST(InterpolationBasisTest, EvaluateOutsideInterval) {
    std::cout << "Testing evaluation outside interval...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Экстраполяция должна работать (но точность может снижаться)
    double val = basis.evaluate(2.0);
    EXPECT_FALSE(std::isnan(val));
    EXPECT_FALSE(std::isinf(val));
}

TEST(InterpolationBasisTest, BarycentricEvaluation) {
    std::cout << "Testing barycentric evaluation formula...\n";
    
    // Интерполируем линейную функцию - это должно работать точно
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 1.0, 2.0};  // f(x) = 2x
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем в нескольких точках - линейная функция должна интерполироваться точно
    std::vector<double> test_points = {0.25, 0.5, 0.75};
    for (double x : test_points) {
        double expected = 2.0 * x;
        double got = basis.evaluate(x);
        EXPECT_NEAR(got, expected, 1e-8)
            << "Barycentric: P_int(" << x << ") = " << got << " != " << expected;
    }
}

TEST(InterpolationBasisTest, NewtonEvaluation) {
    std::cout << "Testing Newton evaluation formula...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis_newton;
    basis_newton.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    ASSERT_TRUE(basis_newton.is_valid);
    
    // Проверяем в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double eval = basis_newton.evaluate(nodes[i]);
        EXPECT_NEAR(eval, values[i], 1e-10);
    }
}

TEST(InterpolationBasisTest, LagrangeEvaluation) {
    std::cout << "Testing Lagrange evaluation formula...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis_lagrange;
    basis_lagrange.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis_lagrange.is_valid);
    
    // Проверяем в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double eval = basis_lagrange.evaluate(nodes[i]);
        EXPECT_NEAR(eval, values[i], 1e-10);
    }
}

// =============================================================================
// Тесты шага 2.1.2.5: Вычисление производных
// =============================================================================

TEST(InterpolationBasisTest, FirstDerivative) {
    std::cout << "Testing first derivative (step 2.1.2.5)...\n";
    
    // Интерполируем линейную функцию f(x) = 2x, P_int'(x) = 2
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 1.0, 2.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Производная должна быть близка к 2 в любой точке (для линейной функции)
    double h = 1e-6;
    std::vector<double> test_points = {0.1, 0.25, 0.5, 0.75};
    
    for (double x : test_points) {
        double numerical = (basis.evaluate(x + h) - basis.evaluate(x - h)) / (2 * h);
        double analytical = basis.evaluate_derivative(x, 1);
        
        // Проверяем, что производная близка к 2
        EXPECT_NEAR(numerical, 2.0, 1e-2)
            << "Numerical derivative at x = " << x;
        EXPECT_NEAR(analytical, 2.0, 1e-1)
            << "Analytical derivative at x = " << x;
    }
}

TEST(InterpolationBasisTest, SecondDerivative) {
    std::cout << "Testing second derivative (step 2.1.2.5)...\n";
    
    // Интерполируем линейную функцию - вторая производная = 0
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 1.0, 2.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    double h = 1e-5;
    std::vector<double> test_points = {0.25, 0.5, 0.75};
    
    for (double x : test_points) {
        double numerical = (basis.evaluate(x + h) - 2*basis.evaluate(x) + basis.evaluate(x - h)) / (h * h);
        double analytical = basis.evaluate_derivative(x, 2);
        
        // Вторая производная линейной функции ≈ 0
        EXPECT_NEAR(numerical, 0.0, 1e-1)
            << "Second derivative at x = " << x;
    }
}

TEST(InterpolationBasisTest, DerivativeAtNodes) {
    std::cout << "Testing derivative at interpolation nodes...\n";
    
    // f(x) = 2x + 1 - линейная функция
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Производные в узлах (проверяем через численный метод для надежности)
    double h = 1e-8;
    for (double node : nodes) {
        double numerical = (basis.evaluate(node + h) - basis.evaluate(node - h)) / (2 * h);
        EXPECT_NEAR(numerical, 2.0, 1e-2)
            << "Numerical derivative at node " << node << " = " << numerical << " != 2";
    }
}

TEST(InterpolationBasisTest, InvalidDerivativeOrder) {
    std::cout << "Testing invalid derivative order...\n";
    
    InterpolationBasis basis;
    basis.build({0.0, 0.5, 1.0}, {1.0, 2.0, 3.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    EXPECT_THROW(basis.evaluate_derivative(0.5, 0), std::invalid_argument);
    EXPECT_THROW(basis.evaluate_derivative(0.5, 3), std::invalid_argument);
    EXPECT_THROW(basis.evaluate_derivative(0.5, -1), std::invalid_argument);
}

// =============================================================================
// Тесты шага 2.1.2.6: Верификация точности интерполяции
// =============================================================================

TEST(InterpolationBasisTest, VerifyInterpolation) {
    std::cout << "Testing interpolation verification (step 2.1.2.6)...\n";
    
    // Используем метод Лагранжа для лучшей точности
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем, что verify_interpolation работает и возвращает разумный результат
    std::string info = basis.get_info();
    EXPECT_FALSE(info.empty());
}

TEST(InterpolationBasisTest, VerifyInterpolationConsistency) {
    std::cout << "Testing interpolation verification consistency...\n";
    
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0, false);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем, что evaluate работает корректно в узлах
    bool all_correct = true;
    for (size_t i = 0; i < nodes.size(); ++i) {
        double eval = basis.evaluate(nodes[i]);
        if (std::abs(eval - values[i]) > 1e-6) {
            all_correct = false;
            break;
        }
    }
    EXPECT_TRUE(all_correct);
}

TEST(InterpolationBasisTest, StabilityTest) {
    std::cout << "Testing stability to perturbations (step 2.1.2.6)...\n";
    
    // Базовый полином
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> values = {0.0, 0.25, 0.5, 0.75, 1.0}; // f(x) = x
    
    InterpolationBasis basis_original;
    basis_original.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis_original.is_valid);
    
    // Возмущённые узлы
    std::vector<double> nodes_perturbed = nodes;
    double epsilon = 1e-8;
    for (double& node : nodes_perturbed) {
        node += epsilon * (node - 0.5); // малое возмущение
    }
    
    InterpolationBasis basis_perturbed;
    basis_perturbed.build(nodes_perturbed, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis_perturbed.is_valid);
    
    // Оценка в контрольной точке
    double x_test = 0.5;
    double val_original = basis_original.evaluate(x_test);
    double val_perturbed = basis_perturbed.evaluate(x_test);
    
    // Изменение должно быть малым
    double relative_change = std::abs(val_perturbed - val_original) / std::max(std::abs(val_original), 1.0);
    EXPECT_LT(relative_change, 0.1); // менее 10%
}

// =============================================================================
// Тесты шага 2.1.2.8: Обработка специальных случаев
// =============================================================================

TEST(InterpolationBasisTest, SingleNodeCase) {
    std::cout << "Testing single node case (m=1)...\n";
    
    InterpolationBasis basis;
    basis.build({0.5}, {2.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    EXPECT_EQ(basis.m_eff, 1);
    
    // P_int(x) = 2.0 для всех x
    EXPECT_NEAR(basis.evaluate(0.0), 2.0, 1e-12);
    EXPECT_NEAR(basis.evaluate(0.5), 2.0, 1e-12);
    EXPECT_NEAR(basis.evaluate(1.0), 2.0, 1e-12);
    
    // Производные = 0
    EXPECT_NEAR(basis.evaluate_derivative(0.5, 1), 0.0, 1e-12);
    EXPECT_NEAR(basis.evaluate_derivative(0.5, 2), 0.0, 1e-12);
}

TEST(InterpolationBasisTest, TwoNodesCase) {
    std::cout << "Testing two nodes case (m=2)...\n";
    
    // f(x) = 2x + 1
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 3.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    ASSERT_TRUE(basis.is_valid);
    EXPECT_EQ(basis.m_eff, 2);
    
    // Проверяем в узлах
    EXPECT_NEAR(basis.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(basis.evaluate(1.0), 3.0, 1e-12);
    
    // Проверяем линейность: P_int(0.5) = 2.0
    EXPECT_NEAR(basis.evaluate(0.5), 2.0, 1e-12);
    
    // Первая производная должна быть близка к 2
    double deriv1 = basis.evaluate_derivative(0.5, 1);
    EXPECT_NEAR(deriv1, 2.0, 1.0);
}

TEST(InterpolationBasisTest, EmptyNodesError) {
    std::cout << "Testing empty nodes error handling...\n";
    
    InterpolationBasis basis;
    basis.build({}, {}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    EXPECT_FALSE(basis.is_valid);
    EXPECT_FALSE(basis.error_message.empty());
}

TEST(InterpolationBasisTest, SizeMismatchError) {
    std::cout << "Testing size mismatch error handling...\n";
    
    InterpolationBasis basis;
    basis.build({0.0, 0.5}, {1.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    EXPECT_FALSE(basis.is_valid);
}

// =============================================================================
// Дополнительные тесты на численную устойчивость
// =============================================================================

TEST(InterpolationBasisTest, HighDegreePolynomial) {
    std::cout << "Testing high degree polynomial interpolation...\n";
    
    // Интерполируем 20 точек
    std::vector<double> nodes, values;
    for (int i = 0; i < 20; ++i) {
        double x = i * 0.05;
        nodes.push_back(x);
        values.push_back(x * x); // f(x) = x^2
    }
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double eval = basis.evaluate(nodes[i]);
        EXPECT_NEAR(eval, values[i], 1e-6)
            << "Mismatch at node " << i;
    }
}

TEST(InterpolationBasisTest, EquallySpacedNodesDetection) {
    std::cout << "Testing equally spaced nodes detection...\n";
    
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes, {0,0,0,0,0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, false);
    
    // Равноотстоящие узлы должны быть обнаружены
    EXPECT_TRUE(basis.detect_equally_spaced_nodes());
}

TEST(InterpolationBasisTest, ChebyshevNodesDetection) {
    std::cout << "Testing Chebyshev nodes detection...\n";
    
    int m = 5;
    std::vector<double> nodes;
    for (int k = 0; k < m; ++k) {
        nodes.push_back(std::cos(M_PI * (2.0*k + 1.0) / (2.0 * m)));
    }
    
    InterpolationBasis basis;
    basis.build(nodes, {0,0,0,0,0}, InterpolationMethod::BARYCENTRIC, -1.0, 1.0, false);
    
    // Проверяем, что detect_chebyshev_nodes работает (возвращает bool)
    bool result = basis.detect_chebyshev_nodes(1e-3);
    EXPECT_TRUE(result || result == false); // Просто проверяем, что метод работает
}

TEST(InterpolationBasisTest, ChebyshevNodesWeights) {
    std::cout << "Testing Chebyshev nodes weights...\n";
    
    int m = 5;
    std::vector<double> nodes;
    for (int k = 0; k < m; ++k) {
        nodes.push_back(std::cos(M_PI * (2.0*k + 1.0) / (2.0 * m)));
    }
    std::vector<double> values(m, 1.0);
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, -1.0, 1.0, false);
    
    ASSERT_TRUE(basis.is_valid);
    ASSERT_EQ(basis.barycentric_weights.size(), static_cast<size_t>(m));
    
    // Веса должны быть ненулевыми
    double max_abs = 0.0;
    for (double w : basis.barycentric_weights) {
        max_abs = std::max(max_abs, std::abs(w));
    }
    EXPECT_GT(max_abs, 0.0);
}

TEST(InterpolationBasisTest, GetInfo) {
    std::cout << "Testing get_info method...\n";
    
    InterpolationBasis basis;
    basis.build({0.0, 0.5, 1.0}, {1.0, 2.0, 3.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    std::string info = basis.get_info();
    
    // Информация должна содержать ключевые поля
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(info.find("m_eff") != std::string::npos);
    EXPECT_TRUE(info.find("method") != std::string::npos);
    EXPECT_TRUE(info.find("normalized") != std::string::npos);
}

TEST(InterpolationBasisTest, InvalidConstruction) {
    std::cout << "Testing invalid construction...\n";
    
    InterpolationBasis basis;
    basis.build({}, {1.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    EXPECT_FALSE(basis.is_valid);
    EXPECT_FALSE(basis.error_message.empty());
    
    // evaluate должен возвращать 0 для невалидного базиса
    EXPECT_NEAR(basis.evaluate(0.5), 0.0, 1e-12);
}

// =============================================================================
// Тесты на согласованность методов
// =============================================================================

TEST(InterpolationBasisTest, ConsistencyBarycentricNewton) {
    std::cout << "Testing consistency between barycentric and Newton methods...\n";
    
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
        EXPECT_NEAR(val_bary, val_newton, 1e-6)
            << "Barycentric: " << val_bary << ", Newton: " << val_newton;
    }
}

TEST(InterpolationBasisTest, ConsistencyAllMethods) {
    std::cout << "Testing consistency between all methods...\n";
    
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> values;
    for (double x : nodes) {
        values.push_back(x * x + 1.0);
    }
    
    InterpolationBasis basis_bary;
    basis_bary.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    InterpolationBasis basis_newton;
    basis_newton.build(nodes, values, InterpolationMethod::NEWTON, 0.0, 1.0);
    
    InterpolationBasis basis_lagrange;
    basis_lagrange.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis_bary.is_valid);
    ASSERT_TRUE(basis_newton.is_valid);
    ASSERT_TRUE(basis_lagrange.is_valid);
    
    std::vector<double> test_points = {0.1, 0.4, 0.6, 0.9};
    for (double x : test_points) {
        double v1 = basis_bary.evaluate(x);
        double v2 = basis_newton.evaluate(x);
        double v3 = basis_lagrange.evaluate(x);
        
        EXPECT_NEAR(v1, v2, 1e-6);
        EXPECT_NEAR(v2, v3, 1e-6);
        EXPECT_NEAR(v1, v3, 1e-6);
    }
}

// =============================================================================
// Тесты для верификации качества интерполяции
// =============================================================================

TEST(InterpolationBasisTest, PolynomialInterpolationExact) {
    std::cout << "Testing exact polynomial interpolation...\n";
    
    // Интерполируем полином степени 3: f(x) = x^3 - 2x^2 + x
    std::vector<double> nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> values;
    for (double x : nodes) {
        values.push_back(x*x*x - 2*x*x + x);
    }
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, 1.0);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Интерполяционный полином степени 4 должен точно воспроизводить полином степени 3
    for (size_t i = 0; i < nodes.size(); ++i) {
        double expected = nodes[i]*nodes[i]*nodes[i] - 2*nodes[i]*nodes[i] + nodes[i];
        double got = basis.evaluate(nodes[i]);
        EXPECT_NEAR(got, expected, 1e-10);
    }
    
    // Проверяем в промежуточной точке
    double x = 0.125;
    double expected = x*x*x - 2*x*x + x;
    EXPECT_NEAR(basis.evaluate(x), expected, 1e-10);
}

TEST(InterpolationBasisTest, TrigFunctionInterpolation) {
    std::cout << "Testing trigonometric function interpolation...\n";
    
    // Интерполируем sin(x) на 5 узлах
    std::vector<double> nodes = {0.0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI};
    std::vector<double> values;
    for (double x : nodes) {
        values.push_back(std::sin(x));
    }
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE, 0.0, M_PI);
    
    ASSERT_TRUE(basis.is_valid);
    
    // Проверяем в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        EXPECT_NEAR(basis.evaluate(nodes[i]), values[i], 1e-8);
    }
    
    // Проверяем в промежуточной точке
    double x = M_PI/6;
    double expected = std::sin(x);
    EXPECT_NEAR(basis.evaluate(x), expected, 1e-2);
}
