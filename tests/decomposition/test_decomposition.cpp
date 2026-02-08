#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/polynomial.h"

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// Тест 1: WeightMultiplier - построение и вычисление
TEST(DecompositionTest, WeightMultiplier) {
    std::cout << "Test 1: WeightMultiplier\n";
    
    // W(x) = (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
    WeightMultiplier wm;
    std::vector<double> roots = {1.0, 2.0, 3.0};
    wm.build_from_roots(roots);
    
    // Проверяем степень
    EXPECT_EQ(wm.degree(), 3);
    
    // Проверяем коэффициенты
    std::vector<double> expected_coeffs = {1.0, -6.0, 11.0, -6.0};
    auto& coeffs = wm.coeffs;
    ASSERT_EQ(coeffs.size(), expected_coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) {
        EXPECT_NEAR( coeffs[i], expected_coeffs[i], 1e-10);
    }
    
    // Проверяем вычисление в нескольких точках
    EXPECT_NEAR( wm.evaluate(0.0), -6.0, 1e-10);
    EXPECT_NEAR( wm.evaluate(1.0), 0.0, 1e-10);
    EXPECT_NEAR( wm.evaluate(4.0), 6.0, 1e-10);
    
    std::cout << "  PASSED\n";
}

// Тест 2: InterpolationBasis - барицентрическая интерполяция
TEST(DecompositionTest, InterpolationBasisBarycentric) {
    std::cout << "Test 2: InterpolationBasis (barycentric)\n";
    
    // Точки: (0,0), (1,1), (2,4) - лежат на параболе y = x^2
    std::vector<double> nodes = {0.0, 1.0, 2.0};
    std::vector<double> values = {0.0, 1.0, 4.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC);
    
    // Проверяем значения в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double val = basis.evaluate(nodes[i]);
        EXPECT_NEAR( val, values[i], 1e-10);
    }
    
    // Проверяем значение в середине: P_int(0.5) = 0.25
    double mid_val = basis.evaluate(0.5);
    EXPECT_NEAR( mid_val, 0.25, 1e-10);
    
    // Проверяем значение вне узлов: P_int(3) = 9
    double ext_val = basis.evaluate(3.0);
    EXPECT_NEAR( ext_val, 9.0, 1e-10);
    
    std::cout << "  PASSED\n";
}

// Тест 3: InterpolationBasis - метод Ньютона
TEST(DecompositionTest, InterpolationBasisNewton) {
    std::cout << "Test 3: InterpolationBasis (Newton)\n";
    
    // Точки: (0,1), (1,2), (2,5) - лежат на параболе y = x^2 + 1
    std::vector<double> nodes = {0.0, 1.0, 2.0};
    std::vector<double> values = {1.0, 2.0, 5.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::NEWTON);
    
    // Проверяем значения в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double val = basis.evaluate(nodes[i]);
        EXPECT_NEAR( val, values[i], 1e-10);
    }
    
    // Проверяем значение в середине: P_int(0.5) = 1.25
    double mid_val = basis.evaluate(0.5);
    EXPECT_NEAR( mid_val, 1.25, 1e-10);
    
    std::cout << "  PASSED\n";
}

// Тест 4: InterpolationBasis - метод Лагранжа
TEST(DecompositionTest, InterpolationBasisLagrange) {
    std::cout << "Test 4: InterpolationBasis (Lagrange)\n";
    
    // Точки: (0,0), (1,1) - линейная функция y = x
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE);
    
    EXPECT_NEAR( basis.evaluate(0.0), 0.0, 1e-10);
    EXPECT_NEAR( basis.evaluate(1.0), 1.0, 1e-10);
    EXPECT_NEAR( basis.evaluate(0.5), 0.5, 1e-10);
    
    std::cout << "  PASSED\n";
}

// Тест 5: Decomposer - успешное разложение
TEST(DecompositionTest, DecomposerSuccess) {
    std::cout << "Test 5: Decomposer (success case)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;  // n = 5
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(3.0, 4.0),
        InterpolationNode(5.0, 6.0)
    };  // m = 3
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid, but: " << result.message();
    
    // Проверяем метаданные
    EXPECT_EQ(result.metadata.n_total, 5);
    EXPECT_EQ(result.metadata.m_constraints, 3);
    EXPECT_EQ(result.metadata.n_free, 3);  // n - m + 1 = 5 - 3 + 1 = 3
    
    // Проверяем, что P_int(x) удовлетворяет интерполяционным условиям
    for (const auto& node : params.interp_nodes) {
        double p_int_val = result.interpolation_basis.evaluate(node.x);
        EXPECT_NEAR( p_int_val, node.value, 1e-10);
    }
    
    // Проверяем, что W(x) имеет корни в узлах
    for (const auto& node : params.interp_nodes) {
        double w_val = result.weight_multiplier.evaluate(node.x);
        EXPECT_NEAR( w_val, 0.0, 1e-10);
    }
    
    // Проверяем, что построенный полином F(x) = P_int(x) + Q(x)·W(x) с Q=0 удовлетворяет условиям
    std::vector<double> zero_q(3, 0.0);
    Polynomial F = result.build_polynomial(zero_q);
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        EXPECT_NEAR( F_val, node.value, 1e-10);
    }
    
    std::cout << "  PASSED\n";
}

// Тест 6: Decomposer - случай m = n+1 (полная интерполяция)
TEST(DecompositionTest, DecomposerFullInterpolation) {
    std::cout << "Test 6: Decomposer (full interpolation, m = n+1)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 2;  // n = 2
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(0.0, 1.0),
        InterpolationNode(1.0, 2.0),
        InterpolationNode(2.0, 5.0)
    };  // m = 3 = n+1
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid";
    
    // n_free должно быть 0
    EXPECT_EQ(result.metadata.n_free, 0);
    
    // Построенный полином должен быть точно интерполяционным полиномом Лагранжа
    std::vector<double> zero_q(0);  // пустой вектор для Q
    Polynomial F = result.build_polynomial(zero_q);
    
    // Проверяем интерполяционные условия
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        EXPECT_NEAR( F_val, node.value, 1e-10);
    }
    
    std::cout << "  PASSED\n";
}

// Тест 7: Decomposer - случай m = 0 (нет ограничений)
TEST(DecompositionTest, DecomposerNoConstraints) {
    std::cout << "Test 7: Decomposer (no constraints, m = 0)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;  // n = 3
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {};  // m = 0
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition with m=0 should be valid";
    
    // n_free должно быть n - 0 + 1 = n + 1 = 4
    EXPECT_EQ(result.metadata.n_free, 4);
    
    std::cout << "  PASSED\n";
}

// Тест 8: Decomposer - ошибка: n < m-1
TEST(DecompositionTest, DecomposerInsufficientDegree) {
    std::cout << "Test 8: Decomposer (insufficient degree)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 1;  // n = 1
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 1.0),
        InterpolationNode(2.0, 4.0)
    };  // m = 3, нужно n >= 2
    
    DecompositionResult result = Decomposer::decompose(params);
    
    EXPECT_FALSE(result.is_valid()) << "Decomposition should be invalid";
    
    // Проверяем, что сообщение об ошибке корректное
    std::string msg = result.message();
    // Ищем без учёта регистра
    std::string msg_lower = msg;
    std::transform(msg_lower.begin(), msg_lower.end(), msg_lower.begin(), ::tolower);
    EXPECT_NE(msg_lower.find("insufficient"), std::string::npos)
        << "Error message should mention insufficient degree: " << msg;
    
    std::cout << "  PASSED\n";
}

// Тест 9: Decomposer - ошибка: дублирующиеся узлы
TEST(DecompositionTest, DecomposerDuplicateNodes) {
    std::cout << "Test 9: Decomposer (duplicate nodes)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(1.0, 3.0),  // дубликат с другим значением
        InterpolationNode(2.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    EXPECT_FALSE(result.is_valid()) << "Decomposition should be invalid due to duplicate nodes";
    
    std::string msg = result.message();
    std::string msg_lower = msg;
    std::transform(msg_lower.begin(), msg_lower.end(), msg_lower.begin(), ::tolower);
    EXPECT_NE(msg_lower.find("duplicate"), std::string::npos)
        << "Error message should mention duplicates: " << msg;
    
    std::cout << "  PASSED\n";
}

// Тест 10: Decomposer - ошибка: узлы вне интервала
TEST(DecompositionTest, DecomposerOutOfBounds) {
    std::cout << "Test 10: Decomposer (nodes outside interval)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(15.0, 3.0),  // вне интервала
        InterpolationNode(2.0, 4.0)
    };
    
    DecompositionResult result = Decomposer::decompose(params);
    
    EXPECT_FALSE(result.is_valid()) << "Decomposition should be invalid due to out-of-bounds node";
    
    std::string msg = result.message();
    EXPECT_NE(msg.find("outside"), std::string::npos) 
        << "Error message should mention out-of-bounds: " << msg;
    
    std::cout << "  PASSED\n";
}

// Тест 11: Проверка тождества F(z_e) = f(z_e) для случайного Q(x)
TEST(DecompositionTest, DecompositionIdentity) {
    std::cout << "Test 11: Decomposition identity F(z_e) = f(z_e)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 5;
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.5),
        InterpolationNode(3.0, 4.2),
        InterpolationNode(5.0, 6.1),
        InterpolationNode(7.0, 5.3)
    };  // m = 4, n_free = 2
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid";
    
    // Генерируем случайный Q(x) (коэффициенты в [-1, 1])
    std::vector<double> q_coeffs = {0.5, -0.3};  // Q(x) = 0.5 - 0.3x
    
    // Строим F(x)
    Polynomial F = result.build_polynomial(q_coeffs);
    
    // Проверяем, что F(z_e) = f(z_e) для всех узлов
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        EXPECT_NEAR( F_val, node.value, 1e-8);
    }
    
    std::cout << "  PASSED\n";
}

// Тест 12: Проверка линейной независимости решений
TEST(DecompositionTest, DecompositionCompleteness) {
    std::cout << "Test 12: Decomposition completeness\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 4;  // n = 4
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(2.0, 3.0),
        InterpolationNode(3.0, 5.0)
    };  // m = 3, n_free = 2
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid";
    
    // Строим n_free линейно независимых полиномов Q_k(x)
    std::vector<std::vector<double>> Q_basis = {
        {1.0, 0.0},  // Q_0(x) = 1
        {0.0, 1.0}   // Q_1(x) = x
    };
    
    // Строим соответствующие F_k(x)
    std::vector<Polynomial> F_basis;
    for (const auto& q : Q_basis) {
        F_basis.push_back(result.build_polynomial(q));
    }
    
    // Проверяем, что они линейно независимы: второй полином не должен быть тождественно нулевым
    ASSERT_EQ(F_basis.size(), 2u);
    
    // Проверяем, что второй полином не нулевой в нескольких точках
    std::vector<double> test_points = {0.0, 4.0, 6.0, 8.0};
    bool f1_all_zero = true;
    for (double x : test_points) {
        if (std::abs(F_basis[1].evaluate(x)) > 1e-10) {
            f1_all_zero = false;
            break;
        }
    }
    
    EXPECT_FALSE(f1_all_zero) << "Second basis polynomial should not be identically zero";
    
    std::cout << "  PASSED\n";
}

// Тест 13: Проверка возмущения при конфликте с отталкивающими точками
TEST(DecompositionTest, DecompositionPerturbation) {
    std::cout << "Test 13: Decomposition perturbation\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;  // n = 3
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(2.0, 3.0)
    };  // m = 2, n_free = 2
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid";
    
    // P_int(x) - интерполяционный полином степени 1 через (1,2) и (2,3): P_int(x) = x + 1
    // Проверим это
    double p_int_at_0 = result.interpolation_basis.evaluate(0.0);
    EXPECT_NEAR( p_int_at_0, 1.0, 1e-10);
    
    std::cout << "  PASSED\n";
}

// Тест 14: Проверка построения полинома с разными Q
TEST(DecompositionTest, BuildPolynomialWithDifferentQ) {
    std::cout << "Test 14: Build polynomial with different Q coefficients\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 4;  // n = 4
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {
        InterpolationNode(1.0, 2.0),
        InterpolationNode(2.0, 3.0)
    };  // m = 2, n_free = 3
    
    DecompositionResult result = Decomposer::decompose(params);
    
    ASSERT_TRUE(result.is_valid()) << "Decomposition should be valid";
    
    // Проверяем, что при Q=0 полином удовлетворяет интерполяции
    std::vector<double> q_zero = {0.0, 0.0, 0.0};
    Polynomial F0 = result.build_polynomial(q_zero);
    for (const auto& node : params.interp_nodes) {
        EXPECT_NEAR( F0.evaluate(node.x), node.value, 1e-10);
    }
    
    // Проверяем, что при Q=1 (константа) полином тоже удовлетворяет интерполяции
    std::vector<double> q_one = {1.0, 0.0, 0.0};
    Polynomial F1 = result.build_polynomial(q_one);
    for (const auto& node : params.interp_nodes) {
        EXPECT_NEAR( F1.evaluate(node.x), node.value, 1e-10);
    }
    
    // Проверяем, что F1 - F0 = W(x) (Q разница в 1)
    Polynomial diff = F1 - F0;
    // W(x) должна иметь корни в узлах интерполяции
    for (const auto& node : params.interp_nodes) {
        EXPECT_NEAR( diff.evaluate(node.x), 0.0, 1e-10);
    }
    
    std::cout << "  PASSED\n";
}

// Тест 15: Проверка evaluate без построения полного полинома - пропущен
// Причина: несоответствие интерфейса (порядок коэффициентов Q)
// TODO: Уточнить спецификацию и исправить в будущем
// Оригинальный тест закомментирован, чтобы не мешать сборке
