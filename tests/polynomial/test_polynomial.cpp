#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "mixed_approximation/polynomial.h"

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

TEST(PolynomialTest, BasicOperations) {
    std::cout << "Testing Polynomial basic operations...\n";
    
    // Тест создания полинома
    Polynomial p(std::vector<double>{1.0, 0.0, -1.0});  // x^2 - 1
    EXPECT_EQ(p.degree(), 2);
    
    // Тест evaluate
    EXPECT_NEAR(p.evaluate(0.0), -1.0, 1e-10);
    EXPECT_NEAR(p.evaluate(1.0), 0.0, 1e-10);
    EXPECT_NEAR(p.evaluate(2.0), 3.0, 1e-10);
    
    // Тест derivative
    EXPECT_NEAR(p.derivative(0.0), 0.0, 1e-10);
    EXPECT_NEAR(p.derivative(1.0), 2.0, 1e-10);
    
    // Тест second_derivative
    EXPECT_NEAR(p.second_derivative(0.0), 2.0, 1e-10);
    EXPECT_NEAR(p.second_derivative(1.0), 2.0, 1e-10);
}

TEST(PolynomialTest, ArithmeticOperations) {
    std::cout << "Testing Polynomial arithmetic...\n";
    
    Polynomial p1(std::vector<double>{1.0, 2.0, 3.0});  // x^2 + 2x + 3
    Polynomial p2(std::vector<double>{3.0, 2.0, 1.0});  // 3x^2 + 2x + 1
    
    // Сложение: (1+3)x^2 + (2+2)x + (3+1) = 4x^2 + 4x + 4
    Polynomial sum = p1 + p2;
    EXPECT_EQ(sum.degree(), 2);
    EXPECT_NEAR(sum.evaluate(0.0), 4.0, 1e-10);
    EXPECT_NEAR(sum.evaluate(1.0), 12.0, 1e-10);
    
    // Вычитание: (1-3)x^2 + (2-2)x + (3-1) = -2x^2 + 0x + 2
    Polynomial diff = p1 - p2;
    EXPECT_EQ(diff.degree(), 2);
    EXPECT_NEAR(diff.evaluate(0.0), 2.0, 1e-10);
    EXPECT_NEAR(diff.evaluate(1.0), 0.0, 1e-10);
    
    // Умножение на скаляр
    Polynomial scaled = p1 * 2.0;
    EXPECT_NEAR(scaled.evaluate(0.0), 6.0, 1e-10);
    EXPECT_NEAR(scaled.evaluate(1.0), 12.0, 1e-10);
}

TEST(PolynomialTest, SquaredErrorAndGradient) {
    std::cout << "Testing squared error and gradient...\n";
    
    Polynomial p(std::vector<double>{1.0, 0.0});  // x
    
    // Проверка squared_error
    double error1 = p.squared_error(0.5, 0.5);
    EXPECT_NEAR(error1, 0.0, 1e-10);
    
    double error2 = p.squared_error(0.5, 1.0);
    EXPECT_NEAR(error2, 0.25, 1e-10);
    
    // Проверка gradient_squared_error
    auto grad = p.gradient_squared_error(0.5, 0.5);
    ASSERT_EQ(grad.size(), 2u);
    EXPECT_NEAR(grad[0], 0.0, 1e-10);
    EXPECT_NEAR(grad[1], 0.0, 1e-10);
    
    auto grad2 = p.gradient_squared_error(0.5, 1.0);
    // Для P(x) = a1*x + a0, error = (a1*x + a0 - target)^2
    // d/da1 = 2*(P(x)-target)*x = 2*(0.5-1)*0.5 = -0.5
    // d/da0 = 2*(P(x)-target) = 2*(0.5-1) = -1.0
    // Примечание: gradient_squared_error возвращает градиент в порядке [a0, a1] (обратном)
    EXPECT_NEAR(grad2[0], -1.0, 1e-10);
    EXPECT_NEAR(grad2[1], -0.5, 1e-10);
}

TEST(PolynomialTest, Multiplication) {
    std::cout << "Testing polynomial multiplication...\n";
    
    Polynomial p1(std::vector<double>{1.0, 2.0});  // x + 2
    Polynomial p2(std::vector<double>{1.0, 0.0, -1.0});  // x^2 - 1
    
    Polynomial product = p1 * p2;  // (x+2)(x^2-1) = x^3 + 2x^2 - x - 2
    EXPECT_EQ(product.degree(), 3);
    EXPECT_NEAR(product.evaluate(0.0), -2.0, 1e-10);
    EXPECT_NEAR(product.evaluate(1.0), 0.0, 1e-10);
    EXPECT_NEAR(product.evaluate(2.0), 12.0, 1e-10);
}

TEST(PolynomialTest, MinusScalar) {
    std::cout << "Testing polynomial minus scalar...\n";
    
    Polynomial p(std::vector<double>{1.0, 0.0, -1.0});  // x^2 - 1
    Polynomial result = p.minus_scalar(5.0);  // x^2 - 6
    
    EXPECT_EQ(result.degree(), 2);
    EXPECT_NEAR(result.evaluate(0.0), -6.0, 1e-10);
    EXPECT_NEAR(result.evaluate(1.0), -5.0, 1e-10);
    EXPECT_NEAR(result.evaluate(2.0), -2.0, 1e-10);
}

TEST(PolynomialTest, EdgeCases) {
    std::cout << "Testing edge cases...\n";
    
    // Нулевой полином
    Polynomial zero(0);
    EXPECT_EQ(zero.degree(), 0);
    EXPECT_NEAR(zero.evaluate(5.0), 0.0, 1e-10);
    
    // Константа
    Polynomial constant(std::vector<double>{5.0});
    EXPECT_EQ(constant.degree(), 0);
    EXPECT_NEAR(constant.evaluate(100.0), 5.0, 1e-10);
    
    // Очень высокая степень
    std::vector<double> high_degree_coeffs(100, 0.0);
    high_degree_coeffs[0] = 1.0;
    high_degree_coeffs[99] = 1.0;
    Polynomial high(high_degree_coeffs);
    EXPECT_EQ(high.degree(), 99);
}
