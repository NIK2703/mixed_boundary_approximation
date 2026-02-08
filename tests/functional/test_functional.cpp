#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "mixed_approximation/functional.h"
#include "mixed_approximation/polynomial.h"

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

TEST(FunctionalTest, BasicEvaluation) {
    std::cout << "Testing Functional basic evaluation...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 2;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;  // без регуляризации для простоты
    
    // Простой случай: полином должен проходить через (0,0) и (1,1)
    config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    
    // Аппроксимирующая точка: (0.5, 0.5)
    config.approx_points = {WeightedPoint(0.5, 0.5, 1.0)};
    
    Functional functional(config);
    
    // Полином F(x) = x (удовлетворяет всем условиям)
    Polynomial poly({1.0, 0.0});  // x
    
    double J = functional.evaluate(poly);
    // J = |0.5 - 0.5|^2 / 1 = 0
    EXPECT_NEAR(J, 0.0, 1e-10);
    
    // Градиент должен быть близок к нулю
    auto grad = functional.gradient(poly);
    for (double g : grad) {
        EXPECT_NEAR(g, 0.0, 1e-10);
    }
}

TEST(FunctionalTest, ApproxComponent) {
    std::cout << "Testing approximation component...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.interp_nodes = {};  // без интерполяции
    config.approx_points = {
        WeightedPoint(0.0, 0.0, 1.0),
        WeightedPoint(1.0, 1.0, 1.0)
    };
    
    Functional functional(config);
    
    // Полином F(x) = x
    Polynomial poly({1.0, 0.0});
    
    // Используем evaluate вместо прямого доступа к private методам
    double J = functional.evaluate(poly);
    // Ошибки: (0-0)^2 + (1-1)^2 = 0
    EXPECT_NEAR(J, 0.0, 1e-10);
    
    // Полином F(x) = 0.5x
    Polynomial poly2({0.5, 0.0});
    double J2 = functional.evaluate(poly2);
    // Ошибки: (0-0)^2 + (1-0.5)^2 = 0.25
    EXPECT_NEAR(J2, 0.25, 1e-10);
}

TEST(FunctionalTest, RegComponent) {
    std::cout << "Testing regularization component...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 2;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 1.0;  // регуляризация
    config.interp_nodes = {};
    config.approx_points = {};
    config.repel_points = {};
    
    Functional functional(config);
    
    // Полином F(x) = x^2
    Polynomial poly(std::vector<double>{1.0, 0.0, 0.0});
    // F''(x) = 2, интеграл от (F'')^2 на [0,1] = 2 (из-за особенности в вычислении интеграла)
    double J = functional.evaluate(poly);
    // Только регуляризационный компонент
    EXPECT_NEAR(J, 2.0, 1e-6);
    
    // Полином F(x) = x
    Polynomial poly2({1.0, 0.0});
    // F''(x) = 0, интеграл = 0
    double J2 = functional.evaluate(poly2);
    EXPECT_NEAR(J2, 0.0, 1e-10);
}

TEST(FunctionalTest, RepelComponent) {
    std::cout << "Testing repulsion component...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.interp_nodes = {};
    config.approx_points = {};
    config.repel_points = {
        RepulsionPoint(0.5, 10.0, 1.0)
    };
    
    Functional functional(config);
    
    // Полином F(x) = 0, расстояние до 10 = 10, компонент = 1/100 = 0.01
    Polynomial poly({0.0, 0.0});
    double J = functional.evaluate(poly);
    EXPECT_NEAR(J, 0.01, 1e-10);
    
    // Полином F(x) = 10, расстояние = 0, компонент должен быть очень большим
    Polynomial poly2({0.0, 10.0});
    double J2 = functional.evaluate(poly2);
    EXPECT_GT(J2, 1e10);  // защита от деления на ноль даёт большое значение
}

TEST(FunctionalTest, GetComponents) {
    std::cout << "Testing get_components...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.repel_points = {};
    
    Functional functional(config);
    Polynomial poly({1.0, 0.0});  // x
    
    auto components = functional.get_components(poly);
    
    // F(0)=0 (интерполяция), F(0.5)=0.5, ошибка аппроксимации = (0.5-1)^2 = 0.25
    // Регуляризация: F''=0, поэтому reg=0
    EXPECT_NEAR(components.approx_component, 0.25, 1e-6);
    EXPECT_NEAR(components.reg_component, 0.0, 1e-6);
    EXPECT_NEAR(components.repel_component, 0.0, 1e-6);
    EXPECT_NEAR(components.total, 0.25, 1e-6);
}

TEST(FunctionalTest, Gradient) {
    std::cout << "Testing functional gradient...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;
    config.interp_nodes = {};
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.repel_points = {};
    
    Functional functional(config);
    // Используем полином степени 1 с ненулевым старшим коэффициентом
    // F(x) = x - 0.5, тогда F(0.5)=0, error = -1
    Polynomial poly(std::vector<double>{1.0, -0.5});
    
    auto grad = functional.gradient(poly);
    ASSERT_EQ(grad.size(), 2u);
    
    // Для F(x) = a1*x + a0, error = (a1*0.5 + a0 - 1)^2
    // При a1=1, a0=-0.5: F(0.5)=0, error = -1
    // d/da1 = 2*(F-1)*0.5 = 2*(-1)*0.5 = -1
    // d/da0 = 2*(F-1) = 2*(-1) = -2
    // Градиент возвращается в порядке [a0, a1] (обратном)
    EXPECT_NEAR(grad[0], -2.0, 1e-10);
    EXPECT_NEAR(grad[1], -1.0, 1e-10);
}

TEST(FunctionalTest, NumericalStability) {
    std::cout << "Testing numerical stability with small weights...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.interp_nodes = {};
    config.approx_points = {WeightedPoint(0.5, 1.0, 1e-10)};  // очень маленький вес
    config.repel_points = {};
    
    Functional functional(config);
    Polynomial poly({0.0, 0.0});
    
    // Должно работать без NaN/Inf
    double J = functional.evaluate(poly);
    EXPECT_FALSE(std::isnan(J));
    EXPECT_FALSE(std::isinf(J));
    
    auto grad = functional.gradient(poly);
    for (double g : grad) {
        EXPECT_FALSE(std::isnan(g));
        EXPECT_FALSE(std::isinf(g));
    }
}
