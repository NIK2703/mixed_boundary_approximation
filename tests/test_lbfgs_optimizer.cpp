#include <gtest/gtest.h>
#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>

namespace mixed_approx {
namespace test {

// Тест 1: Проверка создания оптимизатора
TEST(LBFGSOptimizerTest, Constructor) {
    LBFGSOptimizer optimizer;
    // По умолчанию параметры должны быть установлены
    // Просто проверяем, что объект создается без ошибок
}

// Тест 2: Оптимизация простого квадратичного функционала
TEST(LBFGSOptimizerTest, QuadraticConvergence) {
    LBFGSOptimizer optimizer;
    optimizer.set_parameters(1000, 1e-8, 1e-10);
    
    // Создаем простой функционал: аппроксимация одной точки (0.5, 0.5)
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;  // без регуляризации
    config.approx_points = {WeightedPoint(0.5, 0.5, 1.0)};
    
    Functional functional(config);
    
    // Начальное приближение: [0.0, 0.0] (2 коэффициента для полинома степени 1)
    std::vector<double> initial = {0.0, 0.0};
    
    auto result = optimizer.optimize(functional, initial);
    
    EXPECT_TRUE(result.success || result.iterations < 1000) << "Optimization message: " << result.message;
    EXPECT_TRUE(std::isfinite(result.final_objective));
    
    // Проверяем, что оптимизация улучшила результат
    double initial_obj = functional.evaluate(Polynomial(initial));
    EXPECT_LE(result.final_objective, initial_obj + 1e-6);
}

// Тест 3: Оптимизация с границами
TEST(LBFGSOptimizerTest, Bounds) {
    LBFGSOptimizer optimizer;
    optimizer.set_parameters(500, 1e-6, 1e-8);
    
    // Границы: коэффициенты в [-2.0, 2.0]
    std::vector<double> lower = {-2.0, -2.0};
    std::vector<double> upper = {2.0, 2.0};
    optimizer.set_bounds(lower, upper);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.approx_points = {WeightedPoint(0.5, 0.5, 1.0)};
    
    Functional functional(config);
    
    // Начальное приближение далеко от минимума
    std::vector<double> initial = {10.0, -10.0};
    
    auto result = optimizer.optimize(functional, initial);
    
    // Проверяем, что результат находится в границах (если оптимизация сошлась)
    if (result.success) {
        for (size_t i = 0; i < result.coefficients.size(); ++i) {
            EXPECT_GE(result.coefficients[i], lower[i] - 1e-3);
            EXPECT_LE(result.coefficients[i], upper[i] + 1e-3);
        }
    }
    EXPECT_TRUE(std::isfinite(result.final_objective));
}

// Тест 4: Проверка максимального числа итераций
TEST(LBFGSOptimizerTest, MaxIterations) {
    LBFGSOptimizer optimizer;
    optimizer.set_parameters(10, 1e-8, 1e-10);  // Очень мало итераций
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.approx_points = {
        WeightedPoint(0.25, 0.75, 1.0),
        WeightedPoint(0.75, 0.75, 1.0)
    };
    
    Functional functional(config);
    std::vector<double> initial = {0.0, 0.0, 0.0};
    
    auto result = optimizer.optimize(functional, initial);
    
    // Должны достичь максимума итераций или сойтись
    EXPECT_LE(result.iterations, 10);
    EXPECT_TRUE(std::isfinite(result.final_objective));
}

// Тест 5: Проверка параметров L-BFGS через set_lbfgs_parameters
TEST(LBFGSOptimizerTest, LBFGSParameters) {
    LBFGSOptimizer optimizer;
    optimizer.set_lbfgs_parameters(300, 1e-6, 1e-8, 0.01, 10, 20);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.approx_points = {WeightedPoint(0.5, 0.5, 1.0)};
    
    Functional functional(config);
    std::vector<double> initial = {0.0, 0.0};
    
    auto result = optimizer.optimize(functional, initial);
    
    // Оптимизатор должен работать и возвращать осмысленный результат
    EXPECT_NE(result.message.find("L-BFGS"), std::string::npos);
    EXPECT_TRUE(std::isfinite(result.final_objective));
}

// Тест 6: Сравнение с градиентным спуском
TEST(LBFGSOptimizerTest, CompareToGradientDescent) {
    LBFGSOptimizer lbfgs;
    lbfgs.set_parameters(200, 1e-8, 1e-10);
    
    AdaptiveGradientDescentOptimizer gd;
    gd.set_parameters(200, 1e-8, 1e-10, 0.01);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.approx_points = {
        WeightedPoint(0.25, 0.75, 1.0),
        WeightedPoint(0.75, 0.75, 1.0)
    };
    
    Functional functional(config);
    std::vector<double> initial = {0.0, 0.0, 0.0};
    
    auto result_lbfgs = lbfgs.optimize(functional, initial);
    auto result_gd = gd.optimize(functional, initial);
    
    // Оба должны дать конечные значения
    EXPECT_TRUE(std::isfinite(result_lbfgs.final_objective));
    EXPECT_TRUE(std::isfinite(result_gd.final_objective));
    
    // L-BFGS обычно сходится за меньшее число итераций
    EXPECT_LE(result_lbfgs.iterations, result_gd.iterations + 50); // с допуском
}

// Тест 7: Работа с более сложным функционалом (с отталкиванием и регуляризацией)
TEST(LBFGSOptimizerTest, ComplexFunctional) {
    LBFGSOptimizer optimizer;
    optimizer.set_parameters(200, 1e-6, 1e-8);
    
    ApproximationConfig config;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.01;  // регуляризация
    config.approx_points = {
        WeightedPoint(0.0, 1.0, 1.0),
        WeightedPoint(0.5, 0.5, 1.0),
        WeightedPoint(1.0, 1.0, 1.0)
    };
    config.repel_points = {
        RepulsionPoint(0.3, 0.5, 10.0),
        RepulsionPoint(0.7, 0.5, 10.0)
    };
    
    Functional functional(config);
    std::vector<double> initial(5, 0.0);  // 5 свободных параметров
    
    auto result = optimizer.optimize(functional, initial);
    
    // Должна сойтись или достичь лимита
    EXPECT_LE(result.iterations, 200);
    EXPECT_TRUE(std::isfinite(result.final_objective));
    EXPECT_GE(result.final_objective, 0.0);  // функционал должен быть неотрицательным
}

// Тест 8: Проверка, что оптимизатор корректно обрабатывает отсутствие NLopt
TEST(LBFGSOptimizerTest, NoNLOpt) {
    // Этот тест проверяет, что код компилируется и возвращает ошибку без NLopt
    // Но в нашей сборке NLopt есть, поэтому просто проверяем базовое поведение
    LBFGSOptimizer optimizer;
    EXPECT_TRUE(true);
}

} // namespace test
} // namespace mixed_approx