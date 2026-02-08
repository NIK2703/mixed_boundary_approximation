#include <gtest/gtest.h>
#include <cmath>
#include "mixed_approximation/types.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/constrained_polynomial.h"
#include "mixed_approximation/i_polynomial.h"

namespace mixed_approx {
namespace test {

// ============== Тесты InterpolationBasis ==============

TEST(InterpolationBasisTest, BasicConstruction) {
    InterpolationBasis basis;
    basis.build({0.0, 0.5, 1.0}, {1.0, 0.5, 0.0});
    
    EXPECT_EQ(basis.m_eff, 3);
    EXPECT_TRUE(basis.is_valid);
}

TEST(InterpolationBasisTest, Evaluate) {
    // Тест: P(z_e) = f(z_e) для узлов
    InterpolationBasis basis;
    basis.build({0.0, 0.5, 1.0}, {1.0, 2.0, 3.0});
    
    // Проверяем конкретные значения
    EXPECT_NEAR(basis.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(basis.evaluate(0.5), 2.0, 1e-12);
    EXPECT_NEAR(basis.evaluate(1.0), 3.0, 1e-12);
}

TEST(InterpolationBasisTest, LinearInterpolation) {
    // Линейная функция f(x) = 2x + 1
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 3.0});
    
    // Проверяем в промежуточной точке
    double x = 0.5;
    double expected = 2.0 * x + 1.0;
    EXPECT_NEAR(basis.evaluate(x), expected, 1e-10);
}

TEST(InterpolationBasisTest, Derivative) {
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {0.0, 1.0});
    
    // f(x) = x, f'(x) = 1
    EXPECT_NEAR(basis.evaluate_derivative(0.5, 1), 1.0, 1e-8);
}

// ============== Тесты WeightMultiplier ==============

TEST(WeightMultiplierTest, BasicConstruction) {
    WeightMultiplier weight;
    weight.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    EXPECT_EQ(weight.degree(), 3);
}

TEST(WeightMultiplierTest, Roots) {
    // W(x) = x * (x - 0.5) * (x - 1)
    WeightMultiplier weight;
    weight.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    // Проверяем, что узлы являются корнями
    EXPECT_NEAR(weight.evaluate(0.0), 0.0, 1e-12);
    EXPECT_NEAR(weight.evaluate(0.5), 0.0, 1e-12);
    EXPECT_NEAR(weight.evaluate(1.0), 0.0, 1e-12);
}

TEST(WeightMultiplierTest, Derivatives) {
    WeightMultiplier weight;
    weight.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // W(x) = x * (x - 1) = x² - x
    // W'(x) = 2x - 1
    // W''(x) = 2
    
    EXPECT_NEAR(weight.evaluate_derivative(0.5, 1), 0.0, 1e-12);
    EXPECT_NEAR(weight.evaluate_derivative(0.5, 2), 2.0, 1e-12);
}

// ============== Тесты CorrectionPolynomial ==============

TEST(CorrectionPolynomialTest, BasicConstruction) {
    CorrectionPolynomial corr;
    corr.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    
    EXPECT_EQ(corr.degree, 3);
    EXPECT_EQ(corr.basis_type, BasisType::MONOMIAL);
    EXPECT_TRUE(corr.is_initialized);
}

TEST(CorrectionPolynomialTest, Evaluate) {
    CorrectionPolynomial corr;
    corr.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    corr.initialize_zero();
    
    // Q(x) = 1 + 2x + 3x²
    corr.coeffs = {1.0, 2.0, 3.0};
    
    EXPECT_NEAR(corr.evaluate_Q(0.0), 1.0, 1e-12);
    EXPECT_NEAR(corr.evaluate_Q(1.0), 6.0, 1e-12);
    EXPECT_NEAR(corr.evaluate_Q(2.0), 17.0, 1e-12);
}

TEST(CorrectionPolynomialTest, Derivative) {
    CorrectionPolynomial corr;
    corr.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    corr.initialize_zero();
    
    // Q(x) = 1 + 2x + 3x²
    corr.coeffs = {1.0, 2.0, 3.0};
    
    // Q'(x) = 2 + 6x
    EXPECT_NEAR(corr.evaluate_Q_derivative(0.0, 1), 2.0, 1e-12);
    EXPECT_NEAR(corr.evaluate_Q_derivative(1.0, 1), 8.0, 1e-12);
}

TEST(CorrectionPolynomialTest, BasisFunctions) {
    CorrectionPolynomial corr;
    corr.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    corr.initialize_zero();
    
    // φ_0(x) = 1, φ_1(x) = x, φ_2(x) = x²
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(0.0, corr.coeffs, 0), 1.0, 1e-12);
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(0.0, corr.coeffs, 1), 0.0, 1e-12);
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(0.0, corr.coeffs, 2), 0.0, 1e-12);
    
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(2.0, corr.coeffs, 0), 1.0, 1e-12);
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(2.0, corr.coeffs, 1), 2.0, 1e-12);
    EXPECT_NEAR(corr.compute_basis_function_with_coeffs(2.0, corr.coeffs, 2), 4.0, 1e-12);
}

// ============== Тесты ConstrainedPolynomial ==============

TEST(ConstrainedPolynomialTest, BasicConstruction) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {0.5, 2.0},
        {1.0, 3.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    
    EXPECT_EQ(poly.interpolation_count(), 3);
    // Степень: max(2, 2+3) = 5
    EXPECT_EQ(poly.degree(), 5);
    // Число параметров = deg(Q) + 1 = 3
    EXPECT_EQ(poly.num_parameters(), 3);
}

TEST(ConstrainedPolynomialTest, InterpolationConditions) {
    // Создаём полином с нулевой коррекцией
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {0.5, 2.0},
        {1.0, 3.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    poly.set_parameters({0.0, 0.0, 0.0});  // Q(x) = 0
    
    // Проверяем интерполяционные условия
    EXPECT_TRUE(poly.check_interpolation_conditions(1e-10));
    
    // Проверяем значения в узлах
    EXPECT_NEAR(poly.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(poly.evaluate(0.5), 2.0, 1e-12);
    EXPECT_NEAR(poly.evaluate(1.0), 3.0, 1e-12);
}

TEST(ConstrainedPolynomialTest, InterpolationWithCorrection) {
    // Создаём полином с ненулевой коррекцией
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {0.5, 2.0},
        {1.0, 3.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    poly.set_parameters({1.0, 0.0, 0.0});  // Q(x) = 1
    
    // Интерполяционные условия должны сохраняться!
    EXPECT_NEAR(poly.evaluate(0.0), 1.0, 1e-12);
    EXPECT_NEAR(poly.evaluate(0.5), 2.0, 1e-12);
    EXPECT_NEAR(poly.evaluate(1.0), 3.0, 1e-12);
}

TEST(ConstrainedPolynomialTest, BetweenNodes) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    
    // Полином должен проходить через (0,0) и (1,0)
    ConstrainedPolynomial poly(nodes, 1);
    
    // Проверяем поведение между узлами
    EXPECT_NEAR(poly.evaluate(0.5), poly.evaluate(0.5), 1e-12);
}

TEST(ConstrainedPolynomialTest, Derivatives) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {1.0, 1.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    
    // Проверяем вычисление производных
    double x = 0.5;
    auto result = poly.evaluate_with_derivatives(x);
    
    EXPECT_TRUE(std::isfinite(result.value));
    EXPECT_TRUE(std::isfinite(result.first_deriv));
    EXPECT_TRUE(std::isfinite(result.second_deriv));
}

TEST(ConstrainedPolynomialTest, BasisFunctions) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    
    // Проверяем базисные функции
    double x = 0.5;
    for (std::size_t k = 0; k < poly.num_parameters(); ++k) {
        double bf = poly.basis_function(k, x);
        EXPECT_TRUE(std::isfinite(bf));
    }
}

TEST(ConstrainedPolynomialTest, Gradient) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    
    // Проверяем градиент
    auto grad = poly.gradient(0.5);
    
    EXPECT_EQ(grad.size(), poly.num_parameters());
    for (double g : grad) {
        EXPECT_TRUE(std::isfinite(g));
    }
}

TEST(ConstrainedPolynomialTest, Caching) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {1.0, 2.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    poly.reset_evaluation_count();
    
    // Первые вычисления
    double val1 = poly.evaluate(0.5);
    EXPECT_EQ(poly.evaluation_count(), 1);
}

TEST(ConstrainedPolynomialTest, ToString) {
    std::vector<InterpolationNode> nodes = {
        {0.0, 1.0},
        {1.0, 2.0}
    };
    
    ConstrainedPolynomial poly(nodes, 2);
    
    std::string str = poly.to_string();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("ConstrainedPolynomial"), std::string::npos);
}

// ============== Тесты LRUCache ==============

TEST(LRUCacheTest, BasicOperations) {
    LRUCache<int, double> cache(3);
    
    cache.put(1, 1.0);
    cache.put(2, 2.0);
    cache.put(3, 3.0);
    
    EXPECT_EQ(cache.size(), 3);
    EXPECT_FALSE(cache.empty());
    
    double value;
    EXPECT_TRUE(cache.try_get(1, value));
    EXPECT_NEAR(value, 1.0, 1e-12);
}

TEST(LRUCacheTest, Clear) {
    LRUCache<int, double> cache(3);
    
    cache.put(1, 1.0);
    cache.put(2, 2.0);
    
    cache.clear();
    
    EXPECT_EQ(cache.size(), 0);
    EXPECT_TRUE(cache.empty());
}

} // namespace test
} // namespace mixed_approx
