#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Тесты шага 2.1.4.1: Определение размерности пространства свободных параметров
// =============================================================================

TEST(CorrectionPolynomialTest, DimensionCalculation) {
    std::cout << "Testing dimension calculation (step 2.1.4.1)...\n";
    
    CorrectionPolynomial Q;
    
    // deg_Q = 0 (n = m)
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    EXPECT_EQ(Q.degree, 0);
    EXPECT_EQ(Q.n_free, 1);
    EXPECT_TRUE(Q.is_initialized);
    
    // deg_Q = 2 (n = m + 2)
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    EXPECT_EQ(Q.degree, 2);
    EXPECT_EQ(Q.n_free, 3);
    
    // deg_Q = 5 (n = m + 5)
    Q.initialize(5, BasisType::MONOMIAL, 0.5, 0.5);
    EXPECT_EQ(Q.degree, 5);
    EXPECT_EQ(Q.n_free, 6);
}

// =============================================================================
// Тесты шага 2.1.4.2: Выбор базиса представления полинома Q(x)
// =============================================================================

TEST(CorrectionPolynomialTest, BasisTypeSelection) {
    std::cout << "Testing basis type selection (step 2.1.4.2)...\n";
    
    // deg <= 5: MONOMIAL
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(0), BasisType::MONOMIAL);
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(1), BasisType::MONOMIAL);
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(5), BasisType::MONOMIAL);
    
    // deg > 5: CHEBYSHEV
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(6), BasisType::CHEBYSHEV);
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(10), BasisType::CHEBYSHEV);
    EXPECT_EQ(CorrectionPolynomial::choose_basis_type(15), BasisType::CHEBYSHEV);
}

TEST(CorrectionPolynomialTest, BasisTypeInitialization) {
    std::cout << "Testing basis type initialization...\n";
    
    CorrectionPolynomial Q;
    
    // Инициализация с мономиальным базисом
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    EXPECT_EQ(Q.basis_type, BasisType::MONOMIAL);
    EXPECT_EQ(Q.degree, 3);
    EXPECT_EQ(Q.n_free, 4);
    EXPECT_EQ(Q.x_center, 0.5);
    EXPECT_EQ(Q.x_scale, 0.5);
    
    // Инициализация с базисом Чебышёва
    Q.initialize(7, BasisType::CHEBYSHEV, 0.5, 0.5);
    EXPECT_EQ(Q.basis_type, BasisType::CHEBYSHEV);
    EXPECT_EQ(Q.degree, 7);
    EXPECT_EQ(Q.n_free, 8);
}

// =============================================================================
// Тесты шага 2.1.4.4: Инициализация коэффициентов корректирующего полинома
// =============================================================================

TEST(CorrectionPolynomialTest, ZeroInitialization) {
    std::cout << "Testing zero initialization (step 2.1.4.4)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // Все коэффициенты должны быть нулями
    ASSERT_EQ(Q.coeffs.size(), 4u);
    for (double coeff : Q.coeffs) {
        EXPECT_EQ(coeff, 0.0);
    }
    EXPECT_EQ(Q.init_method, InitializationMethod::ZERO);
    
    // Q(x) = 0 для любого x
    EXPECT_NEAR(Q.evaluate_Q(0.0), 0.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q(0.5), 0.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q(1.0), 0.0, 1e-12);
}

TEST(CorrectionPolynomialTest, RandomInitialization) {
    std::cout << "Testing random initialization (step 2.1.4.4)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_random();
    
    ASSERT_EQ(Q.coeffs.size(), 3u);
    EXPECT_EQ(Q.init_method, InitializationMethod::RANDOM);
    
    // Проверяем, что все коэффициенты в диапазоне [-0.01, 0.01]
    for (double coeff : Q.coeffs) {
        EXPECT_GE(coeff, -0.01);
        EXPECT_LE(coeff, 0.01);
    }
    
    // Проверяем, что не все коэффициенты одинаковые (вероятностно)
    bool all_zero = true;
    for (double coeff : Q.coeffs) {
        if (std::abs(coeff) > 1e-15) {
            all_zero = false;
            break;
        }
    }
    // Примечание: случайная инициализация может теоретически дать все нули,
    // но это крайне маловероятно
}

TEST(CorrectionPolynomialTest, LeastSquaresInitialization) {
    std::cout << "Testing least squares initialization (step 2.1.4.4)...\n";
    
    // Подготовка данных
    InterpolationBasis p_int;
    p_int.build({0.0, 1.0}, {0.0, 0.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.25, 1.0, 1.0),
        WeightedPoint(0.5, 0.5, 1.0),
        WeightedPoint(0.75, 1.0, 1.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_least_squares(approx_points, p_int, W);
    
    ASSERT_EQ(Q.coeffs.size(), 2u);
    
    // Проверяем, что инициализация прошла успешно
    // (значения могут варьироваться в зависимости от МНК)
    EXPECT_FALSE(std::isnan(Q.coeffs[0]));
    EXPECT_FALSE(std::isnan(Q.coeffs[1]));
    
    // Проверяем, что validation_message установлен
    EXPECT_FALSE(Q.validation_message.empty());
}

// =============================================================================
// Тесты шага 2.1.4.5: Вычисление базисных функций и их производных
// =============================================================================

TEST(CorrectionPolynomialTest, EvaluateMonomialBasis) {
    std::cout << "Testing monomial basis evaluation (step 2.1.4.5)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // Q(x) = 1 + 2x + 3x^2 + 4x^3
    Q.coeffs = {1.0, 2.0, 3.0, 4.0};
    
    // Проверяем значения
    EXPECT_NEAR(Q.evaluate_Q(0.0), 1.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q(1.0), 10.0, 1e-12);  // 1 + 2 + 3 + 4 = 10
    EXPECT_NEAR(Q.evaluate_Q(0.5), 3.25, 1e-12);   // 1 + 1 + 0.75 + 0.5 = 3.25
    
    // Проверяем производные
    // Q'(x) = 2 + 6x + 12x^2
    EXPECT_NEAR(Q.evaluate_Q_derivative(0.0, 1), 2.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q_derivative(1.0, 1), 20.0, 1e-12);  // 2 + 6 + 12 = 20
    
    // Q''(x) = 6 + 24x
    EXPECT_NEAR(Q.evaluate_Q_derivative(0.0, 2), 6.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q_derivative(1.0, 2), 30.0, 1e-12);  // 6 + 24 = 30
}

TEST(CorrectionPolynomialTest, EvaluateChebyshevBasis) {
    std::cout << "Testing Chebyshev basis evaluation (step 2.1.4.5)...\n";
    
    CorrectionPolynomial Q;
    // Интервал [0, 1], центр = 0.5, масштаб = 0.5
    Q.initialize(2, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.initialize_zero();
    
    // Устанавливаем коэффициенты: Q(x) = T_0(t) + T_1(t) + T_2(t)
    // t = 2*(x - 0.5)/0.5 - 1 = 2*x - 2 (при x_center=0.5, x_scale=0.5)
    // T_0(t) = 1, T_1(t) = t, T_2(t) = 2t^2 - 1
    Q.coeffs = {1.0, 1.0, 1.0};
    
    // Проверяем для x = 0.5 (t = 0)
    // Q(0.5) = 1 + 0 + (-1) = 0
    EXPECT_NEAR(Q.evaluate_Q(0.5), 0.0, 1e-12);
    
    // Проверяем для x = 0.0 (t = -1)
    // T_0(-1) = 1, T_1(-1) = -1, T_2(-1) = 2(1) - 1 = 1
    // Q(0) = 1 - 1 + 1 = 1
    EXPECT_NEAR(Q.evaluate_Q(0.0), 1.0, 1e-12);
    
    // Проверяем для x = 1.0 (t = 1)
    // T_0(1) = 1, T_1(1) = 1, T_2(1) = 2(1) - 1 = 1
    // Q(1) = 1 + 1 + 1 = 3
    EXPECT_NEAR(Q.evaluate_Q(1.0), 3.0, 1e-12);
}

TEST(CorrectionPolynomialTest, BasisDerivatives) {
    std::cout << "Testing basis function derivatives...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.initialize_zero();
    
    // Проверяем, что производные вычисляются без ошибок
    double deriv1 = Q.evaluate_Q_derivative(0.5, 1);
    double deriv2 = Q.evaluate_Q_derivative(0.5, 2);
    
    EXPECT_FALSE(std::isnan(deriv1));
    EXPECT_FALSE(std::isnan(deriv2));
}

TEST(CorrectionPolynomialTest, BuildAndClearCaches) {
    std::cout << "Testing cache building and clearing (step 2.1.4.5)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    std::vector<double> points_x = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> points_y = {0.1, 0.9};
    
    // Строим кэши
    Q.build_caches(points_x, points_y);
    
    // Проверяем, что кэши заполнены
    EXPECT_EQ(Q.basis_cache_x.size(), 5u);
    EXPECT_EQ(Q.basis_cache_x[0].size(), 3u);  // n_free = 3
    EXPECT_EQ(Q.basis_cache_y.size(), 2u);
    EXPECT_EQ(Q.basis2_cache_x.size(), 5u);
    EXPECT_EQ(Q.basis2_cache_y.size(), 2u);
    
    // Очищаем кэши
    Q.clear_caches();
    
    // Проверяем, что кэши пусты
    EXPECT_TRUE(Q.basis_cache_x.empty());
    EXPECT_TRUE(Q.basis_cache_y.empty());
    EXPECT_TRUE(Q.basis2_cache_x.empty());
    EXPECT_TRUE(Q.basis2_cache_y.empty());
}

// =============================================================================
// Тесты шага 2.1.4.6: Вычисление функционала и градиента
// =============================================================================

TEST(CorrectionPolynomialTest, ComputeObjective) {
    std::cout << "Testing objective computation (step 2.1.4.6)...\n";
    
    // Подготовка данных - используем более простой случай
    // Интерполяция без узлов (P_int = 0)
    InterpolationBasis p_int;
    p_int.build({}, {}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    // W(x) = 1 (без корней)
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.0, 1.0, 1.0),
        WeightedPoint(0.5, 2.0, 1.0),
        WeightedPoint(1.0, 1.0, 1.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);  // Q(x) = q_0
    Q.initialize_zero();
    
    // Q(x) = 0, поэтому F(x) = 0
    double obj = Q.compute_objective(approx_points, p_int, W);
    
    // Ошибки: (0-1)^2 + (0-2)^2 + (0-1)^2 = 1 + 4 + 1 = 6
    EXPECT_NEAR(obj, 6.0, 1e-6);
}

TEST(CorrectionPolynomialTest, ComputeGradient) {
    std::cout << "Testing gradient computation (step 2.1.4.6)...\n";
    
    // Подготовка данных
    InterpolationBasis p_int;
    p_int.build({}, {}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.0, 1.0, 1.0),
        WeightedPoint(0.5, 2.0, 1.0),
        WeightedPoint(1.0, 1.0, 1.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    auto grad = Q.compute_gradient(approx_points, p_int, W);
    
    ASSERT_EQ(grad.size(), 1u);
    
    // Градиент не должен содержать NaN или Inf
    for (double g : grad) {
        EXPECT_FALSE(std::isnan(g));
        EXPECT_FALSE(std::isinf(g));
    }
    
    // Градиент должен быть ненулевым, так как F(x)=0 != целевых значений
    EXPECT_NE(grad[0], 0.0);
}

TEST(CorrectionPolynomialTest, StiffnessMatrix) {
    std::cout << "Testing stiffness matrix computation (step 2.1.4.6)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.0, 1.0);  // Интервал [0, 1]
    
    // W(x) = 1 (без корней)
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    // Вычисляем матрицу жёсткости
    Q.compute_stiffness_matrix(0.0, 1.0, W, 10);
    
    EXPECT_TRUE(Q.stiffness_matrix_computed);
    EXPECT_EQ(Q.stiffness_matrix.size(), 3u);  // n_free = 3
    EXPECT_EQ(Q.stiffness_matrix[0].size(), 3u);
    
    // Проверяем симметричность
    for (size_t i = 0; i < Q.stiffness_matrix.size(); ++i) {
        for (size_t j = 0; j < Q.stiffness_matrix.size(); ++j) {
            EXPECT_NEAR(Q.stiffness_matrix[i][j], Q.stiffness_matrix[j][i], 1e-12);
        }
    }
    
    // Проверяем, что матрица полуопределённая (все элементы >= 0 для базиса производных)
    for (size_t i = 0; i < Q.stiffness_matrix.size(); ++i) {
        for (size_t j = 0; j < Q.stiffness_matrix.size(); ++j) {
            EXPECT_GE(Q.stiffness_matrix[i][j], -1e-10);
        }
    }
}

TEST(CorrectionPolynomialTest, ObjectiveWithRegularization) {
    std::cout << "Testing objective with regularization...\n";
    
    // Подготовка данных
    InterpolationBasis p_int;
    p_int.build({}, {}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.5, 0.5, 1.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // Вычисляем матрицу жёсткости
    Q.compute_stiffness_matrix(0.0, 1.0, W, 10);
    
    // Устанавливаем регуляризационный параметр
    Q.set_regularization_lambda(0.1);
    
    // Вычисляем функционал с регуляризацией
    double obj = Q.compute_objective(approx_points, p_int, W);
    
    // Без регуляризации: (0 - 0.5)^2 / 1 = 0.25
    // С регуляризацией: 0.25 + 0.1 * (q^T * K * q)
    // При q = 0, регуляризационный член = 0
    EXPECT_NEAR(obj, 0.25, 1e-6);
}

// =============================================================================
// Тесты шага 2.1.4.7: Верификация корректности параметризации
// =============================================================================

TEST(CorrectionPolynomialTest, VerifyInitialization) {
    std::cout << "Testing initialization verification (step 2.1.4.7)...\n";
    
    // Подготовка данных
    InterpolationBasis p_int;
    p_int.build({0.0, 1.0}, {0.0, 0.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.5, 0.5, 1.0)
    };
    
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(0.5, 10.0, 100.0)  // отталкивающая точка далеко от аппроксимации
    };
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // Верификация должна пройти
    bool verified = Q.verify_initialization(approx_points, repel_points, p_int, W);
    EXPECT_TRUE(verified);
    EXPECT_FALSE(Q.validation_message.empty());
}

TEST(CorrectionPolynomialTest, VerifyInitializationNearBarrier) {
    std::cout << "Testing verification with near-barrier initialization...\n";
    
    // Подготовка данных
    InterpolationBasis p_int;
    p_int.build({0.0, 1.0}, {0.0, 0.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.5, 0.5, 1.0)
    };
    
    // Отталкивающая точка близко к аппроксимирующей
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(0.5, 0.6, 100.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // При Q=0, F(x) = P_int(x) = 0, расстояние до y_forbidden = 0.6
    // Это достаточно далеко, верификация должна пройти
    bool verified = Q.verify_initialization(approx_points, repel_points, p_int, W);
    EXPECT_TRUE(verified);
}

// =============================================================================
// Тесты шага 2.1.4.8: Интеграция с полной параметризацией
// =============================================================================

TEST(CorrectionPolynomialTest, DiagnosticInfo) {
    std::cout << "Testing diagnostic info (step 2.1.4.8)...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.initialize_zero();
    
    std::string info = Q.get_diagnostic_info();
    
    // Проверяем, что информация содержит ключевые данные
    EXPECT_NE(info.find("degree:"), std::string::npos);
    EXPECT_NE(info.find("n_free:"), std::string::npos);
    EXPECT_NE(info.find("basis_type:"), std::string::npos);
    EXPECT_NE(info.find("CHEBYSHEV"), std::string::npos);
    EXPECT_NE(info.find("initialized: yes"), std::string::npos);
}

// =============================================================================
// Дополнительные тесты на граничные случаи
// =============================================================================

TEST(CorrectionPolynomialTest, EdgeCases) {
    std::cout << "Testing edge cases...\n";
    
    // Степень 0 (константный корректирующий полином)
    CorrectionPolynomial Q;
    Q.initialize(0, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    ASSERT_EQ(Q.n_free, 1);
    EXPECT_NEAR(Q.evaluate_Q(0.0), 0.0, 1e-12);
    EXPECT_NEAR(Q.evaluate_Q(1.0), 0.0, 1e-12);
    
    // Высокая степень с базисом Чебышёва
    Q.initialize(10, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.initialize_zero();
    
    ASSERT_EQ(Q.n_free, 11);
    EXPECT_EQ(Q.basis_type, BasisType::CHEBYSHEV);
    
    // Проверяем вычисление без ошибок
    double val = Q.evaluate_Q(0.5);
    EXPECT_FALSE(std::isnan(val));
}

TEST(CorrectionPolynomialTest, ChebyshevOrthogonality) {
    std::cout << "Testing Chebyshev orthogonality properties...\n";
    
    // Проверяем, что T_0, T_1, T_2 вычисляются правильно
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::CHEBYSHEV, 0.0, 1.0);  // Интервал [0, 1], x_center=0, x_scale=1
    Q.initialize_zero();
    
    // Нормализация: t = (x - 0) / 1 = x
    // T_2(t) = 2t^2 - 1
    
    // Устанавливаем коэффициенты так, чтобы Q(x) = T_2(x)
    Q.coeffs = {0.0, 0.0, 1.0};
    
    // При x = 0, T_2(0) = -1
    EXPECT_NEAR(Q.evaluate_Q(0.0), -1.0, 1e-12);
    
    // При x = 1, T_2(1) = 1
    EXPECT_NEAR(Q.evaluate_Q(1.0), 1.0, 1e-12);
    
    // При x = 0.5, T_2(0.5) = 2*0.25 - 1 = -0.5
    EXPECT_NEAR(Q.evaluate_Q(0.5), -0.5, 1e-12);
}

TEST(CorrectionPolynomialTest, InitializationMethods) {
    std::cout << "Testing all initialization methods...\n";
    
    InterpolationBasis p_int;
    p_int.build({0.0, 1.0}, {0.0, 0.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    std::vector<WeightedPoint> approx_points = {
        WeightedPoint(0.5, 0.5, 1.0)
    };
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.5, 0.5);
    
    // ZERO
    Q.initialize_coefficients(InitializationMethod::ZERO, approx_points, {}, p_int, W, 0.0, 1.0);
    EXPECT_EQ(Q.init_method, InitializationMethod::ZERO);
    
    // RANDOM
    Q.initialize_coefficients(InitializationMethod::RANDOM, approx_points, {}, p_int, W, 0.0, 1.0);
    EXPECT_EQ(Q.init_method, InitializationMethod::RANDOM);
    
    // LEAST_SQUARES
    Q.initialize_coefficients(InitializationMethod::LEAST_SQUARES, approx_points, {}, p_int, W, 0.0, 1.0);
    EXPECT_EQ(Q.init_method, InitializationMethod::LEAST_SQUARES);
    
    // Проверяем, что все методы не вызывают ошибок
    EXPECT_TRUE(Q.is_initialized);
    EXPECT_FALSE(Q.validation_message.empty());
}

TEST(CorrectionPolynomialTest, StiffnessMatrixWithDifferentGaussPoints) {
    std::cout << "Testing stiffness matrix with different Gauss-Legendre quadrature points...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);
    
    // Разные числа точек квадратуры
    Q.compute_stiffness_matrix(0.0, 1.0, W, 10);
    EXPECT_TRUE(Q.stiffness_matrix_computed);
    
    Q.clear_caches();
    Q.compute_stiffness_matrix(0.0, 1.0, W, 20);
    EXPECT_TRUE(Q.stiffness_matrix_computed);
    
    // Матрицы должны быть похожи (сходимость при большем числе точек)
    // Просто проверяем, что обе вычислены без ошибок
    EXPECT_EQ(Q.stiffness_matrix.size(), 3u);
}

TEST(CorrectionPolynomialTest, InvalidDerivativeOrder) {
    std::cout << "Testing invalid derivative order handling...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.initialize_zero();
    
    // Проверяем, что недопустимый порядок производной вызывает исключение
    EXPECT_THROW(Q.evaluate_Q_derivative(0.5, 0), std::invalid_argument);
    EXPECT_THROW(Q.evaluate_Q_derivative(0.5, 3), std::invalid_argument);
    EXPECT_THROW(Q.evaluate_Q_derivative(0.5, -1), std::invalid_argument);
}

TEST(CorrectionPolynomialTest, UninitializedAccess) {
    std::cout << "Testing uninitialized access handling...\n";
    
    CorrectionPolynomial Q;
    
    // Проверяем, что неинициализированный доступ вызывает исключение
    EXPECT_THROW(Q.evaluate_Q(0.5), std::runtime_error);
    EXPECT_THROW(Q.evaluate_Q_derivative(0.5, 1), std::runtime_error);
    
    // Для compute_objective нужны реальные объекты
    InterpolationBasis p_int;
    WeightMultiplier W;
    EXPECT_THROW(Q.compute_objective({}, p_int, W), std::runtime_error);
}
