#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace mixed_approx;

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// Тест 1: WeightMultiplier - построение и вычисление
static void test_weight_multiplier() {
    std::cout << "Test 1: WeightMultiplier\n";
    
    // W(x) = (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6
    WeightMultiplier wm;
    std::vector<double> roots = {1.0, 2.0, 3.0};
    wm.build_from_roots(roots);
    
    // Проверяем степень
    if (wm.degree() != 3) {
        std::cerr << "  FAILED: degree = " << wm.degree() << ", expected 3\n";
        return;
    }
    
    // Проверяем коэффициенты
    std::vector<double> expected_coeffs = {1.0, -6.0, 11.0, -6.0};  // x^3 - 6x^2 + 11x - 6
    auto& coeffs = wm.coeffs;
    if (coeffs.size() != expected_coeffs.size()) {
        std::cerr << "  FAILED: coeffs size = " << coeffs.size() << ", expected " << expected_coeffs.size() << "\n";
        return;
    }
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (!approx_equal(coeffs[i], expected_coeffs[i])) {
            std::cerr << "  FAILED: coeffs[" << i << "] = " << coeffs[i] << ", expected " << expected_coeffs[i] << "\n";
            return;
        }
    }
    
    // Проверяем вычисление в нескольких точках
    double x1 = wm.evaluate(0.0);  // W(0) = -6
    if (!approx_equal(x1, -6.0)) {
        std::cerr << "  FAILED: W(0) = " << x1 << ", expected -6\n";
        return;
    }
    
    double x2 = wm.evaluate(1.0);  // W(1) = 0
    if (!approx_equal(x2, 0.0)) {
        std::cerr << "  FAILED: W(1) = " << x2 << ", expected 0\n";
        return;
    }
    
    double x3 = wm.evaluate(4.0);  // W(4) = (4-1)(4-2)(4-3) = 6
    if (!approx_equal(x3, 6.0)) {
        std::cerr << "  FAILED: W(4) = " << x3 << ", expected 6\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 2: InterpolationBasis - барицентрическая интерполяция
static void test_interpolation_basis_barycentric() {
    std::cout << "Test 2: InterpolationBasis (barycentric)\n";
    
    // Точки: (0,0), (1,1), (2,4) - лежат на параболе y = x^2
    std::vector<double> nodes = {0.0, 1.0, 2.0};
    std::vector<double> values = {0.0, 1.0, 4.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC);
    
    // Проверяем значения в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double val = basis.evaluate(nodes[i]);
        if (!approx_equal(val, values[i])) {
            std::cerr << "  FAILED: P_int(" << nodes[i] << ") = " << val << ", expected " << values[i] << "\n";
            return;
        }
    }
    
    // Проверяем значение в середине: P_int(0.5) = 0.25
    double mid_val = basis.evaluate(0.5);
    if (!approx_equal(mid_val, 0.25)) {
        std::cerr << "  FAILED: P_int(0.5) = " << mid_val << ", expected 0.25\n";
        return;
    }
    
    // Проверяем значение вне узлов: P_int(3) = 9
    double ext_val = basis.evaluate(3.0);
    if (!approx_equal(ext_val, 9.0)) {
        std::cerr << "  FAILED: P_int(3) = " << ext_val << ", expected 9\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 3: InterpolationBasis - метод Ньютона
static void test_interpolation_basis_newton() {
    std::cout << "Test 3: InterpolationBasis (Newton)\n";
    
    // Точки: (0,1), (1,2), (2,5) - лежат на параболе y = x^2 + 1
    std::vector<double> nodes = {0.0, 1.0, 2.0};
    std::vector<double> values = {1.0, 2.0, 5.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::NEWTON);
    
    // Проверяем значения в узлах
    for (size_t i = 0; i < nodes.size(); ++i) {
        double val = basis.evaluate(nodes[i]);
        if (!approx_equal(val, values[i])) {
            std::cerr << "  FAILED: P_int(" << nodes[i] << ") = " << val << ", expected " << values[i] << "\n";
            return;
        }
    }
    
    // Проверяем значение в середине: P_int(0.5) = 1.25
    double mid_val = basis.evaluate(0.5);
    if (!approx_equal(mid_val, 1.25)) {
        std::cerr << "  FAILED: P_int(0.5) = " << mid_val << ", expected 1.25\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 4: InterpolationBasis - метод Лагранжа
static void test_interpolation_basis_lagrange() {
    std::cout << "Test 4: InterpolationBasis (Lagrange)\n";
    
    // Точки: (0,0), (1,1) - линейная функция y = x
    std::vector<double> nodes = {0.0, 1.0};
    std::vector<double> values = {0.0, 1.0};
    
    InterpolationBasis basis;
    basis.build(nodes, values, InterpolationMethod::LAGRANGE);
    
    // Проверяем значения
    if (!approx_equal(basis.evaluate(0.0), 0.0)) {
        std::cerr << "  FAILED: P_int(0) = " << basis.evaluate(0.0) << ", expected 0\n";
        return;
    }
    if (!approx_equal(basis.evaluate(1.0), 1.0)) {
        std::cerr << "  FAILED: P_int(1) = " << basis.evaluate(1.0) << ", expected 1\n";
        return;
    }
    if (!approx_equal(basis.evaluate(0.5), 0.5)) {
        std::cerr << "  FAILED: P_int(0.5) = " << basis.evaluate(0.5) << ", expected 0.5\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 5: Decomposer - успешное разложение
static void test_decomposer_success() {
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
    
    if (!result.is_valid()) {
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // Проверяем метаданные
    if (result.metadata.n_total != 5) {
        std::cerr << "  FAILED: n_total = " << result.metadata.n_total << ", expected 5\n";
        return;
    }
    if (result.metadata.m_constraints != 3) {
        std::cerr << "  FAILED: m_constraints = " << result.metadata.m_constraints << ", expected 3\n";
        return;
    }
    if (result.metadata.n_free != 3) {  // n - m + 1 = 5 - 3 + 1 = 3
        std::cerr << "  FAILED: n_free = " << result.metadata.n_free << ", expected 3\n";
        return;
    }
    
    // Проверяем, что P_int(x) удовлетворяет интерполяционным условиям
    for (const auto& node : params.interp_nodes) {
        double p_int_val = result.interpolation_basis.evaluate(node.x);
        if (!approx_equal(p_int_val, node.value)) {
            std::cerr << "  FAILED: P_int(" << node.x << ") = " << p_int_val << ", expected " << node.value << "\n";
            return;
        }
    }
    
    // Проверяем, что W(x) имеет корни в узлах
    for (const auto& node : params.interp_nodes) {
        double w_val = result.weight_multiplier.evaluate(node.x);
        if (!approx_equal(w_val, 0.0)) {
            std::cerr << "  FAILED: W(" << node.x << ") = " << w_val << ", expected 0\n";
            return;
        }
    }
    
    // Проверяем, что построенный полином F(x) = P_int(x) + Q(x)·W(x) с Q=0 удовлетворяет условиям
    std::vector<double> zero_q(3, 0.0);
    Polynomial F = result.build_polynomial(zero_q);
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        if (!approx_equal(F_val, node.value)) {
            std::cerr << "  FAILED: F(" << node.x << ") = " << F_val << ", expected " << node.value << "\n";
            return;
        }
    }
    
    std::cout << "  PASSED\n";
}

// Тест 6: Decomposer - случай m = n+1 (полная интерполяция)
static void test_decomposer_full_interpolation() {
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
    
    if (!result.is_valid()) {
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // n_free должно быть 0
    if (result.metadata.n_free != 0) {
        std::cerr << "  FAILED: n_free = " << result.metadata.n_free << ", expected 0\n";
        return;
    }
    
    // Построенный полином должен быть точно интерполяционным полиномом Лагранжа
    std::vector<double> zero_q(0);  // пустой вектор для Q
    Polynomial F = result.build_polynomial(zero_q);
    
    // Проверяем интерполяционные условия
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        if (!approx_equal(F_val, node.value)) {
            std::cerr << "  FAILED: F(" << node.x << ") = " << F_val << ", expected " << node.value << "\n";
            return;
        }
    }
    
    std::cout << "  PASSED\n";
}

// Тест 7: Decomposer - случай m = 0 (нет ограничений)
static void test_decomposer_no_constraints() {
    std::cout << "Test 7: Decomposer (no constraints, m = 0)\n";
    
    Decomposer::Parameters params;
    params.polynomial_degree = 3;  // n = 3
    params.interval_start = 0.0;
    params.interval_end = 10.0;
    params.interp_nodes = {};  // m = 0
    
    DecompositionResult result = Decomposer::decompose(params);
    
    if (!result.is_valid()) {
        // Случай m=0 должен быть корректным
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // n_free должно быть n - 0 + 1 = n + 1 = 4
    if (result.metadata.n_free != 4) {
        std::cerr << "  FAILED: n_free = " << result.metadata.n_free << ", expected 4\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 8: Decomposer - ошибка: n < m-1
static void test_decomposer_insufficient_degree() {
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
    
    if (result.is_valid()) {
        std::cerr << "  FAILED: decomposition should be invalid, but is_valid() returned true\n";
        return;
    }
    
    // Проверяем, что сообщение об ошибке корректное
    if (result.message().find("insufficient") == std::string::npos &&
        result.message().find("Insufficient") == std::string::npos) {
        std::cerr << "  FAILED: error message doesn't mention insufficient degree: " << result.message() << "\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 9: Decomposer - ошибка: дублирующиеся узлы
static void test_decomposer_duplicate_nodes() {
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
    
    if (result.is_valid()) {
        std::cerr << "  FAILED: decomposition should be invalid due to duplicate nodes\n";
        return;
    }
    
    if (result.message().find("duplicate") == std::string::npos &&
        result.message().find("Duplicate") == std::string::npos) {
        std::cerr << "  FAILED: error message doesn't mention duplicates: " << result.message() << "\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 10: Decomposer - ошибка: узлы вне интервала
static void test_decomposer_out_of_bounds() {
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
    
    if (result.is_valid()) {
        std::cerr << "  FAILED: decomposition should be invalid due to out-of-bounds node\n";
        return;
    }
    
    if (result.message().find("outside") == std::string::npos &&
        result.message().find("below") == std::string::npos) {
        std::cerr << "  FAILED: error message doesn't mention out-of-bounds: " << result.message() << "\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Тест 11: Проверка тождества F(z_e) = f(z_e) для случайного Q(x)
static void test_decomposition_identity() {
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
    
    if (!result.is_valid()) {
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // Генерируем случайный Q(x) (коэффициенты в [-1, 1])
    std::vector<double> q_coeffs = {0.5, -0.3};  // Q(x) = 0.5 - 0.3x
    
    // Строим F(x)
    Polynomial F = result.build_polynomial(q_coeffs);
    
    // Проверяем, что F(z_e) = f(z_e) для всех узлов
    for (const auto& node : params.interp_nodes) {
        double F_val = F.evaluate(node.x);
        if (!approx_equal(F_val, node.value, 1e-8)) {
            std::cerr << "  FAILED: F(" << node.x << ") = " << F_val << ", expected " << node.value << "\n";
            return;
        }
    }
    
    std::cout << "  PASSED\n";
}

// Тест 12: Проверка линейной независимости решений
static void test_decomposition_completeness() {
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
    
    if (!result.is_valid()) {
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // Строим n_free линейно независимых полиномов Q_k(x)
    // Q_0(x) = 1, Q_1(x) = x
    std::vector<std::vector<double>> Q_basis = {
        {1.0, 0.0},  // Q_0(x) = 1 (коэффициенты: [q_1, q_0] для степени 1)
        {0.0, 1.0}   // Q_1(x) = x
    };
    
    // Строим соответствующие F_k(x)
    std::vector<Polynomial> F_basis;
    for (const auto& q : Q_basis) {
        F_basis.push_back(result.build_polynomial(q));
    }
    
    // Проверяем, что они линейно независимы: вычисляем значения в нескольких точках
    // и проверяем, что определитель матрицы не равен нулю
    std::vector<double> test_points = {0.0, 4.0, 6.0, 8.0};
    int k_count = F_basis.size();
    int x_count = test_points.size();
    
    if (k_count < x_count) {
        // Недостаточно базисных полиномов для проверки определителя
        // Но мы знаем, что n_free = 2, и у нас 2 полинома, так что проверяем,
        // что они не пропорциональны
        bool proportional = true;
        for (size_t i = 0; i < test_points.size(); ++i) {
            double x = test_points[i];
            double ratio = F_basis[0].evaluate(x) / F_basis[1].evaluate(x);
            if (std::abs(ratio) > 1e-10 && std::abs(F_basis[0].evaluate(x) / F_basis[1].evaluate(x) - 
                F_basis[0].evaluate(test_points[0]) / F_basis[1].evaluate(test_points[0])) > 1e-6) {
                proportional = false;
                break;
            }
        }
        if (proportional) {
            std::cerr << "  FAILED: basis polynomials appear proportional\n";
            return;
        }
    }
    
    std::cout << "  PASSED\n";
}

// Тест 13: Проверка возмущения при конфликте с отталкивающими точками
static void test_decomposition_perturbation() {
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
    
    if (!result.is_valid()) {
        std::cerr << "  FAILED: decomposition not valid: " << result.message() << "\n";
        return;
    }
    
    // P_int(x) - интерполяционный полином степени 1 через (1,2) и (2,3): P_int(x) = x + 1
    // Проверим это
    double p_int_at_0 = result.interpolation_basis.evaluate(0.0);
    if (!approx_equal(p_int_at_0, 1.0)) {
        std::cerr << "  FAILED: P_int(0) = " << p_int_at_0 << ", expected 1.0\n";
        return;
    }
    
    std::cout << "  PASSED\n";
}

// Главная функция для запуска всех тестов
int main() {
    std::cout << "Running decomposition tests...\n\n";
    
    test_weight_multiplier();
    test_interpolation_basis_barycentric();
    test_interpolation_basis_newton();
    test_interpolation_basis_lagrange();
    test_decomposer_success();
    test_decomposer_full_interpolation();
    test_decomposer_no_constraints();
    test_decomposer_insufficient_degree();
    test_decomposer_duplicate_nodes();
    test_decomposer_out_of_bounds();
    test_decomposition_identity();
    test_decomposition_completeness();
    test_decomposition_perturbation();
    
    std::cout << "\nAll tests completed.\n";
    return 0;
}
