#include <iostream>
#include <cassert>
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/validator.h"
#include "mixed_approximation/functional.h"

using namespace mixed_approx;

void test_polynomial_basic() {
    std::cout << "Testing Polynomial basic operations...\n";
    
    // Тест создания полинома
    Polynomial p({1.0, 0.0, -1.0});  // x^2 - 1
    assert(p.degree() == 2);
    
    // Тест evaluate
    assert(std::abs(p.evaluate(0.0) - (-1.0)) < 1e-10);
    assert(std::abs(p.evaluate(1.0) - 0.0) < 1e-10);
    assert(std::abs(p.evaluate(2.0) - 3.0) < 1e-10);
    
    // Тест derivative
    assert(std::abs(p.derivative(0.0) - 0.0) < 1e-10);
    assert(std::abs(p.derivative(1.0) - 2.0) < 1e-10);
    
    // Тест second_derivative
    assert(std::abs(p.second_derivative(0.0) - 2.0) < 1e-10);
    assert(std::abs(p.second_derivative(1.0) - 2.0) < 1e-10);
    
    std::cout << "  Polynomial basic operations: PASSED\n";
}

void test_polynomial_operations() {
    std::cout << "Testing Polynomial arithmetic...\n";
    
    Polynomial p1({1.0, 2.0, 3.0});  // x^2 + 2x + 3
    Polynomial p2({3.0, 2.0, 1.0});  // 3x^2 + 2x + 1
    
    // Сложение: (1+3)x^2 + (2+2)x + (3+1) = 4x^2 + 4x + 4
    Polynomial sum = p1 + p2;
    assert(sum.degree() == 2);
    assert(std::abs(sum.evaluate(0.0) - 4.0) < 1e-10);
    assert(std::abs(sum.evaluate(1.0) - 12.0) < 1e-10);  // 4+4+4=12
    
    // Вычитание: (1-3)x^2 + (2-2)x + (3-1) = -2x^2 + 0x + 2
    Polynomial diff = p1 - p2;
    assert(diff.degree() == 2);
    assert(std::abs(diff.evaluate(0.0) - 2.0) < 1e-10);
    assert(std::abs(diff.evaluate(1.0) - 0.0) < 1e-10);  // -2+0+2=0
    
    // Умножение на скаляр
    Polynomial scaled = p1 * 2.0;
    assert(std::abs(scaled.evaluate(0.0) - 6.0) < 1e-10);
    assert(std::abs(scaled.evaluate(1.0) - 12.0) < 1e-10);  // 2+4+6=12
    
    std::cout << "  Polynomial arithmetic: PASSED\n";
}

void test_validator() {
    std::cout << "Testing Validator...\n";
    
    // Корректная конфигурация
    ApproximationConfig valid_config;
    valid_config.polynomial_degree = 3;
    valid_config.interval_start = 0.0;
    valid_config.interval_end = 1.0;
    valid_config.gamma = 0.1;
    valid_config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    valid_config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    std::string error = Validator::validate(valid_config);
    assert(error.empty());
    
    // Неверные веса
    ApproximationConfig invalid_weights = valid_config;
    invalid_weights.approx_points[0].weight = -1.0;
    error = Validator::validate(invalid_weights);
    assert(!error.empty());
    
    // Пересекающиеся точки
    ApproximationConfig invalid_intersect = valid_config;
    invalid_intersect.approx_points.push_back(WeightedPoint(0.0, 2.0, 1.0));
    error = Validator::validate(invalid_intersect);
    assert(!error.empty());
    
    // Слишком много интерполяционных узлов
    ApproximationConfig invalid_count = valid_config;
    invalid_count.polynomial_degree = 1;
    invalid_count.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(0.5, 1.0), InterpolationNode(1.0, 0.0)};
    error = Validator::validate(invalid_count);
    assert(!error.empty());
    
    std::cout << "  Validator: PASSED\n";
}

void test_functional_basic() {
    std::cout << "Testing Functional basic operations...\n";
    
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
    assert(std::abs(J) < 1e-10);
    
    // Градиент должен быть близок к нулю
    auto grad = functional.gradient(poly);
    for (double g : grad) {
        assert(std::abs(g) < 1e-6);
    }
    
    std::cout << "  Functional basic: PASSED\n";
}

int main() {
    std::cout << "=== Running Basic Tests ===\n\n";
    
    try {
        test_polynomial_basic();
        test_polynomial_operations();
        test_validator();
        test_functional_basic();
        
        std::cout << "\n=== All tests PASSED ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTest FAILED: Unknown exception\n";
        return 1;
    }
}
