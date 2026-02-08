#include <iostream>
#include <cassert>
#include <stdexcept>
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/config_reader.h"

using namespace mixed_approx;

void test_mixed_approximation_validation() {
    std::cout << "Testing MixedApproximation integration with Validator...\n";
    
    // Тест 1: Невалидная конфигурация должна вызывать исключение при создании MixedApproximation
    ApproximationConfig invalid_config;
    invalid_config.polynomial_degree = 3;
    invalid_config.interval_start = 0.0;
    invalid_config.interval_end = 1.0;
    invalid_config.gamma = 0.1;
    invalid_config.approx_points = {WeightedPoint(0.5, 1.0, -1.0)};  // отрицательный вес
    
    try {
        MixedApproximation method(invalid_config);
        assert(false && "Should have thrown exception for invalid config");
    } catch (const std::invalid_argument& e) {
        std::cout << "  Caught expected exception: " << e.what() << "\n";
        // Проверяем, что сообщение содержит информацию о валидации
        std::string what_str(e.what());
        assert(what_str.find("Invalid configuration") != std::string::npos);
    }
    
    // Тест 2: Валидная конфигурация должна создаваться без ошибок
    ApproximationConfig valid_config;
    valid_config.polynomial_degree = 3;
    valid_config.interval_start = 0.0;
    valid_config.interval_end = 1.0;
    valid_config.gamma = 0.1;
    valid_config.approx_points = {WeightedPoint(0.1, 1.0, 1.0), WeightedPoint(0.5, 2.0, 1.0)};
    valid_config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    
    try {
        MixedApproximation method(valid_config);
        std::cout << "  Valid config accepted successfully\n";
    } catch (...) {
        assert(false && "Valid config should not throw");
    }
    
    std::cout << "  MixedApproximation validation integration: PASSED\n";
}

void test_validation_report_usage() {
    std::cout << "Testing ValidationReport usage patterns...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.0;  // предупреждение
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    // Получаем полный отчёт
    ValidationReport report = Validator::validate_full(config);
    
    // Проверяем статистику
    assert(report.approx_points_count == 1);
    assert(report.interp_nodes_count == 1);
    assert(report.polynomial_degree == 3);
    assert(report.free_parameters == 2);  // 3 - 1 = 2
    
    // Должно быть предупреждение о gamma=0
    assert(report.has_warnings());
    assert(!report.has_errors());
    
    std::string formatted = report.format();
    assert(formatted.find("Warnings") != std::string::npos);
    assert(formatted.find("Gamma") != std::string::npos);  // сообщение содержит "Gamma"
    
    // Strict mode должен превратить предупреждение в ошибку
    ValidationReport strict_report = Validator::validate_full(config, true);
    assert(strict_report.has_errors());
    assert(!strict_report.has_warnings());
    
    std::cout << "  ValidationReport usage: PASSED\n";
}

// Вспомогательная функция для создания валидного конфига
ApproximationConfig create_valid_config() {
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    return config;
}

void test_edge_cases() {
    std::cout << "Testing edge cases...\n";
    
    // Тест 1: Пустые множества
    ApproximationConfig empty_config;
    empty_config.polynomial_degree = 3;
    empty_config.interval_start = 0.0;
    empty_config.interval_end = 1.0;
    empty_config.gamma = 0.1;
    // Ни approx_points, ни interp_nodes
    
    std::string error = Validator::validate(empty_config);
    assert(!error.empty() && "Empty both sets should fail");
    
    // Тест 2: Только интерполяционные узлы (допустимо)
    empty_config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    error = Validator::validate(empty_config);
    assert(error.empty() && "Only interp nodes should pass");
    
    // Тест 3: Только аппроксимация (допустимо)
    empty_config.interp_nodes.clear();
    empty_config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    error = Validator::validate(empty_config);
    assert(error.empty() && "Only approx points should pass");
    
    // Тест 4: Очень маленький интервал (предупреждение)
    ApproximationConfig tiny_config;
    tiny_config.polynomial_degree = 2;
    tiny_config.interval_start = 0.0;
    tiny_config.interval_end = 1e-12;  // очень маленький
    tiny_config.gamma = 0.1;
    tiny_config.approx_points = {WeightedPoint(0.0, 0.0, 1.0)};
    
    ValidationReport report = Validator::validate_full(tiny_config);
    // Должно быть предупреждение о маленьком интервале
    assert(report.has_warnings());
    
    // Тест 5: Нечисловые значения
    ApproximationConfig nan_config = create_valid_config();
    nan_config.interval_start = std::numeric_limits<double>::quiet_NaN();
    error = Validator::validate(nan_config);
    assert(!error.empty() && "NaN interval start should fail");
    
    std::cout << "  Edge cases: PASSED\n";
}

void test_initial_approximation_perturbation() {
    std::cout << "Testing initial approximation perturbation (step 1.2.6)...\n";
    
    // Создаём конфигурацию, где начальный полином P_int(x) будет близок к запрещённому значению
    ApproximationConfig config;
    config.polynomial_degree = 3;  // n=3
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Интерполяционные узлы: m=2 (< n+1=4), поэтому есть свободные параметры
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 0.0)
    };
    
    // Отталкивающая точка: x=0.5, y_forbidden=0.0
    // P_int(x) для этих узлов будет P_int(x) = 0 (интерполяция нуля)
    // Поэтому F(0.5) = 0, что равно y_forbidden. Расстояние = 0 < epsilon_init.
    // Ожидается, что свободные коэффициенты (степени 2 и 3) будут возмущены.
    config.repel_points = {RepulsionPoint(0.5, 0.0, 100.0)};
    
    // Создаём метод - в конструкторе строится начальное приближение
    MixedApproximation method(config);
    Polynomial poly = method.get_polynomial();
    
    // Проверяем, что полином не равен точно P_int (который был бы нулевым)
    double value_at_05 = poly.evaluate(0.5);
    // После возмущения значение должно отличаться от 0
    // Возмущение 1e-6 к коэффициентам степени 2 и 3 даст вклад порядка 1e-6 * (0.5^2) = 2.5e-7
    // Поэтому значение должно быть не равно 0 (с точностью до машинного эпсилона)
    assert(std::abs(value_at_05) > 1e-12 && "Perturbation should change polynomial value");
    
    // Проверяем, что интерполяционные условия всё ещё выполняются
    bool interp_ok = method.check_interpolation_conditions();
    assert(interp_ok && "Interpolation conditions must still be satisfied");
    
    std::cout << "  Initial approximation perturbation: PASSED\n";
}

void test_repel_interp_value_conflict_integration() {
    std::cout << "Testing repel-interp value conflict integration with MixedApproximation...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    
    // Конфликт: repel и interp в одной точке с одинаковым значением
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.5, 10.0)};
    
    try {
        MixedApproximation method(config);
        assert(false && "Should have thrown exception for value conflict");
    } catch (const std::invalid_argument& e) {
        std::string what_str(e.what());
        assert(what_str.find("FATAL") != std::string::npos || what_str.find("conflict") != std::string::npos);
        std::cout << "  Caught expected exception: " << e.what() << "\n";
    }
    
    // Без конфликта - должно работать (разные x)
    config.repel_points[0].x = 0.6;  // меняем x, чтобы избежать конфликта координат
    config.interp_nodes[0].x = 0.5;
    config.interp_nodes[0].value = 10.1;
    try {
        MixedApproximation method(config);
        std::cout << "  Valid config without conflict accepted\n";
    } catch (...) {
        assert(false && "Valid config should not throw");
    }
    
    std::cout << "  Repel-interp value conflict integration: PASSED\n";
}

int main() {
    std::cout << "=== Running Integration Validator Tests ===\n\n";
    
    try {
        test_mixed_approximation_validation();
        test_validation_report_usage();
        test_edge_cases();
        test_initial_approximation_perturbation();
        test_repel_interp_value_conflict_integration();
        
        std::cout << "\n=== All integration tests PASSED ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTest FAILED: Unknown exception\n";
        return 1;
    }
}
