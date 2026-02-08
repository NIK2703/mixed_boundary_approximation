#include <iostream>
#include <cassert>
#include <cmath>
#include "mixed_approximation/validator.h"
#include "mixed_approximation/types.h"

using namespace mixed_approx;

void test_interval_validation() {
    std::cout << "Testing interval validation...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.gamma = 0.1;
    
    // Тест 1: Некорректный интервал (a >= b)
    config.interval_start = 1.0;
    config.interval_end = 1.0;
    std::string error = Validator::check_interval(config);
    assert(!error.empty() && "Degenerate interval should fail");
    
    // Тест 2: a > b
    config.interval_start = 2.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    assert(!error.empty() && "Inverted interval should fail");
    
    // Тест 3: Нечисловые значения
    config.interval_start = std::numeric_limits<double>::infinity();
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    assert(!error.empty() && "Infinity start should fail");
    
    config.interval_start = 0.0;
    config.interval_end = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_interval(config);
    assert(!error.empty() && "NaN end should fail");
    
    // Тест 4: Корректный интервал
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    assert(error.empty() && "Valid interval should pass");
    
    // Тест 5: Очень маленький интервал (должен пройти, т.к. проверяется с точностью)
    config.interval_start = 0.0;
    config.interval_end = 1e-13;
    error = Validator::check_interval(config);
    // Может пройти или нет в зависимости от адаптивного epsilon
    std::cout << "    Small interval check: " << (error.empty() ? "passed" : "failed") << "\n";
    
    std::cout << "  Interval validation: PASSED\n";
}

void test_points_in_interval() {
    std::cout << "Testing points in interval...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Точка вне интервала
    config.approx_points = {WeightedPoint(1.5, 2.0, 1.0)};
    std::string error = Validator::check_points_in_interval(config);
    assert(!error.empty() && "Point outside interval should fail");
    
    // Тест 2: Точка на границе (должна пройти)
    config.approx_points = {WeightedPoint(0.0, 0.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    assert(error.empty() && "Point at boundary should pass");
    
    config.approx_points = {WeightedPoint(1.0, 1.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    assert(error.empty() && "Point at upper boundary should pass");
    
    // Тест 3: Несколько точек, все внутри
    config.approx_points = {
        WeightedPoint(0.1, 1.0, 1.0),
        WeightedPoint(0.5, 2.0, 1.0),
        WeightedPoint(0.9, 1.5, 1.0)
    };
    config.repel_points = {
        WeightedPoint(0.3, 10.0, 100.0)
    };
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_points_in_interval(config);
    assert(error.empty() && "All points inside should pass");
    
    // Тест 4: Нечисловые координаты
    config.approx_points[0].x = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_points_in_interval(config);
    assert(!error.empty() && "NaN coordinate should fail");
    
    std::cout << "  Points in interval: PASSED\n";
}

void test_disjoint_sets() {
    std::cout << "Testing disjoint sets (optimized O(N log N))...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Пересечение approx и interp (фатальное)
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    std::string error = Validator::check_disjoint_sets(config);
    assert(!error.empty() && "Approx-interp intersection should fail");
    assert(error.find("FATAL") != std::string::npos && "Should be marked as FATAL");
    
    // Тест 2: Пересечение repel и interp (фатальное)
    config.approx_points.clear();
    config.repel_points = {WeightedPoint(0.3, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.3, 0.0)};
    error = Validator::check_disjoint_sets(config);
    assert(!error.empty() && "Repel-interp intersection should fail");
    assert(error.find("FATAL") != std::string::npos && "Should be marked as FATAL");
    
    // Тест 3: Пересечение approx и repel (предупреждение)
    config.repel_points.clear();
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes.clear();
    config.approx_points.push_back(WeightedPoint(0.5, 2.0, 1.0)); // Теперь два approx - не должно быть ошибки
    // Нужен repel
    config.repel_points = {WeightedPoint(0.5, 10.0, 100.0)};
    error = Validator::check_disjoint_sets(config);
    // Должно быть предупреждение (не фатальное)
    assert(!error.empty() && "Approx-repel intersection should warn");
    assert(error.find("FATAL") == std::string::npos && "Should not be marked as FATAL");
    
    // Тест 4: Нет пересечений
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.repel_points = {WeightedPoint(0.9, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    error = Validator::check_disjoint_sets(config);
    assert(error.empty() && "No intersections should pass");
    
    // Тест 5: Дубликаты в одном множестве (должны обрабатываться в check_unique_interpolation_nodes)
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0), WeightedPoint(0.1, 2.0, 1.0)};
    config.repel_points.clear();
    config.interp_nodes.clear();
    error = Validator::check_disjoint_sets(config);
    // Два approx в одной точке - не обрабатывается здесь
    assert(error.empty() && "Duplicate in same set not checked by disjoint");
    
    std::cout << "  Disjoint sets: PASSED\n";
}

void test_repel_interp_value_conflict() {
    std::cout << "Testing repel-interp value conflict (step 1.2.4)...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Конфликт по координатам и значениям (фатальный)
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};  // x=0.5, y_forbidden=10.0
    config.interp_nodes = {InterpolationNode(0.5, 10.0)};      // x=0.5, f=10.0
    std::string error = Validator::check_repel_interp_value_conflict(config);
    assert(!error.empty() && "Value conflict should fail");
    assert(error.find("FATAL") != std::string::npos && "Should be marked as FATAL");
    
    // Тест 2: Близкие координаты, но разные значения (должно пройти)
    config.repel_points[0].y_forbidden = 10.0;
    config.interp_nodes[0].value = 10.1;
    error = Validator::check_repel_interp_value_conflict(config);
    assert(error.empty() && "Close x but different y should pass");
    
    // Тест 3: Разные координаты (должно пройти)
    config.repel_points[0].x = 0.3;
    config.interp_nodes[0].x = 0.5;
    error = Validator::check_repel_interp_value_conflict(config);
    assert(error.empty() && "Different x should pass");
    
    // Тест 4: Пустое множество repel или interp (должно пройти)
    config.repel_points.clear();
    error = Validator::check_repel_interp_value_conflict(config);
    assert(error.empty() && "Empty repel set should pass");
    
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};
    config.interp_nodes.clear();
    error = Validator::check_repel_interp_value_conflict(config);
    assert(error.empty() && "Empty interp set should pass");
    
    // Тест 5: Конфликт с точностью epsilon_value
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.5, 10.0 + 1e-10)};  // очень близко
    error = Validator::check_repel_interp_value_conflict(config);
    // Может быть конфликт или нет в зависимости от epsilon_value
    std::cout << "    Epsilon value test: " << (error.empty() ? "no conflict" : "conflict detected") << "\n";
    
    std::cout << "  Repel-interp value conflict: PASSED\n";
}

void test_positive_weights() {
    std::cout << "Testing positive weights and anomalies...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Отрицательный вес
    config.approx_points = {WeightedPoint(0.5, 1.0, -1.0)};
    std::string error = Validator::check_positive_weights(config);
    assert(!error.empty() && "Negative weight should fail");
    
    // Тест 2: Нулевой вес
    config.approx_points[0].weight = 0.0;
    error = Validator::check_positive_weights(config);
    assert(!error.empty() && "Zero weight should fail");
    
    // Тест 3: Очень маленький вес
    config.approx_points[0].weight = 1e-20;
    error = Validator::check_positive_weights(config);
    assert(!error.empty() && "Very small weight should fail");
    
    // Тест 4: Отрицательная gamma
    config.approx_points[0].weight = 1.0;
    config.gamma = -0.1;
    error = Validator::check_positive_weights(config);
    assert(!error.empty() && "Negative gamma should fail");
    
    // Тест 5: Нечисловая gamma
    config.gamma = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_positive_weights(config);
    assert(!error.empty() && "NaN gamma should fail");
    
    // Тест 6: Корректные значения
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0), WeightedPoint(0.5, 2.0, 2.0)};
    config.repel_points = {WeightedPoint(0.9, 10.0, 100.0)};
    error = Validator::check_positive_weights(config);
    assert(error.empty() && "Valid weights should pass");
    
    std::cout << "  Positive weights: PASSED\n";
}

void test_interpolation_nodes_count() {
    std::cout << "Testing interpolation nodes count...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: m > n+1 (ошибка)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.33, 1.0),
        InterpolationNode(0.66, 2.0),
        InterpolationNode(1.0, 1.0)  // m=4, n+1=4 -> предупреждение, не ошибка
    };
    config.polynomial_degree = 2;  // n=2, n+1=3, m=4 -> ошибка
    std::string error = Validator::check_interpolation_nodes_count(config);
    assert(!error.empty() && "Too many nodes should fail");
    assert(error.find("exceeds") != std::string::npos && "Should mention 'exceeds'");
    
    // Тест 2: m == n+1 (предупреждение)
    config.polynomial_degree = 3;  // n=3, n+1=4, m=4
    error = Validator::check_interpolation_nodes_count(config);
    assert(!error.empty() && "m == n+1 should produce warning");
    assert(error.find("WARNING") != std::string::npos && "Should be a warning");
    
    // Тест 3: m < n+1 (OK)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 1.0)
    };  // m=2, n+1=4
    error = Validator::check_interpolation_nodes_count(config);
    assert(error.empty() && "m < n+1 should pass");
    
    // Тест 4: m == 0 (OK)
    config.interp_nodes.clear();
    error = Validator::check_interpolation_nodes_count(config);
    assert(error.empty() && "No interp nodes should pass");
    
    std::cout << "  Interpolation nodes count: PASSED\n";
}

void test_unique_interpolation_nodes() {
    std::cout << "Testing unique interpolation nodes...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Дубликаты узлов
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 1.0),
        InterpolationNode(0.5, 2.0),  // дубликат x
        InterpolationNode(1.0, 1.0)
    };
    std::string error = Validator::check_unique_interpolation_nodes(config);
    assert(!error.empty() && "Duplicate nodes should fail");
    
    // Тест 2: Очень близкие узлы (должны считаться дубликатами из-за epsilon)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 1.0),
        InterpolationNode(0.5 + 1e-11, 2.0),  // очень близко
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    assert(!error.empty() && "Very close nodes should fail");
    
    // Тест 3: Все узлы уникальны
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.33, 1.0),
        InterpolationNode(0.66, 2.0),
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    assert(error.empty() && "Unique nodes should pass");
    
    // Тест 4: Нечисловые координаты
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(std::numeric_limits<double>::quiet_NaN(), 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    assert(!error.empty() && "NaN coordinate should fail");
    
    std::cout << "  Unique interpolation nodes: PASSED\n";
}

void test_nonempty_sets() {
    std::cout << "Testing non-empty sets...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Оба множества пусты
    config.approx_points.clear();
    config.interp_nodes.clear();
    std::string error = Validator::check_nonempty_sets(config);
    assert(!error.empty() && "Both empty should fail");
    
    // Тест 2: Только approx
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    error = Validator::check_nonempty_sets(config);
    assert(error.empty() && "Only approx should pass");
    
    // Тест 3: Только interp
    config.approx_points.clear();
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    error = Validator::check_nonempty_sets(config);
    assert(error.empty() && "Only interp should pass");
    
    // Тест 4: Оба непустые
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.9, 0.9)};
    error = Validator::check_nonempty_sets(config);
    assert(error.empty() && "Both non-empty should pass");
    
    std::cout << "  Non-empty sets: PASSED\n";
}

void test_numerical_anomalies() {
    std::cout << "Testing numerical anomalies...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Экстремальный разброс весов
    config.approx_points = {
        WeightedPoint(0.1, 1.0, 1e-6),
        WeightedPoint(0.5, 2.0, 1e6)  // ratio = 1e12
    };
    std::string warning = Validator::check_numerical_anomalies(config);
    // Должно быть предупреждение о разбросе
    assert(!warning.empty() && "Extreme weight ratio should warn");
    
    // Тест 2: Очень большой repel вес
    config.approx_points.clear();
    config.repel_points = {WeightedPoint(0.5, 10.0, 1e9)};
    warning = Validator::check_numerical_anomalies(config);
    assert(!warning.empty() && "Large repel weight should warn");
    
    // Тест 3: Очень маленький repel вес
    config.repel_points[0].weight = 1e-10;
    warning = Validator::check_numerical_anomalies(config);
    assert(!warning.empty() && "Small repel weight should warn");
    
    // Тест 4: gamma = 0
    config.repel_points.clear();
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.gamma = 0.0;
    warning = Validator::check_numerical_anomalies(config);
    assert(!warning.empty() && "gamma=0 should warn");
    
    // Тест 5: gamma очень большое
    config.gamma = 1e7;
    warning = Validator::check_numerical_anomalies(config);
    assert(!warning.empty() && "Large gamma should warn");
    
    // Тест 6: Очень маленький интервал
    config.interval_start = 0.0;
    config.interval_end = 1e-11;
    warning = Validator::check_numerical_anomalies(config);
    assert(!warning.empty() && "Very small interval should warn");
    
    // Тест 7: Нормальные значения
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.gamma = 0.1;
    warning = Validator::check_numerical_anomalies(config);
    assert(warning.empty() && "Normal values should pass");
    
    std::cout << "  Numerical anomalies: PASSED\n";
}

void test_validate_full() {
    std::cout << "Testing validate_full() with report...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    // Тест 1: Корректная конфигурация
    ValidationReport report = Validator::validate_full(config);
    assert(!report.has_errors() && "Valid config should have no errors");
    assert(!report.has_warnings() && "Valid config should have no warnings");
    assert(report.approx_points_count == 1);
    assert(report.interp_nodes_count == 1);
    assert(report.polynomial_degree == 3);
    assert(report.free_parameters == 2);  // n - m = 3 - 1 = 2
    
    std::string formatted = report.format();
    assert(formatted.find("Validation passed") != std::string::npos);
    
    // Тест 2: Конфигурация с ошибками
    config.approx_points[0].weight = -1.0;
    report = Validator::validate_full(config);
    assert(report.has_errors() && "Invalid weight should produce error");
    assert(!report.has_warnings());
    
    // Тест 3: Конфигурация с предупреждениями
    config.approx_points[0].weight = 1.0;
    config.gamma = 0.0;
    report = Validator::validate_full(config);
    assert(!report.has_errors() && "gamma=0 should be warning, not error");
    assert(report.has_warnings() && "gamma=0 should produce warning");
    
    // Тест 4: Strict mode
    report = Validator::validate_full(config, false);
    assert(!report.has_errors() && "Without strict mode, warnings are not errors");
    assert(report.has_warnings());
    
    report = Validator::validate_full(config, true);
    assert(report.has_errors() && "With strict mode, warnings become errors");
    assert(!report.has_warnings() && "Warnings moved to errors");
    
    std::cout << "  validate_full(): PASSED\n";
}

void test_validate_interface() {
    std::cout << "Testing validate() interface compatibility...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    // Корректная конфигурация
    std::string error = Validator::validate(config);
    assert(error.empty() && "Valid config should pass");
    
    // Ошибка
    config.approx_points[0].weight = -1.0;
    error = Validator::validate(config);
    assert(!error.empty() && "Invalid weight should fail");
    
    // Strict mode
    config.approx_points[0].weight = 1.0;
    config.gamma = 0.0;
    error = Validator::validate(config, false);
    assert(error.empty() && "Warning without strict mode should pass");
    
    error = Validator::validate(config, true);
    assert(!error.empty() && "Warning with strict mode should fail");
    
    std::cout << "  validate() interface: PASSED\n";
}

int main() {
    std::cout << "=== Running Advanced Validator Tests ===\n\n";
    
    try {
        test_interval_validation();
        test_points_in_interval();
        test_disjoint_sets();
        test_repel_interp_value_conflict();
        test_positive_weights();
        test_interpolation_nodes_count();
        test_unique_interpolation_nodes();
        test_nonempty_sets();
        test_numerical_anomalies();
        test_validate_full();
        test_validate_interface();
        
        std::cout << "\n=== All advanced validator tests PASSED ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTest FAILED: Unknown exception\n";
        return 1;
    }
}
