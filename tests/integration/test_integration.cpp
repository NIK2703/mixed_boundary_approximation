#include <gtest/gtest.h>
#include <fstream>
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/config_reader.h"

using namespace mixed_approx;

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

// Вспомогательная функция для проверки близости чисел
static bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

TEST(IntegrationTest, MixedApproximationValidation) {
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
        FAIL() << "Should have thrown exception for invalid config";
    } catch (const std::invalid_argument& e) {
        std::cout << "  Caught expected exception: " << e.what() << "\n";
        // Проверяем, что сообщение содержит информацию о валидации
        std::string what_str(e.what());
        EXPECT_NE(what_str.find("Invalid configuration"), std::string::npos)
            << "Exception message should mention invalid configuration";
    }
    
    // Тест 2: Валидная конфигурация должна создаваться без ошибок
    ApproximationConfig valid_config = create_valid_config();
    
    try {
        MixedApproximation method(valid_config);
        std::cout << "  Valid config accepted successfully\n";
    } catch (...) {
        FAIL() << "Valid config should not throw";
    }
}

TEST(IntegrationTest, ValidationReportUsage) {
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
    EXPECT_EQ(report.approx_points_count, 1u);
    EXPECT_EQ(report.interp_nodes_count, 1u);
    EXPECT_EQ(report.polynomial_degree, 3);
    EXPECT_EQ(report.free_parameters, 2);  // 3 - 1 = 2
    
    // Должно быть предупреждение о gamma=0
    EXPECT_TRUE(report.has_warnings());
    EXPECT_FALSE(report.has_errors());
    
    std::string formatted = report.format();
    EXPECT_NE(formatted.find("Warnings"), std::string::npos);
    EXPECT_NE(formatted.find("Gamma"), std::string::npos);
    
    // Strict mode должен превратить предупреждение в ошибку
    ValidationReport strict_report = Validator::validate_full(config, true);
    EXPECT_TRUE(strict_report.has_errors());
    EXPECT_FALSE(strict_report.has_warnings());
}

TEST(IntegrationTest, EdgeCases) {
    std::cout << "Testing edge cases...\n";
    
    // Тест 1: Пустые множества
    ApproximationConfig empty_config;
    empty_config.polynomial_degree = 3;
    empty_config.interval_start = 0.0;
    empty_config.interval_end = 1.0;
    empty_config.gamma = 0.1;
    
    std::string error = Validator::validate(empty_config);
    EXPECT_FALSE(error.empty()) << "Empty both sets should fail";
    
    // Тест 2: Только интерполяционные узлы (допустимо)
    empty_config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    error = Validator::validate(empty_config);
    EXPECT_TRUE(error.empty()) << "Only interp nodes should pass";
    
    // Тест 3: Только аппроксимация (допустимо)
    empty_config.interp_nodes.clear();
    empty_config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    error = Validator::validate(empty_config);
    EXPECT_TRUE(error.empty()) << "Only approx points should pass";
    
    // Тест 4: Нечисловые значения
    ApproximationConfig nan_config = create_valid_config();
    nan_config.interval_start = std::numeric_limits<double>::quiet_NaN();
    error = Validator::validate(nan_config);
    EXPECT_FALSE(error.empty()) << "NaN interval start should fail";
}

TEST(IntegrationTest, InitialApproximationPerturbation) {
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
    EXPECT_NE(value_at_05, 0.0) << "Perturbation should change polynomial value";
    
    // Проверяем, что интерполяционные условия всё ещё выполняются
    bool interp_ok = method.check_interpolation_conditions();
    EXPECT_TRUE(interp_ok) << "Interpolation conditions must still be satisfied";
}

TEST(IntegrationTest, RepelInterpValueConflictIntegration) {
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
        FAIL() << "Should have thrown exception for value conflict";
    } catch (const std::invalid_argument& e) {
        std::string what_str(e.what());
        EXPECT_NE(what_str.find("FATAL"), std::string::npos) 
            << "Exception should mention FATAL: " << what_str;
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
        FAIL() << "Valid config should not throw";
    }
}

TEST(IntegrationTest, SolveAndGetPolynomial) {
    std::cout << "Testing solve() and get_polynomial()...\n";
    
    ApproximationConfig config = create_valid_config();
    config.polynomial_degree = 1;  // степень 1
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 1.0)
    };  // Два узла: полином полностью определён (F(x)=x), свободных параметров 0
    config.gamma = 0.0;  // без регуляризации
    
    std::cout << "  Config: degree=" << config.polynomial_degree
              << ", interp_nodes=" << config.interp_nodes.size()
              << ", approx_points=" << config.approx_points.size()
              << ", gamma=" << config.gamma << "\n";
    
    MixedApproximation method(config);
    Polynomial initial = method.get_polynomial();
    std::cout << "  Initial polynomial coefficients: ";
    for (double c : initial.coefficients()) std::cout << c << " ";
    std::cout << "\n";
    std::cout << "  Initial interp check: ";
    for (const auto& node : config.interp_nodes) {
        double val = initial.evaluate(node.x);
        std::cout << "F(" << node.x << ")=" << val << " (expected " << node.value << ") ";
    }
    std::cout << "\n";
    
    OptimizationResult result = method.solve();
    
    std::cout << "  Optimization result: success=" << result.success
              << ", iterations=" << result.iterations
              << ", final_objective=" << result.final_objective << "\n";
    
    // Проверяем, что оптимизация прошла успешно
    EXPECT_TRUE(result.success)
        << "Optimization should succeed";
    
    Polynomial poly = method.get_polynomial();
    
    std::cout << "  Final polynomial coefficients: ";
    for (double c : poly.coefficients()) std::cout << c << " ";
    std::cout << "\n";
    
    // Проверяем интерполяционные условия
    for (const auto& node : config.interp_nodes) {
        double val = poly.evaluate(node.x);
        std::cout << "  Check interp at x=" << node.x << ": F(x)=" << val << ", expected=" << node.value << "\n";
        EXPECT_NEAR( val, node.value, 1e-6)
            << "Interpolation condition at x=" << node.x << " failed";
    }
    
    // Проверяем, что функционал конечен
    Functional functional(config);
    double J = functional.evaluate(poly);
    std::cout << "  Functional value: " << J << "\n";
    EXPECT_FALSE(std::isnan(J));
    EXPECT_FALSE(std::isinf(J));
    EXPECT_GE(J, 0.0) << "Functional should be non-negative";
}

TEST(IntegrationTest, ComputeRepelDistances) {
    std::cout << "Testing compute_repel_distances()...\n";
    
    ApproximationConfig config = create_valid_config();
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    config.repel_points = {
        RepulsionPoint(0.3, 10.0, 1.0),
        RepulsionPoint(0.7, 5.0, 2.0)
    };
    
    MixedApproximation method(config);
    // method.build_initial_approximation() вызывается в конструкторе
    // Для config с interp_nodes = {InterpolationNode(0.0, 0.0)} P_int(x) = 0
    // Поэтому compute_repel_distances() использует polynomial_ = 0
    
    std::vector<double> distances = method.compute_repel_distances();
    
    EXPECT_EQ(distances.size(), 2u);
    
    // F(0.3) = 0, расстояние до 10.0 = 10.0
    EXPECT_NEAR( distances[0], 10.0, 1e-6);
    // F(0.7) = 0, расстояние до 5.0 = 5.0
    EXPECT_NEAR( distances[1], 5.0, 1e-6);
}

TEST(IntegrationTest, GetFunctionalComponents) {
    std::cout << "Testing get_functional_components()...\n";
    
    ApproximationConfig config = create_valid_config();
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    config.repel_points = {};
    config.gamma = 0.1;
    
    MixedApproximation method(config);
    // polynomial_ = P_int(x) = 0 (интерполяция через (0,0))
    
    auto components = method.get_functional_components();
    
    // F(0)=0 (интерполяция), F(0.5)=0, ошибка аппроксимации = (0-1)^2 = 1.0
    // Регуляризация: F''=0, поэтому reg=0
    EXPECT_NEAR( components.approx_component, 1.0, 1e-6);
    EXPECT_NEAR( components.reg_component, 0.0, 1e-6);
    EXPECT_NEAR( components.repel_component, 0.0, 1e-6);
    EXPECT_NEAR( components.total, 1.0, 1e-6);
}

TEST(IntegrationTest, CheckInterpolationConditions) {
    std::cout << "Testing check_interpolation_conditions()...\n";
    
    ApproximationConfig config = create_valid_config();
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 0.5),
        InterpolationNode(1.0, 1.0)
    };
    config.approx_points = {};
    config.repel_points = {};
    
    MixedApproximation method(config);
    
    // Полином должен удовлетворять интерполяции
    EXPECT_TRUE(method.check_interpolation_conditions(1e-10))
        << "Polynomial should satisfy interpolation conditions";
    
    // Проверим, что условие действительно выполняется
    Polynomial poly = method.get_polynomial();
    for (const auto& node : config.interp_nodes) {
        double val = poly.evaluate(node.x);
        EXPECT_NEAR( val, node.value, 1e-8)
            << "At x=" << node.x;
    }
}

TEST(IntegrationTest, ConfigReaderIntegration) {
    std::cout << "Testing ConfigReader integration...\n";
    
    // Создадим временный файл в простом формате (не YAML)
    std::string config_content = R"(
polynomial_degree = 3
interval_start = 0.0
interval_end = 1.0
gamma = 0.1
approx_points_count = 1
0.1 1.0 1.0
interp_nodes_count = 1
0.0 0.0
)";
    
    // Запишем временный файл
    std::string temp_file = "/tmp/test_config.txt";
    std::ofstream ofs(temp_file);
    ofs << config_content;
    ofs.close();
    
    // Читаем конфигурацию
    try {
        ApproximationConfig config = ConfigReader::read_from_file(temp_file);
        
        // Проверяем параметры
        EXPECT_EQ(config.polynomial_degree, 3);
        EXPECT_DOUBLE_EQ(config.interval_start, 0.0);
        EXPECT_DOUBLE_EQ(config.interval_end, 1.0);
        EXPECT_DOUBLE_EQ(config.gamma, 0.1);
        EXPECT_EQ(config.approx_points.size(), 1u);
        EXPECT_EQ(config.interp_nodes.size(), 1u);
        EXPECT_DOUBLE_EQ(config.approx_points[0].x, 0.1);
        EXPECT_DOUBLE_EQ(config.approx_points[0].value, 1.0);
        EXPECT_DOUBLE_EQ(config.approx_points[0].weight, 1.0);
        EXPECT_DOUBLE_EQ(config.interp_nodes[0].x, 0.0);
        EXPECT_DOUBLE_EQ(config.interp_nodes[0].value, 0.0);
        
        // Пробуем создать MixedApproximation
        MixedApproximation method(config);
        std::cout << "  ConfigReader integration successful\n";
        
        // Удаляем временный файл
        std::remove(temp_file.c_str());
    } catch (const std::exception& e) {
        std::remove(temp_file.c_str());
        FAIL() << "ConfigReader integration failed: " << e.what();
    }
}
