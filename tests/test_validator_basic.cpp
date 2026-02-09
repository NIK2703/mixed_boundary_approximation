#include <gtest/gtest.h>
#include <limits>
#include <algorithm>
#include "mixed_approximation/validator.h"
#include "mixed_approximation/types.h"

using namespace mixed_approx;

TEST(ValidatorBasicTest, SimpleValidation) {
    std::cout << "Testing Validator::validate basic functionality...\n";
    
    // Корректная конфигурация
    ApproximationConfig valid_config;
    valid_config.polynomial_degree = 3;
    valid_config.interval_start = 0.0;
    valid_config.interval_end = 1.0;
    valid_config.gamma = 0.1;
    valid_config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    valid_config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    std::string error = Validator::validate(valid_config);
    EXPECT_TRUE(error.empty()) << "Valid config should pass, but got error: " << error;
    
    // Неверные веса
    ApproximationConfig invalid_weights = valid_config;
    invalid_weights.approx_points[0].weight = -1.0;
    error = Validator::validate(invalid_weights);
    EXPECT_FALSE(error.empty()) << "Negative weight should fail";
    EXPECT_NE(error.find("weight"), std::string::npos);
    
    // Пересекающиеся точки (approx и interp в одной точке)
    ApproximationConfig invalid_intersect = valid_config;
    invalid_intersect.approx_points.push_back(WeightedPoint(0.0, 2.0, 1.0));
    error = Validator::validate(invalid_intersect);
    EXPECT_FALSE(error.empty()) << "Intersecting points should fail";
    // Ищем без учёта регистра
    std::string err_lower = error;
    std::transform(err_lower.begin(), err_lower.end(), err_lower.begin(), ::tolower);
    EXPECT_NE(err_lower.find("fatal"), std::string::npos)
        << "Error should be marked as FATAL: " << error;
    
    // Слишком много интерполяционных узлов
    ApproximationConfig invalid_count = valid_config;
    invalid_count.polynomial_degree = 1;
    invalid_count.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 1.0),
        InterpolationNode(1.0, 0.0)
    };
    error = Validator::validate(invalid_count);
    EXPECT_FALSE(error.empty()) << "Too many interp nodes should fail";
}

TEST(ValidatorBasicTest, IntervalValidation) {
    std::cout << "Testing interval validation...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.gamma = 0.1;
    
    // Некорректный интервал (a >= b)
    config.interval_start = 1.0;
    config.interval_end = 1.0;
    std::string error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Degenerate interval should fail";
    
    // a > b
    config.interval_start = 2.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Inverted interval should fail";
    
    // Нечисловые значения
    config.interval_start = std::numeric_limits<double>::infinity();
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Infinity start should fail";
    
    config.interval_start = 0.0;
    config.interval_end = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "NaN end should fail";
    
    // Корректный интервал
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_TRUE(error.empty()) << "Valid interval should pass";
}

TEST(ValidatorBasicTest, PointsInInterval) {
    std::cout << "Testing points in interval check...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Точка вне интервала
    config.approx_points = {WeightedPoint(1.5, 2.0, 1.0)};
    std::string error = Validator::check_points_in_interval(config);
    EXPECT_FALSE(error.empty()) << "Point outside interval should fail";
    
    // Точка на границе (должна пройти)
    config.approx_points = {WeightedPoint(0.0, 0.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "Point at lower boundary should pass";
    
    config.approx_points = {WeightedPoint(1.0, 1.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "Point at upper boundary should pass";
    
    // Несколько точек, все внутри
    config.approx_points = {
        WeightedPoint(0.1, 1.0, 1.0),
        WeightedPoint(0.5, 2.0, 1.0),
        WeightedPoint(0.9, 1.5, 1.0)
    };
    config.repel_points = {WeightedPoint(0.3, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "All points inside should pass";
    
    // Нечисловые координаты
    config.approx_points[0].x = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_points_in_interval(config);
    EXPECT_FALSE(error.empty()) << "NaN coordinate should fail";
}

TEST(ValidatorBasicTest, DisjointSets) {
    std::cout << "Testing disjoint sets check...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Пересечение approx и interp (фатальное)
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    std::string error = Validator::check_disjoint_sets(config);
    EXPECT_FALSE(error.empty()) << "Approx-interp intersection should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Пересечение repel и interp (фатальное)
    config.approx_points.clear();
    config.repel_points = {WeightedPoint(0.3, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.3, 0.0)};
    error = Validator::check_disjoint_sets(config);
    EXPECT_FALSE(error.empty()) << "Repel-interp intersection should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Нет пересечений
    config.repel_points.clear();
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    error = Validator::check_disjoint_sets(config);
    EXPECT_TRUE(error.empty()) << "No intersections should pass";
}

TEST(ValidatorBasicTest, PositiveWeights) {
    std::cout << "Testing positive weights check...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Отрицательный вес
    config.approx_points = {WeightedPoint(0.5, 1.0, -1.0)};
    std::string error = Validator::check_positive_weights(config);
    EXPECT_FALSE(error.empty()) << "Negative weight should fail";
    
    // Нулевой вес
    config.approx_points[0].weight = 0.0;
    error = Validator::check_positive_weights(config);
    EXPECT_FALSE(error.empty()) << "Zero weight should fail";
    
    // Очень маленький вес
    config.approx_points[0].weight = 1e-20;
    error = Validator::check_positive_weights(config);
    EXPECT_FALSE(error.empty()) << "Very small weight should fail";
    
    // Отрицательная gamma
    config.approx_points[0].weight = 1.0;
    config.gamma = -0.1;
    error = Validator::check_positive_weights(config);
    EXPECT_FALSE(error.empty()) << "Negative gamma should fail";
    
    // Нечисловая gamma
    config.gamma = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_positive_weights(config);
    EXPECT_FALSE(error.empty()) << "NaN gamma should fail";
    
    // Корректные значения
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    error = Validator::check_positive_weights(config);
    EXPECT_TRUE(error.empty()) << "Valid weights should pass";
}

TEST(ValidatorBasicTest, InterpolationNodesCount) {
    std::cout << "Testing interpolation nodes count...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // m > n+1 (ошибка)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.33, 1.0),
        InterpolationNode(0.66, 2.0),
        InterpolationNode(1.0, 1.0)
    };
    config.polynomial_degree = 2;  // n=2, n+1=3, m=4 -> ошибка
    std::string error = Validator::check_interpolation_nodes_count(config);
    EXPECT_FALSE(error.empty()) << "Too many nodes should fail";
    EXPECT_NE(error.find("exceeds"), std::string::npos) << "Should mention 'exceeds'";
    
    // m == n+1 (предупреждение)
    config.polynomial_degree = 3;  // n=3, n+1=4, m=4
    error = Validator::check_interpolation_nodes_count(config);
    EXPECT_FALSE(error.empty()) << "m == n+1 should produce warning";
    EXPECT_NE(error.find("WARNING"), std::string::npos) << "Should be a warning";
    
    // m < n+1 (OK)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 1.0)
    };  // m=2, n+1=4
    error = Validator::check_interpolation_nodes_count(config);
    EXPECT_TRUE(error.empty()) << "m < n+1 should pass";
    
    // m == 0 (OK)
    config.interp_nodes.clear();
    error = Validator::check_interpolation_nodes_count(config);
    EXPECT_TRUE(error.empty()) << "No interp nodes should pass";
}

TEST(ValidatorBasicTest, UniqueInterpolationNodes) {
    std::cout << "Testing unique interpolation nodes...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Дубликаты узлов
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 1.0),
        InterpolationNode(0.5, 2.0),  // дубликат x
        InterpolationNode(1.0, 1.0)
    };
    std::string error = Validator::check_unique_interpolation_nodes(config);
    EXPECT_FALSE(error.empty()) << "Duplicate nodes should fail";
    
    // Очень близкие узлы (должны считаться дубликатами из-за epsilon)
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.5, 1.0),
        InterpolationNode(0.5 + 1e-11, 2.0),  // очень близко
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    EXPECT_FALSE(error.empty()) << "Very close nodes should fail";
    
    // Все узлы уникальны
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.33, 1.0),
        InterpolationNode(0.66, 2.0),
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    EXPECT_TRUE(error.empty()) << "Unique nodes should pass";
    
    // Нечисловые координаты
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(std::numeric_limits<double>::quiet_NaN(), 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    EXPECT_FALSE(error.empty()) << "NaN coordinate should fail";
}

TEST(ValidatorBasicTest, NonemptySets) {
    std::cout << "Testing non-empty sets...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Оба множества пусты
    config.approx_points.clear();
    config.interp_nodes.clear();
    std::string error = Validator::check_nonempty_sets(config);
    EXPECT_FALSE(error.empty()) << "Both empty should fail";
    
    // Только approx
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    error = Validator::check_nonempty_sets(config);
    EXPECT_TRUE(error.empty()) << "Only approx should pass";
    
    // Только interp
    config.approx_points.clear();
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    error = Validator::check_nonempty_sets(config);
    EXPECT_TRUE(error.empty()) << "Only interp should pass";
    
    // Оба непустые
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.9, 0.9)};
    error = Validator::check_nonempty_sets(config);
    EXPECT_TRUE(error.empty()) << "Both non-empty should pass";
}

TEST(ValidatorBasicTest, ValidateFull) {
    std::cout << "Testing validate_full()...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0)};
    
    // Корректная конфигурация
    ValidationReport report = Validator::validate_full(config);
    EXPECT_FALSE(report.has_errors()) << "Valid config should have no errors";
    EXPECT_FALSE(report.has_warnings()) << "Valid config should have no warnings";
    EXPECT_EQ(report.approx_points_count, 1u);
    EXPECT_EQ(report.interp_nodes_count, 1u);
    EXPECT_EQ(report.polynomial_degree, 3);
    EXPECT_EQ(report.free_parameters, 2);  // n - m = 3 - 1 = 2
    
    std::string formatted = report.format();
    EXPECT_NE(formatted.find("Validation passed"), std::string::npos);
    
    // Конфигурация с ошибками
    config.approx_points[0].weight = -1.0;
    report = Validator::validate_full(config);
    EXPECT_TRUE(report.has_errors()) << "Invalid weight should produce error";
    EXPECT_FALSE(report.has_warnings());
    
    // Конфигурация с предупреждениями
    config.approx_points[0].weight = 1.0;
    config.gamma = 0.0;
    report = Validator::validate_full(config);
    EXPECT_FALSE(report.has_errors()) << "gamma=0 should be warning, not error";
    EXPECT_TRUE(report.has_warnings()) << "gamma=0 should produce warning";
    
    // Strict mode
    report = Validator::validate_full(config, false);
    EXPECT_FALSE(report.has_errors()) << "Without strict mode, warnings are not errors";
    EXPECT_TRUE(report.has_warnings());
    
    report = Validator::validate_full(config, true);
    EXPECT_TRUE(report.has_errors()) << "With strict mode, warnings become errors";
    EXPECT_FALSE(report.has_warnings()) << "Warnings moved to errors";
}

TEST(ValidatorBasicTest, RepelInterpValueConflict) {
    std::cout << "Testing repel-interp value conflict...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Конфликт по координатам и значениям (фатальный)
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};  // x=0.5, y_forbidden=10.0
    config.interp_nodes = {InterpolationNode(0.5, 10.0)};      // x=0.5, f=10.0
    std::string error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_FALSE(error.empty()) << "Value conflict should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Близкие координаты, но разные значения (должно пройти)
    config.repel_points[0].y_forbidden = 10.0;
    config.interp_nodes[0].value = 10.1;
    error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_TRUE(error.empty()) << "Close x but different y should pass";
    
    // Разные координаты (должно пройти)
    config.repel_points[0].x = 0.3;
    config.interp_nodes[0].x = 0.5;
    error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_TRUE(error.empty()) << "Different x should pass";
    
    // Пустое множество repel или interp (должно пройти)
    config.repel_points.clear();
    error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_TRUE(error.empty()) << "Empty repel set should pass";
    
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};
    config.interp_nodes.clear();
    error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_TRUE(error.empty()) << "Empty interp set should pass";
}
