#include <gtest/gtest.h>
#include <limits>
#include "mixed_approximation/validator.h"
#include "mixed_approximation/types.h"

using namespace mixed_approx;

TEST(ValidatorAdvancedTest, IntervalValidationDetailed) {
    std::cout << "Testing detailed interval validation...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.gamma = 0.1;
    
    // Тест 1: Некорректный интервал (a >= b)
    config.interval_start = 1.0;
    config.interval_end = 1.0;
    std::string error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Degenerate interval should fail";
    
    // Тест 2: a > b
    config.interval_start = 2.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Inverted interval should fail";
    
    // Тест 3: Нечисловые значения
    config.interval_start = std::numeric_limits<double>::infinity();
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "Infinity start should fail";
    
    config.interval_start = 0.0;
    config.interval_end = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_interval(config);
    EXPECT_FALSE(error.empty()) << "NaN end should fail";
    
    // Тест 4: Корректный интервал
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    error = Validator::check_interval(config);
    EXPECT_TRUE(error.empty()) << "Valid interval should pass";
}

TEST(ValidatorAdvancedTest, PointsInIntervalDetailed) {
    std::cout << "Testing detailed points in interval...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Точка вне интервала
    config.approx_points = {WeightedPoint(1.5, 2.0, 1.0)};
    std::string error = Validator::check_points_in_interval(config);
    EXPECT_FALSE(error.empty()) << "Point outside interval should fail";
    
    // Тест 2: Точка на границе (должна пройти)
    config.approx_points = {WeightedPoint(0.0, 0.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "Point at lower boundary should pass";
    
    config.approx_points = {WeightedPoint(1.0, 1.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "Point at upper boundary should pass";
    
    // Тест 3: Несколько точек, все внутри
    config.approx_points = {
        WeightedPoint(0.1, 1.0, 1.0),
        WeightedPoint(0.5, 2.0, 1.0),
        WeightedPoint(0.9, 1.5, 1.0)
    };
    config.repel_points = {WeightedPoint(0.3, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.0, 0.0), InterpolationNode(1.0, 1.0)};
    error = Validator::check_points_in_interval(config);
    EXPECT_TRUE(error.empty()) << "All points inside should pass";
    
    // Тест 4: Нечисловые координаты
    config.approx_points[0].x = std::numeric_limits<double>::quiet_NaN();
    error = Validator::check_points_in_interval(config);
    EXPECT_FALSE(error.empty()) << "NaN coordinate should fail";
}

TEST(ValidatorAdvancedTest, DisjointSetsDetailed) {
    std::cout << "Testing disjoint sets with O(N log N) algorithm...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Тест 1: Пересечение approx и interp (фатальное)
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    std::string error = Validator::check_disjoint_sets(config);
    EXPECT_FALSE(error.empty()) << "Approx-interp intersection should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Тест 2: Пересечение repel и interp (фатальное)
    config.approx_points.clear();
    config.repel_points = {WeightedPoint(0.3, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.3, 0.0)};
    error = Validator::check_disjoint_sets(config);
    EXPECT_FALSE(error.empty()) << "Repel-interp intersection should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Тест 3: Нет пересечений
    config.repel_points.clear();
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    config.interp_nodes = {InterpolationNode(0.5, 0.5)};
    error = Validator::check_disjoint_sets(config);
    EXPECT_TRUE(error.empty()) << "No intersections should pass";
}

TEST(ValidatorAdvancedTest, RepelInterpValueConflictDetailed) {
    std::cout << "Testing repel-interp value conflict (step 1.2.4)...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Конфликт по координатам и значениям (фатальный)
    config.repel_points = {RepulsionPoint(0.5, 10.0, 100.0)};
    config.interp_nodes = {InterpolationNode(0.5, 10.0)};
    std::string error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_FALSE(error.empty()) << "Value conflict should fail";
    EXPECT_NE(error.find("FATAL"), std::string::npos) << "Should be marked as FATAL";
    
    // Близкие координаты, но разные значения (должно пройти)
    config.repel_points[0].y_forbidden = 10.0;
    config.interp_nodes[0].value = 10.1;
    error = Validator::check_repel_interp_value_conflict(config);
    EXPECT_TRUE(error.empty()) << "Close x but different y should pass";
}

TEST(ValidatorAdvancedTest, PositiveWeightsDetailed) {
    std::cout << "Testing positive weights and numerical anomalies...\n";
    
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
    
    // Корректные значения
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.1, 1.0, 1.0)};
    error = Validator::check_positive_weights(config);
    EXPECT_TRUE(error.empty()) << "Valid weights should pass";
}

TEST(ValidatorAdvancedTest, InterpolationNodesCountDetailed) {
    std::cout << "Testing interpolation nodes count constraints...\n";
    
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
    };
    error = Validator::check_interpolation_nodes_count(config);
    EXPECT_TRUE(error.empty()) << "m < n+1 should pass";
}

TEST(ValidatorAdvancedTest, UniqueInterpolationNodesDetailed) {
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
    
    // Все узлы уникальны
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(0.33, 1.0),
        InterpolationNode(0.66, 2.0),
        InterpolationNode(1.0, 1.0)
    };
    error = Validator::check_unique_interpolation_nodes(config);
    EXPECT_TRUE(error.empty()) << "Unique nodes should pass";
}

TEST(ValidatorAdvancedTest, NonemptySetsDetailed) {
    std::cout << "Testing non-empty sets validation...\n";
    
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
}

TEST(ValidatorAdvancedTest, NumericalAnomalies) {
    std::cout << "Testing numerical anomalies detection...\n";
    
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Экстремальный разброс весов
    config.approx_points = {
        WeightedPoint(0.1, 1.0, 1e-6),
        WeightedPoint(0.5, 2.0, 1e6)
    };
    std::string warning = Validator::check_numerical_anomalies(config);
    // Должно быть предупреждение о разбросе
    EXPECT_FALSE(warning.empty()) << "Extreme weight ratio should warn";
    
    // Очень большой repel вес
    config.approx_points.clear();
    config.repel_points = {WeightedPoint(0.5, 10.0, 1e9)};
    warning = Validator::check_numerical_anomalies(config);
    EXPECT_FALSE(warning.empty()) << "Large repel weight should warn";
    
    // gamma = 0
    config.repel_points.clear();
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    config.gamma = 0.0;
    warning = Validator::check_numerical_anomalies(config);
    EXPECT_FALSE(warning.empty()) << "gamma=0 should warn";
    
    // Нормальные значения
    config.gamma = 0.1;
    config.approx_points = {WeightedPoint(0.5, 1.0, 1.0)};
    warning = Validator::check_numerical_anomalies(config);
    EXPECT_TRUE(warning.empty()) << "Normal values should pass";
}

TEST(ValidatorAdvancedTest, ValidateInterfaceCompatibility) {
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
    EXPECT_TRUE(error.empty()) << "Valid config should pass";
    
    // Ошибка
    config.approx_points[0].weight = -1.0;
    error = Validator::validate(config);
    EXPECT_FALSE(error.empty()) << "Invalid weight should fail";
    
    // Strict mode
    config.approx_points[0].weight = 1.0;
    config.gamma = 0.0;
    error = Validator::validate(config, false);
    EXPECT_TRUE(error.empty()) << "Warning without strict mode should pass";
    
    error = Validator::validate(config, true);
    EXPECT_FALSE(error.empty()) << "Warning with strict mode should fail";
}
