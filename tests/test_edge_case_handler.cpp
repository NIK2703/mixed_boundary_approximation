#include <gtest/gtest.h>
#include "mixed_approximation/edge_case_handler.h"
#include "mixed_approximation/parameterization_data.h"

namespace mixed_approx {
namespace test {

// ============== Тесты для NumericalAnomalyMonitor ==============

class NumericalAnomalyMonitorTest : public ::testing::Test {
protected:
    NumericalAnomalyMonitor monitor;
};

TEST_F(NumericalAnomalyMonitorTest, NoAnomaly) {
    monitor.reset();
    
    AnomalyMonitorResult result = monitor.check_anomaly(
        0,                    // iteration
        1.0,                  // current_value
        1.0,                  // gradient_norm
        {0.1, 0.2, 0.3},      // params
        {0.0, 0.0, 0.0}       // prev_params
    );
    
    EXPECT_FALSE(result.anomaly_detected);
    EXPECT_FALSE(result.needs_recovery);
    EXPECT_FALSE(result.needs_stop);
}

TEST_F(NumericalAnomalyMonitorTest, NaNDetected) {
    monitor.reset();
    
    AnomalyMonitorResult result = monitor.check_anomaly(
        0,                    // iteration
        std::numeric_limits<double>::quiet_NaN(),  // current_value
        1.0,                  // gradient_norm
        {0.1, 0.2, 0.3},      // params
        {0.0, 0.0, 0.0}       // prev_params
    );
    
    EXPECT_TRUE(result.anomaly_detected);
    EXPECT_EQ(result.last_anomaly_type, AnomalyType::NAN_DETECTED);
    EXPECT_TRUE(result.needs_recovery);
}

TEST_F(NumericalAnomalyMonitorTest, InfinityDetected) {
    monitor.reset();
    
    AnomalyMonitorResult result = monitor.check_anomaly(
        0,                    // iteration
        std::numeric_limits<double>::infinity(),  // current_value
        1.0,                  // gradient_norm
        {0.1, 0.2, 0.3},      // params
        {0.0, 0.0, 0.0}       // prev_params
    );
    
    EXPECT_TRUE(result.anomaly_detected);
    EXPECT_EQ(result.last_anomaly_type, AnomalyType::INF_DETECTED);
    EXPECT_TRUE(result.needs_recovery);
}

TEST_F(NumericalAnomalyMonitorTest, GradientExplosion) {
    monitor.reset();
    monitor.gradient_threshold = 100.0;
    
    AnomalyMonitorResult result = monitor.check_anomaly(
        0,                    // iteration
        1.0,                  // current_value
        1e12,                 // gradient_norm (exploded)
        {0.1, 0.2, 0.3},      // params
        {0.0, 0.0, 0.0}       // prev_params
    );
    
    EXPECT_TRUE(result.anomaly_detected);
    EXPECT_EQ(result.last_anomaly_type, AnomalyType::GRADIENT_EXPLOSION);
    EXPECT_TRUE(result.needs_recovery);
}

TEST_F(NumericalAnomalyMonitorTest, MultipleAnomalies) {
    monitor.reset();
    monitor.max_consecutive_anomalies = 2;  // Уменьшаем порог для теста
    
    // Вызываем с NaN и бесконечностью одновременно - должны получить 2 аномалии
    AnomalyMonitorResult result = monitor.check_anomaly(
        0,
        std::numeric_limits<double>::quiet_NaN(),  // Первая аномалия: NaN
        std::numeric_limits<double>::infinity(),   // Вторая аномалия: Inf
        {0.1, 0.2, 0.3},
        {0.0, 0.0, 0.0}
    );
    
    // Должны быть 2 аномалии, что превышает порог 2
    EXPECT_EQ(result.anomaly_iterations, 2);
    EXPECT_TRUE(result.needs_stop);
}

// ============== Тесты для EdgeCaseHandler ==============

class EdgeCaseHandlerTest : public ::testing::Test {
protected:
    EdgeCaseHandler handler;
};

TEST_F(EdgeCaseHandlerTest, NoEdgeCases) {
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    config.interp_nodes = {
        {0.0, 0.0},
        {0.5, 0.5},
        {1.0, 1.0}
    };
    
    std::vector<double> interp_values = {0.0, 0.5, 1.0};
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(5, 3, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.has_critical_errors());
}

TEST_F(EdgeCaseHandlerTest, ZeroNodes) {
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.interp_nodes.clear();
    
    std::vector<double> interp_values;
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(5, 0, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.adapted_m, 0);
    EXPECT_TRUE(result.parameters_modified);
    
    const auto& cases = handler.get_detected_cases();
    EXPECT_FALSE(cases.empty());
}

TEST_F(EdgeCaseHandlerTest, FullInterpolation) {
    ApproximationConfig config;
    config.polynomial_degree = 4;  // n = 4
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    // m = n + 1 = 5
    config.interp_nodes = {
        {0.0, 0.0},
        {0.25, 0.25},
        {0.5, 0.5},
        {0.75, 0.75},
        {1.0, 1.0}
    };
    
    std::vector<double> interp_values = {0.0, 0.25, 0.5, 0.75, 1.0};
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(4, 5, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.has_critical_errors());
}

TEST_F(EdgeCaseHandlerTest, OverconstrainedStrict) {
    ApproximationConfig config;
    config.polynomial_degree = 4;  // n = 4, max m = 5
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    // m = 10 > n + 1 = 5
    for (int i = 0; i < 10; ++i) {
        config.interp_nodes.push_back({static_cast<double>(i) / 9.0, static_cast<double>(i) / 9.0});
    }
    
    std::vector<double> interp_values;
    for (int i = 0; i < 10; ++i) {
        interp_values.push_back(static_cast<double>(i) / 9.0);
    }
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(4, 10, interp_values, config);
    
    // Теперь используется STRICT стратегия, должен быть failure
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.has_critical_errors());
}

TEST_F(EdgeCaseHandlerTest, HighDegreePolynomial) {
    ApproximationConfig config;
    config.polynomial_degree = 35;  // n > 30
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    config.interp_nodes = {
        {0.0, 0.0},
        {0.5, 0.5},
        {1.0, 1.0}
    };
    
    std::vector<double> interp_values = {0.0, 0.5, 1.0};
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(35, 3, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.has_warnings());  // Предупреждение о высокой степени
}

TEST_F(EdgeCaseHandlerTest, CloseNodes) {
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    // Создаём очень близкие узлы (расстояние ~1e-9 от интервала)
    config.interp_nodes = {
        {0.0, 0.0},
        {1e-9, 1e-9},  // Очень близко к предыдущему
        {2e-9, 2e-9},  // Очень близко к предыдущим
        {0.5, 0.5},
        {1.0, 1.0}
    };
    
    std::vector<double> interp_values = {0.0, 1e-9, 2e-9, 0.5, 1.0};
    
    CloseNodesResult result = handler.handle_close_nodes(
        {0.0, 1e-9, 2e-9, 0.5, 1.0},
        interp_values,
        1.0
    );
    
    EXPECT_TRUE(result.has_close_nodes);
    EXPECT_LT(result.min_distance, 1e-8);
}

TEST_F(EdgeCaseHandlerTest, ConstantValues) {
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    config.interp_nodes = {
        {0.0, 1.0},
        {0.5, 1.0},
        {1.0, 1.0}
    };
    
    DegeneracyResult result = handler.analyze_degeneracy({1.0, 1.0, 1.0});
    
    EXPECT_TRUE(result.is_degenerate);
    EXPECT_EQ(result.type, DegeneracyType::CONSTANT);
    EXPECT_DOUBLE_EQ(result.constant_value, 1.0);
}

TEST_F(EdgeCaseHandlerTest, NonConstantValues) {
    DegeneracyResult result = handler.analyze_degeneracy({1.0, 2.0, 3.0});
    
    // 1, 2, 3 - разности: 1, 1 - одинаковые, это линейная зависимость!
    // Проверяем что это либо не вырождено, либо линейное
    EXPECT_TRUE(result.is_degenerate);  // Линейная зависимость detected
    EXPECT_EQ(result.type, DegeneracyType::LINEAR);
}

TEST_F(EdgeCaseHandlerTest, LinearValues) {
    // Значения на линии y = 2*x + 1
    DegeneracyResult result = handler.analyze_degeneracy({1.0, 3.0, 5.0, 7.0, 9.0});
    
    // Все разности равны 2, это линейная зависимость
    EXPECT_TRUE(result.is_degenerate);
    EXPECT_EQ(result.type, DegeneracyType::LINEAR);
}

// ============== Тесты для форматтеров ==============

TEST(FormattingTest, ZeroNodesResult) {
    ZeroNodesResult result;
    result.degree = 5;
    result.info_message = "INFO: No interpolation nodes (m = 0).\n";
    
    std::string formatted = format_zero_nodes_result(result);
    
    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("INFO"), std::string::npos);
}

TEST(FormattingTest, FullInterpolationResult) {
    FullInterpolationResult result;
    result.n_free = 0;
    result.recommendations = "Test recommendation";
    result.info_message = "WARNING: Full interpolation";
    
    std::string formatted = format_full_interpolation_result(result);
    
    EXPECT_FALSE(formatted.empty());
}

TEST(FormattingTest, HighDegreeResult) {
    HighDegreeResult result;
    result.original_degree = 35;
    result.requires_adaptation = true;
    result.switch_to_chebyshev = true;
    result.suggest_splines = true;
    result.recommendations = {"Rec 1", "Rec 2"};
    
    std::string formatted = format_high_degree_result(result);
    
    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("35"), std::string::npos);
}

TEST(FormattingTest, DegeneracyResult) {
    DegeneracyResult result;
    result.is_degenerate = true;
    result.type = DegeneracyType::CONSTANT;
    result.constant_value = 2.5;
    
    std::string formatted = format_degeneracy_result(result);
    
    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("YES"), std::string::npos);
}

TEST(FormattingTest, EdgeCaseHandlingResult) {
    EdgeCaseHandlingResult result;
    result.success = true;
    result.adapted_m = 3;
    result.adapted_n = 5;
    
    EdgeCaseInfo info;
    info.level = EdgeCaseLevel::WARNING;
    info.message = "Test warning";
    result.detected_cases.push_back(info);
    
    result.warnings.push_back("Test warning message");
    
    std::string formatted = format_edge_case_result(result);
    
    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("SUCCESS"), std::string::npos);
}

// ============== Тесты для обработчика близких узлов ==============

TEST(CloseNodesTest, NoCloseNodes) {
    EdgeCaseHandler handler;
    
    CloseNodesResult result = handler.handle_close_nodes(
        {0.0, 0.3, 0.6, 1.0},
        {0.0, 0.3, 0.6, 1.0},
        1.0
    );
    
    EXPECT_FALSE(result.has_close_nodes);
    EXPECT_EQ(result.original_m, 4);
    EXPECT_EQ(result.effective_m, 4);
}

TEST(CloseNodesTest, AllCloseNodes) {
    EdgeCaseHandler handler;
    
    // Все узлы очень близки
    CloseNodesResult result = handler.handle_close_nodes(
        {0.0, 1e-9, 2e-9, 3e-9},
        {0.0, 0.0, 0.0, 0.0},
        1.0
    );
    
    EXPECT_TRUE(result.has_close_nodes);
    EXPECT_LT(result.effective_m, result.original_m);
}

// ============== Тесты для обработки высокой степени ==============

TEST(HighDegreeTest, NormalDegree) {
    EdgeCaseHandler handler;
    
    HighDegreeResult result = handler.handle_high_degree(10);
    
    EXPECT_FALSE(result.requires_adaptation);
    EXPECT_FALSE(result.switch_to_chebyshev);
}

TEST(HighDegreeTest, HighDegree) {
    EdgeCaseHandler handler;
    
    HighDegreeResult result = handler.handle_high_degree(35);
    
    EXPECT_TRUE(result.requires_adaptation);
    EXPECT_TRUE(result.switch_to_chebyshev);
}

TEST(HighDegreeTest, VeryHighDegree) {
    EdgeCaseHandler handler;
    
    HighDegreeResult result = handler.handle_high_degree(50);
    
    EXPECT_TRUE(result.requires_adaptation);
    EXPECT_TRUE(result.switch_to_chebyshev);
    EXPECT_TRUE(result.suggest_splines);
}

// ============== Интеграционные тесты ==============

TEST(IntegrationTest, FullParameterizationWithEdgeCases) {
    EdgeCaseHandler handler;
    ApproximationConfig config;
    
    // Тест с обычными параметрами
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.interp_nodes = {{0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0}};
    
    std::vector<double> interp_values = {0.0, 0.5, 1.0};
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(5, 3, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.has_critical_errors());
}

TEST(IntegrationTest, ZeroNodesWithHighDegree) {
    EdgeCaseHandler handler;
    ApproximationConfig config;
    
    config.polynomial_degree = 35;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.interp_nodes.clear();
    
    std::vector<double> interp_values;
    
    EdgeCaseHandlingResult result = handler.handle_all_cases(35, 0, interp_values, config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.adapted_m, 0);
    EXPECT_TRUE(result.has_warnings());  // Предупреждение о высокой степени
}

} // namespace test
} // namespace mixed_approx
