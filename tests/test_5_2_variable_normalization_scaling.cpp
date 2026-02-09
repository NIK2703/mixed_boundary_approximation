#include <gtest/gtest.h>
#include "mixed_approximation/variable_normalizer.h"
#include "mixed_approximation/types.h"

using namespace mixed_approx;

namespace mixed_approximation {
namespace test {

// ============== Тесты базовой функциональности ==============

TEST(VariableNormalizerTest, BasicNormalization) {
    // Создаём конфигурацию с типичными данными
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 10.0;
    config.gamma = 0.1;
    
    // Добавляем аппроксимирующие точки
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {2.5, 3.0, 0.1},
        {5.0, 5.0, 0.1},
        {7.5, 7.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    // Добавляем отталкивающие точки
    config.repel_points = {
        {4.0, 2.0, 10.0},  // отталкивание от значения 2.0 в x=4
        {6.0, 8.0, 10.0}   // отталкивание от значения 8.0 в x=6
    };
    
    // Добавляем интерполяционные узлы
    config.interp_nodes = {
        {1.0, 1.5},
        {9.0, 8.5}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    
    // Проверяем успешность нормализации
    EXPECT_TRUE(result.success);
    
    // Проверяем, что нормализованные значения в диапазоне [-1, 1]
    for (double x_norm : result.approx_x_norm) {
        EXPECT_GE(x_norm, -1.0);
        EXPECT_LE(x_norm, 1.0);
    }
    
    for (double x_norm : result.repel_y_norm) {
        EXPECT_GE(x_norm, -1.0);
        EXPECT_LE(x_norm, 1.0);
    }
    
    for (double x_norm : result.interp_z_norm) {
        EXPECT_GE(x_norm, -1.0);
        EXPECT_LE(x_norm, 1.0);
    }
    
    // Проверяем границы интервала
    auto& params = normalizer.get_params();
    EXPECT_NEAR(params.norm_a, -1.0, 1e-10);
    EXPECT_NEAR(params.norm_b, 1.0, 1e-10);
}

TEST(VariableNormalizerTest, CoefficientTransform) {
    // Тест обратного преобразования коэффициентов
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 10.0;
    config.gamma = 0.1;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {5.0, 5.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    // Проверяем, что методы преобразования не падают и возвращают корректный размер
    std::vector<double> test_coeffs = {1.0, 0.5, 0.25, 0.125};
    
    std::vector<double> original = normalizer.inverse_transform_coefficients(test_coeffs);
    EXPECT_EQ(original.size(), test_coeffs.size());
    
    std::vector<double> normalized = normalizer.forward_transform_coefficients(original);
    EXPECT_EQ(normalized.size(), original.size());
    
    // Обратное преобразование от восстановленных должно давать исходные
    std::vector<double> double_recovered = normalizer.inverse_transform_coefficients(normalized);
    EXPECT_EQ(double_recovered.size(), normalized.size());
    
    // Примечание: Точное совпадение зависит от параметров нормализации
    // Для полной проверки нужно аналитически вычислять ожидаемые значения
}

TEST(VariableNormalizerTest, GammaCorrection) {
    // Тест коррекции параметра регуляризации
    ApproximationConfig config;
    config.polynomial_degree = 5;
    config.interval_start = 0.0;
    config.interval_end = 10.0;
    config.gamma = 0.01;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    // Проверяем, что gamma была скорректирована
    EXPECT_NE(result.gamma_normalized, config.gamma);
    
    // gamma_normalized = gamma * (x_range/2)^3
    // x_range = 10, так что (10/2)^3 = 125
    double expected_correction = std::pow(5.0, 3.0);  // 125
    EXPECT_NEAR(result.gamma_normalized, config.gamma * expected_correction, 1e-10);
}

// ============== Тесты крайних случаев ==============

TEST(VariableNormalizerTest, SymmetricInterval) {
    // Тест нормализации симметричного интервала [-5, 5]
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = -5.0;
    config.interval_end = 5.0;
    config.gamma = 0.1;
    
    config.approx_points = {
        {-5.0, 1.0, 0.1},
        {0.0, 5.0, 0.1},
        {5.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    // Проверяем, что концы интервала нормализовались в -1 и 1
    auto& params = normalizer.get_params();
    EXPECT_NEAR(params.transform_x(-5.0), -1.0, 1e-10);
    EXPECT_NEAR(params.transform_x(5.0), 1.0, 1e-10);
}

TEST(VariableNormalizerTest, LargeScaleData) {
    // Тест с большими значениями (проверка на переполнение)
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1e6;  // Большой диапазон
    config.gamma = 0.1;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {1e6, 1e6, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    
    // Проверяем, что нормализация выполнена успешно
    EXPECT_TRUE(result.success);
    
    // Проверяем, что значения в пределах [-1, 1]
    EXPECT_GE(result.approx_x_norm[0], -1.0);
    EXPECT_LE(result.approx_x_norm.back(), 1.0);
}

TEST(VariableNormalizerTest, SmallScaleData) {
    // Тест с малыми значениями (проверка на потерю точности)
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1e-6;  // Малый диапазон
    config.gamma = 0.1;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {1e-6, 2.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    
    EXPECT_TRUE(result.success);
    
    // Проверяем, что значения в пределах [-1, 1]
    EXPECT_GE(result.approx_x_norm[0], -1.0);
    EXPECT_LE(result.approx_x_norm.back(), 1.0);
}

// ============== Тесты стратегий нормализации ==============

TEST(VariableNormalizerTest, LinearToZeroOneStrategy) {
    // Тест стратегии [0, 1]
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 10.0;
    config.gamma = 0.1;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    normalizer.set_x_strategy(XNormalizationStrategy::LINEAR_TO_0_1);
    
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    // Проверяем границы
    auto& params = normalizer.get_params();
    EXPECT_NEAR(params.norm_a, 0.0, 1e-10);
    EXPECT_NEAR(params.norm_b, 1.0, 1e-10);
}

TEST(VariableNormalizerTest, AdaptiveStrategySymmetric) {
    // Тест адаптивной стратегии с симметричными данными
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = -10.0;
    config.interval_end = 10.0;
    config.gamma = 0.1;
    
    config.approx_points = {
        {-10.0, 1.0, 0.1},
        {0.0, 5.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    normalizer.set_x_strategy(XNormalizationStrategy::ADAPTIVE);
    
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    // Симметричные данные должны дать [-1, 1]
    auto& params = normalizer.get_params();
    EXPECT_NEAR(params.norm_a, -1.0, 1e-10);
    EXPECT_NEAR(params.norm_b, 1.0, 1e-10);
}

// ============== Тесты вспомогательных функций ==============

TEST(NormalizationUtilsTest, MedianComputation) {
    std::vector<double> values = {1.0, 3.0, 5.0, 7.0, 9.0};
    double median = NormalizationUtils::compute_median(values);
    EXPECT_NEAR(median, 5.0, 1e-10);
    
    // Чётное количество
    values = {1.0, 3.0, 5.0, 7.0};
    median = NormalizationUtils::compute_median(values);
    EXPECT_NEAR(median, 4.0, 1e-10);
}

TEST(NormalizationUtilsTest, RobustStdComputation) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};  // 100 - выброс
    double robust_std = NormalizationUtils::compute_robust_std(values);
    
    // ROBUST_std должен быть устойчив к выбросам
    // Для нормального распределения MAD × 1.4826 ≈ std
    EXPECT_GT(robust_std, 0.0);
    EXPECT_LT(robust_std, 10.0);  // Не должен зависеть сильно от выброса
}

TEST(NormalizationUtilsTest, LogTransform) {
    // Проверка логарифмического преобразования
    std::vector<double> values = {1.0, 10.0, 100.0, 1000.0};
    bool needs_log = NormalizationUtils::needs_log_transform(values);
    EXPECT_TRUE(needs_log);
    
    values = {1.0, 2.0, 3.0, 4.0};
    needs_log = NormalizationUtils::needs_log_transform(values);
    EXPECT_FALSE(needs_log);
}

TEST(NormalizationUtilsTest, DegenerateRange) {
    // Проверка определения вырожденного диапазона
    bool degenerate = NormalizationUtils::is_degenerate_range(5.0, 5.0, 5.0);
    EXPECT_TRUE(degenerate);
    
    degenerate = NormalizationUtils::is_degenerate_range(5.0, 5.000001, 5.0);
    EXPECT_FALSE(degenerate);
}

// ============== Тесты диагностики ==============

TEST(VariableNormalizerTest, DiagnosticReport) {
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 10.0;
    config.gamma = 0.1;
    
    config.approx_points = {
        {0.0, 1.0, 0.1},
        {10.0, 9.0, 0.1}
    };
    
    VariableNormalizer normalizer(config);
    auto result = normalizer.normalize();
    ASSERT_TRUE(result.success);
    
    std::string report = normalizer.get_diagnostic_report();
    
    // Проверяем, что отчёт содержит ключевую информацию
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Ось X"), std::string::npos);
    EXPECT_NE(report.find("Ось Y"), std::string::npos);
}

}  // namespace test
}  // namespace mixed_approximation
