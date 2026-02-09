#include <gtest/gtest.h>
#include <cmath>
#include "mixed_approximation/barrier_safety_monitor.h"
#include "mixed_approximation/types.h"

namespace mixed_approx {

// Тест 5.1.1: Базовая инициализация и классификация зон
TEST(BarrierSafetyMonitorTest, BasicInitialization) {
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0),
        RepulsionPoint(2.0, 3.0, 50.0),
        RepulsionPoint(3.0, 4.0, 200.0)
    };
    
    BarrierSafetyConfig config;
    BarrierSafetyMonitor monitor(repel_points, config);
    
    // Проверка начального состояния
    EXPECT_EQ(monitor.get_adaptive_thresholds().first, 1e-8);
    EXPECT_EQ(monitor.get_adaptive_thresholds().second, 1e-4);
}

// Тест 5.1.2: Классификация зон барьера
TEST(BarrierSafetyMonitorTest, ZoneClassification) {
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points);
    
    // Создаём distances для проверки
    std::vector<double> distances_critical = {1e-9};   // < epsilon_critical (1e-8)
    std::vector<double> distances_warning = {1e-5};    // > epsilon_critical, < epsilon_warning
    std::vector<double> distances_safe = {1e-3};       // > epsilon_warning
    
    std::vector<double> weights = {100.0};
    
    auto result_critical = monitor.check_safety(distances_critical, weights, 0.0);
    auto result_warning = monitor.check_safety(distances_warning, weights, 0.0);
    auto result_safe = monitor.check_safety(distances_safe, weights, 0.0);
    
    EXPECT_EQ(result_critical.critical_count, 1);
    EXPECT_EQ(result_warning.warning_count, 1);
    EXPECT_EQ(result_safe.critical_count, 0);
    EXPECT_EQ(result_safe.warning_count, 0);
}

// Тест 5.1.3: Сглаживание барьера
TEST(BarrierSafetyMonitorTest, BarrierSmoothing) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    // Тестируем сглаженный барьер в критической зоне
    BarrierZone zone;
    double dist_critical = 1e-9;
    double term_critical = monitor.compute_smoothed_barrier(dist_critical, 100.0, zone);
    
    EXPECT_EQ(zone, BarrierZone::CRITICAL);
    EXPECT_FALSE(std::isnan(term_critical));
    EXPECT_FALSE(std::isinf(term_critical));
    
    // Тестируем в предупредительной зоне
    double dist_warning = 1e-5;
    double term_warning = monitor.compute_smoothed_barrier(dist_warning, 100.0, zone);
    
    EXPECT_EQ(zone, BarrierZone::WARNING);
    EXPECT_GT(term_warning, 0.0);
    
    // Тестируем в безопасной зоне
    double dist_safe = 1e-3;
    double term_safe = monitor.compute_smoothed_barrier(dist_safe, 100.0, zone);
    
    EXPECT_EQ(zone, BarrierZone::SAFE);
    EXPECT_GT(term_safe, 0.0);
}

// Тест 5.1.4: Защита градиента
TEST(BarrierSafetyMonitorTest, GradientProtection) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points(5, RepulsionPoint(1.0, 2.0, 100.0));
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    // Создаём градиент с взрывными значениями
    std::vector<double> raw_gradient = {1e20, 2e20, -1.5e20, 8e19, -3e20};
    std::vector<double> distances = {1e-8, 1e-8, 1e-8, 1e-8, 1e-8};
    
    auto protected_gradient = monitor.protect_gradient(raw_gradient, distances);
    
    // Проверяем, что градиент ограничен
    double max_component = 0.0;
    for (double g : protected_gradient) {
        max_component = std::max(max_component, std::abs(g));
    }
    
    EXPECT_LT(max_component, config.gradient_max_per_component);
    
    // Проверяем, что знаки сохранены
    for (size_t i = 0; i < raw_gradient.size(); ++i) {
        EXPECT_EQ(
            (raw_gradient[i] >= 0) == (protected_gradient[i] >= 0),
            true
        );
    }
}

// Тест 5.1.5: Обнаружение коллапса
TEST(BarrierSafetyMonitorTest, CollapseDetection) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    std::vector<double> distances = {1e-9};
    
    // Нормальный функционал
    EXPECT_FALSE(monitor.detect_collapse(100.0, 99.0, distances));
    
    // Резкий скачок функционала (признак коллапса)
    EXPECT_TRUE(monitor.detect_collapse(10000.0, 100.0, distances));
    
    // Слишком маленькое расстояние (признак коллапса)
    std::vector<double> collapse_distances = {1e-12};
    EXPECT_TRUE(monitor.detect_collapse(100.0, 99.0, collapse_distances));
}

// Тест 5.1.6: Восстановление после коллапса
TEST(BarrierSafetyMonitorTest, CollapseRecovery) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points(5, RepulsionPoint(1.0, 2.0, 100.0));
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    std::vector<double> current_params = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> safe_params = {0.5, 1.5, 2.5, 3.5, 4.5};
    std::vector<double> distances = {1e-9, 1e-9, 1e-9, 1e-9, 1e-9};
    std::vector<double> weights(5, 100.0);
    
    auto result = monitor.recover_from_collapse(current_params, safe_params, distances, weights);
    
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.rollback_performed);
    EXPECT_EQ(result.corrected_params, safe_params);
}

// Тест 5.1.7: Предотвращающая коррекция
TEST(BarrierSafetyMonitorTest, PreventiveCorrection) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0),
        RepulsionPoint(2.0, 3.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    std::vector<double> distances = {1e-10, 1e-3};  // Первый слишком близок
    std::vector<double> weights = {100.0, 100.0};
    
    bool corrected = monitor.apply_preventive_correction(distances, weights, 1.0);
    
    EXPECT_TRUE(corrected);
    EXPECT_GE(distances[0], 5.0 * 1e-8);  // Должен быть увеличен до безопасного уровня
}

// Тест 5.1.8: Адаптивные пороги
TEST(BarrierSafetyMonitorTest, AdaptiveThresholds) {
    BarrierSafetyConfig config;
    config.scale_factor_small = 1e-10;
    config.scale_factor_large = 1e-6;
    
    ApproximationConfig app_config;
    app_config.repel_points.push_back(RepulsionPoint(1.0, 1000.0, 100.0));  // Большое значение
    
    BarrierSafetyConfig adaptive_config = BarrierSafetyConfig::create_adaptive(app_config);
    
    // Проверяем, что адаптивные пороги больше базовых для больших масштабов
    EXPECT_GE(adaptive_config.epsilon_critical_base, 1e-8);
    EXPECT_GE(adaptive_config.epsilon_warning_base, 1e-4);
}

// Тест 5.1.9: Интеграция с несколькими барьерами
TEST(BarrierSafetyMonitorTest, MultipleBarriersIntegration) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0),
        RepulsionPoint(2.0, 3.0, 200.0),
        RepulsionPoint(3.0, 4.0, 50.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    // Разные расстояния для разных барьеров
    std::vector<double> distances = {1e-9, 1e-5, 1e-3};
    std::vector<double> weights = {100.0, 200.0, 50.0};
    
    auto result = monitor.check_safety(distances, weights, 0.0);
    
    EXPECT_EQ(result.critical_count, 1);   // Только первый барьер критический
    EXPECT_EQ(result.warning_count, 1);     // Только второй предупредительный
    EXPECT_EQ(result.is_safe, false);       // Не безопасно из-за критического барьера
    
    // Проверяем вычисление оценки опасности
    EXPECT_GT(result.danger_score, 0.0);
}

// Тест 5.1.10: Генерация отчёта
TEST(BarrierSafetyMonitorTest, ReportGeneration) {
    BarrierSafetyConfig config;
    config.enable_logging = true;
    
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    std::vector<double> distances = {1e-9};
    std::vector<double> weights = {100.0};
    
    monitor.check_safety(distances, weights, 0.0);
    
    std::string report = monitor.generate_report();
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Barrier Safety Monitor Report"), std::string::npos);
}

// Тест 5.1.11: Пустой список барьеров
TEST(BarrierSafetyMonitorTest, EmptyBarrierList) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points;
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    std::vector<double> distances;
    std::vector<double> weights;
    
    auto result = monitor.check_safety(distances, weights, 0.0);
    
    EXPECT_TRUE(result.is_safe);
    EXPECT_EQ(result.critical_count, 0);
    EXPECT_EQ(result.warning_count, 0);
}

// Тест 5.1.12: Сброс состояния
TEST(BarrierSafetyMonitorTest, StateReset) {
    BarrierSafetyConfig config;
    std::vector<RepulsionPoint> repel_points = {
        RepulsionPoint(1.0, 2.0, 100.0)
    };
    
    BarrierSafetyMonitor monitor(repel_points, config);
    
    // Вызываем check_safety для заполнения истории
    std::vector<double> distances = {1e-9};
    std::vector<double> weights = {100.0};
    monitor.check_safety(distances, weights, 0.0);
    
    // Сбрасываем состояние
    monitor.reset();
    
    // Проверяем, что состояние сброшено
    EXPECT_EQ(monitor.get_adaptive_thresholds().first, config.epsilon_critical_base);
    EXPECT_EQ(monitor.get_adaptive_thresholds().second, config.epsilon_warning_base);
    EXPECT_TRUE(monitor.get_events().empty());
}

} // namespace mixed_approx
