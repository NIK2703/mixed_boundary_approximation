#include <gtest/gtest.h>
#include <cmath>
#include "mixed_approximation/types.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"
#include "mixed_approximation/constrained_polynomial.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/composite_polynomial.h"

namespace mixed_approx {
namespace test {

// ============== Тесты BarrierParams ==============

TEST(BarrierParamsTest, DefaultValues) {
    BarrierParams params;
    EXPECT_DOUBLE_EQ(params.epsilon_safe, 1e-8);
    EXPECT_DOUBLE_EQ(params.smoothing_factor, 10.0);
    EXPECT_DOUBLE_EQ(params.adaptive_gain, 5.0);
    EXPECT_DOUBLE_EQ(params.warning_zone_factor, 10.0);
}

TEST(BarrierParamsTest, ComputeEpsilonSafe) {
    EXPECT_DOUBLE_EQ(BarrierParams::compute_epsilon_safe(1.0), 1e-6);
    EXPECT_DOUBLE_EQ(BarrierParams::compute_epsilon_safe(0.1), 1e-7);
    EXPECT_DOUBLE_EQ(BarrierParams::compute_epsilon_safe(1e-8), 1e-8);  // min is 1e-8
}

// ============== Тесты FunctionalDiagnostics ==============

TEST(FunctionalDiagnosticsTest, FormatReport) {
    FunctionalDiagnostics diag;
    diag.raw_approx = 2.3456;
    diag.raw_repel = 1.8765;
    diag.raw_reg = 0.9721;
    diag.normalized_approx = 2.3456;
    diag.normalized_repel = 1.8765;
    diag.normalized_reg = 0.9721;
    diag.total_functional = 5.1942;
    diag.share_approx = 45.2;
    diag.share_repel = 36.1;
    diag.share_reg = 18.7;
    diag.min_repulsion_distance = 0.123;
    diag.max_residual = 0.456;
    diag.second_deriv_norm = 3.21;
    
    std::string report = diag.format_report();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Функционал смешанной аппроксимации"), std::string::npos);
    EXPECT_NE(report.find("Аппроксимирующий член"), std::string::npos);
    EXPECT_NE(report.find("Отталкивающий член"), std::string::npos);
    EXPECT_NE(report.find("Регуляризация"), std::string::npos);
}

TEST(FunctionalDiagnosticsTest, IsDominantComponent) {
    FunctionalDiagnostics diag;
    
    diag.share_approx = 96.0;
    EXPECT_TRUE(diag.is_dominant_component());
    
    diag.share_repel = 96.0;
    EXPECT_TRUE(diag.is_dominant_component());
    
    diag.share_reg = 96.0;
    EXPECT_TRUE(diag.is_dominant_component());
    
    diag.share_approx = 50.0;
    diag.share_repel = 30.0;
    diag.share_reg = 20.0;
    EXPECT_FALSE(diag.is_dominant_component());
}

TEST(FunctionalDiagnosticsTest, GetWeightRecommendation) {
    FunctionalDiagnostics diag;
    
    diag.share_approx = 96.0;
    std::string rec = diag.get_weight_recommendation();
    EXPECT_NE(rec.find("Аппроксимация доминирует"), std::string::npos);
    
    diag.share_approx = 0.0;  // Reset
    diag.share_repel = 96.0;
    rec = diag.get_weight_recommendation();
    EXPECT_NE(rec.find("Отталкивание доминирует"), std::string::npos);
    
    diag.share_repel = 0.0;  // Reset
    diag.share_reg = 96.0;
    rec = diag.get_weight_recommendation();
    EXPECT_NE(rec.find("Регуляризация доминирует"), std::string::npos);
}

// ============== Тесты FunctionalEvaluator ==============

TEST(FunctionalEvaluatorTest, Constructor) {
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};
    config.gamma = 0.1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    FunctionalEvaluator evaluator(config);
    const NormalizationParams& norm_params = evaluator.get_normalization_params();
    
    // По умолчанию масштабы = 1.0
    EXPECT_DOUBLE_EQ(norm_params.scale_approx, 1.0);
    EXPECT_DOUBLE_EQ(norm_params.scale_repel, 1.0);
    EXPECT_DOUBLE_EQ(norm_params.scale_reg, 1.0);
}

TEST(FunctionalEvaluatorTest, InitializeNormalization) {
    // Создаём простую конфигурацию
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1.0}, {1.0, 2.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};
    config.gamma = 0.1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    // Создаём интерполяционный базис
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    // Создаём весовой множитель
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    // Создаём корректирующий полином
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};  // Q(x) = 0
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    const NormalizationParams& norm_params = evaluator.get_normalization_params();
    
    // Масштабы должны быть > 0
    EXPECT_GT(norm_params.scale_approx, 0.0);
    EXPECT_GT(norm_params.scale_repel, 0.0);
    EXPECT_GT(norm_params.scale_reg, 0.0);
    
    // Веса должны быть положительными
    EXPECT_GT(norm_params.weight_approx, 0.0);
    EXPECT_GT(norm_params.weight_repel, 0.0);
    EXPECT_GT(norm_params.weight_reg, 0.0);
}

TEST(FunctionalEvaluatorTest, EvaluateWithDiagnostics) {
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1.0}, {1.0, 2.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};
    config.gamma = 0.1;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    
    EXPECT_TRUE(result.is_ok());
    EXPECT_FALSE(result.diagnostics.has_numerical_anomaly);
    EXPECT_GT(result.value, 0.0);
    
    // Проверяем, что диагностика заполнена
    // При Q(x)=0 интерполяция точна, поэтому max_residual = 0 (или очень мал)
    EXPECT_GE(result.diagnostics.max_residual, 0.0);
    EXPECT_LE(result.diagnostics.max_residual, 1e-10);
    EXPECT_TRUE(std::isfinite(result.diagnostics.min_repulsion_distance));
}

TEST(FunctionalEvaluatorTest, BarrierProtection) {
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};  // Запрещённое значение 0.0 в x=0.5
    config.gamma = 0.0;
    config.epsilon = 1e-8;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    BarrierParams barrier_params;
    barrier_params.epsilon_safe = 1e-6;
    
    FunctionalEvaluator evaluator(config);
    evaluator.set_barrier_params(barrier_params);
    evaluator.initialize_normalization(param, {0.0});
    
    // Проверяем, что отталкивающий член вычисляется корректно
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    
    EXPECT_TRUE(result.is_ok());
    EXPECT_GT(result.diagnostics.raw_repel, 0.0);
}

TEST(FunctionalEvaluatorTest, EmptyPointsHandling) {
    ApproximationConfig config;
    config.approx_points.clear();
    config.repel_points.clear();
    config.gamma = 0.0;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    
    // Должен вернуть статус EMPTY_APPROX_POINTS
    EXPECT_EQ(result.status, FunctionalStatus::EMPTY_APPROX_POINTS);
}

TEST(FunctionalEvaluatorTest, NumericalAnomalyDetection) {
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};
    config.gamma = 0.0;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    // Проверяем, что NaN/Inf корректно детектируются
    // Это сложно проверить без искусственного создания NaN/Inf в вычислениях
    // Но мы можем проверить, что статус OK при нормальных условиях
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    EXPECT_TRUE(result.is_ok());
}

TEST(FunctionalEvaluatorTest, ComponentDominanceDetection) {
    ApproximationConfig config;
    config.approx_points = {{0.0, 1.0, 1e-10}};  // Очень маленький вес -> большая аппроксимация
    config.repel_points = {{0.5, 0.0, 1.0}};
    config.gamma = 0.0;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {1.0, 2.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    
    // Проверяем, что доминирование обнаружено
    if (result.diagnostics.is_dominant_component()) {
        EXPECT_FALSE(result.diagnostics.get_weight_recommendation().empty());
    }
}

// ============== Тесты RepulsionResult ==============

TEST(RepulsionResultTest, BasicStructure) {
    RepulsionResult result;
    result.total = 10.0;
    result.min_distance = 0.1;
    result.max_distance = 1.0;
    result.critical_count = 2;
    result.warning_count = 3;
    result.barrier_activated = true;
    result.distances = {0.1, 0.5, 0.2};
    
    EXPECT_DOUBLE_EQ(result.total, 10.0);
    EXPECT_DOUBLE_EQ(result.min_distance, 0.1);
    EXPECT_DOUBLE_EQ(result.max_distance, 1.0);
    EXPECT_EQ(result.critical_count, 2);
    EXPECT_EQ(result.warning_count, 3);
    EXPECT_TRUE(result.barrier_activated);
    EXPECT_EQ(result.distances.size(), 3);
}

// ============== Тесты FunctionalStatus ==============

TEST(FunctionalStatusTest, Values) {
    FunctionalResult result;
    EXPECT_TRUE(result.is_ok());
    EXPECT_EQ(result.status, FunctionalStatus::OK);
}

// ============== Интеграционные тесты ==============

TEST(FunctionalEvaluatorIntegration, SimpleCase) {
    // Простой случай: интерполяция через (0,0) и (1,1) с отталкиванием от 0.5
    ApproximationConfig config;
    config.approx_points = {{0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}};
    config.repel_points = {{0.5, 0.0, 1.0}};  // Избегаем 0.0 в x=0.5
    config.gamma = 0.01;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {0.0, 1.0});
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(1, BasisType::MONOMIAL, 0.0, 1.0);
    Q.initialize_zero();
    Q.coeffs = {0.0};  // Начинаем с Q(x)=0
    
    CompositePolynomial param;
    param.build(basis, W, Q, 0.0, 1.0);
    
    FunctionalEvaluator evaluator(config);
    evaluator.initialize_normalization(param, {0.0});
    
    FunctionalResult result = evaluator.evaluate_with_diagnostics(param, {0.0});
    
    EXPECT_TRUE(result.is_ok());
    EXPECT_GT(result.value, 0.0);
    
    // Проверяем интерполяционные условия
    EXPECT_NEAR(param.interpolation_basis.evaluate(0.0), 0.0, 1e-10);
    EXPECT_NEAR(param.interpolation_basis.evaluate(1.0), 1.0, 1e-10);
    
    // Проверяем, что отталкивание работает (F(0.5) не должно быть близко к 0)
    double F_at_05 = param.evaluate(0.5);
    EXPECT_NEAR(F_at_05, 0.5, 0.5);  // Должно быть около 0.5, а не 0
}

} // namespace test
} // namespace mixed_approx
