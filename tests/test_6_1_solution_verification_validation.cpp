#include <gtest/gtest.h>
#include "mixed_approximation/solution_validator.h"
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/optimization_problem_data.h"
#include <cmath>

using namespace mixed_approx;

// ============== Тесты базовой валидации ==============

TEST(BasicValidationTest, ValidLinearPolynomial) {
    // Линейный полином F(x) = 2x + 1
    Polynomial poly({2.0, 1.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {1.0, 2.0, 3.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.numerical_correct);
    EXPECT_TRUE(result.interpolation_ok);
    EXPECT_TRUE(result.barriers_safe);
    EXPECT_TRUE(result.physically_plausible);
    EXPECT_TRUE(result.numerically_stable);
}

TEST(BasicValidationTest, ValidQuadraticPolynomial) {
    // Квадратичный полином F(x) = x²
    Polynomial poly({1.0, 0.0, 0.0});
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.approx_x = {-1.0, 0.0, 1.0};
    data.approx_f = {1.0, 0.0, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.numerical_correct);
}

TEST(BasicValidationTest, InterpolationViolation) {
    // Полином, который не проходит через интерполяционные узлы
    Polynomial poly({1.0, 0.0, 0.0});  // F(x) = x²
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.interp_z = {0.0};  // Узел в точке 0
    data.interp_f = {1.0};  // Но ожидаем значение 1.0
    // Однако x² в точке 0 даёт 0, не 1
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    EXPECT_FALSE(result.interpolation_ok);
    EXPECT_NEAR(result.max_interpolation_error, 1.0, 1e-10);
}

TEST(BasicValidationTest, BarrierSafetyViolation) {
    // Полином, который слишком близко подходит к запрещённой точке
    Polynomial poly({1.0, 0.0});  // F(x) = x
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 2.0;
    data.repel_y = {1.0};           // Точка отталкивания
    data.repel_forbidden = {1.0};   // Запрещённое значение
    data.repel_weight = {100.0};    // Большой вес
    
    SolutionValidator validator(1e-3);  // Маленький порог
    auto result = validator.validate(poly, data);
    
    // F(1) = 1, запрещённое значение = 1, расстояние = 0
    EXPECT_FALSE(result.barriers_safe);
}

// ============== Тесты интерполяции ==============

TEST(InterpolationTest, NoInterpolationNodes) {
    Polynomial poly({1.0, 2.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    
    SolutionValidator validator;
    double max_error;
    bool ok = validator.check_interpolation(poly, data, max_error);
    
    EXPECT_TRUE(ok);
    EXPECT_EQ(max_error, 0.0);
}

TEST(InterpolationTest, PerfectInterpolation) {
    // Полином, точно проходящий через узлы
    Polynomial poly({1.0, 0.0, 0.0});  // F(x) = x²
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.interp_z = {-1.0, 0.0, 1.0};
    data.interp_f = {1.0, 0.0, 1.0};  // x² в этих точках
    
    SolutionValidator validator;
    double max_error;
    bool ok = validator.check_interpolation(poly, data, max_error);
    
    EXPECT_TRUE(ok);
    EXPECT_NEAR(max_error, 0.0, 1e-12);
}

// ============== Тесты безопасности барьеров ==============

TEST(BarrierSafetyTest, NoRepelPoints) {
    Polynomial poly({1.0, 2.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    
    SolutionValidator validator;
    double min_distance;
    bool safe = validator.check_barrier_safety(poly, data, min_distance);
    
    EXPECT_TRUE(safe);
    EXPECT_TRUE(std::isinf(min_distance));
}

TEST(BarrierSafetyTest, SafeDistance) {
    Polynomial poly({1.0, 0.0});  // F(x) = x
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 2.0;
    data.repel_y = {1.0};           // Точка
    data.repel_forbidden = {10.0}; // Запрещённое значение далеко
    data.repel_weight = {1.0};
    
    SolutionValidator validator;
    double min_distance;
    bool safe = validator.check_barrier_safety(poly, data, min_distance);
    
    EXPECT_TRUE(safe);
    EXPECT_NEAR(min_distance, 9.0, 1e-10);
}

// ============== Тесты численной корректности ==============

TEST(NumericalCorrectnessTest, ValidPolynomial) {
    Polynomial poly({1.0, 2.0, 3.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {3.0, 4.75, 9.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    bool correct = validator.check_numerical_correctness(poly, data);
    
    EXPECT_TRUE(correct);
}

TEST(NumericalCorrectnessTest, NaNInCoefficients) {
    std::vector<double> coeffs = {1.0, std::nan(""), 0.0};
    Polynomial poly(coeffs);
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    
    SolutionValidator validator;
    bool correct = validator.check_numerical_correctness(poly, data);
    
    EXPECT_FALSE(correct);
}

// ============== Тесты физической правдоподобности ==============

TEST(PhysicalPlausibilityTest, NormalValues) {
    Polynomial poly({1.0, 0.0, 0.0});  // F(x) = x²
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.approx_x = {-1.0, 0.0, 1.0};
    data.approx_f = {1.0, 0.0, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    double max_value;
    bool plausible = validator.check_physical_plausibility(poly, data, max_value);
    
    EXPECT_TRUE(plausible);
    EXPECT_NEAR(max_value, 1.0, 1e-10);
}

// ============== Тесты численной стабильности ==============

TEST(NumericalStabilityTest, StablePolynomial) {
    Polynomial poly({1.0, 0.0, 0.0});  // F(x) = x²
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.approx_x = {-1.0, 0.0, 1.0};
    data.approx_f = {1.0, 0.0, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    bool stable = validator.check_numerical_stability(poly, data);
    
    EXPECT_TRUE(stable);
}

// ============== Тесты анализа баланса ==============

TEST(FunctionalBalanceTest, BasicFunctionality) {
    Polynomial poly({1.0, 0.0});  // F(x) = x
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.gamma = 0.1;
    
    SolutionValidator validator;
    FunctionalDiagnostics diagnostics;
    bool success = validator.analyze_functional_balance(poly, data, diagnostics);
    
    EXPECT_TRUE(success);
    EXPECT_GE(diagnostics.share_approx, 0.0);
    EXPECT_GE(diagnostics.share_repel, 0.0);
    EXPECT_GE(diagnostics.share_reg, 0.0);
}

TEST(FunctionalBalanceTest, ClassifyBalance) {
    SolutionValidator validator;
    
    // Идеальный баланс
    std::string status = validator.classify_balance(30.0, 40.0, 30.0);
    EXPECT_EQ(status, "IDEAL BALANCE");
    
    // Умеренный дисбаланс (макс < 75%, разрыв < 50%)
    status = validator.classify_balance(65.0, 20.0, 15.0);
    EXPECT_EQ(status, "MODERATE IMBALANCE");
    
    // Сильный дисбаланс
    status = validator.classify_balance(80.0, 10.0, 10.0);
    EXPECT_EQ(status, "STRONG IMBALANCE");
    
    // Сильный дисбаланс (разрыв > 50%)
    status = validator.classify_balance(70.0, 15.0, 15.0);
    EXPECT_EQ(status, "STRONG IMBALANCE");
}

// ============== Тесты оценки качества ==============

TEST(QualityScoreTest, ExcellentSolution) {
    SolutionValidator validator;
    
    ValidationResult validation;
    validation.is_valid = true;
    validation.interpolation_ok = true;
    validation.barriers_safe = true;
    validation.numerically_stable = true;
    validation.max_interpolation_error = 1e-15;
    
    FunctionalDiagnostics diagnostics;
    diagnostics.share_approx = 33.0;
    diagnostics.share_repel = 33.0;
    diagnostics.share_reg = 34.0;
    
    double quality = validator.compute_quality_score(validation, diagnostics);
    
    EXPECT_GE(quality, 0.95);
    EXPECT_LE(quality, 1.0);
}

TEST(QualityScoreTest, UnacceptableSolution) {
    SolutionValidator validator;
    
    ValidationResult validation;
    validation.is_valid = false;
    validation.interpolation_ok = false;
    validation.barriers_safe = true;
    validation.numerically_stable = true;
    validation.max_interpolation_error = 1e-3;
    
    FunctionalDiagnostics diagnostics;
    diagnostics.share_approx = 99.0;
    diagnostics.share_repel = 0.5;
    diagnostics.share_reg = 0.5;
    
    double quality = validator.compute_quality_score(validation, diagnostics);
    
    EXPECT_LT(quality, 0.70);
}

TEST(QualityClassificationTest, AllLevels) {
    SolutionValidator validator;
    
    // EXCELLENT
    EXPECT_EQ(validator.classify_quality(0.97), SolutionQuality::EXCELLENT);
    EXPECT_EQ(validator.classify_quality(0.95), SolutionQuality::EXCELLENT);
    
    // GOOD
    EXPECT_EQ(validator.classify_quality(0.94), SolutionQuality::GOOD);
    EXPECT_EQ(validator.classify_quality(0.85), SolutionQuality::GOOD);
    
    // SATISFACTORY
    EXPECT_EQ(validator.classify_quality(0.84), SolutionQuality::SATISFACTORY);
    EXPECT_EQ(validator.classify_quality(0.70), SolutionQuality::SATISFACTORY);
    
    // UNACCEPTABLE
    EXPECT_EQ(validator.classify_quality(0.69), SolutionQuality::UNACCEPTABLE);
    EXPECT_EQ(validator.classify_quality(0.0), SolutionQuality::UNACCEPTABLE);
}

TEST(QualityRecommendationsTest, GenerateRecommendations) {
    SolutionValidator validator;
    
    // Рекомендации для отличного решения
    auto recs_excellent = validator.generate_quality_recommendations(SolutionQuality::EXCELLENT);
    EXPECT_FALSE(recs_excellent.empty());
    EXPECT_EQ(recs_excellent[0], "Solution passed all checks. Ready for use.");
    
    // Рекомендации для неприемлемого решения
    auto recs_unacceptable = validator.generate_quality_recommendations(SolutionQuality::UNACCEPTABLE);
    EXPECT_FALSE(recs_unacceptable.empty());
    EXPECT_EQ(recs_unacceptable[0], "CRITICAL: Solution does not meet requirements.");
}

// ============== Тесты генерации отчётов ==============

TEST(ReportGenerationTest, BasicReport) {
    Polynomial poly({1.0, 2.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {1.0, 2.0, 3.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    std::string report = validator.generate_report(result);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("SOLUTION VERIFICATION REPORT"), std::string::npos);
    EXPECT_NE(report.find("Overall Status"), std::string::npos);
}

TEST(ReportGenerationTest, StatusToString) {
    std::string status;
    
    status = SolutionValidator::status_to_string(VerificationStatus::VERIFICATION_OK);
    EXPECT_NE(status.find("VERIFICATION_OK"), std::string::npos);
    
    status = SolutionValidator::status_to_string(VerificationStatus::VERIFICATION_WARNING);
    EXPECT_NE(status.find("VERIFICATION_WARNING"), std::string::npos);
    
    status = SolutionValidator::status_to_string(VerificationStatus::VERIFICATION_CRITICAL);
    EXPECT_NE(status.find("VERIFICATION_CRITICAL"), std::string::npos);
}

TEST(ReportGenerationTest, QualityToString) {
    std::string status;
    
    status = SolutionValidator::quality_to_string(SolutionQuality::EXCELLENT);
    EXPECT_NE(status.find("EXCELLENT"), std::string::npos);
    
    status = SolutionValidator::quality_to_string(SolutionQuality::GOOD);
    EXPECT_NE(status.find("GOOD"), std::string::npos);
    
    status = SolutionValidator::quality_to_string(SolutionQuality::UNACCEPTABLE);
    EXPECT_NE(status.find("UNACCEPTABLE"), std::string::npos);
}

// ============== Тесты проекционной коррекции ==============

TEST(ProjectionCorrectionTest, NoInterpolationNodes) {
    Polynomial poly({1.0, 2.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    
    SolutionValidator validator;
    bool corrected = validator.apply_projection_correction(poly, data);
    
    EXPECT_FALSE(corrected);
}

TEST(ProjectionCorrectionTest, WithInterpolationNodes) {
    // Полином, который нужно скорректировать
    Polynomial poly({1.0, 0.0});  // F(x) = x, не проходит через (0.5, 0.25)
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.interp_z = {0.5};
    data.interp_f = {0.25};  // Ожидаем x²
    
    SolutionValidator validator;
    bool corrected = validator.apply_projection_correction(poly, data);
    
    // Проверяем, что после коррекции полином проходит через узел
    if (corrected) {
        double value_at_node = poly.evaluate(0.5);
        EXPECT_NEAR(value_at_node, 0.25, 1e-10);
    }
}

// ============== Тесты полной верификации ==============

TEST(FullVerificationTest, LinearPolynomial) {
    Polynomial poly({1.0, 2.0});  // F(x) = 2x + 1
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {1.0, 2.0, 3.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    data.gamma = 0.01;
    
    SolutionValidator validator;
    FunctionalDiagnostics diagnostics;
    auto result = validator.verify_full(poly, data, diagnostics);
    
    EXPECT_TRUE(result.validation.is_valid);
    EXPECT_GE(result.quality_score, 0.0);
    EXPECT_LE(result.quality_score, 1.0);
}

TEST(FullVerificationTest, HighDegreePolynomial) {
    // Полином высокой степени
    std::vector<double> coeffs(11, 0.0);
    coeffs[0] = 1.0;  // x^10
    Polynomial poly(coeffs);
    
    OptimizationProblemData data;
    data.interval_a = -1.0;
    data.interval_b = 1.0;
    data.approx_x = {-1.0, -0.5, 0.0, 0.5, 1.0};
    data.approx_f = {1.0, 0.5, 0.0, 0.5, 1.0};
    data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
    data.gamma = 0.1;
    
    SolutionValidator validator;
    FunctionalDiagnostics diagnostics;
    auto result = validator.verify_full(poly, data, diagnostics);
    
    EXPECT_TRUE(result.validation.is_valid);
    EXPECT_GE(result.quality_score, 0.0);
}

// ============== Граничные случаи ==============

TEST(EdgeCasesTest, EmptyApproximationPoints) {
    Polynomial poly({1.0, 2.0});
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {};
    data.approx_f = {};
    data.approx_weight = {};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    EXPECT_TRUE(result.numerical_correct);
}

TEST(EdgeCasesTest, ZeroInterval) {
    Polynomial poly(std::vector<double>{5.0});  // F(x) = 5
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 0.0;  // Нулевой интервал
    data.approx_x = {0.0};
    data.approx_f = {5.0};
    data.approx_weight = {1.0};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    // Не должно крашиться
    EXPECT_TRUE(result.numerical_correct);
}

TEST(EdgeCasesTest, ConstantPolynomial) {
    Polynomial poly(std::vector<double>{5.0});  // F(x) = 5
    
    OptimizationProblemData data;
    data.interval_a = 0.0;
    data.interval_b = 1.0;
    data.approx_x = {0.0, 0.5, 1.0};
    data.approx_f = {5.0, 5.0, 5.0};
    data.approx_weight = {1.0, 1.0, 1.0};
    
    SolutionValidator validator;
    auto result = validator.validate(poly, data);
    
    EXPECT_TRUE(result.is_valid);
    EXPECT_TRUE(result.physically_plausible);
}

// ============== Главный тестовый файл ==============

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
