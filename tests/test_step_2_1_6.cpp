#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "mixed_approximation/parameterization_verification.h"
#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/interpolation_basis.h"
#include "mixed_approximation/weight_multiplier.h"
#include "mixed_approximation/correction_polynomial.h"

using namespace mixed_approx;

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.1: ФОРМАЛИЗАЦИЯ КРИТЕРИЕВ КОРРЕКТНОСТИ
// =============================================================================

TEST(Step2_1_6_1, VerificationStatusEnum) {
    std::cout << "Testing VerificationStatus enum values...\n";
    
    // Проверяем, что все статусы определены
    EXPECT_EQ(static_cast<int>(VerificationStatus::PASSED), 0);
    EXPECT_EQ(static_cast<int>(VerificationStatus::WARNING), 1);
    EXPECT_EQ(static_cast<int>(VerificationStatus::FAILED), 2);
}

TEST(Step_1_6_1, RecommendationTypeEnum) {
    std::cout << "Testing RecommendationType enum values...\n";
    
    // Проверяем, что все типы рекомендаций определены
    EXPECT_GE(static_cast<int>(RecommendationType::NONE), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::CHANGE_BASIS), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::MERGE_NODES), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::REDUCE_DEGREE), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::INCREASE_GAMMA), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::USE_LONG_DOUBLE), 0);
    EXPECT_GE(static_cast<int>(RecommendationType::REDUCE_TOLERANCE), 0);
}

TEST(Step2_1_6_1, DefaultVerifierParameters) {
    std::cout << "Testing default verifier parameters...\n";
    
    ParameterizationVerifier verifier;
    
    // Проверяем, что верификатор создаётся без ошибок
    EXPECT_NO_THROW(ParameterizationVerification result = verifier.verify(
        CompositePolynomial(), std::vector<InterpolationNode>()));
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.2: АЛГОРИТМ ТЕСТИРОВАНИЯ ТОЧНОСТИ ИНТЕРПОЛЯЦИИ
// =============================================================================

TEST(Step2_1_6_2, InterpolationTestPassed) {
    std::cout << "Testing interpolation test with perfect interpolation...\n";
    
    // Создаём композитный полином с корректной параметризацией
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.1, 0.2, 0.3};  // Ненулевые коэффициенты
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    InterpolationTestResult result = verifier.test_interpolation(F, interp_nodes);
    
    // Тест должен пройти
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.failed_nodes, 0);
    EXPECT_LT(result.max_absolute_error, 1e-8);
}

TEST(Step2_1_6_2, InterpolationTestFailed) {
    std::cout << "Testing interpolation test with intentional failure...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    // Создаём корректирующий полином с большими коэффициентами
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {1000.0, 2000.0, 3000.0};  // Большие значения
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // В узлах интерполяции W(z_e) = 0, поэтому F(z_e) = P_int(z_e) независимо от Q
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    InterpolationTestResult result = verifier.test_interpolation(F, interp_nodes);
    
    // Интерполяционные условия всё равно должны выполняться!
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.failed_nodes, 0);
}

TEST(Step2_1_6_2, InterpolationTestNodeErrors) {
    std::cout << "Testing interpolation test error reporting...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {0.0, 0.0, 0.0};  // Нулевые коэффициенты
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    InterpolationTestResult result = verifier.test_interpolation(F, interp_nodes);
    
    // Проверяем структуру ошибок
    EXPECT_EQ(result.node_errors.size(), nodes.size());
    EXPECT_EQ(result.total_nodes, 3);
    
    // Все W(z_e) должны быть близки к нулю
    for (const auto& err : result.node_errors) {
        EXPECT_TRUE(err.W_acceptable);
        EXPECT_LT(std::abs(err.W_value), 1e-10);
    }
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.3: АЛГОРИТМ ТЕСТИРОВАНИЯ ПОЛНОТЫ ПРОСТРАНСТВА
// =============================================================================

TEST(Step2_1_6_3, CompletenessTestPassed) {
    std::cout << "Testing completeness test with well-conditioned basis...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::CHEBYSHEV, 0.5, 0.5);  // n_free = 3
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    ParameterizationVerifier verifier;
    CompletenessTestResult result = verifier.test_completeness(Q, W, 0.0, 1.0);
    
    // Тест должен пройти с базисом Чебышёва
    EXPECT_GE(result.actual_rank, 1);  //至少有一个有效的秩
    EXPECT_GE(result.expected_rank, 1);
}

TEST(Step2_1_6_3, CompletenessTestHighDegree) {
    std::cout << "Testing completeness test with high degree polynomial...\n";
    
    // Высокая степень с мономиальным базисом может давать плохую обусловленность
    CorrectionPolynomial Q;
    Q.initialize(8, BasisType::MONOMIAL, 0.5, 0.5);  // n_free = 9
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.25, 0.5, 0.75, 1.0}, 0.0, 1.0);
    
    ParameterizationVerifier verifier;
    CompletenessTestResult result = verifier.test_completeness(Q, W, 0.0, 1.0);
    
    // Проверяем, что тест выполнен
    EXPECT_GE(result.expected_rank, 1);
}

TEST(Step2_1_6_3, CompletenessTestNoFreeParams) {
    std::cout << "Testing completeness test with no free parameters...\n";
    
    CorrectionPolynomial Q;
    Q.initialize(-1, BasisType::MONOMIAL, 0.5, 0.5);  // n_free = 0 (вырожденный случай)
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    ParameterizationVerifier verifier;
    CompletenessTestResult result = verifier.test_completeness(Q, W, 0.0, 1.0);
    
    // Должен пройти (нет свободных параметров)
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.actual_rank, 0);
    EXPECT_EQ(result.expected_rank, 0);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.4: АЛГОРИТМ ТЕСТИРОВАНИЯ ЧИСЛЕННОЙ УСТОЙЧИВОСТИ
// =============================================================================

TEST(Step2_1_6_4, StabilityTestPassed) {
    std::cout << "Testing stability test with well-behaved polynomial...\n";
    
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1.0, 1.5, 2.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.coeffs = {0.1, 0.0, 0.0};  // Малые коэффициенты
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    StabilityTestResult result = verifier.test_stability(F, interp_nodes);
    
    // Проверяем, что тест выполнен
    EXPECT_GE(result.max_component_scale, 0.0);
}

TEST(Step2_1_6_4, StabilityTestLargeScaleRatio) {
    std::cout << "Testing stability test with potentially poor scale balance...\n";
    
    InterpolationBasis basis;
    // Сильно различающиеся значения
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {1e6, 1.5e6, 2e6};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {1e6, 1e6, 1e6};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    StabilityTestResult result = verifier.test_stability(F, interp_nodes);
    
    // Проверяем, что тест выполнен
    EXPECT_GE(result.scale_balance_ratio, 0.0);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.5: СТРУКТУРА ДАННЫХ ДЛЯ РЕЗУЛЬТАТОВ
// =============================================================================

TEST(Step2_1_6_5, ParameterizationVerificationStructure) {
    std::cout << "Testing ParameterizationVerification structure...\n";
    
    ParameterizationVerification verification;
    
    // Проверяем начальное состояние
    EXPECT_EQ(verification.overall_status, VerificationStatus::PASSED);
    EXPECT_EQ(verification.polynomial_degree, 0);
    EXPECT_EQ(verification.num_constraints, 0);
    EXPECT_EQ(verification.num_free_params, 0);
    
    // Проверяем методы
    EXPECT_FALSE(verification.has_warnings());
    EXPECT_FALSE(verification.has_errors());
    EXPECT_TRUE(verification.is_passed());
}

TEST(Step2_1_6_5, VerificationResultFormat) {
    std::cout << "Testing verification result formatting...\n";
    
    ParameterizationVerification verification;
    verification.polynomial_degree = 5;
    verification.num_constraints = 3;
    verification.num_free_params = 3;
    verification.interval_a = 0.0;
    verification.interval_b = 1.0;
    verification.overall_status = VerificationStatus::PASSED;
    
    // Форматируем результат
    std::string output = verification.format();
    
    // Проверяем, что вывод содержит ключевые элементы
    EXPECT_TRUE(output.find("PASSED") != std::string::npos);
    EXPECT_TRUE(output.find("5") != std::string::npos);  // degree
    EXPECT_TRUE(output.find("3") != std::string::npos);  // constraints
    
    // Проверяем подробный вывод
    std::string detailed = verification.format(true);
    EXPECT_TRUE(detailed.find("Interpolation Test:") != std::string::npos);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.6: СТРАТЕГИИ КОРРЕКЦИИ
// =============================================================================

TEST(Step2_1_6_6, VerifierWithCustomParameters) {
    std::cout << "Testing verifier with custom parameters...\n";
    
    // Создаём верификатор с пользовательскими параметрами
    ParameterizationVerifier verifier(1e-8, 1e-10, 1e6, 1e-6);
    
    InterpolationBasis basis;
    basis.build({0.0, 0.5, 1.0}, {1.0, 1.5, 2.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 0.5, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::MONOMIAL, 0.5, 0.5);
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes = {
        {0.0, 1.0}, {0.5, 1.5}, {1.0, 2.0}
    };
    
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    // Проверяем, что верификация выполнена
    EXPECT_TRUE(result.overall_status == VerificationStatus::PASSED ||
                result.overall_status == VerificationStatus::WARNING);
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.7: ГЕНЕРАЦИЯ ДИАГНОСТИЧЕСКОГО ОТЧЁТА
// =============================================================================

TEST(Step2_1_6_7, DiagnosticReportGeneration) {
    std::cout << "Testing diagnostic report generation...\n";
    
    ParameterizationVerification verification;
    verification.polynomial_degree = 5;
    verification.num_constraints = 3;
    verification.num_free_params = 3;
    verification.interval_a = 0.0;
    verification.interval_b = 1.0;
    
    // Добавляем рекомендации
    verification.recommendations.push_back(
        Recommendation(RecommendationType::CHANGE_BASIS, 
                       "Switch to Chebyshev basis", 
                       "For improved numerical stability"));
    
    verification.warnings.push_back("High condition number detected");
    
    std::string report = verification.format(true);
    
    // Проверяем наличие секций
    EXPECT_TRUE(report.find("Recommendations:") != std::string::npos);
    EXPECT_TRUE(report.find("Warnings:") != std::string::npos);
}

TEST(Step2_1_6_7, RecommendationStructure) {
    std::cout << "Testing recommendation structure...\n";
    
    Recommendation rec(RecommendationType::CHANGE_BASIS,
                       "Test message",
                       "Test rationale");
    
    EXPECT_EQ(rec.type, RecommendationType::CHANGE_BASIS);
    EXPECT_EQ(rec.message, "Test message");
    EXPECT_EQ(rec.rationale, "Test rationale");
}

// =============================================================================
// ТЕСТЫ ШАГА 2.1.6.8: ИНТЕГРАЦИЯ В ОСНОВНОЙ АЛГОРИТМ
// =============================================================================

TEST(Step2_1_6_8, FullVerificationPipeline) {
    std::cout << "Testing full verification pipeline...\n";
    
    // Создаём полную параметризацию
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.3, 0.7, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::CHEBYSHEV, 0.5, 0.5);  // n = 6, m = 4, deg_Q = 3
    Q.initialize_zero();
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Создаём интерполяционные узлы
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    // Выполняем полную верификацию
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    // Выводим отчёт
    std::cout << result.format(true);
    
    // Проверяем результаты тестов
    EXPECT_TRUE(result.interpolation_test.passed);  // Интерполяция должна выполняться
    EXPECT_GE(result.completeness_test.actual_rank, 1);  //至少有一个有效的秩
}

TEST(Step2_1_6_8, VerificationWithChebyshevBasis) {
    std::cout << "Testing verification with Chebyshev basis...\n";
    
    InterpolationBasis basis;
    basis.build({0.0, 1.0}, {0.0, 1.0}, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({0.0, 1.0}, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(5, BasisType::CHEBYSHEV, 0.5, 0.5);  // 高阶多项式
    Q.coeffs = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes = {{0.0, 0.0}, {1.0, 1.0}};
    
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    std::cout << result.format();
    
    // Проверяем, что интерполяционный тест пройден
    EXPECT_TRUE(result.interpolation_test.passed);
}

TEST(Step2_1_6_8, EdgeCaseNoInterpolationNodes) {
    std::cout << "Testing edge case: no interpolation nodes...\n";
    
    InterpolationBasis basis;  // Пустой базис
    std::vector<double> empty_nodes, empty_values;
    basis.build(empty_nodes, empty_values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots({}, 0.0, 1.0);  // W(x) = 1
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::MONOMIAL, 0.5, 0.5);
    Q.coeffs = {1.0, 0.0, 0.0, 0.0};  // Q(x) = 1
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    // Верификация без интерполяционных узлов
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, {});
    
    std::cout << result.format();
    
    // Проверяем структуру результата
    EXPECT_TRUE(result.interpolation_test.total_nodes == 0);
}

TEST(Step2_1_6_8, EdgeCaseFullInterpolation) {
    std::cout << "Testing edge case: full interpolation (m = n + 1)...\n";
    
    // n = 2, m = 3 -> Q вырожден
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.5, 1.0};
    std::vector<double> values = {0.0, 0.25, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(-1, BasisType::MONOMIAL, 0.5, 0.5);  // deg_Q = -1
    Q.n_free = 0;
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes = {
        {0.0, 0.0}, {0.5, 0.25}, {1.0, 1.0}
    };
    
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    std::cout << result.format();
    
    // Интерполяция должна выполняться
    EXPECT_TRUE(result.interpolation_test.passed);
}

// =============================================================================
// ИНТЕГРАЦИОННЫЕ ТЕСТЫ
// =============================================================================

TEST(IntegrationTest, ParameterizationVerificationIntegration) {
    std::cout << "Testing full parameterization verification integration...\n";
    
    // Типичная задача аппроксимации
    InterpolationBasis basis;
    std::vector<double> nodes = {0.0, 0.4, 0.7, 1.0};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(3, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.initialize_zero();
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    // Верификация
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    std::cout << "\n=== Full Verification Report ===\n";
    std::cout << result.format(true);
    
    // Критический тест: интерполяция должна выполняться!
    ASSERT_TRUE(result.interpolation_test.passed)
        << "Interpolation conditions must be satisfied!";
    
    // Проверяем общий статус
    EXPECT_TRUE(result.is_passed())
        << "Verification should pass for correct parameterization";
}

TEST(IntegrationTest, VerificationFailureScenarios) {
    std::cout << "Testing verification with problematic configurations...\n";
    
    // Создаём параметризацию с потенциальными проблемами
    InterpolationBasis basis;
    // Близкие узлы могут вызвать проблемы
    std::vector<double> nodes = {0.0, 0.001, 0.002};
    std::vector<double> values = {1.0, 1.0, 1.0};
    basis.build(nodes, values, InterpolationMethod::BARYCENTRIC, 0.0, 1.0, true, false);
    
    WeightMultiplier W;
    W.build_from_roots(nodes, 0.0, 1.0);
    
    CorrectionPolynomial Q;
    Q.initialize(2, BasisType::CHEBYSHEV, 0.5, 0.5);
    Q.coeffs = {0.1, 0.0, 0.0};
    
    CompositePolynomial F;
    F.build(basis, W, Q, 0.0, 1.0);
    
    std::vector<InterpolationNode> interp_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        interp_nodes.emplace_back(nodes[i], values[i]);
    }
    
    ParameterizationVerifier verifier;
    ParameterizationVerification result = verifier.verify(F, interp_nodes);
    
    std::cout << result.format();
    
    // Проверяем, что результат получен
    EXPECT_TRUE(result.interpolation_test.total_nodes > 0);
}

// =============================================================================
// ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ
// =============================================================================

TEST(Step2_1_6, NodeErrorStructure) {
    std::cout << "Testing NodeError structure...\n";
    
    NodeError error;
    error.node_index = 0;
    error.coordinate = 0.5;
    error.target_value = 1.0;
    error.computed_value = 1.0000001;
    error.absolute_error = 1e-7;
    error.relative_error = 1e-7;
    error.W_value = 1e-15;
    error.W_acceptable = true;
    
    EXPECT_EQ(error.node_index, 0);
    EXPECT_DOUBLE_EQ(error.absolute_error, 1e-7);
    EXPECT_TRUE(error.W_acceptable);
}

TEST(Step2_1_6, InterpolationTestResultStructure) {
    std::cout << "Testing InterpolationTestResult structure...\n";
    
    InterpolationTestResult result;
    result.passed = true;
    result.total_nodes = 5;
    result.failed_nodes = 0;
    result.max_absolute_error = 1e-12;
    result.max_relative_error = 1e-13;
    result.tolerance = 1e-10;
    
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_nodes, 5);
    EXPECT_LT(result.max_absolute_error, result.tolerance);
}

TEST(Step2_1_6, CompletenessTestResultStructure) {
    std::cout << "Testing CompletenessTestResult structure...\n";
    
    CompletenessTestResult result;
    result.passed = true;
    result.expected_rank = 4;
    result.actual_rank = 4;
    result.condition_number = 1e3;
    result.min_singular_value = 1e-4;
    result.relative_min_sv = 1e-7;
    
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.expected_rank, result.actual_rank);
}

TEST(Step2_1_6, StabilityTestResultStructure) {
    std::cout << "Testing StabilityTestResult structure...\n";
    
    StabilityTestResult result;
    result.passed = true;
    result.perturbation_sensitivity = 1e-9;
    result.scale_balance_ratio = 1.5;
    result.gradient_condition_number = 100.0;
    result.max_component_scale = 2.0;
    result.min_component_scale = 1.0;
    
    EXPECT_TRUE(result.passed);
    EXPECT_LT(result.perturbation_sensitivity, 1e-4);
}

// Конец файла
