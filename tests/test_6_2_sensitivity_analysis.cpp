/**
 * @file test_6_2_sensitivity_analysis.cpp
 * @brief Тесты для модуля анализа чувствительности (Шаг 6.2)
 */

#include <gtest/gtest.h>
#include "mixed_approximation/sensitivity_analysis.h"
#include "mixed_approximation/optimization_problem_data.h"
#include "mixed_approximation/polynomial.h"

namespace mixed_approx {
namespace test {

class SensitivityAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        analyzer = std::make_unique<SensitivityAnalyzer>();
    }
    
    OptimizationProblemData create_test_data() {
        OptimizationProblemData data;
        data.interval_a = 0.0;
        data.interval_b = 1.0;
        data.gamma = 0.1;
        
        // Аппроксимирующие данные
        data.approx_x = {0.0, 0.25, 0.5, 0.75, 1.0};
        data.approx_f = {0.0, 0.25, 0.5, 0.75, 1.0};
        
        // Барьерные данные
        data.repel_y = {0.3, 0.7};
        data.repel_forbidden = {0.4, 0.6};
        data.repel_weight = {100.0, 100.0};
        
        return data;
    }
    
    std::shared_ptr<Polynomial> create_test_polynomial() {
        std::vector<double> coeffs = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        return std::make_shared<Polynomial>(coeffs);
    }
    
    std::unique_ptr<SensitivityAnalyzer> analyzer;
};

TEST_F(SensitivityAnalysisTest, GammaTrajectoryGeneration) {
    auto data = create_test_data();
    auto trajectory = analyzer->generate_gamma_trajectory(data.gamma);
    
    EXPECT_FALSE(trajectory.empty());
    EXPECT_GE(trajectory.size(), 3);
    
    // Trajectory should be sorted
    for (size_t i = 1; i < trajectory.size(); ++i) {
        EXPECT_LE(trajectory[i-1], trajectory[i]);
    }
    
    // Trajectory should include the current gamma value
    bool contains_current = false;
    for (double g : trajectory) {
        if (std::abs(g - data.gamma) < 1e-10) {
            contains_current = true;
            break;
        }
    }
    EXPECT_TRUE(contains_current);
}

TEST_F(SensitivityAnalysisTest, ClusterBuilding) {
    auto data = create_test_data();
    auto clusters = analyzer->build_clusters(data);
    
    EXPECT_FALSE(clusters.empty());
    
    // All points should be assigned to some cluster
    size_t total_points = 0;
    for (const auto& cluster : clusters) {
        total_points += cluster.size();
    }
    EXPECT_EQ(total_points, data.approx_x.size());
}

TEST_F(SensitivityAnalysisTest, QualityMetricsComputation) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto metrics = analyzer->compute_quality_metrics(*poly, data);
    
    EXPECT_GE(metrics.approx_error, 0.0);
    EXPECT_GE(metrics.curvature_norm, 0.0);
    EXPECT_GE(metrics.min_distance, 0.0);
}

TEST_F(SensitivityAnalysisTest, SensitivityLevelClassification) {
    EXPECT_EQ(SensitivityAnalyzer::get_sensitivity_level(0.1), SensitivityLevel::LOW);
    EXPECT_EQ(SensitivityAnalyzer::get_sensitivity_level(0.5), SensitivityLevel::MODERATE);
    EXPECT_EQ(SensitivityAnalyzer::get_sensitivity_level(1.5), SensitivityLevel::HIGH);
}

TEST_F(SensitivityAnalysisTest, BarrierCriticalityClassification) {
    EXPECT_EQ(SensitivityAnalyzer::get_barrier_criticality(0.05), BarrierCriticality::NON_CRITICAL);
    EXPECT_EQ(SensitivityAnalyzer::get_barrier_criticality(0.3), BarrierCriticality::MODERATE);
    EXPECT_EQ(SensitivityAnalyzer::get_barrier_criticality(0.7), BarrierCriticality::CRITICAL);
}

TEST_F(SensitivityAnalysisTest, StabilityLevelClassification) {
    EXPECT_EQ(SensitivityAnalyzer::get_stability_level(0.01), StabilityLevel::HIGH);
    EXPECT_EQ(SensitivityAnalyzer::get_stability_level(0.05), StabilityLevel::MODERATE);
    EXPECT_EQ(SensitivityAnalyzer::get_stability_level(0.15), StabilityLevel::LOW);
}

TEST_F(SensitivityAnalysisTest, GammaSensitivityAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto result = analyzer->analyze_gamma_sensitivity(poly, data, {});
    
    EXPECT_EQ(result.parameter_name, "gamma (regulyarizaciya)");
    EXPECT_GE(result.sensitivity_coefficient, 0.0);
    EXPECT_LE(result.sensitivity_coefficient, 1.0);
    EXPECT_FALSE(result.recommendation.empty());
}

TEST_F(SensitivityAnalysisTest, BarrierSensitivityAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto results = analyzer->analyze_barrier_sensitivity(poly, data, {});
    
    EXPECT_EQ(results.size(), data.repel_y.size());
    
    for (const auto& result : results) {
        EXPECT_GE(result.transfer_coefficient, 0.0);
        EXPECT_LE(result.transfer_coefficient, 1.0);
        EXPECT_FALSE(result.recommendation.empty());
    }
}

TEST_F(SensitivityAnalysisTest, ClusterSensitivityAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto results = analyzer->analyze_cluster_sensitivity(poly, data);
    
    EXPECT_FALSE(results.empty());
    
    for (const auto& result : results) {
        EXPECT_GE(result.locality_coefficient, 0.0);
        EXPECT_LE(result.locality_coefficient, 1.0);
        EXPECT_FALSE(result.impact_description.empty());
    }
}

TEST_F(SensitivityAnalysisTest, StochasticStabilityAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto result = analyzer->analyze_stochastic_stability(poly, data, {});
    
    EXPECT_EQ(result.sample_count, 50);
    EXPECT_GE(result.shape_variation_coef, 0.0);
    EXPECT_LE(result.shape_variation_coef, 1.0);
}

TEST_F(SensitivityAnalysisTest, WorstCaseAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto [delta_x, delta_y] = analyzer->analyze_worst_case(poly, data, {});
    
    EXPECT_GE(delta_x, 0.0);
    EXPECT_GE(delta_y, 0.0);
}

TEST_F(SensitivityAnalysisTest, SensitivityMatrixBuilding) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto elements = analyzer->build_sensitivity_matrix(poly, data, {});
    
    // Should have elements for gamma vs each barrier
    EXPECT_EQ(elements.size(), data.repel_y.size());
}

TEST_F(SensitivityAnalysisTest, CompensationDetection) {
    // Test with empty elements
    std::vector<SensitivityMatrixElement> elements;
    auto results = analyzer->detect_compensations(elements);
    // Should return empty vector for empty input
    EXPECT_TRUE(results.empty());
}

TEST_F(SensitivityAnalysisTest, ProblemClassification) {
    ParameterSensitivityResult gamma_result;
    gamma_result.level = SensitivityLevel::MODERATE;
    
    std::vector<BarrierSensitivityResult> barrier_results;
    BarrierSensitivityResult barrier;
    barrier.criticality = BarrierCriticality::CRITICAL;
    barrier_results.push_back(barrier);
    
    StochasticStabilityResult stability;
    stability.stability_level = StabilityLevel::MODERATE;
    
    auto problems = analyzer->classify_problems(gamma_result, barrier_results, stability);
    
    // Should detect the critical barrier
    bool found_excessive_barrier = false;
    for (const auto& problem : problems) {
        if (problem.type == ProblemType::EXCESSIVE_BARRIER) {
            found_excessive_barrier = true;
            break;
        }
    }
    EXPECT_TRUE(found_excessive_barrier);
}

TEST_F(SensitivityAnalysisTest, RecommendationGeneration) {
    std::vector<IdentifiedProblem> problems;
    ParameterSensitivityResult gamma_result;
    gamma_result.level = SensitivityLevel::MODERATE;
    
    std::vector<BarrierSensitivityResult> barrier_results;
    
    StochasticStabilityResult stability;
    stability.stability_level = StabilityLevel::MODERATE;
    
    auto recommendations = analyzer->generate_recommendations(
        problems, gamma_result, barrier_results);
    
    EXPECT_FALSE(recommendations.empty());
}

TEST_F(SensitivityAnalysisTest, OverallStabilityComputation) {
    ParameterSensitivityResult gamma_result;
    gamma_result.level = SensitivityLevel::MODERATE;
    
    std::vector<BarrierSensitivityResult> barrier_results;
    
    StochasticStabilityResult stability;
    stability.stability_level = StabilityLevel::MODERATE;
    
    auto score = analyzer->compute_overall_stability(gamma_result, barrier_results, stability);
    
    EXPECT_GE(score, 0.0);
    EXPECT_LE(score, 100.0);
}

TEST_F(SensitivityAnalysisTest, FullAnalysis) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto result = analyzer->analyze_full(poly, data, {});
    
    EXPECT_FALSE(result.timestamp.empty());
    EXPECT_FALSE(result.overall_assessment.empty());
    
    // Check that all sub-analyses are present
    EXPECT_GE(result.gamma_sensitivity.sensitivity_coefficient, 0.0);
    EXPECT_FALSE(result.prioritized_recommendations.empty());
    EXPECT_GE(result.overall_stability_score, 0.0);
    EXPECT_LE(result.overall_stability_score, 100.0);
}

TEST_F(SensitivityAnalysisTest, ReportFormatting) {
    auto poly = create_test_polynomial();
    auto data = create_test_data();
    
    auto result = analyzer->analyze_full(poly, data, {});
    auto report = result.format_report();
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("ANALIZ CHUVSTVITELNOSTI"), std::string::npos);
}

TEST_F(SensitivityAnalysisTest, DefaultConfiguration) {
    SensitivityAnalyzer default_analyzer;
    
    EXPECT_EQ(default_analyzer.gamma_trajectory_points, 9);
    EXPECT_EQ(default_analyzer.stochastic_samples, 50);
    EXPECT_EQ(default_analyzer.analysis_level, 2);
}

} // namespace test
} // namespace mixed_approx

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
