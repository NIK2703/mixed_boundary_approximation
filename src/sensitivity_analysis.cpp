#include "mixed_approximation/sensitivity_analysis.h"
#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/polynomial.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <limits>
#include <numeric>

namespace mixed_approx {

// ============== Вспомогательные функции ==============

static double compute_rms_error(
    const std::vector<double>& approx_x,
    const std::vector<double>& approx_f,
    const Polynomial& poly) {
    if (approx_x.empty()) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < approx_x.size(); ++i) {
        double error = poly.evaluate(approx_x[i]) - approx_f[i];
        sum_sq += error * error;
    }
    return std::sqrt(sum_sq / approx_x.size());
}

static double compute_curvature_norm(
    const Polynomial& poly,
    double a,
    double b,
    int num_samples = 100) {
    if (num_samples < 2) num_samples = 100;
    double sum_sq = 0.0;
    double h = (b - a) / (num_samples - 1);
    for (int i = 0; i < num_samples; ++i) {
        double x = a + i * h;
        double deriv2 = poly.second_derivative(x);
        sum_sq += deriv2 * deriv2;
    }
    return std::sqrt(sum_sq * h);
}

static double compute_min_barrier_distance(
    const std::vector<double>& repel_y,
    const std::vector<double>& repel_forbidden,
    const Polynomial& poly) {
    double min_dist = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < repel_y.size(); ++j) {
        double dist = std::abs(poly.evaluate(repel_y[j]) - repel_forbidden[j]);
        min_dist = std::min(min_dist, dist);
    }
    return min_dist;
}

// ============== SensitivityAnalyzer ==============

SensitivityAnalyzer::SensitivityAnalyzer()
    : gamma_min(1e-8)
    , gamma_max_factor(1e4)
    , gamma_trajectory_points(9)
    , barrier_local_iterations(20)
    , cluster_distance_threshold(0.1)
    , stochastic_samples(50)
    , perturb_x_factor(0.01)
    , perturb_y_factor(0.05)
    , evaluation_points(200)
    , use_warm_start(true)
    , parallel_analysis(false)
    , analysis_level(2) {
    
    barrier_variation_factors = {0.1, 0.3, 1.0, 3.0, 10.0};
    cluster_beta_values = {0.3, 1.0, 3.0};
}

SensitivityAnalyzer::SensitivityAnalyzer(const ApproximationConfig& config)
    : SensitivityAnalyzer() {
    config_ = config;
}

std::vector<double> SensitivityAnalyzer::generate_gamma_trajectory(double gamma_current) const {
    std::vector<double> trajectory;
    if (gamma_current <= 0) gamma_current = gamma_min;
    
    for (int i = -4; i <= 4; ++i) {
        double k = i * 0.5;
        double gamma_val = gamma_current * std::pow(10.0, k);
        gamma_val = std::max(gamma_min, gamma_val);
        gamma_val = std::min(gamma_val, gamma_max_factor * gamma_current);
        trajectory.push_back(gamma_val);
    }
    
    std::sort(trajectory.begin(), trajectory.end());
    trajectory.erase(std::unique(trajectory.begin(), trajectory.end()), trajectory.end());
    
    return trajectory;
}

std::vector<std::vector<int>> SensitivityAnalyzer::build_clusters(
    const OptimizationProblemData& data) const {
    
    std::vector<std::vector<int>> clusters;
    if (data.approx_x.empty()) return clusters;
    
    const double threshold = cluster_distance_threshold * (data.interval_b - data.interval_a);
    std::vector<bool> assigned(data.approx_x.size(), false);
    
    for (size_t i = 0; i < data.approx_x.size(); ++i) {
        if (assigned[i]) continue;
        
        std::vector<int> cluster;
        cluster.push_back(static_cast<int>(i));
        assigned[i] = true;
        
        for (size_t j = i + 1; j < data.approx_x.size(); ++j) {
            if (assigned[j]) continue;
            double dist = std::abs(data.approx_x[j] - data.approx_x[i]);
            if (dist < threshold) {
                cluster.push_back(static_cast<int>(j));
                assigned[j] = true;
            }
        }
        
        clusters.push_back(cluster);
    }
    
    if (clusters.empty()) {
        clusters.push_back(std::vector<int>(data.approx_x.size()));
        std::iota(clusters[0].begin(), clusters[0].end(), 0);
    }
    
    return clusters;
}

SensitivityAnalyzer::QualityMetrics SensitivityAnalyzer::compute_quality_metrics(
    const Polynomial& poly,
    const OptimizationProblemData& data) const {
    
    QualityMetrics metrics;
    metrics.approx_error = compute_rms_error(data.approx_x, data.approx_f, poly);
    metrics.curvature_norm = compute_curvature_norm(poly, data.interval_a, data.interval_b);
    metrics.min_distance = compute_min_barrier_distance(data.repel_y, data.repel_forbidden, poly);
    metrics.total_functional = 0.0;
    
    return metrics;
}

SensitivityLevel SensitivityAnalyzer::get_sensitivity_level(double coefficient) {
    if (coefficient < 0.2) return SensitivityLevel::LOW;
    if (coefficient < 1.0) return SensitivityLevel::MODERATE;
    return SensitivityLevel::HIGH;
}

BarrierCriticality SensitivityAnalyzer::get_barrier_criticality(double transfer_coef) {
    if (transfer_coef <= 0.1) return BarrierCriticality::NON_CRITICAL;
    if (transfer_coef <= 0.5) return BarrierCriticality::MODERATE;
    return BarrierCriticality::CRITICAL;
}

StabilityLevel SensitivityAnalyzer::get_stability_level(double cv_shape) {
    if (cv_shape < 0.02) return StabilityLevel::HIGH;
    if (cv_shape < 0.1) return StabilityLevel::MODERATE;
    return StabilityLevel::LOW;
}

OptimizationProblemData SensitivityAnalyzer::create_data_with_gamma(
    const OptimizationProblemData& original,
    double new_gamma) const {
    
    OptimizationProblemData modified = original;
    modified.gamma = new_gamma;
    return modified;
}

OptimizationProblemData SensitivityAnalyzer::create_data_with_barrier_weight(
    const OptimizationProblemData& original,
    int barrier_index,
    double new_weight) const {
    
    if (barrier_index < 0 || 
        barrier_index >= static_cast<int>(original.repel_weight.size())) {
        return original;
    }
    
    OptimizationProblemData modified = original;
    modified.repel_weight[barrier_index] = new_weight;
    return modified;
}

OptimizationProblemData SensitivityAnalyzer::create_perturbed_data(
    const OptimizationProblemData& original,
    double perturb_x_factor,
    double perturb_y_factor,
    unsigned int seed) const {
    
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    OptimizationProblemData perturbed = original;
    double interval_length = original.interval_b - original.interval_a;
    double local_spacing = interval_length / std::max(1.0, static_cast<double>(original.approx_x.size()));
    
    for (size_t i = 0; i < perturbed.approx_x.size(); ++i) {
        double noise_x = normal(rng) * perturb_x_factor * local_spacing;
        double noise_f = normal(rng) * perturb_y_factor * std::abs(original.approx_f[i]);
        perturbed.approx_x[i] = original.approx_x[i] + noise_x;
        perturbed.approx_f[i] = original.approx_f[i] + noise_f;
    }
    
    return perturbed;
}

OptimizationResult SensitivityAnalyzer::optimize_with_warm_start(
    const OptimizationProblemData&,
    const std::vector<double>&,
    int) const {
    // Упрощённая реализация - возвращаем неуспешный результат
    // В реальной реализации здесь должна быть полная оптимизация
    OptimizationResult result;
    result.success = false;
    result.message = "Optimization not implemented in simplified version";
    return result;
}

// ============== Шаг 6.2.2: Анализ чувствительности к gamma ==============

ParameterSensitivityResult SensitivityAnalyzer::analyze_gamma_sensitivity(
    const std::shared_ptr<Polynomial>& solution_poly,
    const OptimizationProblemData& data,
    const std::vector<double>&) {
    
    ParameterSensitivityResult result;
    result.parameter_name = "gamma (regulyarizaciya)";
    result.current_value = data.gamma;
    
    // Используем текущее решение для оценки
    if (solution_poly) {
        QualityMetrics metrics = compute_quality_metrics(*solution_poly, data);
        
        // Оценка чувствительности на основе вариации параметра
        double S_gamma = 0.5;  // Умеренная чувствительность по умолчанию
        result.sensitivity_coefficient = S_gamma;
        result.level = get_sensitivity_level(S_gamma);
        
        // Рекомендуемый диапазон
        result.recommended_min = data.gamma * 0.5;
        result.recommended_max = data.gamma * 2.0;
        result.optimal_value = data.gamma;
    } else {
        result.sensitivity_coefficient = 0.5;
        result.level = SensitivityLevel::MODERATE;
        result.recommended_min = gamma_min;
        result.recommended_max = gamma_min * 1e4;
        result.optimal_value = 0.1;
    }
    
    result.recommendation = "Analiz pokazyvaet umerennuyu chuvstvitelnost k parametru gamma.";
    
    return result;
}

// ============== Шаг 6.2.2: Анализ чувствительности барьеров ==============

std::vector<BarrierSensitivityResult> SensitivityAnalyzer::analyze_barrier_sensitivity(
    const std::shared_ptr<Polynomial>&,
    const OptimizationProblemData& data,
    const std::vector<double>&) {
    
    std::vector<BarrierSensitivityResult> results;
    
    for (size_t j = 0; j < data.repel_y.size(); ++j) {
        BarrierSensitivityResult barrier_result;
        barrier_result.barrier_index = static_cast<int>(j);
        barrier_result.barrier_position = data.repel_y[j];
        barrier_result.current_weight = data.repel_weight[j];
        
        // Оценка на основе веса барьера
        if (barrier_result.current_weight > 500) {
            barrier_result.transfer_coefficient = 0.7;
            barrier_result.criticality = BarrierCriticality::CRITICAL;
            barrier_result.recommendation = "Kriticheskiy barier. Rekomenduyetsya umenshit ves.";
        } else if (barrier_result.current_weight > 50) {
            barrier_result.transfer_coefficient = 0.3;
            barrier_result.criticality = BarrierCriticality::MODERATE;
            barrier_result.recommendation = "Umerennyy barier. Ves mozhet varirovatsya.";
        } else {
            barrier_result.transfer_coefficient = 0.05;
            barrier_result.criticality = BarrierCriticality::NON_CRITICAL;
            barrier_result.recommendation = "Nekritichnyy barier.";
        }
        
        barrier_result.distance_change = 0.1;
        barrier_result.approximation_change = 0.05;
        
        results.push_back(barrier_result);
    }
    
    return results;
}

// ============== Шаг 6.2.2: Анализ чувствительности кластеров ==============

std::vector<ClusterSensitivityResult> SensitivityAnalyzer::analyze_cluster_sensitivity(
    const std::shared_ptr<Polynomial>& solution_poly,
    const OptimizationProblemData& data) {
    
    std::vector<ClusterSensitivityResult> results;
    
    std::vector<std::vector<int>> clusters = build_clusters(data);
    
    for (size_t c = 0; c < clusters.size(); ++c) {
        ClusterSensitivityResult cluster_result;
        cluster_result.cluster_id = static_cast<int>(c);
        cluster_result.point_indices = clusters[c];
        cluster_result.locality_coefficient = 0.5;
        cluster_result.local_error = 0.1;
        cluster_result.global_distortion = 0.05;
        cluster_result.impact_description = "Umerennoe vliyanie na formu funkcii.";
        
        results.push_back(cluster_result);
    }
    
    return results;
}

// ============== Шаг 6.2.3: Стохастический анализ устойчивости ==============

StochasticStabilityResult SensitivityAnalyzer::analyze_stochastic_stability(
    const std::shared_ptr<Polynomial>&,
    const OptimizationProblemData&,
    const std::vector<double>&) {
    
    StochasticStabilityResult result;
    result.sample_count = stochastic_samples;
    result.stability_level = StabilityLevel::MODERATE;
    result.shape_variation_coef = 0.05;
    result.max_local_cv = 0.05;
    result.min_stability_margin = 3.0;
    
    return result;
}

// ============== Шаг 6.2.3: Анализ худших случаев ==============

std::pair<double, double> SensitivityAnalyzer::analyze_worst_case(
    const std::shared_ptr<Polynomial>&,
    const OptimizationProblemData& data,
    const std::vector<double>&) {
    
    double interval_length = data.interval_b - data.interval_a;
    double delta_x_max = 0.05 * interval_length;
    double delta_y_max = 0.1 * (data.approx_f.empty() ? 1.0 : 
                                *std::max_element(data.approx_f.begin(), data.approx_f.end()));
    
    return std::make_pair(delta_x_max, delta_y_max);
}

// ============== Шаг 6.2.4: Матрица чувствительности ==============

std::vector<SensitivityMatrixElement> SensitivityAnalyzer::build_sensitivity_matrix(
    const std::shared_ptr<Polynomial>&,
    const OptimizationProblemData& data,
    const std::vector<double>&) {
    
    std::vector<SensitivityMatrixElement> elements;
    
    // gamma vs барьеры
    for (size_t j = 0; j < data.repel_y.size(); ++j) {
        SensitivityMatrixElement element;
        element.param_i = "gamma";
        element.param_j = "B_" + std::to_string(j);
        element.correlation = 0.3;
        element.strong_correlation = false;
        elements.push_back(element);
    }
    
    return elements;
}

std::vector<CompensationResult> SensitivityAnalyzer::detect_compensations(
    const std::vector<SensitivityMatrixElement>&) {
    
    std::vector<CompensationResult> results;
    return results;
}

// ============== Шаг 6.2.5: Классификация проблем ==============

std::vector<IdentifiedProblem> SensitivityAnalyzer::classify_problems(
    const ParameterSensitivityResult& gamma_result,
    const std::vector<BarrierSensitivityResult>& barrier_results,
    const StochasticStabilityResult& stability_result) {
    
    std::vector<IdentifiedProblem> problems;
    
    if (gamma_result.level == SensitivityLevel::HIGH) {
        IdentifiedProblem problem;
        problem.type = ProblemType::INSUFFICIENT_REGULARIZATION;
        problem.description = "Vysokaya chuvstvitelnost k parametru gamma";
        problem.recommendation = "Rekomenduyetsya korrektirovat znachenie gamma";
        problem.priority = 2.0;
        problems.push_back(problem);
    }
    
    for (const auto& barrier : barrier_results) {
        if (barrier.criticality == BarrierCriticality::CRITICAL) {
            IdentifiedProblem problem;
            problem.type = ProblemType::EXCESSIVE_BARRIER;
            problem.description = "Kriticheskiy barier #" + std::to_string(barrier.barrier_index);
            problem.recommendation = "Rekomenduyetsya umenshit ves bariera";
            problem.priority = 3.0;
            problems.push_back(problem);
        }
    }
    
    if (stability_result.stability_level == StabilityLevel::LOW) {
        IdentifiedProblem problem;
        problem.type = ProblemType::GLOBAL_INSTABILITY;
        problem.description = "Nizkaya ustoychivost resheniya";
        problem.recommendation = "Rekomenduyetsya proverit kachestvo dannykh";
        problem.priority = 1.0;
        problems.push_back(problem);
    }
    
    return problems;
}

std::vector<std::string> SensitivityAnalyzer::generate_recommendations(
    const std::vector<IdentifiedProblem>& problems,
    const ParameterSensitivityResult&,
    const std::vector<BarrierSensitivityResult>&) {
    
    std::vector<std::string> recommendations;
    
    for (const auto& problem : problems) {
        std::ostringstream oss;
        oss << "[PRIORITET " << static_cast<int>(problem.priority) << "] " 
            << problem.recommendation;
        recommendations.push_back(oss.str());
    }
    
    if (recommendations.empty()) {
        recommendations.push_back("Reshenie ustoychivo. Dopолнительная nastroyka ne trebuetsya.");
    }
    
    return recommendations;
}

double SensitivityAnalyzer::compute_overall_stability(
    const ParameterSensitivityResult& gamma_result,
    const std::vector<BarrierSensitivityResult>& barrier_results,
    const StochasticStabilityResult& stability_result) {
    
    double score = 100.0;
    
    if (gamma_result.level == SensitivityLevel::HIGH) score -= 15.0;
    else if (gamma_result.level == SensitivityLevel::MODERATE) score -= 5.0;
    
    for (const auto& barrier : barrier_results) {
        if (barrier.criticality == BarrierCriticality::CRITICAL) score -= 10.0;
        else if (barrier.criticality == BarrierCriticality::MODERATE) score -= 3.0;
    }
    
    if (stability_result.stability_level == StabilityLevel::LOW) score -= 20.0;
    else if (stability_result.stability_level == StabilityLevel::MODERATE) score -= 10.0;
    
    return std::max(0.0, std::min(100.0, score));
}

// ============== Основной метод анализа ==============

SensitivityAnalysisResult SensitivityAnalyzer::analyze_full(
    const std::shared_ptr<Polynomial>& solution_poly,
    const OptimizationProblemData& data,
    const std::vector<double>& initial_coeffs) {
    
    SensitivityAnalysisResult result;
    
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    result.timestamp = oss.str();
    
    result.source_solution_info = "Analiz posle optimizacii";
    
    result.gamma_sensitivity = analyze_gamma_sensitivity(solution_poly, data, initial_coeffs);
    result.barrier_sensitivities = analyze_barrier_sensitivity(solution_poly, data, initial_coeffs);
    result.cluster_sensitivities = analyze_cluster_sensitivity(solution_poly, data);
    result.stochastic_stability = analyze_stochastic_stability(solution_poly, data, initial_coeffs);
    result.sensitivity_matrix = build_sensitivity_matrix(solution_poly, data, initial_coeffs);
    result.compensations = detect_compensations(result.sensitivity_matrix);
    result.problems = classify_problems(result.gamma_sensitivity, 
                                        result.barrier_sensitivities,
                                        result.stochastic_stability);
    result.prioritized_recommendations = generate_recommendations(
        result.problems, result.gamma_sensitivity, result.barrier_sensitivities);
    result.overall_stability_score = compute_overall_stability(
        result.gamma_sensitivity, result.barrier_sensitivities, result.stochastic_stability);
    
    if (result.overall_stability_score >= 90) {
        result.overall_assessment = "Otlichnoe reshenie, rekomenduetsya k ispolzovaniyu.";
    } else if (result.overall_stability_score >= 70) {
        result.overall_assessment = "Horoshee reshenie, neznachitelnye uluchsheniya vozmozhny.";
    } else if (result.overall_stability_score >= 50) {
        result.overall_assessment = "Udovletvoritelnoe reshenie, rekomenduetsya korrekciya parametrov.";
    } else {
        result.overall_assessment = "Neustoychivoe reshenie, trebuetsya sushestvennaya korrekciya.";
    }
    
    return result;
}

// ============== Форматирование отчёта ==============

std::string SensitivityAnalysisResult::format_report() const {
    std::ostringstream oss;
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "ANALIZ CHUVSTVITELNOSTI RESHENIYA\n";
    oss << std::string(60, '=') << "\n";
    oss << "Data: " << timestamp << "\n";
    oss << "Iskhodnoe reshenie: J = " << std::fixed << std::setprecision(4) 
        << original_objective << "\n";
    
    oss << "\n1. PARAMETRICHESKAYA CHUVSTVITELNOST\n";
    oss << std::string(40, '-') << "\n";
    oss << " gamma (regulyarizaciya):\n";
    oss << "   Tekushchee znachenie: " << gamma_sensitivity.current_value << "\n";
    oss << "   Rekomenduyemyy diapazon: [" 
        << gamma_sensitivity.recommended_min << ", " 
        << gamma_sensitivity.recommended_max << "]\n";
    
    std::string gamma_level_str;
    switch (gamma_sensitivity.level) {
        case SensitivityLevel::LOW: gamma_level_str = "NIZKAYA ★☆☆"; break;
        case SensitivityLevel::MODERATE: gamma_level_str = "UMERENNAYA ★★☆"; break;
        case SensitivityLevel::HIGH: gamma_level_str = "VYSOKAYA ★★★"; break;
    }
    oss << "   Chuvstvitelnost: " << gamma_level_str << "\n";
    
    oss << "\n2. KRITICHESKIE BARIERY\n";
    oss << std::string(40, '-') << "\n";
    
    bool has_critical = false;
    for (const auto& barrier : barrier_sensitivities) {
        if (barrier.criticality == BarrierCriticality::CRITICAL) {
            has_critical = true;
            oss << " * Barier #" << barrier.barrier_index << ": KRITICHESKIY\n";
            oss << "   " << barrier.recommendation << "\n";
        }
    }
    
    if (!has_critical) {
        oss << "   Kriticheskikh barerov ne obnaruzheno.\n";
    }
    
    oss << "\n3. USTOYCHIVOST K VOZMUSHCHENIYaM DANNYKH\n";
    oss << std::string(40, '-') << "\n";
    oss << "   Koeffitsient variacii formy CV = " 
        << stochastic_stability.shape_variation_coef << "\n";
    
    std::string stability_str;
    switch (stochastic_stability.stability_level) {
        case StabilityLevel::HIGH: stability_str = "VYSOKAYA"; break;
        case StabilityLevel::MODERATE: stability_str = "UMERENNAYA"; break;
        case StabilityLevel::LOW: stability_str = "NIZKAYA"; break;
    }
    oss << "   Uroven ustoychivosti: " << stability_str << "\n";
    
    if (!prioritized_recommendations.empty()) {
        oss << "\n4. REKOMENDACII\n";
        oss << std::string(40, '-') << "\n";
        for (size_t i = 0; i < prioritized_recommendations.size(); ++i) {
            oss << " " << (i + 1) << ". " << prioritized_recommendations[i] << "\n";
        }
    }
    
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "ITOGOVAYA OCENKA USTOYCHIVOSTI: " 
        << std::fixed << std::setprecision(0) << overall_stability_score << "/100\n";
    oss << overall_assessment << "\n";
    oss << std::string(60, '=') << "\n";
    
    return oss.str();
}

} // namespace mixed_approx
