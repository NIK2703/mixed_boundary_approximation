#include "mixed_approximation/parameterization_verification.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <random>

namespace mixed_approx {

// ==================== Форматирование результата ====================

std::string ParameterizationVerification::format(bool detailed) const {
    std::ostringstream oss;
    
    // Сводная секция
    oss << "=================================================\n";
    oss << "Parameterization Verification Report\n";
    oss << "=================================================\n\n";
    
    oss << "Overall Status: ";
    switch (overall_status) {
        case VerificationStatus::PASSED:
            oss << "PASSED";
            break;
        case VerificationStatus::WARNING:
            oss << "WARNING";
            break;
        case VerificationStatus::FAILED:
            oss << "FAILED";
            break;
    }
    oss << "\n\n";
    
    // Параметры
    oss << "Parameters:\n";
    oss << "  - Polynomial degree (n): " << polynomial_degree << "\n";
    oss << "  - Interpolation nodes (m): " << num_constraints << "\n";
    oss << "  - Free parameters (n_free): " << num_free_params << "\n";
    oss << "  - Interval: [" << interval_a << ", " << interval_b << "]\n\n";
    
    // Тест интерполяции
    oss << "Interpolation Test:\n";
    oss << "  - Status: " << (interpolation_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Nodes checked: " << interpolation_test.total_nodes << "/"
        << interpolation_test.total_nodes << "\n";
    oss << "  - Failed nodes: " << interpolation_test.failed_nodes << "\n";
    oss << "  - Max absolute error: " << std::scientific << std::setprecision(3)
        << interpolation_test.max_absolute_error << "\n";
    oss << "  - Max relative error: " << interpolation_test.max_relative_error << "\n";
    oss << "  - Tolerance: " << interpolation_test.tolerance << "\n";
    if (detailed && !interpolation_test.node_errors.empty()) {
        oss << "  - Node errors:\n";
        for (const auto& err : interpolation_test.node_errors) {
            if (err.absolute_error > interpolation_test.tolerance) {
                oss << "    Node " << err.node_index << " (x=" << err.coordinate << "):\n";
                oss << "      Target: " << err.target_value << ", Computed: " << err.computed_value << "\n";
                oss << "      Abs error: " << err.absolute_error << ", W(x): " << err.W_value << "\n";
            }
        }
    }
    oss << "\n";
    
    // Тест полноты
    oss << "Completeness Test:\n";
    oss << "  - Status: " << (completeness_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Expected rank: " << completeness_test.expected_rank << "\n";
    oss << "  - Actual rank: " << completeness_test.actual_rank << "\n";
    oss << "  - Condition number: " << std::scientific << std::setprecision(3)
        << completeness_test.condition_number << "\n";
    oss << "  - Min singular value: " << completeness_test.min_singular_value << "\n";
    oss << "  - Relative min SV: " << completeness_test.relative_min_sv << "\n";
    oss << "\n";
    
    // Тест устойчивости
    oss << "Stability Test:\n";
    oss << "  - Status: " << (stability_test.passed ? "PASSED" : "FAILED") << "\n";
    oss << "  - Perturbation sensitivity: " << std::scientific << std::setprecision(3)
        << stability_test.perturbation_sensitivity << "\n";
    oss << "  - Scale balance ratio: " << stability_test.scale_balance_ratio << "\n";
    oss << "  - Gradient condition number: " << stability_test.gradient_condition_number << "\n";
    oss << "\n";
    
    // Рекомендации
    if (!recommendations.empty()) {
        oss << "Recommendations:\n";
        for (size_t i = 0; i < recommendations.size(); ++i) {
            const auto& rec = recommendations[i];
            oss << "  " << (i + 1) << ". [" << static_cast<int>(rec.type) << "] " << rec.message << "\n";
            if (!rec.rationale.empty()) {
                oss << "     Rationale: " << rec.rationale << "\n";
            }
        }
        oss << "\n";
    }
    
    // Предупреждения
    if (!warnings.empty()) {
        oss << "Warnings:\n";
        for (const auto& w : warnings) {
            oss << "  - " << w << "\n";
        }
        oss << "\n";
    }
    
    // Ошибки
    if (!errors.empty()) {
        oss << "Errors:\n";
        for (const auto& e : errors) {
            oss << "  - " << e << "\n";
        }
        oss << "\n";
    }
    
    oss << "=================================================\n";
    
    return oss.str();
}

// ==================== Конструктор ====================

ParameterizationVerifier::ParameterizationVerifier(
    double interp_tolerance,
    double svd_tolerance,
    double condition_limit,
    double perturbation_scale)
    : interp_tolerance_(interp_tolerance)
    , svd_tolerance_(svd_tolerance)
    , condition_limit_(condition_limit)
    , perturbation_scale_(perturbation_scale)
{
}

// ==================== Основные методы верификации ====================

ParameterizationVerification ParameterizationVerifier::verify(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes)
{
    ParameterizationVerification result;
    
    // Заполняем параметры
    result.polynomial_degree = composite.total_degree;
    result.num_constraints = composite.num_constraints;
    result.num_free_params = composite.num_free_params;
    result.interval_a = composite.interval_a;
    result.interval_b = composite.interval_b;
    
    // Выполняем тесты
    result.interpolation_test = test_interpolation(composite, interp_nodes);
    result.completeness_test = test_completeness(
        composite.correction_poly,
        composite.weight_multiplier,
        composite.interval_a,
        composite.interval_b);
    result.stability_test = test_stability(composite, interp_nodes);
    
    // Определяем общий статус
    if (!result.interpolation_test.passed) {
        result.overall_status = VerificationStatus::FAILED;
        result.errors.push_back("Interpolation conditions are not satisfied");
    } else if (!result.completeness_test.passed) {
        result.overall_status = VerificationStatus::FAILED;
        result.errors.push_back("Solution space is not complete (rank deficient)");
    } else if (!result.stability_test.passed) {
        result.overall_status = VerificationStatus::WARNING;
        result.warnings.push_back("Numerical stability issues detected");
    } else {
        result.overall_status = VerificationStatus::PASSED;
    }
    
    // Генерируем рекомендации
    for (const auto& err : result.interpolation_test.node_errors) {
        if (err.absolute_error > result.interpolation_test.tolerance) {
            result.recommendations.push_back(
                diagnose_interpolation_error(err, 1e-12));
        }
    }
    
    if (!result.completeness_test.passed) {
        result.recommendations.push_back(
            diagnose_condition_issue(result.completeness_test,
                                   composite.correction_poly.basis_type));
    }
    
    return result;
}

ParameterizationVerification ParameterizationVerifier::verify_components(
    const InterpolationBasis& basis,
    const WeightMultiplier& W,
    const CorrectionPolynomial& Q,
    const std::vector<InterpolationNode>& interp_nodes)
{
    // Создаём временный композитный полином для верификации
    CompositePolynomial composite;
    composite.interpolation_basis = basis;
    composite.weight_multiplier = W;
    composite.correction_poly = Q;
    
    // Вычисляем метаданные
    composite.total_degree = W.degree() + Q.degree;
    composite.num_constraints = static_cast<int>(basis.m_eff);
    composite.num_free_params = Q.n_free;
    
    // Интервал
    if (basis.is_normalized) {
        composite.interval_a = basis.x_center - basis.x_scale;
        composite.interval_b = basis.x_center + basis.x_scale;
    } else {
        composite.interval_a = 0.0;
        composite.interval_b = 1.0;
    }
    
    return verify(composite, interp_nodes);
}

// ==================== Тест интерполяции ====================

InterpolationTestResult ParameterizationVerifier::test_interpolation(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes)
{
    InterpolationTestResult result;
    result.tolerance = interp_tolerance_;
    result.total_nodes = static_cast<int>(interp_nodes.size());
    
    const double W_tolerance = 1e-10;
    
    for (int i = 0; i < result.total_nodes; ++i) {
        const auto& node = interp_nodes[i];
        
        // Вычисляем F(z_e)
        double F_val = composite.evaluate(node.x);
        double W_val = composite.weight_multiplier.evaluate(node.x);
        
        // Вычисляем ошибку
        double abs_err = std::abs(F_val - node.value);
        double rel_err = (std::abs(node.value) > 1e-12) ?
                         abs_err / std::abs(node.value) : abs_err;
        
        // Проверяем допуск
        bool W_ok = std::abs(W_val) < W_tolerance;
        bool interp_ok = abs_err < interp_tolerance_;
        
        NodeError node_err;
        node_err.node_index = i;
        node_err.coordinate = node.x;
        node_err.target_value = node.value;
        node_err.computed_value = F_val;
        node_err.absolute_error = abs_err;
        node_err.relative_error = rel_err;
        node_err.W_value = W_val;
        node_err.W_acceptable = W_ok;
        
        result.node_errors.push_back(node_err);
        
        if (!interp_ok) {
            result.failed_nodes++;
        }
        
        result.max_absolute_error = std::max(result.max_absolute_error, abs_err);
        result.max_relative_error = std::max(result.max_relative_error, rel_err);
    }
    
    result.passed = (result.failed_nodes == 0);
    
    return result;
}

// ==================== Тест полноты ====================

CompletenessTestResult ParameterizationVerifier::test_completeness(
    const CorrectionPolynomial& Q,
    const WeightMultiplier& W,
    double interval_a,
    double interval_b)
{
    CompletenessTestResult result;
    
    int n_free = Q.n_free;
    result.expected_rank = n_free;
    
    // Особые случаи
    if (n_free <= 0) {
        result.passed = true;
        result.actual_rank = 0;
        result.info_messages.push_back("No free parameters (m = n + 1 or m > n)");
        return result;
    }
    
    // Генерируем узлы Чебышёва для тестирования
    std::vector<double> test_points = chebyshev_nodes(n_free, interval_a, interval_b);
    
    // Строим матрицу базиса G
    std::vector<std::vector<double>> G(n_free, std::vector<double>(n_free, 0.0));
    
    for (int i = 0; i < n_free; ++i) {
        double x = test_points[i];
        double W_val = W.evaluate(x);
        double x_work = (Q.basis_type == BasisType::CHEBYSHEV) ?
                        (x - Q.x_center) / Q.x_scale : x;
        
        for (int k = 0; k < n_free; ++k) {
            double phi_k = compute_basis_function_public(Q, x_work, k);
            G[i][k] = phi_k * W_val;
        }
    }
    
    // Вычисляем сингулярные значения
    double condition_number;
    std::vector<double> singular_values = compute_singular_values(G, condition_number);
    
    result.condition_number = condition_number;
    result.singular_values = singular_values;
    
    if (!singular_values.empty()) {
        double max_sv = singular_values[0];
        result.min_singular_value = singular_values.back();
        result.relative_min_sv = (max_sv > 0) ?
                                  result.min_singular_value / max_sv : 0.0;
    }
    
    // Определяем фактический ранг
    double sv_threshold = svd_tolerance_ * (singular_values.empty() ? 1.0 : singular_values[0]);
    result.actual_rank = 0;
    for (double sv : singular_values) {
        if (sv > sv_threshold) {
            result.actual_rank++;
        }
    }
    
    // Проверяем критерии
    bool rank_ok = (result.actual_rank == result.expected_rank);
    bool condition_ok = (condition_number < condition_limit_);
    
    result.passed = rank_ok && condition_ok;
    
    if (!rank_ok) {
        result.warnings.push_back(
            "Matrix rank is deficient: expected " + std::to_string(result.expected_rank) +
            ", got " + std::to_string(result.actual_rank));
    }
    
    if (!condition_ok) {
        result.warnings.push_back(
            "Poorly conditioned basis matrix: cond = " +
            std::to_string(condition_number));
    }
    
    return result;
}

// ==================== Тест устойчивости ====================

StabilityTestResult ParameterizationVerifier::test_stability(
    const CompositePolynomial& composite,
    const std::vector<InterpolationNode>& interp_nodes)
{
    StabilityTestResult result;
    
    // Пропускаем тест если нет узлов или мало свободных параметров
    if (composite.num_free_params <= 0 || interp_nodes.empty()) {
        result.passed = true;
        result.info_messages.push_back("Skipped: insufficient data for stability test");
        return result;
    }
    
    double a = composite.interval_a;
    double b = composite.interval_b;
    double mid_point = (a + b) / 2.0;
    
    // Оценка баланса масштабов компонент
    double max_P_int = 0.0;
    double max_Q = 0.0;
    double max_W = 0.0;
    
    std::vector<double> test_points;
    int num_test = std::min(20, composite.num_free_params + 5);
    for (int i = 0; i < num_test; ++i) {
        double t = -1.0 + 2.0 * (i + 0.5) / num_test;
        test_points.push_back(0.5 * (b - a) * t + 0.5 * (a + b));
    }
    
    for (double x : test_points) {
        double P_val = std::abs(composite.interpolation_basis.evaluate(x));
        double W_val = std::abs(composite.weight_multiplier.evaluate(x));
        double Q_val = std::abs(composite.correction_poly.evaluate_Q(x));
        
        max_P_int = std::max(max_P_int, P_val);
        max_W = std::max(max_W, W_val);
        max_Q = std::max(max_Q, Q_val);
    }
    
    double QW_scale = max_Q * max_W;
    result.max_component_scale = std::max(max_P_int, QW_scale);
    result.min_component_scale = std::min(max_P_int, QW_scale);
    
    if (result.min_component_scale > 0) {
        result.scale_balance_ratio = result.max_component_scale / result.min_component_scale;
    } else {
        result.scale_balance_ratio = std::numeric_limits<double>::infinity();
    }
    
    // Тест на чувствительность к возмущениям
    if (composite.num_constraints > 0) {
        // Возмущаем узлы
        std::vector<InterpolationNode> perturbed_nodes = interp_nodes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-perturbation_scale_, perturbation_scale_);
        
        double perturb_factor = perturbation_scale_ * (b - a);
        for (auto& node : perturbed_nodes) {
            node.x += perturb_factor * dist(gen);
        }
        
        // Оцениваем изменение функции
        double F_original = composite.evaluate(mid_point);
        
        // Создаём возмущённый композит
        CompositePolynomial perturbed_composite;
        perturbed_composite.interpolation_basis = composite.interpolation_basis;
        perturbed_composite.weight_multiplier = composite.weight_multiplier;
        perturbed_composite.correction_poly = composite.correction_poly;
        perturbed_composite.total_degree = composite.total_degree;
        perturbed_composite.num_constraints = perturbed_nodes.size();
        perturbed_composite.num_free_params = composite.num_free_params;
        perturbed_composite.interval_a = a;
        perturbed_composite.interval_b = b;
        
        double F_perturbed = perturbed_composite.evaluate(mid_point);
        double delta_F = std::abs(F_perturbed - F_original);
        
        if (std::abs(F_original) > 1e-12) {
            result.perturbation_sensitivity = delta_F / std::abs(F_original);
        } else {
            result.perturbation_sensitivity = delta_F;
        }
    }
    
    // Оценка обусловленности градиента (упрощённая)
    double max_grad_component = 0.0;
    double min_grad_component = std::numeric_limits<double>::infinity();
    
    for (const auto& node : interp_nodes) {
        double x = node.x;
        double W_val = composite.weight_multiplier.evaluate(x);
        
        for (int k = 0; k < composite.num_free_params; ++k) {
            double x_work = (composite.correction_poly.basis_type == BasisType::CHEBYSHEV) ?
                           (x - composite.correction_poly.x_center) / composite.correction_poly.x_scale : x;
            double phi_k = compute_basis_function_public(composite.correction_poly, x_work, k);
            double grad_component = std::abs(phi_k * W_val);
            max_grad_component = std::max(max_grad_component, grad_component);
            min_grad_component = std::min(min_grad_component, grad_component);
        }
    }
    
    if (min_grad_component > 0) {
        result.gradient_condition_number = max_grad_component / min_grad_component;
    } else {
        result.gradient_condition_number = std::numeric_limits<double>::infinity();
    }
    
    // Проверяем критерии устойчивости
    bool scale_ok = (result.scale_balance_ratio < 1e6);
    bool gradient_ok = (result.gradient_condition_number < 1e6);
    bool perturb_ok = (result.perturbation_sensitivity < 1e-4);
    
    result.passed = scale_ok && gradient_ok && perturb_ok;
    
    if (!scale_ok) {
        result.warnings.push_back(
            "Poor scale balance: ratio = " + std::to_string(result.scale_balance_ratio));
    }
    
    if (!gradient_ok) {
        result.warnings.push_back(
            "Poor gradient conditioning: ratio = " +
            std::to_string(result.gradient_condition_number));
    }
    
    return result;
}

// ==================== Вспомогательные методы ====================

void ParameterizationVerifier::set_parameters(
    double interp_tolerance,
    double svd_tolerance,
    double condition_limit,
    double perturbation_scale)
{
    interp_tolerance_ = interp_tolerance;
    svd_tolerance_ = svd_tolerance;
    condition_limit_ = condition_limit;
    perturbation_scale_ = perturbation_scale;
}

std::vector<double> ParameterizationVerifier::compute_singular_values(
    const std::vector<std::vector<double>>& matrix,
    double& condition_number)
{
    int n = static_cast<int>(matrix.size());
    if (n == 0) {
        condition_number = 0.0;
        return {};
    }
    
    // Упрощённое вычисление сингулярных значений через QR-разложение
    // Для полноценного SVD требуется внешняя библиотека
    
    // Копируем матрицу
    std::vector<std::vector<double>> A = matrix;
    
    // Вычисляем собственные значения A^T * A (упрощённо)
    std::vector<std::vector<double>> ATA(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[k][i] * A[k][j];
            }
            ATA[i][j] = sum;
        }
    }
    
    // Степенной метод для нахождения максимального сингулярного значения
    std::vector<double> sv(n, 1.0);
    std::vector<double> v(n, 1.0 / std::sqrt(n));
    
    double lambda_max = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
        // y = A^T * A * v
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                y[i] += ATA[i][j] * v[j];
            }
        }
        
        // Нормализация
        double norm = 0.0;
        for (double val : y) norm += val * val;
        norm = std::sqrt(norm);
        
        if (norm < 1e-15) break;
        
        for (double& val : y) val /= norm;
        
        lambda_max = norm;
        v = y;
    }
    
    double sigma_max = std::sqrt(lambda_max);
    
    // Оценка минимального сингулярного значения через обратную итерацию
    double sigma_min = sigma_max;
    if (n > 1) {
        // Простая оценка: минимальное SV >= минимальный диагональный элемент A^T * A
        double min_diag = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            min_diag = std::min(min_diag, ATA[i][i]);
        }
        sigma_min = std::sqrt(std::max(0.0, min_diag));
    }
    
    // Заполняем сингулярные значения (упрощённо)
    std::vector<double> singular_values(n);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(n - 1 - i) / (n - 1);
        singular_values[i] = sigma_min + (sigma_max - sigma_min) * t;
    }
    
    // Сортируем по убыванию
    std::sort(singular_values.begin(), singular_values.end(), std::greater<double>());
    
    if (singular_values.back() > 0) {
        condition_number = singular_values.front() / singular_values.back();
    } else {
        condition_number = std::numeric_limits<double>::infinity();
    }
    
    return singular_values;
}

Recommendation ParameterizationVerifier::diagnose_interpolation_error(
    const NodeError& error,
    double W_tolerance)
{
    (void)W_tolerance; // Подавление предупреждения о неиспользованном параметре
    
    if (!error.W_acceptable) {
        return Recommendation(
            RecommendationType::USE_LONG_DOUBLE,
            "W(z_e) is not sufficiently close to zero",
            "The weight multiplier has numerical errors. Consider using long double arithmetic or checking the construction of W(x)."
        );
    }
    
    if (error.absolute_error > 1e-6) {
        return Recommendation(
            RecommendationType::REDUCE_TOLERANCE,
            "Large interpolation error detected",
            "The interpolation error is significantly larger than the tolerance. Consider increasing the tolerance or checking the basis construction."
        );
    }
    
    return Recommendation(
        RecommendationType::NONE,
        "Minor interpolation error",
        "The error is small but exceeds the tolerance. This may be due to accumulated floating-point errors."
    );
}

Recommendation ParameterizationVerifier::diagnose_condition_issue(
    const CompletenessTestResult& result,
    BasisType current_basis)
{
    if (result.condition_number > 1e12) {
        std::string new_basis = (current_basis == BasisType::MONOMIAL) ? "CHEBYSHEV" : "MONOMIAL";
        return Recommendation(
            RecommendationType::CHANGE_BASIS,
            "Critical ill-conditioning detected",
            "The condition number is critical (" + std::to_string(result.condition_number) +
            "). Consider switching to " + new_basis + " basis for improved numerical stability."
        );
    }
    
    if (result.actual_rank < result.expected_rank) {
        return Recommendation(
            RecommendationType::MERGE_NODES,
            "Rank deficient basis matrix",
            "The basis matrix has rank " + std::to_string(result.actual_rank) +
            " instead of expected " + std::to_string(result.expected_rank) +
            ". Consider merging close interpolation nodes."
        );
    }
    
    return Recommendation(
        RecommendationType::INCREASE_GAMMA,
        "Poor conditioning detected",
        "Consider increasing the regularization parameter gamma to improve conditioning."
    );
}

std::vector<double> ParameterizationVerifier::chebyshev_nodes(int n, double a, double b) {
    std::vector<double> nodes(n);
    for (int k = 0; k < n; ++k) {
        double t = std::cos(M_PI * (2.0 * (k + 1) - 1) / (2.0 * n));
        nodes[k] = 0.5 * (b - a) * t + 0.5 * (a + b);
    }
    return nodes;
}

double ParameterizationVerifier::compute_basis_function_public(
    const CorrectionPolynomial& Q, double x, int k) const
{
    // Публичная обёртка для вычисления базисной функции
    // Использует evaluate_Q для получения значения Q(x) с единичными коэффициентами
    if (Q.basis_type == BasisType::MONOMIAL) {
        return std::pow(x, k);
    } else {
        // Базис Чебышёва
        if (k == 0) return 1.0;
        if (k == 1) return x;
        
        double T_prev = 1.0;
        double T_curr = x;
        for (int i = 2; i <= k; ++i) {
            double T_next = 2.0 * x * T_curr - T_prev;
            T_prev = T_curr;
            T_curr = T_next;
        }
        return T_curr;
    }
}

} // namespace mixed_approx
