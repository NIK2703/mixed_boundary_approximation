#include "mixed_approximation/functional.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace mixed_approx {

// ============== Реализация FunctionalEvaluator ==============

FunctionalEvaluator::FunctionalEvaluator(const ApproximationConfig& config,
                                         const BarrierParams& barrier_params)
    : config_(config), barrier_params_(barrier_params), normalization_params_() {}

void FunctionalEvaluator::set_barrier_params(const BarrierParams& params) {
    barrier_params_ = params;
}

void FunctionalEvaluator::set_normalization_params(const NormalizationParams& params) {
    normalization_params_ = params;
}

const NormalizationParams& FunctionalEvaluator::get_normalization_params() const {
    return normalization_params_;
}

double FunctionalEvaluator::evaluate_objective(const CompositePolynomial& param,
                                               const std::vector<double>& q) const {
    Components comp = evaluate_components(param, q);
    return comp.total;
}

void FunctionalEvaluator::evaluate_gradient(const CompositePolynomial& param,
                                           const std::vector<double>& q,
                                           std::vector<double>& grad) const {
    int n_free = static_cast<int>(q.size());
    grad.assign(n_free, 0.0);
    
    compute_approx_gradient(param, q, grad);
    compute_repel_gradient(param, q, grad);
    compute_reg_gradient(param, q, grad);
}

void FunctionalEvaluator::evaluate_objective_and_gradient(const CompositePolynomial& param,
                                                         const std::vector<double>& q,
                                                         double& f,
                                                         std::vector<double>& grad) const {
    evaluate_gradient(param, q, grad);
    f = evaluate_objective(param, q);
}

FunctionalEvaluator::Components FunctionalEvaluator::evaluate_components(
    const CompositePolynomial& param,
    const std::vector<double>& q) const {
    
    Components comp;
    comp.approx_component = compute_approx(param, q);
    comp.repel_component = compute_repel(param, q);
    comp.reg_component = compute_regularization(param, q);
    comp.total = comp.approx_component + comp.repel_component + comp.reg_component;
    return comp;
}

double FunctionalEvaluator::compute_approx(const CompositePolynomial& param,
                                           const std::vector<double>& q) const {
    double sum = 0.0;
    for (const auto& point : config_.approx_points) {
        // Вычисляем F(x_i) через ленивую оценку
        // F(x) = P_int(x) + Q(x) * W(x)
        double P_int = param.interpolation_basis.evaluate(point.x);
        double W = param.weight_multiplier.evaluate(point.x);
        double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
        double F = P_int + Q * W;
        
        double error = F - point.value;
        sum += (error * error) / point.weight;
    }
    return sum;
}

double FunctionalEvaluator::compute_repel(const CompositePolynomial& param,
                                           const std::vector<double>& q) const {
    double sum = 0.0;
    for (const auto& point : config_.repel_points) {
        double P_int = param.interpolation_basis.evaluate(point.x);
        double W = param.weight_multiplier.evaluate(point.x);
        double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
        double F = P_int + Q * W;
        
        double diff = point.y_forbidden - F;
        double dist_sq = diff * diff;
        double safe_dist_sq = std::max(dist_sq, config_.epsilon * config_.epsilon);
        sum += point.weight / safe_dist_sq;
    }
    return sum;
}

double FunctionalEvaluator::compute_regularization(const CompositePolynomial& param,
                                                    const std::vector<double>& q) const {
    if (config_.gamma == 0.0) {
        return 0.0;
    }
    
    // J_reg = γ ∫ (F''(x))^2 dx через компоненты
    // F''(x) = P_int''(x) + Q''(x)·W(x) + 2·Q'(x)·W'(x) + Q(x)·W''(x)
    
    double a = config_.interval_start;
    double b = config_.interval_end;
    int quad_points = 20;
    
    // Узлы квадратуры Гаусса-Лежандра (упрощённо - равномерная сетка)
    double integral = 0.0;
    double h = (b - a) / quad_points;
    
    for (int i = 0; i <= quad_points; ++i) {
        double x = a + i * h;
        
        double P_int = param.interpolation_basis.evaluate(x);
        double P_int2 = param.interpolation_basis.evaluate_derivative(x, 2);
        
        double W = param.weight_multiplier.evaluate(x);
        double W1 = param.weight_multiplier.evaluate_derivative(x, 1);
        double W2 = param.weight_multiplier.evaluate_derivative(x, 2);
        
        double Q = param.correction_poly.evaluate_Q_with_coeffs(x, q);
        double Q1 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 1);
        double Q2 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 2);
        
        // F''(x) = P_int''(x) + Q''(x)·W(x) + 2·Q'(x)·W'(x) + Q(x)·W''(x)
        double F2 = P_int2 + Q2 * W + 2.0 * Q1 * W1 + Q * W2;
        
        integral += F2 * F2;
    }
    
    integral *= h;
    return config_.gamma * integral;
}

void FunctionalEvaluator::compute_approx_gradient(const CompositePolynomial& param,
                                                  const std::vector<double>& q,
                                                  std::vector<double>& grad) const {
    int n_free = static_cast<int>(q.size());
    
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (const auto& point : config_.approx_points) {
            double x = point.x;
            double target = point.value;
            double weight = point.weight;
            
            double P_int = param.interpolation_basis.evaluate(x);
            double W = param.weight_multiplier.evaluate(x);
            
            // F(x) = P_int(x) + Q(x)·W(x)
            // ∂F/∂q_k = φ_k(x)·W(x), где φ_k - базисная функция Q(x)
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
            
            // ∂J_approx/∂q_k = 2·(F(x) - target)·φ_k(x)·W(x) / σ_i
            double F = P_int + param.correction_poly.evaluate_Q_with_coeffs(x, q) * W;
            double error = F - target;
            
            grad_k += 2.0 * error * phi_k * W / weight;
        }
        
        grad[k] += grad_k;
    }
}

void FunctionalEvaluator::compute_repel_gradient(const CompositePolynomial& param,
                                                  const std::vector<double>& q,
                                                  std::vector<double>& grad) const {
    int n_free = static_cast<int>(q.size());
    
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (const auto& point : config_.repel_points) {
            double x = point.x;
            double target = point.y_forbidden;
            double weight = point.weight;  // B_j
            
            double P_int = param.interpolation_basis.evaluate(x);
            double W = param.weight_multiplier.evaluate(x);
            
            double F = P_int + param.correction_poly.evaluate_Q_with_coeffs(x, q) * W;
            double diff = target - F;  // y_j^* - F(y_j)
            double dist_sq = diff * diff;
            double safe_dist_sq = std::max(dist_sq, config_.epsilon * config_.epsilon);
            
            // ∂J_repel/∂q_k = 2·B_j·(y_j^* - F)·φ_k(x)·W(x) / |y_j^* - F|^4
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
            
            grad_k += 2.0 * weight * diff * phi_k * W / (safe_dist_sq * safe_dist_sq);
        }
        
        grad[k] += grad_k;
    }
}

void FunctionalEvaluator::compute_reg_gradient(const CompositePolynomial& param,
                                                const std::vector<double>& q,
                                                std::vector<double>& grad) const {
    if (config_.gamma == 0.0) {
        return;
    }
    
    int n_free = static_cast<int>(q.size());
    int quad_points = 20;
    
    double a = config_.interval_start;
    double b = config_.interval_end;
    double h = (b - a) / quad_points;
    
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (int i = 0; i <= quad_points; ++i) {
            double x = a + i * h;
            
            // Компоненты F''(x)
            double P_int2 = param.interpolation_basis.evaluate_derivative(x, 2);
            
            double W = param.weight_multiplier.evaluate(x);
            double W1 = param.weight_multiplier.evaluate_derivative(x, 1);
            double W2 = param.weight_multiplier.evaluate_derivative(x, 2);
            
            double Q = param.correction_poly.evaluate_Q_with_coeffs(x, q);
            double Q1 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 1);
            double Q2 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 2);
            
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
            double phi_k1 = param.correction_poly.compute_basis_derivative_with_coeffs(x, q, k, 1);
            double phi_k2 = param.correction_poly.compute_basis_derivative_with_coeffs(x, q, k, 2);
            
            // F''(x) = P_int''(x) + Q''(x)·W(x) + 2·Q'(x)·W'(x) + Q(x)·W''(x)
            double F2 = P_int2 + Q2 * W + 2.0 * Q1 * W1 + Q * W2;
            
            // ∂F''(x)/∂q_k = φ_k''(x)·W(x) + 2·φ_k'(x)·W'(x) + φ_k(x)·W''(x)
            double dF2_dqk = phi_k2 * W + 2.0 * phi_k1 * W1 + phi_k * W2;
            
            grad_k += 2.0 * config_.gamma * F2 * dF2_dqk;
        }
        
        grad[k] += grad_k * h;
    }
}

double FunctionalEvaluator::compute_barrier_term(double dist, double weight, int& zone) const {
    const double eps = barrier_params_.epsilon_safe;
    const double k = barrier_params_.smoothing_factor;
    const double warning_factor = barrier_params_.warning_zone_factor;

    // Классификация зоны
    if (dist <= eps) {
        zone = 2;  // Критическая зона
        // Квадратичное сглаживание в критической зоне:
        // term_j = B_j / (ε² + k·(ε - dist)²)
        double smooth = eps * eps + k * (eps - dist) * (eps - dist);
        return weight / smooth;
    } else if (dist <= warning_factor * eps) {
        zone = 1;  // Предупредительная зона
        // Обычная формула с безопасным знаменателем
        double safe_dist_sq = std::max(dist * dist, eps * eps);
        return weight / safe_dist_sq;
    } else {
        zone = 0;  // Нормальная зона
        double dist_sq = dist * dist;
        return weight / dist_sq;
    }
}

double FunctionalEvaluator::compute_effective_weight(double base_weight, double dist, int zone) const {
    const double eps = barrier_params_.epsilon_safe;
    const double alpha = barrier_params_.adaptive_gain;

    // Динамическая адаптация силы барьера
    // При dist < 2·ε_safe временно увеличиваем вес:
    // effective_Bj = B_j · (1 + α · (2·ε - dist) / ε)
    if (zone == 2 && dist > 0) {
        double factor = 1.0 + alpha * (2.0 * eps - dist) / eps;
        return base_weight * std::max(factor, 1.0);
    }
    return base_weight;
}

RepulsionResult FunctionalEvaluator::compute_repel_withDiagnostics(
    const CompositePolynomial& param,
    const std::vector<double>& q) const {

    RepulsionResult result;
    result.total = 0.0;
    result.min_distance = std::numeric_limits<double>::infinity();
    result.max_distance = 0.0;
    result.critical_count = 0;
    result.warning_count = 0;
    result.barrier_activated = false;
    result.distances.clear();

    for (const auto& point : config_.repel_points) {
        double P_int = param.interpolation_basis.evaluate(point.x);
        double W = param.weight_multiplier.evaluate(point.x);
        double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
        double F = P_int + Q * W;

        double diff = point.y_forbidden - F;
        double dist = std::abs(diff);

        // Классификация зоны
        int zone = 0;
        if (dist <= barrier_params_.epsilon_safe) {
            zone = 2;
            result.critical_count++;
            result.barrier_activated = true;
        } else if (dist <= barrier_params_.warning_zone_factor * barrier_params_.epsilon_safe) {
            zone = 1;
            result.warning_count++;
        }

        // Вычисление барьерного члена
        double effective_weight = compute_effective_weight(point.weight, dist, zone);
        double term = compute_barrier_term(dist, effective_weight, zone);

        result.total += term;
        result.distances.push_back(dist);
        result.min_distance = std::min(result.min_distance, dist);
        result.max_distance = std::max(result.max_distance, dist);
    }

    // Проверка на барьерный коллапс
    if (result.total > 1e15) {
        // Это признак несовместимости критериев
    }

    return result;
}

FunctionalResult FunctionalEvaluator::evaluate_with_diagnostics(
    const CompositePolynomial& param,
    const std::vector<double>& q) const {

    FunctionalResult result;
    FunctionalDiagnostics& diag = result.diagnostics;

    // Вычисляем компоненты
    RepulsionResult repel_result = compute_repel_withDiagnostics(param, q);

    diag.raw_approx = compute_approx(param, q);
    diag.raw_repel = repel_result.total;
    diag.raw_reg = compute_regularization(param, q);

    // Проверка на численные аномалии
    bool has_nan = std::isnan(diag.raw_approx) || std::isnan(diag.raw_repel) || std::isnan(diag.raw_reg);
    bool has_inf = std::isinf(diag.raw_approx) || std::isinf(diag.raw_repel) || std::isinf(diag.raw_reg);

    if (has_nan) {
        result.status = FunctionalStatus::NAN_DETECTED;
        diag.has_numerical_anomaly = true;
        diag.anomaly_description = "Обнаружен NaN в компоненте функционала";
        result.value = std::numeric_limits<double>::infinity();
        return result;
    }

    if (has_inf) {
        result.status = FunctionalStatus::INF_DETECTED;
        diag.has_numerical_anomaly = true;
        diag.anomaly_description = "Обнаружен Inf в компоненте функционала";
        result.value = std::numeric_limits<double>::infinity();
        return result;
    }

    // Проверка на барьерный коллапс
    if (diag.raw_repel > 1e15) {
        result.status = FunctionalStatus::BARRIER_COLLAPSE;
        diag.has_numerical_anomaly = true;
        diag.anomaly_description = "Барьерный коллапс: полином слишком близко к запрещённой точке";
    }

    // Применение нормализации
    double norm_approx = diag.raw_approx / normalization_params_.scale_approx;
    double norm_repel = diag.raw_repel / normalization_params_.scale_repel;
    double norm_reg = diag.raw_reg / normalization_params_.scale_reg;

    diag.normalized_approx = norm_approx * normalization_params_.weight_approx;
    diag.normalized_repel = norm_repel * normalization_params_.weight_repel;
    diag.normalized_reg = norm_reg * normalization_params_.weight_reg;

    // Суммарный функционал
    result.value = diag.normalized_approx + diag.normalized_repel + diag.normalized_reg;
    diag.total_functional = result.value;

    // Вычисление долей компонент
    if (result.value > 0) {
        diag.share_approx = (diag.normalized_approx / result.value) * 100.0;
        diag.share_repel = (diag.normalized_repel / result.value) * 100.0;
        diag.share_reg = (diag.normalized_reg / result.value) * 100.0;
    }

    // Диагностика аппроксимации
    fill_approx_diagnostics(param, q, diag);

    // Диагностика отталкивания
    diag.min_repulsion_distance = repel_result.min_distance;
    diag.max_repulsion_distance = repel_result.max_distance;
    diag.repulsion_barrier_active = repel_result.barrier_activated;
    diag.repulsion_violations = repel_result.critical_count;

    // Диагностика регуляризации
    if (config_.gamma > 0) {
        diag.second_deriv_norm = std::sqrt(diag.raw_reg / config_.gamma);
    } else {
        diag.second_deriv_norm = 0.0;
    }

    // Проверка на переполнение
    if (result.value > 1e20) {
        result.status = FunctionalStatus::OVERFLOW;
        diag.has_numerical_anomaly = true;
        diag.anomaly_description = "Переполнение функционала (> 1e20)";
    }

    // Проверка на конфликт критериев
    if (diag.raw_approx > 1e10 && diag.raw_repel > 1e10) {
        result.status = FunctionalStatus::CRITERIA_CONFLICT;
    }

    // Проверка на пустые наборы точек
    if (config_.approx_points.empty() && config_.repel_points.empty()) {
        result.status = FunctionalStatus::EMPTY_APPROX_POINTS;
    } else if (config_.approx_points.empty()) {
        result.status = FunctionalStatus::EMPTY_APPROX_POINTS;
    } else if (config_.repel_points.empty()) {
        result.status = FunctionalStatus::EMPTY_REPEL_POINTS;
    }

    return result;
}

void FunctionalEvaluator::fill_approx_diagnostics(const CompositePolynomial& param,
                                                  const std::vector<double>& q,
                                                  FunctionalDiagnostics& diag) const {
    double max_res = 0.0;
    double min_res = std::numeric_limits<double>::infinity();
    double sum_res = 0.0;
    int count = 0;

    for (const auto& point : config_.approx_points) {
        double P_int = param.interpolation_basis.evaluate(point.x);
        double W = param.weight_multiplier.evaluate(point.x);
        double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
        double F = P_int + Q * W;

        double error = std::abs(F - point.value);
        max_res = std::max(max_res, error);
        min_res = std::min(min_res, error);
        sum_res += error;
        count++;
    }

    diag.max_residual = max_res;
    diag.min_residual = std::isfinite(min_res) ? min_res : 0.0;
    diag.mean_residual = count > 0 ? sum_res / count : 0.0;
}

void FunctionalEvaluator::fill_repel_diagnostics(const CompositePolynomial& param,
                                                const std::vector<double>& q,
                                                FunctionalDiagnostics& diag,
                                                const RepulsionResult& repel_result) const {
    diag.min_repulsion_distance = repel_result.min_distance;
    diag.max_repulsion_distance = repel_result.max_distance;
    diag.repulsion_barrier_active = repel_result.barrier_activated;
    diag.repulsion_violations = repel_result.critical_count;
}

void FunctionalEvaluator::initialize_normalization(const CompositePolynomial& param,
                                                   const std::vector<double>& q) {
    // Вычисляем характерные масштабы на начальном приближении
    Components initial_comp = evaluate_components(param, q);

    normalization_params_.scale_approx = std::max(1.0, initial_comp.approx_component);
    normalization_params_.scale_repel = std::max(1.0, initial_comp.repel_component);
    normalization_params_.scale_reg = std::max(1.0, initial_comp.reg_component);

    // Устанавливаем нормализованные веса
    normalization_params_.weight_approx = 1.0 / normalization_params_.scale_approx;
    normalization_params_.weight_repel = 1.0 / normalization_params_.scale_repel;
    normalization_params_.weight_reg = config_.gamma / normalization_params_.scale_reg;
}

} // namespace mixed_approx
