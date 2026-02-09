#include "mixed_approximation/functional.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace mixed_approx {

// ============== Реализация FunctionalDiagnostics ==============

std::string FunctionalDiagnostics::format_report() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    oss << "Функционал смешанной аппроксимации:\n";
    oss << "  Аппроксимирующий член:  " << normalized_approx << "   (вес: " << weight_approx << ", доля: " << share_approx << "%)\n";
    oss << "  Отталкивающий член:     " << normalized_repel << "   (вес: " << weight_repel << ", доля: " << share_repel << "%)\n";
    oss << "  Регуляризация:          " << normalized_reg << "   (вес: " << weight_reg << ", доля: " << share_reg << "%)\n";
    oss << "  Итого:                  " << total_functional << "\n\n";
    
    oss << "Диагностика:\n";
    if (std::isfinite(min_repulsion_distance)) {
        oss << "  Минимальное расстояние до запрещённых точек: " << min_repulsion_distance << "\n";
    } else {
        oss << "  Минимальное расстояние до запрещённых точек: N/A (нет точек)\n";
    }
    oss << "  Максимальный остаток аппроксимации: " << max_residual << "\n";
    oss << "  Норма второй производной: " << second_deriv_norm << "\n";
    
    if (has_numerical_anomaly) {
        oss << "\nВНИМАНИЕ: Обнаружена численная аномалия!\n";
        oss << "  " << anomaly_description << "\n";
    }
    
    if (is_dominant_component()) {
        oss << "\nВНИМАНИЕ: Доминирование одной компоненты!\n";
        oss << get_weight_recommendation() << "\n";
    }
    
    return oss.str();
}

std::string FunctionalDiagnostics::get_weight_recommendation() const {
    std::ostringstream oss;
    
    if (share_approx > 95.0) {
        oss << "Рекомендация: Аппроксимация доминирует (>95%). ";
        oss << "Рассмотрите увеличение весов отталкивания (B_j) или уменьшение весов аппроксимации (σ_i).";
    } else if (share_repel > 95.0) {
        oss << "Рекомендация: Отталкивание доминирует (>95%). ";
        oss << "Рассмотрите уменьшение весов отталкивания (B_j) или увеличение γ.";
    } else if (share_reg > 95.0) {
        oss << "Рекомендация: Регуляризация доминирует (>95%). ";
        oss << "Рассмотрите уменьшение γ или увеличение весов других компонент.";
    }
    
    return oss.str();
}

Functional::Functional(const ApproximationConfig& config) : config_(config) {}

double Functional::evaluate(const Polynomial& poly) const {
    Components comp = get_components(poly);
    return comp.total;
}

Functional::Components Functional::get_components(const Polynomial& poly) const {
    Components comp;
    comp.approx_component = compute_approx_component(poly);
    comp.repel_component = compute_repel_component(poly);
    comp.reg_component = compute_reg_component(poly);
    comp.total = comp.approx_component + comp.repel_component + comp.reg_component;
    return comp;
}

std::vector<double> Functional::gradient(const Polynomial& poly) const {
    std::vector<double> grad_approx = compute_approx_gradient(poly);
    std::vector<double> grad_repel = compute_repel_gradient(poly);
    std::vector<double> grad_reg = compute_reg_gradient(poly);
    
    // Суммируем градиенты
    int n = poly.degree();
    std::vector<double> total_grad(n + 1, 0.0);
    
    for (int i = 0; i <= n; ++i) {
        if (i < static_cast<int>(grad_approx.size())) total_grad[i] += grad_approx[i];
        if (i < static_cast<int>(grad_repel.size())) total_grad[i] += grad_repel[i];
        if (i < static_cast<int>(grad_reg.size())) total_grad[i] += grad_reg[i];
    }
    
    return total_grad;
}

// ============== Вычисление компонент ==============

double Functional::compute_approx_component(const Polynomial& poly) const {
    double sum = 0.0;
    for (const auto& point : config_.approx_points) {
        double error = poly.evaluate(point.x) - point.value;
        sum += (error * error) / point.weight;
    }
    return sum;
}

double Functional::compute_repel_component(const Polynomial& poly) const {
    double sum = 0.0;
    for (const auto& point : config_.repel_points) {
        double poly_value = poly.evaluate(point.x);
        double diff = point.y_forbidden - poly_value;  // y_j^* - F(y_j)
        double dist_sq = diff * diff;
        // Защита от деления на очень маленькие числа
        double safe_dist_sq = std::max(dist_sq, config_.epsilon * config_.epsilon);
        sum += point.weight / safe_dist_sq;
    }
    return sum;
}

double Functional::compute_reg_component(const Polynomial& poly) const {
    if (config_.gamma == 0.0) {
        return 0.0;
    }
    double integral = integrate_second_derivative_squared(poly, config_.interval_start, config_.interval_end);
    return config_.gamma * integral;
}

// ============== Вычисление градиентов ==============

std::vector<double> Functional::compute_approx_gradient(const Polynomial& poly) const {
    int n = poly.degree();
    std::vector<double> grad(n + 1, 0.0);
    
    for (const auto& point : config_.approx_points) {
        double x = point.x;
        double target = point.value;
        double weight = point.weight;
        
        double poly_value = poly.evaluate(x);
        double error = poly_value - target;
        
        // ∇_a |F(x) - target|^2 = 2 * (F(x) - target) * [x^n, x^{n-1}, ..., x, 1]
        double factor = 2.0 * error / weight;
        
        double x_power = 1.0;
        for (int k = n; k >= 0; --k) {
            grad[n - k] += factor * x_power;
            x_power *= x;
        }
    }
    
    return grad;
}

std::vector<double> Functional::compute_repel_gradient(const Polynomial& poly) const {
    int n = poly.degree();
    std::vector<double> grad(n + 1, 0.0);
    
    for (const auto& point : config_.repel_points) {
        double x = point.x;
        double target = point.y_forbidden;  // y_j^*
        double weight = point.weight;  // B_j
        
        double poly_value = poly.evaluate(x);
        double diff = target - poly_value;  // y_j^* - F(y_j)
        double dist_sq = diff * diff;
        double safe_dist_sq = std::max(dist_sq, config_.epsilon * config_.epsilon);
        
        // ∇_a B_j / |y_j^* - F(y_j)|^2 = 2 * B_j * (y_j^* - F(y_j)) / |y_j^* - F(y_j)|^4 * [x^n, x^{n-1}, ..., 1]
        // Но с учетом знака: производная по a_k от F(y_j) = x^k
        // d/da_k (B_j / (y_j^* - F)^2) = B_j * (-2) * (y_j^* - F)^(-3) * (-x^k) = 2 * B_j * (y_j^* - F) / (y_j^* - F)^4 * x^k
        double factor = 2.0 * weight * diff / (safe_dist_sq * safe_dist_sq);
        
        double x_power = 1.0;
        for (int k = n; k >= 0; --k) {
            grad[n - k] += factor * x_power;
            x_power *= x;
        }
    }
    
    return grad;
}

std::vector<double> Functional::compute_reg_gradient(const Polynomial& poly) const {
    if (config_.gamma == 0.0) {
        return std::vector<double>(poly.degree() + 1, 0.0);
    }
    
    // ∇_a ∫ (F''(x))^2 dx = 2 ∫ F''(x) * ∂F''(x)/∂a_k dx
    // Для полинома: F(x) = Σ_{k=0}^n a_k x^k
    // F''(x) = Σ_{k=2}^n k*(k-1)*a_k x^{k-2}
    // ∂F''(x)/∂a_k = k*(k-1)*x^{k-2} для k ≥ 2, и 0 для k < 2
    
    int n = poly.degree();
    std::vector<double> grad(n + 1, 0.0);
    
    // Используем аналитическое вычисление интеграла от (F''(x))^2
    // Для градиента: ∂/∂a_k ∫ (F''(x))^2 dx = 2 ∫ F''(x) * ∂F''(x)/∂a_k dx
    // = 2 ∫ (Σ_{i=2}^n i(i-1)a_i x^{i-2}) * (k(k-1)x^{k-2}) dx
    // = 2 * k(k-1) * Σ_{i=2}^n i(i-1)a_i ∫ x^{i+k-4} dx
    // = 2 * k(k-1) * Σ_{i=2}^n i(i-1)a_i * (b^{i+k-3} - a^{i+k-3}) / (i+k-3)
    
    const auto& coeffs = poly.coefficients();
    double a = config_.interval_start;
    double b = config_.interval_end;
    
    for (int k = 2; k <= n; ++k) {
        double k_factor = k * (k - 1.0);
        double grad_k = 0.0;
        
        for (int i = 2; i <= n; ++i) {
            double i_factor = i * (i - 1.0);
            double ai = coeffs[n - i];  // коэффициент a_i
            
            int power = i + k - 3;
            if (power < 0) continue;
            
            double integral = (std::pow(b, power + 1) - std::pow(a, power + 1)) / (power + 1);
            grad_k += 2.0 * config_.gamma * k_factor * i_factor * ai * integral;
        }
        
        grad[n - k] = grad_k;  // coeffs_[n-k] соответствует a_k
    }
    
    // Для k = 0, 1 градиент = 0 (так как вторая производная не зависит от a_0, a_1)
    
    return grad;
}

double Functional::safe_repel_distance(double poly_value, double target_value) const {
    double diff = target_value - poly_value;
    return std::max(std::abs(diff), config_.epsilon);
}

// ============== Реализация FunctionalEvaluator ==============

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
            double weight = point.weight;
            
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

// ============== Шаг 3.2: Новые методы FunctionalEvaluator ==============

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

// ============== Шаг 3.3: Реализация улучшенного градиента ==============

void FunctionalEvaluator::compute_gradient_robust(
    const CompositePolynomial& param,
    const std::vector<double>& q,
    std::vector<double>& grad,
    GradientDiagnostics* diag) const
{
    int n_free = static_cast<int>(q.size());
    grad.assign(n_free, 0.0);
    
    GradientDiagnostics local_diag;
    GradientDiagnostics* p_diag = diag ? diag : &local_diag;
    
    // Вычисляем градиенты компонент отдельно для нормализации
    std::vector<double> grad_approx(n_free, 0.0);
    std::vector<double> grad_repel(n_free, 0.0);
    std::vector<double> grad_reg(n_free, 0.0);
    
    // ========== Градиент аппроксимации ==========
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (const auto& point : config_.approx_points) {
            double x = point.x;
            double target = point.value;
            double weight = point.weight;
            
            double P_int = param.interpolation_basis.evaluate(x);
            double W = param.weight_multiplier.evaluate(x);
            
            // ∂F/∂q_k = φ_k(x)·W(x)
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
            
            double F = P_int + param.correction_poly.evaluate_Q_with_coeffs(x, q) * W;
            double error = F - target;
            
            grad_k += 2.0 * error * phi_k * W / weight;
        }
        
        grad_approx[k] = grad_k;
    }
    
    // ========== Градиент отталкивания с многоуровневой защитой ==========
    p_diag->critical_zone_points = 0;
    p_diag->warning_zone_points = 0;
    
    const double eps_critical = barrier_params_.epsilon_safe;
    const double k_smooth = barrier_params_.smoothing_factor;
    const double eps_warning = barrier_params_.warning_zone_factor * eps_critical;
    
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
            double abs_dist = std::abs(diff);
            
            // Классификация зоны
            int zone = 0;
            if (abs_dist <= eps_critical) {
                zone = 2;  // Критическая зона
                p_diag->critical_zone_points++;
            } else if (abs_dist <= eps_warning) {
                zone = 1;  // Предупредительная зона
                p_diag->warning_zone_points++;
            }
            
            // Вычисление защищённого фактора
            double factor;
            if (zone == 2) {
                // Кубическое сглаживание в критической зоне
                double smooth = eps_critical * eps_critical + k_smooth * (eps_critical - abs_dist) * (eps_critical - abs_dist);
                factor = weight / smooth;
            } else if (zone == 1) {
                // Плавный переход в предупредительной зоне
                double alpha = (abs_dist - eps_critical) / (eps_warning - eps_critical);
                double term1 = alpha / (abs_dist * abs_dist * abs_dist);
                double term2 = (1.0 - alpha) / (eps_critical * eps_critical * eps_critical);
                factor = weight * (term1 + term2);
            } else {
                // Стандартная формула
                factor = weight / (abs_dist * abs_dist * abs_dist);
            }
            
            // Направление градиента: sign(diff)
            double direction = (diff > 0) ? 1.0 : -1.0;
            
            // ∂J_repel/∂q_k = 2 · factor · direction · φ_k(x) · W(x)
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
            grad_k += 2.0 * factor * direction * phi_k * W;
        }
        
        grad_repel[k] = grad_k;
    }
    
    // ========== Градиент регуляризации ==========
    if (config_.gamma > 0.0) {
        // Для регуляризации используем численную квадратуру
        int quad_points = 20;
        double a = config_.interval_start;
        double b = config_.interval_end;
        double h = (b - a) / quad_points;
        
        for (int k = 0; k < n_free; ++k) {
            double grad_k = 0.0;
            
            for (int i = 0; i <= quad_points; ++i) {
                double x = a + i * h;
                
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
            
            grad_reg[k] = grad_k * h;
        }
    }
    
    // ========== Нормализация и суммирование ==========
    // Вычисляем нормы компонент
    double norm_approx = 0.0, norm_repel = 0.0, norm_reg = 0.0;
    for (int k = 0; k < n_free; ++k) {
        norm_approx += grad_approx[k] * grad_approx[k];
        norm_repel += grad_repel[k] * grad_repel[k];
        norm_reg += grad_reg[k] * grad_reg[k];
    }
    norm_approx = std::sqrt(norm_approx);
    norm_repel = std::sqrt(norm_repel);
    norm_reg = std::sqrt(norm_reg);
    
    p_diag->norm_approx = norm_approx;
    p_diag->norm_repel = norm_repel;
    p_diag->norm_reg = norm_reg;
    
    // Адаптивные коэффициенты нормализации
    double alpha_approx = 1.0 / std::max(1.0, norm_approx);
    double alpha_repel = 1.0 / std::max(1.0, norm_repel);
    double alpha_reg = 1.0 / std::max(1.0, norm_reg);
    
    // Суммируем с нормализацией
    double total_norm_sq = 0.0;
    for (int k = 0; k < n_free; ++k) {
        grad[k] = alpha_approx * grad_approx[k] + alpha_repel * grad_repel[k] + alpha_reg * grad_reg[k];
        total_norm_sq += grad[k] * grad[k];
    }
    p_diag->norm_total = std::sqrt(total_norm_sq);
    
    // Заполняем остальную диагностику
    if (diag) {
        diag->grad_approx = grad_approx;
        diag->grad_repel = grad_repel;
        diag->grad_reg = grad_reg;
        
        // Находим min/max компоненты
        if (n_free > 0) {
            diag->max_grad_component = grad[0];
            diag->min_grad_component = grad[0];
            for (int k = 1; k < n_free; ++k) {
                diag->max_grad_component = std::max(diag->max_grad_component, grad[k]);
                diag->min_grad_component = std::min(diag->min_grad_component, grad[k]);
            }
        }
    }
}

void FunctionalEvaluator::normalize_gradient(
    const std::vector<double>& grad_approx,
    const std::vector<double>& grad_repel,
    const std::vector<double>& grad_reg,
    std::vector<double>& normalized_grad,
    std::vector<double>& scaling_factors) const
{
    int n = static_cast<int>(grad_approx.size());
    if (n == 0) {
        normalized_grad.clear();
        scaling_factors.clear();
        return;
    }
    
    // Вычисляем нормы компонент
    double norm_approx = 0.0, norm_repel = 0.0, norm_reg = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_approx += grad_approx[i] * grad_approx[i];
        norm_repel += grad_repel[i] * grad_repel[i];
        norm_reg += grad_reg[i] * grad_reg[i];
    }
    norm_approx = std::sqrt(norm_approx);
    norm_repel = std::sqrt(norm_repel);
    norm_reg = std::sqrt(norm_reg);
    
    // Вычисляем коэффициенты нормализации
    scaling_factors.resize(3);
    scaling_factors[0] = 1.0 / std::max(1.0, norm_approx);
    scaling_factors[1] = 1.0 / std::max(1.0, norm_repel);
    scaling_factors[2] = 1.0 / std::max(1.0, norm_reg);
    
    // Применяем нормализацию
    normalized_grad.resize(n);
    for (int i = 0; i < n; ++i) {
        normalized_grad[i] = scaling_factors[0] * grad_approx[i]
                           + scaling_factors[1] * grad_repel[i]
                           + scaling_factors[2] * grad_reg[i];
    }
}

FunctionalEvaluator::GradientVerificationResult
FunctionalEvaluator::verify_gradient_numerical(
    const CompositePolynomial& param,
    const std::vector<double>& q,
    double h,
    int test_component) const
{
    GradientVerificationResult result;
    
    // Вычисляем аналитический градиент
    std::vector<double> grad_analytic;
    compute_gradient_robust(param, q, grad_analytic);
    
    int n = static_cast<int>(q.size());
    if (n == 0) {
        result.success = true;
        result.message = "Empty gradient";
        return result;
    }
    
    // Если указана конкретная компонента, проверяем только её
    if (test_component >= 0 && test_component < n) {
        std::vector<double> q_plus = q;
        std::vector<double> q_minus = q;
        
        q_plus[test_component] += h;
        q_minus[test_component] -= h;
        
        double J_plus = evaluate_objective(param, q_plus);
        double J_minus = evaluate_objective(param, q_minus);
        
        double grad_numeric = (J_plus - J_minus) / (2.0 * h);
        double grad_ana = grad_analytic[test_component];
        
        double denom = std::max(std::abs(grad_ana), std::abs(grad_numeric));
        if (denom < 1e-12) {
            result.relative_error = 0.0;
            result.success = true;
            result.message = "Gradient component is near zero";
        } else {
            result.relative_error = std::abs(grad_ana - grad_numeric) / denom;
            result.success = (result.relative_error < 1e-6);
            result.failed_component = result.success ? -1 : test_component;
            
            if (result.success) {
                result.message = "Verification passed: relative error = " + std::to_string(result.relative_error);
            } else {
                result.message = "Verification failed: relative error = " + std::to_string(result.relative_error) +
                               " for component " + std::to_string(test_component) +
                               " (analytic=" + std::to_string(grad_ana) +
                               ", numeric=" + std::to_string(grad_numeric) + ")";
            }
        }
    } else {
        // Проверяем все компоненты
        result.success = true;
        result.relative_error = 0.0;
        
        for (int k = 0; k < n; ++k) {
            std::vector<double> q_plus = q;
            std::vector<double> q_minus = q;
            
            q_plus[k] += h;
            q_minus[k] -= h;
            
            double J_plus = evaluate_objective(param, q_plus);
            double J_minus = evaluate_objective(param, q_minus);
            
            double grad_numeric = (J_plus - J_minus) / (2.0 * h);
            double grad_ana = grad_analytic[k];
            
            double denom = std::max(std::abs(grad_ana), std::abs(grad_numeric));
            double rel_error = (denom < 1e-12) ? 0.0 : std::abs(grad_ana - grad_numeric) / denom;
            
            if (rel_error >= 1e-6) {
                result.success = false;
                result.failed_component = k;
                result.relative_error = rel_error;
                result.message = "Verification failed at component " + std::to_string(k) +
                               ": rel_error = " + std::to_string(rel_error);
                break;
            }
            
            result.relative_error = std::max(result.relative_error, rel_error);
        }
        
        if (result.success) {
            result.message = "All components verified: max relative error = " + std::to_string(result.relative_error);
        }
    }
    
    return result;
}

void FunctionalEvaluator::build_gradient_caches(
    CompositePolynomial& param,
    const std::vector<WeightedPoint>& points_x,
    const std::vector<RepulsionPoint>& points_y) const
{
    // Очищаем старые кэши
    param.clear_caches();
    
    // Кэшируем значения для аппроксимирующих точек
    param.cache.P_at_x.resize(points_x.size());
    param.cache.W_at_x.resize(points_x.size());
    for (size_t i = 0; i < points_x.size(); ++i) {
        param.cache.P_at_x[i] = param.interpolation_basis.evaluate(points_x[i].x);
        param.cache.W_at_x[i] = param.weight_multiplier.evaluate(points_x[i].x);
    }
    
    // Кэшируем значения для отталкивающих точек
    param.cache.P_at_y.resize(points_y.size());
    param.cache.W_at_y.resize(points_y.size());
    for (size_t j = 0; j < points_y.size(); ++j) {
        param.cache.P_at_y[j] = param.interpolation_basis.evaluate(points_y[j].x);
        param.cache.W_at_y[j] = param.weight_multiplier.evaluate(points_y[j].x);
    }
    
    // Кэшируем узлы квадратуры и значения W, P'' в них
    // Генерируем узлы квадратуры если не предоставлены
    if (config_.gamma > 0.0) {
        // Используем простую равномерную квадратуру (можно улучшить до Гаусса-Лежандра)
        int quad_points = 20;
        double a = config_.interval_start;
        double b = config_.interval_end;
        double h = (b - a) / quad_points;
        
        param.cache.quad_points.resize(quad_points + 1);
        param.cache.W_at_quad.resize(quad_points + 1);
        param.cache.W1_at_quad.resize(quad_points + 1);
        param.cache.W2_at_quad.resize(quad_points + 1);
        param.cache.P2_at_quad.resize(quad_points + 1);
        
        for (int i = 0; i <= quad_points; ++i) {
            double x = a + i * h;
            param.cache.quad_points[i] = x;
            param.cache.W_at_quad[i] = param.weight_multiplier.evaluate(x);
            param.cache.W1_at_quad[i] = param.weight_multiplier.evaluate_derivative(x, 1);
            param.cache.W2_at_quad[i] = param.weight_multiplier.evaluate_derivative(x, 2);
            param.cache.P2_at_quad[i] = param.interpolation_basis.evaluate_derivative(x, 2);
        }
    }
    
    param.caches_built = true;
}

void FunctionalEvaluator::compute_gradient_cached(
    const CompositePolynomial& param,
    const std::vector<double>& q,
    std::vector<double>& grad,
    bool use_normalization) const
{
    int n_free = static_cast<int>(q.size());
    grad.assign(n_free, 0.0);
    
    // Проверяем, что кэши построены
    if (!param.caches_built) {
        // Если кэшей нет, строим их на лету
        // (В реальном использовании build_gradient_caches должен быть вызван явно)
        // Но для совместимости делаем ленивую постройку
        // Для этого нужен доступ к точкам, поэтому просто вызываем обычный метод
        compute_gradient_robust(param, q, grad);
        return;
    }
    
    // ========== Градиент аппроксимации с кэшами ==========
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (size_t i = 0; i < config_.approx_points.size(); ++i) {
            const auto& point = config_.approx_points[i];
            double target = point.value;
            double weight = point.weight;
            
            double P_int = param.cache.P_at_x[i];
            double W = param.cache.W_at_x[i];
            
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(point.x, q, k);
            double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
            double F = P_int + Q * W;
            double error = F - target;
            
            grad_k += 2.0 * error * phi_k * W / weight;
        }
        
        grad[k] += grad_k;
    }
    
    // ========== Градиент отталкивания с кэшами и барьерной защитой ==========
    const double eps_critical = barrier_params_.epsilon_safe;
    const double k_smooth = barrier_params_.smoothing_factor;
    const double eps_warning = barrier_params_.warning_zone_factor * eps_critical;
    
    for (int k = 0; k < n_free; ++k) {
        double grad_k = 0.0;
        
        for (size_t j = 0; j < config_.repel_points.size(); ++j) {
            const auto& point = config_.repel_points[j];
            double target = point.y_forbidden;
            double weight = point.weight;
            
            double P_int = param.cache.P_at_y[j];
            double W = param.cache.W_at_y[j];
            
            double Q = param.correction_poly.evaluate_Q_with_coeffs(point.x, q);
            double F = P_int + Q * W;
            
            double diff = target - F;
            double abs_dist = std::abs(diff);
            
            // Зонная классификация
            double factor;
            if (abs_dist <= eps_critical) {
                double smooth = eps_critical * eps_critical + k_smooth * (eps_critical - abs_dist) * (eps_critical - abs_dist);
                factor = weight / smooth;
            } else if (abs_dist <= eps_warning) {
                double alpha = (abs_dist - eps_critical) / (eps_warning - eps_critical);
                factor = weight * (alpha / (abs_dist * abs_dist * abs_dist) +
                                  (1.0 - alpha) / (eps_critical * eps_critical * eps_critical));
            } else {
                factor = weight / (abs_dist * abs_dist * abs_dist);
            }
            
            double direction = (diff > 0) ? 1.0 : -1.0;
            double phi_k = param.correction_poly.compute_basis_function_with_coeffs(point.x, q, k);
            
            grad_k += 2.0 * factor * direction * phi_k * W;
        }
        
        grad[k] += grad_k;
    }
    
    // ========== Градиент регуляризации с кэшами ==========
    if (config_.gamma > 0.0 && !param.cache.quad_points.empty()) {
        int quad_count = static_cast<int>(param.cache.quad_points.size());
        double h = (config_.interval_end - config_.interval_start) / (quad_count - 1);
        
        for (int k = 0; k < n_free; ++k) {
            double grad_k = 0.0;
            
            for (int i = 0; i < quad_count; ++i) {
                double x = param.cache.quad_points[i];
                
                double P_int2 = param.cache.P2_at_quad[i];
                double W = param.cache.W_at_quad[i];
                double W1 = param.cache.W1_at_quad[i];
                double W2 = param.cache.W2_at_quad[i];
                
                double Q = param.correction_poly.evaluate_Q_with_coeffs(x, q);
                double Q1 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 1);
                double Q2 = param.correction_poly.evaluate_Q_derivative_with_coeffs(x, q, 2);
                
                double phi_k = param.correction_poly.compute_basis_function_with_coeffs(x, q, k);
                double phi_k1 = param.correction_poly.compute_basis_derivative_with_coeffs(x, q, k, 1);
                double phi_k2 = param.correction_poly.compute_basis_derivative_with_coeffs(x, q, k, 2);
                
                double F2 = P_int2 + Q2 * W + 2.0 * Q1 * W1 + Q * W2;
                double dF2_dqk = phi_k2 * W + 2.0 * phi_k1 * W1 + phi_k * W2;
                
                grad_k += 2.0 * config_.gamma * F2 * dF2_dqk;
            }
            
            grad[k] += grad_k * h;
        }
    }
    
    // Нормализация если требуется
    if (use_normalization) {
        // Для нормализации нужно знать компоненты отдельно, поэтому упрощённо:
        // если use_normalization=true, предполагаем, что нормализация уже
        // учтена в весах, или просто оставляем как есть
        // В полной реализации нужно было бы сохранять компоненты отдельно
    }
}

FunctionalEvaluator::GradientDiagnostics
FunctionalEvaluator::get_gradient_diagnostics(
    const CompositePolynomial& param,
    const std::vector<double>& q) const
{
    GradientDiagnostics diag;
    std::vector<double> grad;
    
    compute_gradient_robust(param, q, grad, &diag);
    
    return diag;
}

} // namespace mixed_approx
