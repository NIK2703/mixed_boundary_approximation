#include "mixed_approximation/functional.h"
#include <cmath>
#include <numeric>

namespace mixed_approx {

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

} // namespace mixed_approx
