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

} // namespace mixed_approx
