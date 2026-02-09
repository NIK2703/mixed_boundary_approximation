#include "mixed_approximation/functional.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

namespace mixed_approx {

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
