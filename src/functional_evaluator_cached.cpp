#include "mixed_approximation/functional.h"
#include <cmath>

namespace mixed_approx {

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

} // namespace mixed_approx
