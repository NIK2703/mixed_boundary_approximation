#include "mixed_approximation/interpolation_basis.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace mixed_approx {

double InterpolationBasis::evaluate(double x) const {
    if (!is_valid || nodes.empty()) {
        return 0.0;
    }
    
    double x_norm = x;
    if (is_normalized) {
        x_norm = (x - x_center) / x_scale;
    }
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        return evaluate_barycentric(x_norm);
    } else if (method == InterpolationMethod::NEWTON) {
        return evaluate_newton(x_norm);
    } else {
        return evaluate_lagrange(x_norm);
    }
}

double InterpolationBasis::evaluate_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("Derivative order must be 1 or 2");
    }
    
    if (!is_valid || nodes.empty()) {
        return 0.0;
    }
    
    double x_norm = x;
    if (is_normalized) {
        x_norm = (x - x_center) / x_scale;
    }
    
    double deriv_norm = 0.0;
    
    if (method == InterpolationMethod::BARYCENTRIC) {
        deriv_norm = evaluate_barycentric_derivative(x_norm, order);
    } else if (method == InterpolationMethod::NEWTON) {
        if (order == 1) {
            deriv_norm = evaluate_newton_derivative(x_norm);
        } else {
            deriv_norm = evaluate_newton_second_derivative(x_norm);
        }
    } else {
        const double h = 1e-6;
        if (order == 1) {
            double fp = evaluate(x_norm + h) * x_scale;
            double fm = evaluate(x_norm - h) * x_scale;
            deriv_norm = (fp - fm) / (2.0 * h);
        } else {
            double fp = evaluate(x_norm + h) * x_scale;
            double f = evaluate(x_norm) * x_scale;
            double fm = evaluate(x_norm - h) * x_scale;
            deriv_norm = (fp - 2.0 * f + fm) / (h * h);
        }
    }
    
    if (is_normalized) {
        if (order == 1) {
            return deriv_norm / x_scale;
        } else {
            return deriv_norm / (x_scale * x_scale);
        }
    }
    
    return deriv_norm;
}

double InterpolationBasis::evaluate_barycentric_derivative(double x, int order) const {
    int m = static_cast<int>(nodes.size());
    
    if (m <= 2) {
        if (order == 1) {
            return evaluate_newton_derivative(x);
        } else {
            return evaluate_newton_second_derivative(x);
        }
    }
    
    for (int k = 0; k < m; ++k) {
        if (std::abs(x - nodes[k]) < 1e-12) {
            const double h = 1e-8;
            if (order == 1) {
                double fp = evaluate_barycentric(x + h);
                double fm = evaluate_barycentric(x - h);
                return (fp - fm) / (2.0 * h);
            } else {
                double fp = evaluate_barycentric(x + h);
                double f = evaluate_barycentric(x);
                double fm = evaluate_barycentric(x - h);
                return (fp - 2.0 * f + fm) / (h * h);
            }
        }
    }
    
    if (order == 1) {
        double sum_w_over_dx = 0.0;
        double sum_wf_over_dx = 0.0;
        double sum_w_over_dx2 = 0.0;
        
        for (int k = 0; k < m; ++k) {
            double diff = x - nodes[k];
            double inv_diff = 1.0 / diff;
            double inv_diff2 = inv_diff * inv_diff;
            
            sum_w_over_dx += barycentric_weights[k] * inv_diff;
            sum_wf_over_dx += weighted_values.empty() ? 
                barycentric_weights[k] * values[k] * inv_diff : 
                weighted_values[k] * inv_diff;
            sum_w_over_dx2 += barycentric_weights[k] * inv_diff2;
        }
        
        double denominator = sum_w_over_dx * sum_w_over_dx;
        if (std::abs(denominator) < 1e-14) {
            return 0.0;
        }
        
        double p_x = evaluate_barycentric(x);
        double numerator = sum_wf_over_dx - p_x * sum_w_over_dx;
        return numerator / denominator;
    } else {
        return evaluate_newton_second_derivative(x);
    }
}

} // namespace mixed_approx
