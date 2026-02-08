#include "mixed_approximation/composite_polynomial.h"
#include "mixed_approximation/gauss_quadrature.h"
#include <vector>
#include <cmath>

namespace mixed_approx {

void CompositePolynomial::build_caches(const std::vector<double>& points_x,
                                      const std::vector<double>& points_y,
                                      const std::vector<double>& quad_points) {
    clear_caches();
    
    cache.P_at_x.resize(points_x.size());
    cache.W_at_x.resize(points_x.size());
    for (size_t i = 0; i < points_x.size(); ++i) {
        cache.P_at_x[i] = interpolation_basis.evaluate(points_x[i]);
        cache.W_at_x[i] = weight_multiplier.evaluate(points_x[i]);
    }
    
    cache.P_at_y.resize(points_y.size());
    cache.W_at_y.resize(points_y.size());
    for (size_t i = 0; i < points_y.size(); ++i) {
        cache.P_at_y[i] = interpolation_basis.evaluate(points_y[i]);
        cache.W_at_y[i] = weight_multiplier.evaluate(points_y[i]);
    }
    
    int quad_n;
    if (!quad_points.empty()) {
        quad_n = static_cast<int>(quad_points.size());
        cache.quad_points = quad_points;
    } else {
        quad_n = std::max(10, 2 * total_degree + 1);
        std::vector<double> quad_weights;
        get_gauss_legendre_quadrature(quad_n, cache.quad_points, quad_weights);
    }
    
    cache.W_at_quad.resize(quad_n);
    cache.W1_at_quad.resize(quad_n);
    cache.W2_at_quad.resize(quad_n);
    cache.Q_at_quad.resize(quad_n);
    cache.Q1_at_quad.resize(quad_n);
    cache.Q2_at_quad.resize(quad_n);
    cache.P2_at_quad.resize(quad_n);
    
    for (int i = 0; i < quad_n; ++i) {
        double x = cache.quad_points[i];
        cache.W_at_quad[i] = weight_multiplier.evaluate(x);
        cache.W1_at_quad[i] = weight_multiplier.evaluate_derivative(x, 1);
        cache.W2_at_quad[i] = weight_multiplier.evaluate_derivative(x, 2);
        cache.Q_at_quad[i] = correction_poly.evaluate_Q(x);
        cache.Q1_at_quad[i] = correction_poly.evaluate_Q_derivative(x, 1);
        cache.Q2_at_quad[i] = correction_poly.evaluate_Q_derivative(x, 2);
        cache.P2_at_quad[i] = interpolation_basis.evaluate_derivative(x, 2);
    }
    
    caches_built = true;
}

void CompositePolynomial::clear_caches() {
    cache.P_at_x.clear();
    cache.W_at_x.clear();
    cache.P_at_y.clear();
    cache.W_at_y.clear();
    cache.quad_points.clear();
    cache.W_at_quad.clear();
    cache.W1_at_quad.clear();
    cache.W2_at_quad.clear();
    cache.Q_at_quad.clear();
    cache.Q1_at_quad.clear();
    cache.Q2_at_quad.clear();
    cache.P2_at_quad.clear();
    caches_built = false;
}

double CompositePolynomial::compute_regularization_term(double gamma) const {
    if (gamma <= 0.0 || total_degree < 2) {
        return 0.0;
    }
    
    int quad_n = std::max(10, 2 * total_degree + 1);
    
    std::vector<double> quad_nodes, quad_weights;
    get_gauss_legendre_quadrature(quad_n, quad_nodes, quad_weights);
    
    double integral = 0.0;
    double scale_factor = 0.5 * (interval_b - interval_a);
    
    for (int i = 0; i < quad_n; ++i) {
        double t = quad_nodes[i];
        double weight = quad_weights[i];
        double x = transform_to_standard_interval(t, interval_a, interval_b);
        
        double p2 = interpolation_basis.evaluate_derivative(x, 2);
        double q = correction_poly.evaluate_Q(x);
        double q1 = correction_poly.evaluate_Q_derivative(x, 1);
        double q2 = correction_poly.evaluate_Q_derivative(x, 2);
        double w = weight_multiplier.evaluate(x);
        double w1 = weight_multiplier.evaluate_derivative(x, 1);
        double w2 = weight_multiplier.evaluate_derivative(x, 2);
        
        double f2 = p2 + q2 * w + 2.0 * q1 * w1 + q * w2;
        
        double integrand = f2 * f2;
        integral += integrand * weight;
    }
    
    integral *= scale_factor;
    
    return gamma * integral;
}

} // namespace mixed_approx
