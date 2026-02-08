#include "mixed_approximation/interpolation_basis.h"
#include <vector>
#include <cmath>

namespace mixed_approx {

void InterpolationBasis::compute_divided_differences() {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return;
    
    divided_differences.resize(m);
    
    for (int i = 0; i < m; ++i) {
        divided_differences[i] = values[i];
    }
    
    for (int level = 1; level < m; ++level) {
        for (int i = m - 1; i >= level; --i) {
            double denom = nodes[i] - nodes[i - level];
            if (std::abs(denom) < 1e-14) {
                divided_differences[i] = 0.0;
            } else {
                divided_differences[i] = (divided_differences[i] - divided_differences[i-1]) / denom;
            }
        }
    }
}

double InterpolationBasis::evaluate_newton(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    double result = divided_differences[0];
    double product = 1.0;
    
    for (int level = 1; level < m; ++level) {
        product *= (x - nodes[level-1]);
        result += divided_differences[level] * product;
    }
    
    return result;
}

double InterpolationBasis::evaluate_newton_derivative(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    if (m == 1) {
        return 0.0;
    }
    
    double deriv = divided_differences[1];
    double product = 1.0;
    
    for (int level = 2; level < m; ++level) {
        product *= (x - nodes[level-1]);
        deriv += divided_differences[level] * product;
    }
    
    return deriv;
}

double InterpolationBasis::evaluate_newton_second_derivative(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m < 3) {
        return 0.0;
    }
    
    double deriv2 = 0.0;
    
    for (int k = 2; k < m; ++k) {
        double sum_over_pairs = 0.0;
        
        for (int i = 0; i < k; ++i) {
            for (int l = i + 1; l < k; ++l) {
                double prod = 1.0;
                for (int j = 0; j < k; ++j) {
                    if (j == i || j == l) continue;
                    prod *= (x - nodes[j]);
                }
                sum_over_pairs += prod;
            }
        }
        
        deriv2 += 2.0 * divided_differences[k] * sum_over_pairs;
    }
    
    return deriv2;
}

} // namespace mixed_approx
