#include "mixed_approximation/interpolation_basis.h"
#include <vector>
#include <cmath>

namespace mixed_approx {

double InterpolationBasis::evaluate_lagrange(double x) const {
    int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    
    double result = 0.0;
    
    for (int e = 0; e < m; ++e) {
        double Le = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == e) continue;
            double denom = nodes[e] - nodes[j];
            if (std::abs(denom) < 1e-14) {
                return values[e];
            }
            Le *= (x - nodes[j]) / denom;
        }
        result += values[e] * Le;
    }
    
    return result;
}

} // namespace mixed_approx
