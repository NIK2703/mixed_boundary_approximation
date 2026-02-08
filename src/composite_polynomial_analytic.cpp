#include "mixed_approximation/composite_polynomial.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace mixed_approx {

bool CompositePolynomial::build_analytic_coefficients(int max_degree_for_analytic) {
    analytic_coeffs.clear();
    analytic_coeffs_valid = false;
    
    if (total_degree > max_degree_for_analytic) {
        validation_message = "Degree " + std::to_string(total_degree) + 
                            " exceeds maximum for analytic assembly (" + 
                            std::to_string(max_degree_for_analytic) + "). Use lazy evaluation.";
        return false;
    }
    
    if (num_constraints == total_degree + 1) {
        analytic_coeffs = extract_p_int_coefficients();
        analytic_coeffs_valid = !analytic_coeffs.empty();
        if (analytic_coeffs_valid) {
            validation_message = "Analytic coefficients: F(x) = P_int(x) (full interpolation)";
        }
        return analytic_coeffs_valid;
    }
    
    std::vector<double> p_int_coeffs = extract_p_int_coefficients();
    if (p_int_coeffs.empty() && num_constraints > 0) {
        validation_message = "Failed to extract P_int coefficients";
        return false;
    }
    
    std::vector<double> w_coeffs = weight_multiplier.get_coeffs_ascending();
    std::vector<double> q_coeffs = correction_poly.coeffs;
    
    std::vector<double> qw_coeffs;
    if (q_coeffs.empty() || q_coeffs.size() == 1 && q_coeffs[0] == 0.0) {
        qw_coeffs.clear();
    } else {
        qw_coeffs = convolve_coefficients(q_coeffs, w_coeffs);
    }
    
    std::cout << "    [DEBUG build_analytic] QW coeffs (" << qw_coeffs.size() << "): ";
    for (double c : qw_coeffs) std::cout << c << " "; std::cout << "\n";
    
    int p_int_size = static_cast<int>(p_int_coeffs.size());
    int qw_size = static_cast<int>(qw_coeffs.size());
    int result_size = std::max(p_int_size, qw_size);
    
    if (result_size == 0) {
        analytic_coeffs = {0.0};
        analytic_coeffs_valid = true;
        validation_message = "Analytic coefficients: zero polynomial";
        return true;
    }
    
    analytic_coeffs.resize(result_size, 0.0);
    
    for (int i = 0; i < p_int_size; ++i) {
        analytic_coeffs[i] += p_int_coeffs[i];
    }
    
    for (int i = 0; i < qw_size; ++i) {
        analytic_coeffs[i] += qw_coeffs[i];
    }
    
    while (analytic_coeffs.size() > 1 && 
           std::abs(analytic_coeffs.back()) < 1e-14) {
        analytic_coeffs.pop_back();
    }
    
    analytic_coeffs_valid = true;
    validation_message = "Analytic coefficients built successfully for degree " + 
                        std::to_string(total_degree);
    
    return true;
}

double CompositePolynomial::evaluate_analytic(double x) const {
    if (!analytic_coeffs_valid || analytic_coeffs.empty()) {
        throw std::runtime_error("Analytic coefficients not available. Call build_analytic_coefficients first.");
    }
    
    double result = 0.0;
    for (int i = static_cast<int>(analytic_coeffs.size()) - 1; i >= 0; --i) {
        result = result * x + analytic_coeffs[i];
    }
    return result;
}

void CompositePolynomial::evaluate_batch_analytic(const std::vector<double>& points,
                                                   std::vector<double>& results) const {
    if (!analytic_coeffs_valid) {
        evaluate_batch(points, results);
        return;
    }
    
    results.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        results[i] = evaluate_analytic(points[i]);
    }
}

std::vector<double> CompositePolynomial::extract_p_int_coefficients() const {
    if (interpolation_basis.nodes_original.empty() || interpolation_basis.values.empty()) {
        return std::vector<double>();
    }
    
    const std::vector<double>& work_nodes = interpolation_basis.nodes_original;
    const std::vector<double>& work_values = interpolation_basis.values;
    int m = static_cast<int>(work_nodes.size());
    
    if (m <= 8) {
        std::vector<double> coeffs_ascending(m, 0.0);
        
        for (int j = 0; j < m; ++j) {
            std::vector<double> L_coeff(1, 1.0);
            
            double denom = 1.0;
            for (int k = 0; k < m; ++k) {
                if (k == j) continue;
                denom *= (work_nodes[k] - work_nodes[j]);
                
                double z_k = work_nodes[k];
                std::vector<double> new_L(L_coeff.size() + 1, 0.0);
                for (size_t i = 0; i < L_coeff.size(); ++i) {
                    new_L[i] += -z_k * L_coeff[i];
                    new_L[i + 1] += L_coeff[i];
                }
                L_coeff = new_L;
            }
            
            double weight = work_values[j] / denom;
            
            for (size_t i = 0; i < L_coeff.size(); ++i) {
                coeffs_ascending[i] += weight * L_coeff[i];
            }
        }
        
        while (coeffs_ascending.size() > 1 && 
               std::abs(coeffs_ascending.back()) < 1e-15) {
            coeffs_ascending.pop_back();
        }
        
        return coeffs_ascending;
    }
    
    std::vector<double> nodes = work_nodes;
    std::vector<double> values = work_values;
    
    int n = m - 1;
    
    std::vector<std::vector<double>> V(m, std::vector<double>(m));
    for (int i = 0; i < m; ++i) {
        double power = 1.0;
        for (int j = 0; j <= n; ++j) {
            V[i][j] = power;
            power *= nodes[i];
        }
    }
    
    for (int col = 0; col < m - 1; ++col) {
        int max_row = col;
        double max_val = std::abs(V[col][col]);
        for (int row = col + 1; row < m; ++row) {
            if (std::abs(V[row][col]) > max_val) {
                max_val = std::abs(V[row][col]);
                max_row = row;
            }
        }
        
        if (max_val < 1e-14) {
            return std::vector<double>();
        }
        
        if (max_row != col) {
            std::swap(V[max_row], V[col]);
            std::swap(values[max_row], values[col]);
        }
        
        for (int row = col + 1; row < m; ++row) {
            double factor = V[row][col] / V[col][col];
            values[row] -= factor * values[col];
            for (int j = col; j < m; ++j) {
                V[row][j] -= factor * V[col][j];
            }
        }
    }
    
    std::vector<double> coeffs_ascending(m);
    for (int i = m - 1; i >= 0; --i) {
        double sum = values[i];
        for (int j = i + 1; j < m; ++j) {
            sum -= V[i][j] * coeffs_ascending[j];
        }
        coeffs_ascending[i] = sum / V[i][i];
    }
    
    return coeffs_ascending;
}

std::vector<double> CompositePolynomial::convolve_coefficients(const std::vector<double>& q_coeffs,
                                                               const std::vector<double>& w_coeffs) const {
    if (q_coeffs.empty() || w_coeffs.empty()) {
        return std::vector<double>();
    }
    
    int deg_Q = static_cast<int>(q_coeffs.size()) - 1;
    int deg_W = static_cast<int>(w_coeffs.size()) - 1;
    int deg_result = deg_Q + deg_W;
    
    std::vector<double> result(deg_result + 1, 0.0);
    
    for (int i = 0; i <= deg_Q; ++i) {
        for (int j = 0; j <= deg_W; ++j) {
            int k = i + j;
            result[k] += q_coeffs[i] * w_coeffs[j];
        }
    }
    
    while (result.size() > 1 && 
           std::abs(result.back()) < 1e-14) {
        result.pop_back();
    }
    
    return result;
}

} // namespace mixed_approx
