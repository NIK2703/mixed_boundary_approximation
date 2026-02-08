#include "mixed_approximation/initialization_strategy.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace mixed_approx {

InitializationStrategy InitializationStrategySelector::select(
    const CompositePolynomial& param,
    const OptimizationProblemData& data)
{
    if (data.num_approx_points() == 0) {
        return InitializationStrategy::ZERO;
    }
    
    if (data.num_repel_points() == 0) {
        return InitializationStrategy::LEAST_SQUARES;
    }
    
    for (double w : data.repel_weight) {
        if (w > 100.0) {
            return InitializationStrategy::ZERO;
        }
    }
    
    return InitializationStrategy::LEAST_SQUARES;
}

InitializationResult InitializationStrategySelector::initialize(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationStrategy strategy = select(param, data);
    
    switch (strategy) {
        case InitializationStrategy::ZERO:
            return zero_initialization(param);
        case InitializationStrategy::LEAST_SQUARES:
            return least_squares_initialization(param, data, functor);
        case InitializationStrategy::MULTI_START:
            return multi_start_initialization(param, data, functor);
        default:
            return zero_initialization(param);
    }
}

InitializationResult InitializationStrategySelector::zero_initialization(
    const CompositePolynomial& param)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::ZERO;
    
    int n_free = param.num_free_parameters();
    result.initial_coeffs.assign(n_free, 0.0);
    
    result.success = true;
    result.message = "Zero initialization: all coefficients set to zero";
    
    return result;
}

InitializationResult InitializationStrategySelector::least_squares_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::LEAST_SQUARES;
    
    int n_free = param.num_free_parameters();
    
    size_t N = data.num_approx_points();
    if (N == 0) {
        return zero_initialization(param);
    }
    
    std::vector<std::vector<double>> A(N, std::vector<double>(n_free, 0.0));
    
    for (size_t i = 0; i < N; ++i) {
        double x = data.approx_x[i];
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            A[i][k] = power * functor.parameterization().weight_multiplier.evaluate(x);
            power *= x;
        }
    }
    
    std::vector<double> b(N);
    for (size_t i = 0; i < N; ++i) {
        double F_base = functor.parameterization().interpolation_basis.evaluate(data.approx_x[i]);
        b[i] = data.approx_weight[i] * (data.approx_f[i] - F_base);
    }
    
    std::vector<std::vector<double>> ATA(n_free, std::vector<double>(n_free, 0.0));
    for (size_t i = 0; i < N; ++i) {
        double w = data.approx_weight[i];
        for (int k = 0; k < n_free; ++k) {
            for (int l = 0; l < n_free; ++l) {
                ATA[k][l] += w * A[i][k] * A[i][l];
            }
        }
    }
    
    std::vector<double> ATb(n_free, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (int k = 0; k < n_free; ++k) {
            ATb[k] += A[i][k] * b[i];
        }
    }
    
    double lambda = 1e-8;
    for (int k = 0; k < n_free; ++k) {
        ATA[k][k] += lambda;
    }
    
    result.initial_coeffs = ATb;
    for (int k = 0; k < n_free; ++k) {
        int max_row = k;
        for (int i = k + 1; i < n_free; ++i) {
            if (std::abs(ATA[i][k]) > std::abs(ATA[max_row][k])) {
                max_row = i;
            }
        }
        
        if (std::abs(ATA[max_row][k]) < 1e-15) {
            result.initial_coeffs[k] = 0.0;
            continue;
        }
        
        if (max_row != k) {
            std::swap(ATA[k], ATA[max_row]);
            std::swap(result.initial_coeffs[k], result.initial_coeffs[max_row]);
        }
        
        double piv = ATA[k][k];
        for (int j = k; j < n_free; ++j) {
            ATA[k][j] /= piv;
        }
        result.initial_coeffs[k] /= piv;
        
        for (int i = k + 1; i < n_free; ++i) {
            double factor = ATA[i][k];
            if (std::abs(factor) > 0) {
                for (int j = k; j < n_free; ++j) {
                    ATA[i][j] -= factor * ATA[k][j];
                }
                result.initial_coeffs[i] -= factor * result.initial_coeffs[k];
            }
        }
    }
    
    for (int k = n_free - 2; k >= 0; --k) {
        for (int i = k + 1; i < n_free; ++i) {
            result.initial_coeffs[k] -= ATA[k][i] * result.initial_coeffs[i];
        }
    }
    
    // Barrier protection
    if (data.num_repel_points() > 0) {
        functor.build_caches();
        double min_distance = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double phi_q = 0.0;
            double y = data.repel_y[j];
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += result.initial_coeffs[k] * power;
                power *= y;
            }
            double F_y = functor.parameterization().interpolation_basis.evaluate(y) + 
                         phi_q * functor.parameterization().weight_multiplier.evaluate(y);
            double distance = std::abs(data.repel_forbidden[j] - F_y);
            min_distance = std::min(min_distance, distance);
        }
        
        double d_safe = 0.01;
        if (min_distance < d_safe) {
            double scale = (d_safe / min_distance) * 0.5;
            for (int k = 0; k < n_free; ++k) {
                result.initial_coeffs[k] *= scale;
            }
        }
    }
    
    functor.build_caches();
    result.initial_objective = functor.value(result.initial_coeffs);
    result.success = true;
    result.message = "Least squares initialization with barrier protection";
    
    return result;
}

InitializationResult InitializationStrategySelector::multi_start_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor)
{
    InitializationResult result;
    result.strategy_used = InitializationStrategy::MULTI_START;
    
    int n_free = param.num_free_parameters();
    if (n_free > 10) {
        return least_squares_initialization(param, data, functor);
    }
    
    functor.build_caches();
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> best_coeffs(n_free, 0.0);
    double best_objective = functor.value(best_coeffs);
    
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<double> trial_coeffs(n_free);
        for (int k = 0; k < n_free; ++k) {
            trial_coeffs[k] = dist(rng);
        }
        
        std::vector<double> coeffs = trial_coeffs;
        double step = 0.01;
        
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<double> grad;
            double obj_value;
            functor.value_and_gradient(coeffs, obj_value, grad);
            
            for (int k = 0; k < n_free; ++k) {
                coeffs[k] -= step * grad[k];
            }
        }
        
        double objective = functor.value(coeffs);
        if (objective < best_objective) {
            best_objective = objective;
            best_coeffs = coeffs;
        }
    }
    
    result.initial_coeffs = best_coeffs;
    result.initial_objective = best_objective;
    result.success = true;
    result.message = "Multi-start initialization: best of 10 trials";
    
    return result;
}

} // namespace mixed_approx
