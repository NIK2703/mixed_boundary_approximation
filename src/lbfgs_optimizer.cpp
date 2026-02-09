#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/functional.h"
#include "mixed_approximation/polynomial.h"

#ifdef HAS_NLOPT
#include <nlopt.hpp>
#endif

#include <cmath>
#include <algorithm>
#include <limits>

namespace mixed_approx {

#ifdef HAS_NLOPT

// Вспомогательная структура для передачи данных в NLopt
struct NLoptContext {
    const Functional* functional;
    Polynomial* poly;
};

// Функция цели для NLopt (только значение)
static double nlopt_objective(unsigned n, const double* x, double* grad, void* f_data) {
    NLoptContext* context = static_cast<NLoptContext*>(f_data);
    const Functional* functional = context->functional;
    Polynomial* poly = context->poly;
    
    // Обновляем коэффициенты полинома
    poly->setCoefficients(std::vector<double>(x, x + n));
    
    if (grad) {
        // Вычисляем градиент
        std::vector<double> grad_vec = functional->gradient(*poly);
        std::copy(grad_vec.begin(), grad_vec.end(), grad);
        return functional->evaluate(*poly);
    }
    
    return functional->evaluate(*poly);
}

#endif // HAS_NLOPT

LBFGSOptimizer::LBFGSOptimizer()
    : lbfgs_memory_(10)
    , max_line_search_iters_(20)
    , bounds_set_(false) {
}

LBFGSOptimizer::~LBFGSOptimizer() {
}

void LBFGSOptimizer::set_lbfgs_parameters(int max_iterations,
                                           double tol_gradient,
                                           double tol_objective,
                                           double initial_step,
                                           int lbfgs_memory,
                                           int max_line_search_iters) {
    Optimizer::set_parameters(max_iterations, tol_gradient, tol_objective, initial_step);
    lbfgs_memory_ = lbfgs_memory;
    max_line_search_iters_ = max_line_search_iters;
}

void LBFGSOptimizer::set_bounds(const std::vector<double>& lower_bound, 
                                const std::vector<double>& upper_bound) {
    lower_bounds_ = lower_bound;
    upper_bounds_ = upper_bound;
    bounds_set_ = true;
}

OptimizationResult LBFGSOptimizer::optimize(const Functional& functional,
                                            const std::vector<double>& initial_coeffs) {
    OptimizationResult result;
    result.coefficients = initial_coeffs;
    result.success = false;
    
#ifdef HAS_NLOPT
    const size_t n = initial_coeffs.size();
    if (n == 0) {
        result.message = "Empty initial coefficients";
        result.iterations = 0;
        return result;
    }

    try {
        // Создаем оптимизатор NLopt
        nlopt::opt optimizer(nlopt::LD_LBFGS, n);
        
        // Устанавливаем ограничения на параметры если заданы
        if (bounds_set_ && lower_bounds_.size() == n && upper_bounds_.size() == n) {
            optimizer.set_lower_bounds(lower_bounds_);
            optimizer.set_upper_bounds(upper_bounds_);
        }
        
        // Устанавливаем критерии останова
        optimizer.set_xtol_rel(tol_gradient_);
        optimizer.set_ftol_rel(tol_objective_);
        optimizer.set_maxeval(max_iterations_);
        
        // Настраиваем параметры L-BFGS
        optimizer.set_lbfgs_mem(lbfgs_memory_);
        
        // Создаем полином и контекст
        Polynomial poly(initial_coeffs);
        NLoptContext context{&functional, &poly};
        
        // Запускаем оптимизацию
        std::vector<double> x = initial_coeffs;
        double minf;
        
        try {
            optimizer.optimize(x, minf);
        } catch (const nlopt::roundoff_limited& e) {
            // Некоторые алгоритмы NLopt могут бросить это исключение при проблемах с точностью
            result.message = "Optimization completed with roundoff limitations";
            result.success = true;
        } catch (const nlopt::forced_stop& e) {
            result.message = "Optimization stopped by callback";
            result.success = false;
        } catch (const std::exception& e) {
            result.message = std::string("NLopt error: ") + e.what();
            result.success = false;
        }
        
        // Заполняем результат
        result.coefficients = x;
        result.final_objective = minf;
        result.iterations = optimizer.get_numevals();
        
        // Проверяем статус сходимости
        if (optimizer.get_numevals() >= max_iterations_) {
            result.message = "Max iterations reached";
            result.success = false;
        } else if (result.success) {
            result.message = "L-BFGS-B converged successfully";
        }
        
    } catch (const std::exception& e) {
        result.message = std::string("Exception during optimization: ") + e.what();
        result.iterations = 0;
        result.success = false;
    }
#else // !HAS_NLOPT
    result.message = "NLopt not available. L-BFGS-B optimizer disabled.";
    result.iterations = 0;
    result.success = false;
#endif // HAS_NLOPT
    
    return result;
}

} // namespace mixed_approx