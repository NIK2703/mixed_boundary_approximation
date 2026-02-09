#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/convergence_monitor.h"

#ifdef HAS_NLOPT
#include <nlopt.hpp>
#endif

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <memory>
#include <utility>

namespace mixed_approx {

#ifdef HAS_NLOPT

// Вспомогательная структура для передачи данных в NLopt
struct NLoptContext {
    const ObjectiveFunctor* functor;      // функтор для вычисления J(q) и ∇J(q)
    const Functional* functional;          // функционал для обратной совместимости
    CompositePolynomial* composite;        // композитный полином для построения F(x) (опционально)
    ConvergenceMonitor* monitor;          // монитор сходимости
    double initial_objective;              // начальное значение функционала
    double initial_gradient_norm;          // начальная норма градиента
    double max_barrier_weight;             // максимальный вес барьера
    bool stop_requested_by_monitor;        // флаг запроса остановки
    StopReason monitor_stop_reason;        // причина остановки от монитора
    bool use_functional;                  // использовать Functional вместо ObjectiveFunctor
    
    NLoptContext()
        : functor(nullptr), functional(nullptr), composite(nullptr), monitor(nullptr)
        , initial_objective(0.0), initial_gradient_norm(0.0), max_barrier_weight(0.0)
        , stop_requested_by_monitor(false), monitor_stop_reason(StopReason::NOT_CONVERGED)
        , use_functional(false) {}
};

// Функция цели для NLopt (с поддержкой Functional и ObjectiveFunctor)
static double nlopt_objective(unsigned n, const double* x, double* grad, void* f_data) {
    NLoptContext* context = static_cast<NLoptContext*>(f_data);
    
    // Коэффициенты Q(x) или коэффициенты полинома
    std::vector<double> coeffs(x, x + n);
    
    double objective = 0.0;
    std::vector<double> grad_vec;
    
    if (context->use_functional && context->functional != nullptr) {
        // Functional путь - работаем с Polynomial напрямую
        Polynomial poly(coeffs);
        
        if (grad) {
            grad_vec = context->functional->gradient(poly);
            objective = context->functional->evaluate(poly);
            std::copy(grad_vec.begin(), grad_vec.end(), grad);
        } else {
            objective = context->functional->evaluate(poly);
        }
        
        // Интеграция с ConvergenceMonitor
        if (context->monitor != nullptr) {
            auto comps = context->functional->get_components(poly);
            
            double grad_norm = 0.0;
            if (!grad_vec.empty()) {
                grad_norm = std::sqrt(std::inner_product(grad_vec.begin(), grad_vec.end(), grad_vec.begin(), 0.0));
            }
            
            // Компоненты функционала
            Functional::Components func_comps;
            func_comps.approx_component = comps.approx_component;
            func_comps.repel_component = comps.repel_component;
            func_comps.reg_component = comps.reg_component;
            func_comps.total = comps.total;
            
            context->monitor->record_iteration(objective, grad_norm, func_comps, 0.0);
            
            if (context->monitor->iteration() == 1) {
                context->initial_objective = objective;
                context->initial_gradient_norm = grad_norm;
            }
            
            StopReason reason = context->monitor->check_stop_criteria(
                grad_norm, objective, func_comps, 0.0,
                context->initial_objective, context->initial_gradient_norm,
                context->max_barrier_weight);
            
            if (reason != StopReason::NOT_CONVERGED) {
                context->stop_requested_by_monitor = true;
                context->monitor_stop_reason = reason;
                throw nlopt::forced_stop();
            }
        }
    } else if (context->functor != nullptr) {
        // ObjectiveFunctor путь
        if (grad) {
            context->functor->value_and_gradient(coeffs, objective, grad_vec);
            std::copy(grad_vec.begin(), grad_vec.end(), grad);
        } else {
            objective = context->functor->value(coeffs);
        }
        
        // Интеграция с ConvergenceMonitor
        if (context->monitor != nullptr) {
            auto comp = context->functor->compute_components(coeffs);
            
            double grad_norm = 0.0;
            if (grad) {
                grad_norm = std::sqrt(std::inner_product(grad_vec.begin(), grad_vec.end(), grad_vec.begin(), 0.0));
            } else {
                context->functor->gradient(coeffs, grad_vec);
                grad_norm = std::sqrt(std::inner_product(grad_vec.begin(), grad_vec.end(), grad_vec.begin(), 0.0));
            }
            
            // Преобразуем компоненты
            Functional::Components func_comps;
            func_comps.approx_component = comp.approx;
            func_comps.repel_component = comp.repel;
            func_comps.reg_component = comp.reg;
            func_comps.total = comp.total;
            
            context->monitor->record_iteration(objective, grad_norm, func_comps, 0.0);
            
            if (context->monitor->iteration() == 1) {
                context->initial_objective = objective;
                context->initial_gradient_norm = grad_norm;
            }
            
            StopReason reason = context->monitor->check_stop_criteria(
                grad_norm, objective, func_comps, 0.0,
                context->initial_objective, context->initial_gradient_norm,
                context->max_barrier_weight);
            
            if (reason != StopReason::NOT_CONVERGED) {
                context->stop_requested_by_monitor = true;
                context->monitor_stop_reason = reason;
                throw nlopt::forced_stop();
            }
        }
    }
    
    return objective;
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

// Реализация для ObjectiveFunctor
OptimizationResult LBFGSOptimizer::optimize(const ObjectiveFunctor& functor,
                                            const std::vector<double>& initial_q) {
    OptimizationResult result;
    result.coefficients = initial_q;
    result.success = false;
    
#ifdef HAS_NLOPT
    const size_t n = initial_q.size();
    if (n == 0) {
        result.message = "Empty initial coefficients";
        result.iterations = 0;
        return result;
    }

    try {
        // Определяем монитор: используем внешний если задан, иначе создаём локальный
        ConvergenceMonitor* monitor_ptr = monitor_;
        std::unique_ptr<ConvergenceMonitor> local_monitor_holder;
        
        if (monitor_ptr == nullptr) {
            local_monitor_holder.reset(new ConvergenceMonitor(tol_gradient_, tol_objective_));
            monitor_ptr = local_monitor_holder.get();
            // Настраиваем локальный монитор
            monitor_ptr->start_timer();
            monitor_ptr->max_iterations = max_iterations_;
            monitor_ptr->timeout_seconds = 300.0;  // 5 минут по умолчанию
            monitor_ptr->window_size = 5;
            monitor_ptr->plateau_patience = 3;
            monitor_ptr->barrier_threshold = 0.9;
        }
        
        ConvergenceMonitor& monitor = *monitor_ptr;
        
        // Вычисляем максимальный вес барьера из данных функтора
        double max_B = 0.0;
        const auto& repel_points = functor.data().repel_y;
        const auto& repel_forbidden = functor.data().repel_forbidden;
        const auto& repel_weight = functor.data().repel_weight;
        for (size_t i = 0; i < repel_points.size(); ++i) {
            max_B = std::max(max_B, repel_weight[i]);
        }
        
        // Создаем оптимизатор NLopt
        nlopt::opt optimizer(nlopt::LD_LBFGS, n);
        
        // Устанавливаем ограничения на параметры если заданы
        if (bounds_set_ && lower_bounds_.size() == n && upper_bounds_.size() == n) {
            optimizer.set_lower_bounds(lower_bounds_);
            optimizer.set_upper_bounds(upper_bounds_);
        }
        
        // Устанавливаем критерии останова NLopt (будут перекрываться нашим монитором)
        optimizer.set_xtol_rel(tol_gradient_);
        optimizer.set_ftol_rel(tol_objective_);
        optimizer.set_maxeval(max_iterations_);
        
        // Настраиваем параметры L-BFGS
        optimizer.set_lbfgs_mem(lbfgs_memory_);
        
        // Создаем контекст
        NLoptContext context;
        context.functor = &functor;
        context.functional = nullptr;
        context.composite = nullptr;
        context.monitor = &monitor;
        context.initial_objective = 0.0;
        context.initial_gradient_norm = 0.0;
        context.max_barrier_weight = max_B;
        context.stop_requested_by_monitor = false;
        context.monitor_stop_reason = StopReason::NOT_CONVERGED;
        context.use_functional = false;
        
        // Запускаем оптимизацию
        std::vector<double> x = initial_q;
        double minf;
        
        try {
            optimizer.optimize(x, minf);
        } catch (const nlopt::roundoff_limited& e) {
            result.message = "Optimization completed with roundoff limitations";
            result.success = true;
        } catch (const nlopt::forced_stop& e) {
            if (context.stop_requested_by_monitor) {
                result.success = (context.monitor_stop_reason == StopReason::RELATIVE_OBJECTIVE_CHANGE ||
                                 context.monitor_stop_reason == StopReason::GRADIENT_NORM ||
                                 context.monitor_stop_reason == StopReason::SUCCESS);
                if (result.success) {
                    result.message = "Converged by custom criteria: " + std::to_string(static_cast<int>(context.monitor_stop_reason));
                } else {
                    result.message = "Stopped by custom criteria (diagnostic): " + std::to_string(static_cast<int>(context.monitor_stop_reason));
                    result.success = false;
                }
            } else {
                result.message = "Optimization stopped by NLopt criteria";
                result.success = true;
            }
        } catch (const std::exception& e) {
            result.message = std::string("NLopt error: ") + e.what();
            result.success = false;
        }
        
        // Заполняем результат
        result.coefficients = x;
        result.final_objective = minf;
        result.iterations = optimizer.get_numevals();
        
        if (context.stop_requested_by_monitor) {
            result.iterations = monitor.iteration();
        }
        
        if (!context.stop_requested_by_monitor && optimizer.get_numevals() >= max_iterations_) {
            result.message = "Max iterations reached";
            result.success = false;
        } else if (!result.success && !context.stop_requested_by_monitor) {
            result.message = "L-BFGS-B did not converge";
        }
        
        // Сохраняем отчёт монитора
        if (context.monitor != nullptr) {
            result.message += "\n--- Convergence Monitor Report ---\n";
            result.message += monitor.get_diagnostic_info();
        }
        
    } catch (const std::exception& e) {
        result.message = std::string("Exception during optimization: ") + e.what();
        result.iterations = 0;
        result.success = false;
    }
#else
    result.message = "NLopt not available. L-BFGS-B optimizer disabled.";
    result.iterations = 0;
    result.success = false;
#endif
    
    return result;
}

// Реализация для Functional (для обратной совместимости)
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
        // Определяем монитор: используем внешний если задан, иначе создаём локальный
        ConvergenceMonitor* monitor_ptr = monitor_;
        std::unique_ptr<ConvergenceMonitor> local_monitor_holder;
        
        if (monitor_ptr == nullptr) {
            local_monitor_holder.reset(new ConvergenceMonitor(tol_gradient_, tol_objective_));
            monitor_ptr = local_monitor_holder.get();
            // Настраиваем локальный монитор
            monitor_ptr->start_timer();
            monitor_ptr->max_iterations = max_iterations_;
            monitor_ptr->timeout_seconds = 300.0;  // 5 минут по умолчанию
            monitor_ptr->window_size = 5;
            monitor_ptr->plateau_patience = 3;
            monitor_ptr->barrier_threshold = 0.9;
        }
        
        ConvergenceMonitor& monitor = *monitor_ptr;
        
        // Создаем оптимизатор NLopt
        nlopt::opt optimizer(nlopt::LD_LBFGS, n);
        
        // Устанавливаем ограничения на параметры если заданы
        if (bounds_set_ && lower_bounds_.size() == n && upper_bounds_.size() == n) {
            optimizer.set_lower_bounds(lower_bounds_);
            optimizer.set_upper_bounds(upper_bounds_);
        }
        
        // Устанавливаем критерии останова NLopt (будут перекрываться нашим монитором)
        optimizer.set_xtol_rel(tol_gradient_);
        optimizer.set_ftol_rel(tol_objective_);
        optimizer.set_maxeval(max_iterations_);
        
        // Настраиваем параметры L-BFGS
        optimizer.set_lbfgs_mem(lbfgs_memory_);
        
        // Создаем контекст для Functional
        NLoptContext context;
        context.functor = nullptr;
        context.functional = &functional;
        context.composite = nullptr;
        context.monitor = &monitor;
        context.initial_objective = 0.0;
        context.initial_gradient_norm = 0.0;
        context.max_barrier_weight = 0.0;
        context.stop_requested_by_monitor = false;
        context.monitor_stop_reason = StopReason::NOT_CONVERGED;
        context.use_functional = true;
        
        // Запускаем оптимизацию
        std::vector<double> x = initial_coeffs;
        double minf;
        
        try {
            optimizer.optimize(x, minf);
        } catch (const nlopt::roundoff_limited& e) {
            result.message = "Optimization completed with roundoff limitations";
            result.success = true;
        } catch (const nlopt::forced_stop& e) {
            if (context.stop_requested_by_monitor) {
                result.success = (context.monitor_stop_reason == StopReason::RELATIVE_OBJECTIVE_CHANGE ||
                                 context.monitor_stop_reason == StopReason::GRADIENT_NORM ||
                                 context.monitor_stop_reason == StopReason::SUCCESS);
                if (result.success) {
                    result.message = "Converged by custom criteria: " + std::to_string(static_cast<int>(context.monitor_stop_reason));
                } else {
                    result.message = "Stopped by custom criteria (diagnostic): " + std::to_string(static_cast<int>(context.monitor_stop_reason));
                    result.success = false;
                }
            } else {
                result.message = "Optimization stopped by NLopt criteria";
                result.success = true;
            }
        } catch (const std::exception& e) {
            result.message = std::string("NLopt error: ") + e.what();
            result.success = false;
        }
        
        // Заполняем результат
        result.coefficients = x;
        result.final_objective = minf;
        result.iterations = optimizer.get_numevals();
        
        if (context.stop_requested_by_monitor) {
            result.iterations = monitor.iteration();
        }
        
        if (!context.stop_requested_by_monitor && optimizer.get_numevals() >= max_iterations_) {
            result.message = "Max iterations reached";
            result.success = false;
        } else if (!result.success && !context.stop_requested_by_monitor) {
            result.message = "L-BFGS-B did not converge";
        }
        
        // Сохраняем отчёт монитора
        if (context.monitor != nullptr) {
            result.message += "\n--- Convergence Monitor Report ---\n";
            result.message += monitor.get_diagnostic_info();
        }
        
    } catch (const std::exception& e) {
        result.message = std::string("Exception during optimization: ") + e.what();
        result.iterations = 0;
        result.success = false;
    }
#else
    result.message = "NLopt not available. L-BFGS-B optimizer disabled.";
    result.iterations = 0;
    result.success = false;
#endif
    
    return result;
}

} // namespace mixed_approx
