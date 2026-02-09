#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/decomposition.h"
#include "mixed_approximation/optimizer.h"
#include "mixed_approximation/convergence_monitor.h"
#include "mixed_approximation/solution_validator.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/optimization_problem_data.h"
#include "mixed_approximation/validator.h"
#include "mixed_approximation/initialization_strategy.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include <memory>
#include <chrono>

namespace mixed_approx {

// ============== Реализация методов класса MixedApproximation ==============

MixedApproximation::MixedApproximation()
    : polynomial_(), parametrization_built_(false), has_config_(false) {
}

MixedApproximation::MixedApproximation(const ApproximationConfig& config)
    : polynomial_(), parametrization_built_(false), has_config_(false) {
    initialize_from_config(config);
}

void MixedApproximation::initialize_from_config(const ApproximationConfig& config) {
    config_ = config;
    has_config_ = true;
    
    // Валидация конфигурации - проверяем на критические ошибки
    std::string validation_error = Validator::validate(config);
    if (!validation_error.empty()) {
        throw std::invalid_argument("Invalid configuration: " + validation_error);
    }
    
    // Дополнительная проверка на конфликты repel/interp
    std::string conflict_error = Validator::check_repel_interp_value_conflict(config);
    if (!conflict_error.empty()) {
        throw std::invalid_argument("FATAL: " + conflict_error);
    }
    
    // Преобразуем конфигурацию в OptimizationProblemData
    problem_data_ = OptimizationProblemData(config);
    
    // Вычисляем n_free из конфигурации
    int m = static_cast<int>(config.interp_nodes.size());
    int n_free = config.polynomial_degree - m + 1;
    if (n_free < 0) n_free = 0;
    
    // Строим начальное приближение
    build_initial_approximation(problem_data_, n_free);
}

InitializationResult MixedApproximation::build_initial_approximation(const OptimizationProblemData& data, int n_free) {
    InitializationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Валидация входных данных
        if (n_free < 0) {
            result.message = "Invalid n_free: must be non-negative";
            return result;
        }
        
        if (n_free == 0) {
            // Специальный случай: нет свободных параметров.
            // Полином полностью определяется интерполяционными условиями.
            // Строим разложение и возвращаем P_int(x) как результат.
        }
        
        if (data.approx_x.size() != data.approx_f.size() ||
            data.approx_x.size() != data.approx_weight.size()) {
            result.message = "Mismatched approx_points sizes";
            return result;
        }
        
        if (data.interp_z.size() != data.interp_f.size()) {
            result.message = "Mismatched interp_nodes sizes";
            return result;
        }
        
        if (data.repel_y.size() != data.repel_forbidden.size() ||
            data.repel_y.size() != data.repel_weight.size()) {
            result.message = "Mismatched repel_points sizes";
            return result;
        }
        
        // Сохраняем данные задачи
        problem_data_ = data;
        
        // Построение разложения через Decomposer
        Decomposer::Parameters params;
        int m = static_cast<int>(data.interp_z.size());
        int n = (m > 0) ? (n_free + m - 1) : (n_free - 1); // общая степень полинома F(x)
        params.polynomial_degree = n;
        params.interval_start = data.interval_a;
        params.interval_end = data.interval_b;
        
        // Преобразуем интерполяционные узлы
        params.interp_nodes.clear();
        for (size_t i = 0; i < data.interp_z.size(); ++i) {
            params.interp_nodes.emplace_back(data.interp_z[i], data.interp_f[i]);
        }
        
        params.epsilon_rank = 1e-12;
        params.epsilon_unique = 1e-12;
        params.epsilon_bound = 1e-9;
        
        DecompositionResult decomp = Decomposer::decompose(params);
        
        if (!decomp.is_valid()) {
            result.message = "Decomposition failed: " + decomp.message();
            result.success = false;
            return result;
        }
        
        // Построение композитного полинома
        composite_poly_ = decomp.composite_poly;
        // Устанавливаем корректирующий полином Q(x) с нулевыми коэффициентами
        CorrectionPolynomial zero_Q;
        int deg_Q = n_free - 1;
        if (deg_Q >= 0) {
            zero_Q.initialize(deg_Q, BasisType::MONOMIAL,
                              (data.interval_a + data.interval_b) * 0.5,
                              (data.interval_b - data.interval_a) * 0.5);
            composite_poly_.correction_poly = zero_Q;
        }
        composite_poly_.num_free_params = n_free;
        composite_poly_.total_degree = composite_poly_.correction_poly.degree + composite_poly_.weight_multiplier.degree();
        
        // Начальные коэффициенты Q: нули
        std::vector<double> zero_q(n_free, 0.0);
        polynomial_ = composite_poly_.build_polynomial(zero_q);
        
        parametrization_built_ = true;
        result.success = true;
        result.message = "Initial approximation built successfully";
        result.initial_coeffs = zero_q;
        result.strategy_used = InitializationStrategy::ZERO;
        
        // Проверка безопасности барьеров и возмущение при необходимости (шаг 1.2.6)
        // Если отталкивающая точка слишком близко к интерполяционному полиному,
        // применяем возмущение к свободным коэффициентам
        if (data.repel_y.size() > 0) {
            const InterpolationBasis& basis = composite_poly_.interpolation_basis;
            const WeightMultiplier& weight = composite_poly_.weight_multiplier;
            double eps_safe = 1e-8;
            
            // Проверяем расстояние до каждой отталкивающей точки
            for (size_t j = 0; j < data.repel_y.size(); ++j) {
                double y = data.repel_y[j];
                double y_star = data.repel_forbidden[j];
                
                // Вычисляем F(y) = P_int(y) + Q(y) * W(y) = P_int(y) + 0 * W(y) = P_int(y)
                double F_y = basis.evaluate(y);
                double distance = std::abs(y_star - F_y);
                
                if (distance < eps_safe) {
                    // Критическая ситуация: нужно возмутить коэффициенты Q
                    // Возмущаем в направлении, удаляющем от запрещённого значения
                    double sign = (y_star > F_y) ? 1.0 : -1.0;
                    double W_y = weight.evaluate(y);
                    
                    if (std::abs(W_y) > 1e-15) {
                        // Добавляем небольшое возмущение к младшему коэффициенту Q
                        // F(y) -> F(y) + delta * W(y), где delta выбирается так,
                        // чтобы |y_star - F(y)| увеличилось
                        double delta = sign * eps_safe * 2.0 / std::abs(W_y);
                        zero_q[0] += delta;
                        
                        // Перестраиваем полином с возмущёнными коэффициентами
                        polynomial_ = composite_poly_.build_polynomial(zero_q);
                        result.message += " (applied barrier perturbation)";
                        result.strategy_used = InitializationStrategy::BARRIER_PERTURBATION;
                        break;
                    }
                }
            }
        }
        
        result.initial_coeffs = zero_q; // ZERO strategy
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

OptimizationResult MixedApproximation::solve() {
    // Для обратной совместимости - используем ранее установленную конфигурацию
    if (!has_config_ || !parametrization_built_) {
        OptimizationResult result;
        result.success = false;
        result.message = "No configuration set. Use solve(data, n_free) instead.";
        return result;
    }
    
    // Вычисляем n_free из конфигурации
    // n_free = polynomial_degree - m + 1, где m = число интерполяционных узлов
    int m = static_cast<int>(problem_data_.interp_z.size());
    int n_free = config_.polynomial_degree - m + 1;
    if (n_free < 0) n_free = 0;
    
    // Если нет свободных параметров, оптимизация тривиальна - полином уже определён
    if (n_free == 0) {
        OptimizationResult result;
        result.success = true;
        result.message = "No free parameters - polynomial fully determined by interpolation";
        result.iterations = 0;
        result.final_objective = 0.0;
        result.elapsed_time = 0.0;
        result.converged = true;
        result.coefficients = {};
        result.final_polynomial = std::make_shared<Polynomial>(polynomial_);
        
        // Валидация решения
        SolutionValidator validator;
        ValidationResult validation = validator.validate(polynomial_, problem_data_);
        result.validation = validation;
        
        return result;
    }
    
    return solve(problem_data_, n_free, nullptr);
}

OptimizationResult MixedApproximation::solve(const OptimizationProblemData& data, int n_free, std::unique_ptr<Optimizer> optimizer) {
    OptimizationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Построение параметризации если ещё не построена или данные изменились
        if (!parametrization_built_) {
            auto init_result = build_initial_approximation(data, n_free);
            if (!init_result.success) {
                result.message = "Initialization failed: " + init_result.message;
                result.success = false;
                return result;
            }
        } else {
            // Обновляем данные задачи для функтора
            problem_data_ = data;
        }
        
        // Создаём монитор сходимости
        auto monitor = std::make_unique<ConvergenceMonitor>(1e-6, 1e-6); // пороги по умолчанию
        monitor->start_timer();
        monitor->max_iterations = 1000;
        monitor->timeout_seconds = 300.0;
        
        // Если передан внешний оптимизатор, используем его, иначе создаём LBFGS
        if (!optimizer) {
            optimizer = std::make_unique<LBFGSOptimizer>();
        }
        
        // Устанавливаем монитор в оптимизатор
        optimizer->set_convergence_monitor(monitor.get());
        
        // Создаём функтор для оптимизации по Q
        ObjectiveFunctor functor(composite_poly_, problem_data_);
        functor.build_caches();
        
        // Начальные коэффициенты Q: нулевые
        std::vector<double> initial_q(n_free, 0.0);
        
        // Выполняем оптимизацию
        result = optimizer->optimize(functor, initial_q);
        
        // Если оптимизация успешна, строим полином из оптимизированных коэффициентов
        if (result.success) {
            polynomial_ = composite_poly_.build_polynomial(result.coefficients);
        }
        
        // Сохраняем копию полинома в результат (не указатель, чтобы избежать проблем с lifetime)
        result.final_polynomial = std::make_shared<Polynomial>(polynomial_);
        
        // Выполняем валидацию решения
        SolutionValidator validator;
        ValidationResult validation = validator.validate(polynomial_, problem_data_);
        result.validation = validation;
        
        if (!validation.is_valid) {
            result.success = false;
            result.message += "\nSolution validation failed: " + validation.message;
        }
        
        // Генерация диагностического отчёта
        std::ostringstream report_oss;
        report_oss << "=== DIAGNOSTIC REPORT ===\n\n";
        
        // Convergence status
        report_oss << "Convergence status:\n";
        report_oss << "- Success: " << (result.success ? "YES" : "NO") << "\n";
        report_oss << "- Iterations: " << result.iterations << "\n";
        report_oss << "- Elapsed time: " << result.elapsed_time << " ms\n";
        report_oss << "- Final objective: " << result.final_objective << "\n";
        
        // Components
        auto comps = functor.compute_components(result.coefficients);
        report_oss << "\nFunctional components:\n";
        report_oss << "  Approximation: " << comps.approx << "\n";
        report_oss << "  Repulsion: " << comps.repel << "\n";
        report_oss << "  Regularization: " << comps.reg << "\n";
        report_oss << "  Total: " << comps.total << "\n";
        
        // Validation
        report_oss << "\nSolution validation:\n";
        report_oss << "  Numerical correctness: " << (validation.numerical_correct ? "PASS" : "FAIL") << "\n";
        report_oss << "  Interpolation: " << (validation.interpolation_ok ? "PASS" : "FAIL") << "\n";
        report_oss << "  Barrier safety: " << (validation.barriers_safe ? "PASS" : "FAIL") << "\n";
        report_oss << "  Physical plausibility: " << (validation.physically_plausible ? "PASS" : "FAIL") << "\n";
        if (!validation.warnings.empty()) {
            report_oss << "\nWarnings:\n";
            for (const auto& w : validation.warnings) {
                report_oss << "  - " << w << "\n";
            }
        }
        
        result.diagnostic_report = report_oss.str();
        
        // Синхронизируем converged с success
        result.converged = result.success;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.converged = false;
        result.message = std::string("Exception during solve: ") + e.what();
        result.final_polynomial = std::make_shared<Polynomial>(polynomial_);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

bool MixedApproximation::check_interpolation_conditions(double tolerance) const {
    // Проверяем, что полином удовлетворяет интерполяционным условиям в узлах interp_z
    for (size_t i = 0; i < problem_data_.interp_z.size(); ++i) {
        double x = problem_data_.interp_z[i];
        double target = problem_data_.interp_f[i];
        double computed = polynomial_.evaluate(x);
        if (std::abs(computed - target) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<double> MixedApproximation::compute_repel_distances() const {
    std::vector<double> distances;
    for (size_t i = 0; i < problem_data_.repel_y.size(); ++i) {
        double x = problem_data_.repel_y[i];
        double y_forbidden = problem_data_.repel_forbidden[i];
        double poly_value = polynomial_.evaluate(x);
        distances.push_back(std::abs(y_forbidden - poly_value));
    }
    return distances;
}

FunctionalComponents MixedApproximation::get_functional_components() const {
    FunctionalComponents comps;
    
    if (!parametrization_built_) {
        return comps;
    }
    
    try {
        ObjectiveFunctor functor(composite_poly_, problem_data_);
        functor.build_caches();
        
        // Извлекаем коэффициенты Q из текущего полинома через декомпозицию
        // Polynomial представляет F(x), а нам нужны коэффициенты Q(x)
        // F(x) = P_int(x) + Q(x) * W(x), где W(x) - весовой полином
        
        // Получаем коэффициенты Q из composite_poly_
        int n_free = composite_poly_.num_free_params;
        std::vector<double> q_coeffs;
        
        // Пытаемся извлечь коэффициенты Q из текущего polynomial_
        if (polynomial_.is_initialized() && n_free > 0) {
            // Простой подход: предполагаем, что Q хранится в первых n_free коэффициентах
            // Более точный подход требует декомпозиции polynomial_
            const auto& poly_coeffs = polynomial_.coefficients();
            if (static_cast<int>(poly_coeffs.size()) >= n_free) {
                q_coeffs.assign(poly_coeffs.begin(), poly_coeffs.begin() + n_free);
            } else {
                q_coeffs.assign(n_free, 0.0);
            }
        } else {
            q_coeffs.assign(n_free, 0.0);
        }
        
        auto result = functor.compute_components(q_coeffs);
        comps.approx_component = result.approx;
        comps.repel_component = result.repel;
        comps.reg_component = result.reg;
        comps.total = result.total;
    } catch (...) {
        // В случае ошибки возвращаем нули
    }
    
    return comps;
}

Polynomial MixedApproximation::apply_interpolation_constraints(const Polynomial& poly) const {
    // В данной реализации интерполяционные условия обеспечиваются в build_initial_approximation,
    // поэтому дополнительная коррекция не требуется.
    return poly;
}

} // namespace mixed_approx
