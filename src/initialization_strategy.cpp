#include "mixed_approximation/initialization_strategy.h"
#include "mixed_approximation/objective_functor.h"
#include "mixed_approximation/polynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>

namespace mixed_approx {

// ============== Вспомогательные функции для вычисления метрик ==============

double InitializationStrategySelector::compute_data_density(const OptimizationProblemData& data) {
    if (data.num_approx_points() == 0) return 0.0;
    double interval_length = data.interval_b - data.interval_a;
    return static_cast<double>(data.num_approx_points()) / interval_length;
}

double InitializationStrategySelector::compute_barrier_intensity(const OptimizationProblemData& data) {
    if (data.num_repel_points() == 0) return 0.0;
    
    double max_barrier = 0.0;
    for (double w : data.repel_weight) {
        max_barrier = std::max(max_barrier, w);
    }
    
    double avg_sigma = 0.0;
    // Веса в данных это 1/σ, поэтому σ = 1/weight
    for (double weight : data.approx_weight) {
        if (weight > 1e-15) {
            avg_sigma += 1.0 / weight;
        }
    }
    if (data.approx_weight.size() > 0) {
        avg_sigma /= data.approx_weight.size();
    } else {
        avg_sigma = 1.0;
    }
    
    if (avg_sigma < 1e-15) return std::numeric_limits<double>::max();
    return max_barrier / avg_sigma;
}

bool InitializationStrategySelector::verify_interpolation(const CompositePolynomial& param,
                                                          const OptimizationProblemData& data,
                                                          const std::vector<double>& coeffs) {
    if (data.num_interp_nodes() == 0) return true;
    
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight = param.weight_multiplier;
    int n_free = param.num_free_parameters();
    
    for (size_t i = 0; i < data.num_interp_nodes(); ++i) {
        double x = data.interp_z[i];
        double phi_q = 0.0;
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            phi_q += coeffs[k] * power;
            power *= x;
        }
        double F_x = basis.evaluate(x) + phi_q * weight.evaluate(x);
        if (std::abs(F_x - data.interp_f[i]) > 1e-10) {
            return false;
        }
    }
    return true;
}

bool InitializationStrategySelector::verify_barrier_safety(const CompositePolynomial& param,
                                                            const OptimizationProblemData& data,
                                                            const std::vector<double>& coeffs,
                                                            double safety_ratio) {
    if (data.num_repel_points() == 0) return true;
    
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight = param.weight_multiplier;
    int n_free = param.num_free_parameters();
    double eps_safe = 1e-8;
    
    for (size_t j = 0; j < data.num_repel_points(); ++j) {
        double y = data.repel_y[j];
        double phi_q = 0.0;
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            phi_q += coeffs[k] * power;
            power *= y;
        }
        double F_y = basis.evaluate(y) + phi_q * weight.evaluate(y);
        double distance = std::abs(data.repel_forbidden[j] - F_y);
        if (distance < eps_safe * safety_ratio) {
            return false;
        }
    }
    return true;
}

// ============== Построение нормальной системы (матрица Грама) ==============

Eigen::MatrixXd InitializationStrategySelector::build_normal_matrix(
    const OptimizationProblemData& data,
    const CompositePolynomial& param,
    int n_free,
    double& lambda_regularization) {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_free, n_free);
    const WeightMultiplier& weight_mult = param.weight_multiplier;
    
    // Формирование матрицы Грама: A_kl = Σ_i (x_i^(k+l) / σ_i) * W(x_i)^2
    for (size_t i = 0; i < data.num_approx_points(); ++i) {
        double x = data.approx_x[i];
        double w = data.approx_weight[i];
        double W_val = weight_mult.evaluate(x);
        // Вес уже содержит 1/σ, поэтому используем его напрямую
        double sigma_factor = w;
        
        double power_x = 1.0;
        for (int k = 0; k < n_free; ++k) {
            double power_y = 1.0;
            for (int l = 0; l < n_free; ++l) {
                A(k, l) += (power_x * power_y) * W_val * W_val * sigma_factor;
                power_y *= x;
            }
            power_x *= x;
        }
    }
    
    // Вычисление регуляризации по формуле Тихонова: λ = 1e-6 * trace(A) / (n+1)
    lambda_regularization = 1e-6 * A.trace() / n_free;
    
    return A;
}

// ============== Решение линейной системы с Cholesky/SVD ==============

Eigen::VectorXd InitializationStrategySelector::solve_linear_system(Eigen::MatrixXd& A,
                                                                   const Eigen::VectorXd& b,
                                                                   bool& success,
                                                                   std::string& message) {
    success = false;
    message = "";
    
    int n = A.rows();
    
    // Проверка на симметричность
    if (!A.isApprox(A.transpose(), 1e-12)) {
        message = "Matrix is not symmetric";
        return Eigen::VectorXd::Zero(n);
    }
    
    // Попытка разложения Холецкого
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() == Eigen::Success) {
        // Холецкий успешен, матрица положительно определённая
        Eigen::VectorXd x = llt.solve(b);
        if (std::isfinite(x.sum())) {
            success = true;
            message = "Cholesky decomposition successful";
            return x;
        }
    }
    
    // Если Холецкий не удался, пробуем LDLT (для полуопределённых матриц)
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
        Eigen::VectorXd x = ldlt.solve(b);
        if (std::isfinite(x.sum())) {
            success = true;
            message = "LDLT decomposition successful";
            return x;
        }
    }
    
    // Если не сработало, используем SVD для плохо обусловленных систем
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double cond = svd.singularValues()(0) / svd.singularValues()(n-1);
    
    if (cond > 1e12) {
        message = "Ill-conditioned matrix (cond = " + std::to_string(cond) + "), using truncated SVD";
        // Отсечение малых сингулярных значений
        double threshold = 1e-10 * svd.singularValues()(0);
        Eigen::VectorXd singular_values = svd.singularValues();
        for (int i = 0; i < n; ++i) {
            if (singular_values(i) < threshold) {
                singular_values(i) = 0.0;
            }
        }
        // Пересборка псевдообратной матрицы
        Eigen::MatrixXd S_inv = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            if (singular_values(i) > 0) {
                S_inv(i, i) = 1.0 / singular_values(i);
            }
        }
        Eigen::VectorXd x = svd.matrixV() * S_inv * svd.matrixU().transpose() * b;
        if (std::isfinite(x.sum())) {
            success = true;
            return x;
        }
    } else {
        message = "Used SVD (cond = " + std::to_string(cond) + ")";
        Eigen::VectorXd x = svd.solve(b);
        if (std::isfinite(x.sum())) {
            success = true;
            return x;
        }
    }
    
    message = "Failed to solve linear system";
    return Eigen::VectorXd::Zero(n);
}

// ============== Коррекция барьеров ==============

void InitializationStrategySelector::apply_barrier_correction(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor,
    std::vector<double>& coeffs) {
    
    if (data.num_repel_points() == 0) return;
    
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight = param.weight_multiplier;
    int n_free = param.num_free_parameters();
    double eps_safe = 1e-8;
    double alpha = 0.1;
    int max_iterations = 10;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double min_distance = std::numeric_limits<double>::max();
        std::vector<double> grad_repulse(n_free, 0.0);
        
        // Вычисление антиградиента отталкивающего члена
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double y = data.repel_y[j];
            double y_star = data.repel_forbidden[j];
            double B_j = data.repel_weight[j];
            
            double phi_q = 0.0;
            double power = 1.0;
            std::vector<double> phi_deriv(n_free, 0.0);
            for (int k = 0; k < n_free; ++k) {
                phi_q += coeffs[k] * power;
                phi_deriv[k] = power;
                power *= y;
            }
            
            double W_y = weight.evaluate(y);
            double F_y = basis.evaluate(y) + phi_q * W_y;
            double diff = y_star - F_y;
            double distance = std::abs(diff);
            min_distance = std::min(min_distance, distance);
            
            if (distance > 1e-15) {
                double sign = (diff > 0) ? 1.0 : -1.0;
                double factor = 2.0 * B_j / (distance * distance * distance) * sign;
                for (int k = 0; k < n_free; ++k) {
                    grad_repulse[k] += factor * phi_deriv[k] * W_y;
                }
            }
        }
        
        if (min_distance >= eps_safe) {
            break; // Выход из критической зоны
        }
        
        // Нормализация градиента
        double grad_norm = std::sqrt(std::inner_product(grad_repulse.begin(), grad_repulse.end(), 
                                                        grad_repulse.begin(), 0.0));
        if (grad_norm > 1e-15) {
            for (int k = 0; k < n_free; ++k) {
                coeffs[k] += alpha * grad_repulse[k] / grad_norm;
            }
        }
    }
}

void InitializationStrategySelector::apply_preventive_shift(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor,
    std::vector<double>& coeffs) {
    
    if (data.num_repel_points() == 0) return;
    
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight = param.weight_multiplier;
    int n_free = param.num_free_parameters();
    double eps_safe = 1e-8;
    
    double min_safety_ratio = std::numeric_limits<double>::max();
    std::vector<double> shift_direction(n_free, 0.0);
    
    for (size_t j = 0; j < data.num_repel_points(); ++j) {
        double y = data.repel_y[j];
        double y_star = data.repel_forbidden[j];
        double B_j = data.repel_weight[j];
        
        double phi_q = 0.0;
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            phi_q += coeffs[k] * power;
            power *= y;
        }
        
        double W_y = weight.evaluate(y);
        double F_y = basis.evaluate(y) + phi_q * W_y;
        double distance = std::abs(y_star - F_y);
        double safety_ratio = distance / eps_safe;
        min_safety_ratio = std::min(min_safety_ratio, safety_ratio);
        
        if (safety_ratio < 10.0) {
            double weight_factor = std::exp(-1.0 / safety_ratio);
            double sign = (y_star > F_y) ? 1.0 : -1.0;
            power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                shift_direction[k] += sign * weight_factor * power * W_y;
                power *= y;
            }
        }
    }
    
    if (min_safety_ratio < 10.0) {
        double alpha_risk = 0.5 * (1.0 - min_safety_ratio / 10.0);
        double shift_norm = std::sqrt(std::inner_product(shift_direction.begin(), shift_direction.end(),
                                                         shift_direction.begin(), 0.0));
        if (shift_norm > 1e-15) {
            for (int k = 0; k < n_free; ++k) {
                coeffs[k] += alpha_risk * shift_direction[k] / shift_norm;
            }
        }
    }
}

// ============== Основные стратегии ==============

InitializationResult InitializationStrategySelector::zero_initialization(
    const CompositePolynomial& param) {
    
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
    ObjectiveFunctor& functor) {
    
    InitializationResult result;
    result.strategy_used = InitializationStrategy::LEAST_SQUARES;
    
    int n_free = param.num_free_parameters();
    size_t N = data.num_approx_points();
    
    if (N == 0) {
        return zero_initialization(param);
    }
    
    // Построение матрицы A и вектора b для системы A·q = b
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, n_free);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight_mult = param.weight_multiplier;
    
    for (size_t i = 0; i < N; ++i) {
        double x = data.approx_x[i];
        double w = data.approx_weight[i];  // это 1/σ_i
        double W_val = weight_mult.evaluate(x);
        
        // Заполнение строки матрицы A: [1, x, x^2, ..., x^(n-1)] * W(x) * weight
        double power = 1.0;
        for (int k = 0; k < n_free; ++k) {
            A(i, k) = w * power * W_val;
            power *= x;
        }
        
        // Правая часть: w * (f(x) - P_int(x))
        double F_base = basis.evaluate(x);
        b(i) = w * (data.approx_f[i] - F_base);
    }
    
    // Построение нормальной системы: A^T A
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::VectorXd ATb = A.transpose() * b;
    
    // Регуляризация по Тихонову
    double lambda = 1e-8 * ATA.trace() / n_free;
    ATA.diagonal().array() += lambda;
    
    // Решение системы
    bool solve_success = false;
    std::string solve_message;
    Eigen::VectorXd q = solve_linear_system(ATA, ATb, solve_success, solve_message);
    
    if (!solve_success) {
        result.success = false;
        result.message = "Failed to solve least squares system: " + solve_message;
        return result;
    }
    
    // Преобразование Eigen::VectorXd в std::vector
    result.initial_coeffs.resize(n_free);
    for (int k = 0; k < n_free; ++k) {
        result.initial_coeffs[k] = q(k);
    }
    
    // Коррекция барьеров
    if (data.num_repel_points() > 0) {
        functor.build_caches();
        double min_distance = std::numeric_limits<double>::max();
        
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double y = data.repel_y[j];
            double phi_q = 0.0;
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += result.initial_coeffs[k] * power;
                power *= y;
            }
            double F_y = basis.evaluate(y) + phi_q * weight_mult.evaluate(y);
            double distance = std::abs(data.repel_forbidden[j] - F_y);
            min_distance = std::min(min_distance, distance);
        }
        
        double d_safe = 1e-8;
        if (min_distance < 10.0 * d_safe) {
            // Применяем коррекцию в зависимости от ситуации
            if (min_distance < d_safe) {
                // Критическая ситуация: градиентный подъём
                apply_barrier_correction(param, data, functor, result.initial_coeffs);
            } else if (min_distance < 10.0 * d_safe) {
                // Опасная зона: превентивный сдвиг
                apply_preventive_shift(param, data, functor, result.initial_coeffs);
            }
        }
    }
    
    // Валидация интерполяционных условий
    result.metrics.interpolation_ok = verify_interpolation(param, data, result.initial_coeffs);
    if (!result.metrics.interpolation_ok) {
        // Итеративная коррекция для восстановления интерполяции
        for (int iter = 0; iter < 5; ++iter) {
            bool all_ok = true;
            for (size_t i = 0; i < data.num_interp_nodes(); ++i) {
                double x = data.interp_z[i];
                double phi_q = 0.0;
                double power = 1.0;
                for (int k = 0; k < n_free; ++k) {
                    phi_q += result.initial_coeffs[k] * power;
                    power *= x;
                }
                double F_x = basis.evaluate(x) + phi_q * weight_mult.evaluate(x);
                double error = data.interp_f[i] - F_x;
                if (std::abs(error) > 1e-12) {
                    // Простая коррекция: добавляем поправку к коэффициентам
                    power = 1.0;
                    for (int k = 0; k < n_free; ++k) {
                        result.initial_coeffs[k] += error * power;
                        power *= x;
                    }
                    all_ok = false;
                }
            }
            if (all_ok) break;
        }
        result.metrics.interpolation_ok = verify_interpolation(param, data, result.initial_coeffs);
    }
    
    // Вычисление финального функционала
    functor.build_caches();
    result.initial_objective = functor.value(result.initial_coeffs);
    result.success = true;
    result.message = "Least squares initialization with barrier protection (" + solve_message + ")";
    
    return result;
}

InitializationResult InitializationStrategySelector::random_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor,
    const std::vector<double>& base_coeffs,
    double perturbation_scale) {
    
    InitializationResult result;
    result.strategy_used = InitializationStrategy::RANDOM;
    
    int n_free = param.num_free_parameters();
    std::mt19937 rng(42);
    
    // Оценка масштаба данных
    double scale_y = 0.0;
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();
    for (double f_val : data.approx_f) {
        min_y = std::min(min_y, f_val);
        max_y = std::max(max_y, f_val);
    }
    scale_y = max_y - min_y;
    if (scale_y < 1e-15) scale_y = 1.0;
    
    double scale_x = data.interval_b - data.interval_a;
    if (scale_x < 1e-15) scale_x = 1.0;
    
    double sigma_q = perturbation_scale * scale_y / std::pow(scale_x, n_free > 0 ? n_free - 1 : 1);
    
    std::normal_distribution<double> normal_dist(0.0, sigma_q);
    
    result.initial_coeffs.resize(n_free);
    if (base_coeffs.empty()) {
        for (int k = 0; k < n_free; ++k) {
            result.initial_coeffs[k] = normal_dist(rng);
        }
    } else {
        for (int k = 0; k < n_free; ++k) {
            result.initial_coeffs[k] = base_coeffs[k] + normal_dist(rng);
        }
    }
    
    // Применение коррекции барьеров
    apply_barrier_correction(param, data, functor, result.initial_coeffs);
    
    functor.build_caches();
    result.initial_objective = functor.value(result.initial_coeffs);
    result.success = true;
    result.message = "Random initialization with perturbation (σ = " + std::to_string(sigma_q) + ")";
    
    return result;
}

InitializationResult InitializationStrategySelector::hierarchical_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor) {
    
    InitializationResult result;
    result.strategy_used = InitializationStrategy::HIERARCHICAL;
    
    int n_free = param.num_free_parameters();
    if (n_free <= 5) {
        // Для малых размерностей используем обычный МНК
        return least_squares_initialization(param, data, functor);
    }
    
    // Иерархический подход: начинаем с низкой степени
    int n_start = std::min(5, n_free);
    
    // Создаём временный CompositePolynomial с меньшим числом свободных параметров
    // В реальной реализации нужно модифицировать параметризацию
    // Пока что используем упрощённый подход: инициализируем первые n_start коэффициентов
    
    result.initial_coeffs.assign(n_free, 0.0);
    
    // Инициализация первых n_start коэффициентов через МНК
    // Для этого нужно временно изменить параметризацию
    // В качестве упрощения, используем random с малой амплитудой
    
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.01);
    for (int k = 0; k < n_start; ++k) {
        result.initial_coeffs[k] = dist(rng);
    }
    
    // Постепенное увеличение: оставляем остальные нулями
    result.success = true;
    result.message = "Hierarchical initialization (start with n=" + std::to_string(n_start) + ")";
    result.initial_objective = functor.value(result.initial_coeffs);
    
    return result;
}

InitializationResult InitializationStrategySelector::multi_start_initialization(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor) {
    
    InitializationResult result;
    result.strategy_used = InitializationStrategy::MULTI_START;
    
    int n_free = param.num_free_parameters();
    if (n_free > 10) {
        // Для высоких размерностей возвращаемся к МНК
        return least_squares_initialization(param, data, functor);
    }
    
    functor.build_caches();
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    std::vector<double> best_coeffs(n_free, 0.0);
    double best_objective = functor.value(best_coeffs);
    
    int n_starts = std::min(5, n_free + 1);
    
    for (int trial = 0; trial < n_starts; ++trial) {
        std::vector<double> trial_coeffs(n_free);
        for (int k = 0; k < n_free; ++k) {
            trial_coeffs[k] = dist(rng);
        }
        
        // Небольшой градиентный спуск для уточнения
        std::vector<double> coeffs = trial_coeffs;
        double step = 0.01;
        
        for (int iter = 0; iter < 20; ++iter) {
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
    result.message = "Multi-start initialization: best of " + std::to_string(n_starts) + " trials";
    
    return result;
}

// ============== Адаптивный выбор стратегии ==============

InitializationStrategy InitializationStrategySelector::select(
    const CompositePolynomial& param,
    const OptimizationProblemData& data) {
    
    int n_free = param.num_free_parameters();
    
    // Дерево принятия решений согласно шагу 4.2.5
    
    // Если нет аппроксимирующих точек → нулевая инициализация
    if (data.num_approx_points() == 0) {
        return InitializationStrategy::ZERO;
    }
    
    // Вычисление метрик
    double rho = compute_data_density(data);
    double beta = compute_barrier_intensity(data);
    
    // Экстремально сильные барьеры
    if (beta > 1000.0) {
        // Пробуем МНК, но с агрессивной коррекцией
        return InitializationStrategy::LEAST_SQUARES;
    }
    
    // Разреженные данные
    if (rho < 0.5 * n_free) {
        // Регуляризованный МНК + случайное возмущение
        return InitializationStrategy::RANDOM;
    }
    
    // Сильные барьеры или опасная зона
    if (beta > 100.0) {
        // МНК + агрессивная коррекция + дополнительные случайные точки
        return InitializationStrategy::MULTI_START;
    }
    
    // Высокие степени (n > 15) → иерархический подход
    if (n_free > 15) {
        return InitializationStrategy::HIERARCHICAL;
    }
    
    // Стандартный случай: чистый МНК
    return InitializationStrategy::LEAST_SQUARES;
}

InitializationResult InitializationStrategySelector::initialize(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    ObjectiveFunctor& functor) {
    
    InitializationStrategy strategy = select(param, data);
    
    InitializationResult result;
    
    switch (strategy) {
        case InitializationStrategy::ZERO:
            result = zero_initialization(param);
            break;
        case InitializationStrategy::LEAST_SQUARES:
            result = least_squares_initialization(param, data, functor);
            break;
        case InitializationStrategy::RANDOM: {
            // Сначала пробуем МНК, затем добавляем возмущение
            result = least_squares_initialization(param, data, functor);
            if (result.success) {
                InitializationResult random_result = random_initialization(param, data, functor, result.initial_coeffs, 0.05);
                if (random_result.success) {
                    result = random_result;
                    result.strategy_used = InitializationStrategy::RANDOM;
                }
            }
            break;
        }
        case InitializationStrategy::HIERARCHICAL:
            result = hierarchical_initialization(param, data, functor);
            break;
        case InitializationStrategy::MULTI_START:
            result = multi_start_initialization(param, data, functor);
            break;
        default:
            result = zero_initialization(param);
    }
    
    // Вычисление финальных метрик
    result.metrics = compute_metrics(param, data, result.initial_coeffs, result.initial_objective);
    
    // Генерация рекомендаций
    if (result.metrics.objective_ratio > 0.5) {
        result.recommendations.push_back("Consider using multi-start initialization");
    }
    if (result.metrics.min_barrier_distance < 10.0) {
        result.recommendations.push_back("Barriers are too close, increase safety margin");
    }
    if (result.metrics.rms_residual_norm > 1.0) {
        result.recommendations.push_back("Poor data fit, check data quality or increase polynomial degree");
    }
    if (result.metrics.condition_number > 1e12) {
        result.recommendations.push_back("Ill-conditioned system, use stronger regularization");
    }
    
    return result;
}

InitializationQualityMetrics InitializationStrategySelector::compute_metrics(
    const CompositePolynomial& param,
    const OptimizationProblemData& data,
    const std::vector<double>& coeffs,
    double initial_objective) {
    
    InitializationQualityMetrics metrics;
    
    // Вычисление J_random как среднего значения функционала для случайных коэффициентов
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    double J_random_sum = 0.0;
    int n_random = 10;
    int n_free = param.num_free_parameters();
    
    const InterpolationBasis& basis = param.interpolation_basis;
    const WeightMultiplier& weight = param.weight_multiplier;
    
    for (int i = 0; i < n_random; ++i) {
        std::vector<double> random_coeffs(n_free);
        for (int k = 0; k < n_free; ++k) {
            random_coeffs[k] = dist(rng);
        }
        // Вычисление функционала вручную для случайных коэффициентов
        double obj = 0.0;
        // Аппроксимирующие точки
        for (size_t idx = 0; idx < data.num_approx_points(); ++idx) {
            double x = data.approx_x[idx];
            double phi_q = 0.0;
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += random_coeffs[k] * power;
                power *= x;
            }
            double F_x = basis.evaluate(x) + phi_q * weight.evaluate(x);
            double residual = data.approx_f[idx] - F_x;
            double sigma = data.approx_weight[idx] > 0 ? 1.0 / data.approx_weight[idx] : 1.0;
            obj += (residual * residual) / (sigma * sigma);
        }
        // Барьеры
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double y = data.repel_y[j];
            double phi_q = 0.0;
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += random_coeffs[k] * power;
                power *= y;
            }
            double F_y = basis.evaluate(y) + phi_q * weight.evaluate(y);
            double violation = data.repel_forbidden[j] - F_y;
            if (violation < 0) {
                double B_j = data.repel_weight[j];
                obj += B_j / (violation * violation);
            }
        }
        J_random_sum += obj;
    }
    double J_random_avg = J_random_sum / n_random;
    
    if (J_random_avg > 1e-15) {
        metrics.objective_ratio = initial_objective / J_random_avg;
    } else {
        metrics.objective_ratio = 0.0;
    }
    
    // Минимальное расстояние до барьеров
    if (data.num_repel_points() > 0) {
        double min_distance = std::numeric_limits<double>::max();
        double eps_safe = 1e-8;
        
        for (size_t j = 0; j < data.num_repel_points(); ++j) {
            double y = data.repel_y[j];
            double phi_q = 0.0;
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += coeffs[k] * power;
                power *= y;
            }
            double F_y = basis.evaluate(y) + phi_q * weight.evaluate(y);
            double distance = std::abs(data.repel_forbidden[j] - F_y);
            min_distance = std::min(min_distance, distance);
        }
        metrics.min_barrier_distance = min_distance / eps_safe;
        metrics.barriers_safe = (min_distance > 10.0 * eps_safe);
    } else {
        metrics.min_barrier_distance = std::numeric_limits<double>::max();
        metrics.barriers_safe = true;
    }
    
    // Нормированный остаток аппроксимации
    if (data.num_approx_points() > 0) {
        double sum_sq_residuals = 0.0;
        
        for (size_t i = 0; i < data.num_approx_points(); ++i) {
            double x = data.approx_x[i];
            double phi_q = 0.0;
            double power = 1.0;
            for (int k = 0; k < n_free; ++k) {
                phi_q += coeffs[k] * power;
                power *= x;
            }
            double F_x = basis.evaluate(x) + phi_q * weight.evaluate(x);
            double residual = data.approx_f[i] - F_x;
            sum_sq_residuals += residual * residual;
        }
        double rms_residual = std::sqrt(sum_sq_residuals / data.num_approx_points());
        
        // Вычисление std(f)
        double mean_f = 0.0;
        for (double f_val : data.approx_f) {
            mean_f += f_val;
        }
        mean_f /= data.num_approx_points();
        double var_f = 0.0;
        for (double f_val : data.approx_f) {
            double diff = f_val - mean_f;
            var_f += diff * diff;
        }
        var_f /= data.num_approx_points();
        double std_f = std::sqrt(var_f);
        
        if (std_f > 1e-15) {
            metrics.rms_residual_norm = rms_residual / std_f;
        } else {
            metrics.rms_residual_norm = 0.0;
        }
    } else {
        metrics.rms_residual_norm = 0.0;
    }
    
    // Проверка интерполяции
    metrics.interpolation_ok = verify_interpolation(param, data, coeffs);
    
    // Число обусловленности (приближённо через ATA)
    if (data.num_approx_points() > 0) {
        double lambda;
        Eigen::MatrixXd A = build_normal_matrix(data, param, n_free, lambda);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
        if (svd.singularValues()(0) > 1e-15) {
            metrics.condition_number = svd.singularValues()(0) / svd.singularValues()(n_free-1);
        } else {
            metrics.condition_number = std::numeric_limits<double>::max();
        }
    } else {
        metrics.condition_number = 1.0;
    }
    
    return metrics;
}

} // namespace mixed_approx
