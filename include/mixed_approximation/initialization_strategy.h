#ifndef MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H
#define MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H

#include "types.h"
#include "composite_polynomial.h"
#include <string>
#include <vector>

// Включение Eigen для линейной алгебры
#include <Eigen/Dense>

namespace mixed_approx {

// Forward declarations
class ObjectiveFunctor;
class OptimizationProblemData;

/**
 * @brief Расширенные стратегии инициализации коэффициентов перед оптимизацией
 */
enum class InitializationStrategy {
    ZERO,              ///< Нулевая инициализация (P_int(x))
    LEAST_SQUARES,     ///< Взвешенный МНК с Cholesky/SVD
    RANDOM,            ///< Случайная инициализация с возмущением
    HIERARCHICAL,      ///< Иерархическая инициализация (для n > 15)
    MULTI_START        ///< Многостартовая инициализация (ансамбль)
};

/**
 * @brief Метрики качества инициализации
 */
struct InitializationQualityMetrics {
    double objective_ratio;           ///< J_initial / J_random (цель: < 0.5)
    double min_barrier_distance;      ///< min_dist_to_barrier / ε_safe (цель: > 10)
    double rms_residual_norm;         ///< ||residuals||_RMS / std(f) (цель: < 1.0)
    double condition_number;         ///< Число обусловленности матрицы A
    bool interpolation_ok;            ///< Выполнены ли интерполяционные условия
    bool barriers_safe;               ///< Безопасны ли барьеры
    
    InitializationQualityMetrics()
        : objective_ratio(0.0)
        , min_barrier_distance(0.0)
        , rms_residual_norm(0.0)
        , condition_number(0.0)
        , interpolation_ok(false)
        , barriers_safe(false) {}
};

/**
 * @brief Результат инициализации с расширенной диагностикой
 */
struct InitializationResult {
    std::vector<double> initial_coeffs;
    double initial_objective;
    bool success;
    std::string message;
    InitializationStrategy strategy_used;
    InitializationQualityMetrics metrics;
    std::vector<std::string> warnings;
    std::vector<std::string> recommendations;
    
    InitializationResult()
        : initial_objective(0.0)
        , success(false)
        , strategy_used(InitializationStrategy::ZERO) {}
};

/**
 * @brief Класс для автоматического выбора стратегии инициализации (шаг 4.2)
 */
class InitializationStrategySelector {
public:
    /**
     * @brief Автоматический выбор стратегии инициализации по дереву решений
     * @param param композитный полином
     * @param data данные задачи
     * @return выбранная стратегия
     */
    static InitializationStrategy select(const CompositePolynomial& param,
                                        const OptimizationProblemData& data);
    
    /**
     * @brief Выполнить инициализацию с выбранной стратегией и валидацией
     * @param param композитный полином
     * @param data данные задачи
     * @param functor функтор для вычисления функционала
     * @return результат инициализации с диагностикой
     */
    static InitializationResult initialize(const CompositePolynomial& param,
                                          const OptimizationProblemData& data,
                                          ObjectiveFunctor& functor);
    
    /**
     * @brief Вычисление метрик качества данных для адаптивного выбора
     * @param data данные задачи
     * @return плотность данных ρ = N_approx / (b - a)
     */
    static double compute_data_density(const OptimizationProblemData& data);
    
    /**
     * @brief Вычисление интенсивности барьеров
     * @param data данные задачи
     * @return β = max(B_j) / avg(σ_i)
     */
    static double compute_barrier_intensity(const OptimizationProblemData& data);
    
    /**
     * @brief Проверка интерполяционных условий
     * @param param композитный полином
     * @param data данные задачи
     * @param coeffs коэффициенты Q(x)
     * @return true если все условия выполнены с точностью 1e-10
     */
    static bool verify_interpolation(const CompositePolynomial& param,
                                     const OptimizationProblemData& data,
                                     const std::vector<double>& coeffs);
    
    /**
     * @brief Проверка безопасности барьеров
     * @param param композитный полином
     * @param data данные задачи
     * @param coeffs коэффициенты Q(x)
     * @param safety_ratio порог безопасности (по умолчанию 10.0)
     * @return true если все барьеры безопасны
     */
    static bool verify_barrier_safety(const CompositePolynomial& param,
                                       const OptimizationProblemData& data,
                                       const std::vector<double>& coeffs,
                                       double safety_ratio = 10.0);
    
private:
    // Базовые стратегии
    static InitializationResult zero_initialization(const CompositePolynomial& param);
    
    static InitializationResult least_squares_initialization(const CompositePolynomial& param,
                                                            const OptimizationProblemData& data,
                                                            ObjectiveFunctor& functor);
    
    static InitializationResult random_initialization(const CompositePolynomial& param,
                                                     const OptimizationProblemData& data,
                                                     ObjectiveFunctor& functor,
                                                     const std::vector<double>& base_coeffs = {},
                                                     double perturbation_scale = 0.1);
    
    static InitializationResult hierarchical_initialization(const CompositePolynomial& param,
                                                           const OptimizationProblemData& data,
                                                           ObjectiveFunctor& functor);
    
    static InitializationResult multi_start_initialization(const CompositePolynomial& param,
                                                          const OptimizationProblemData& data,
                                                          ObjectiveFunctor& functor);
    
    // Вспомогательные методы для МНК
    static Eigen::MatrixXd build_normal_matrix(const OptimizationProblemData& data,
                                               const CompositePolynomial& param,
                                               int n_free,
                                               double& lambda_regularization);
    
    static Eigen::VectorXd solve_linear_system(Eigen::MatrixXd& A,
                                              const Eigen::VectorXd& b,
                                              bool& success,
                                              std::string& message);
    
    static void apply_barrier_correction(const CompositePolynomial& param,
                                         const OptimizationProblemData& data,
                                         ObjectiveFunctor& functor,
                                         std::vector<double>& coeffs);
    
    static void apply_preventive_shift(const CompositePolynomial& param,
                                        const OptimizationProblemData& data,
                                        ObjectiveFunctor& functor,
                                        std::vector<double>& coeffs);
    
    // Вычисление метрик
    static InitializationQualityMetrics compute_metrics(const CompositePolynomial& param,
                                                       const OptimizationProblemData& data,
                                                       const std::vector<double>& coeffs,
                                                       double initial_objective);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H
