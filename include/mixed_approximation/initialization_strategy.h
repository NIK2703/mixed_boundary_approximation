#ifndef MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H
#define MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H

#include "types.h"
#include "composite_polynomial.h"
#include <string>

namespace mixed_approx {

// Forward declarations
class ObjectiveFunctor;
class OptimizationProblemData;

/**
 * @brief Стратегии инициализации коэффициентов перед оптимизацией
 */
enum class InitializationStrategy {
    ZERO,              ///< Нулевая инициализация
    LEAST_SQUARES,     ///< Инициализация через взвешенный МНК
    MULTI_START        ///< Многостартовая инициализация (для n_free ≤ 10)
};

/**
 * @brief Результат инициализации
 */
struct InitializationResult {
    std::vector<double> initial_coeffs;
    double initial_objective;
    bool success;
    std::string message;
    InitializationStrategy strategy_used;
    
    InitializationResult()
        : initial_objective(0.0), success(false), strategy_used(InitializationStrategy::ZERO) {}
};

/**
 * @brief Класс для автоматического выбора стратегии инициализации
 */
class InitializationStrategySelector {
public:
    /**
     * @brief Автоматический выбор стратегии инициализации
     * @param param композитный полином
     * @param data данные задачи
     * @return выбранная стратегия
     */
    static InitializationStrategy select(const CompositePolynomial& param,
                                        const OptimizationProblemData& data);
    
    /**
     * @brief Выполнить инициализацию с выбранной стратегией
     * @param param композитный полином
     * @param data данные задачи
     * @param functor функтор для вычисления функционала
     * @return результат инициализации
     */
    static InitializationResult initialize(const CompositePolynomial& param,
                                          const OptimizationProblemData& data,
                                          ObjectiveFunctor& functor);
    
private:
    // Инициализация нулями
    static InitializationResult zero_initialization(const CompositePolynomial& param);
    
    // Инициализация через МНК
    static InitializationResult least_squares_initialization(const CompositePolynomial& param,
                                                            const OptimizationProblemData& data,
                                                            ObjectiveFunctor& functor);
    
    // Многостартовая инициализация
    static InitializationResult multi_start_initialization(const CompositePolynomial& param,
                                                          const OptimizationProblemData& data,
                                                          ObjectiveFunctor& functor);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_INITIALIZATION_STRATEGY_H
