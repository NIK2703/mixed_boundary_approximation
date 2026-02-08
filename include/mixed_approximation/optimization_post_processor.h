#ifndef MIXED_APPROXIMATION_OPTIMIZATION_POST_PROCESSOR_H
#define MIXED_APPROXIMATION_OPTIMIZATION_POST_PROCESSOR_H

#include "types.h"
#include <vector>
#include <string>

namespace mixed_approx {

// Forward declarations
class CompositePolynomial;
class OptimizationProblemData;

/**
 * @brief Результат пост-обработки оптимизации
 */
struct PostOptimizationReport {
    // Качество решения
    double max_interpolation_error;
    double min_barrier_distance;
    double final_gradient_norm;
    
    // Баланс компонентов
    double approx_percentage;
    double repel_percentage;
    double reg_percentage;
    
    // Статус
    bool interpolation_satisfied;
    bool barrier_constraints_satisfied;
    bool converged;
    
    // Рекомендации
    std::vector<std::string> recommendations;
    
    PostOptimizationReport()
        : max_interpolation_error(0.0), min_barrier_distance(0.0)
        , final_gradient_norm(0.0), approx_percentage(0.0)
        , repel_percentage(0.0), reg_percentage(0.0)
        , interpolation_satisfied(false)
        , barrier_constraints_satisfied(false), converged(false) {}
};

/**
 * @brief Класс для пост-обработки результатов оптимизации
 */
class OptimizationPostProcessor {
public:
    /**
     * @brief Конструктор
     * @param param композитный полином
     * @param data данные задачи
     */
    OptimizationPostProcessor(const CompositePolynomial& param,
                              const OptimizationProblemData& data);
    
    /**
     * @brief Генерация отчёта о результатах оптимизации
     * @param final_coeffs финальные коэффициенты
     * @param final_objective финальное значение функционала
     * @return отчёт
     */
    PostOptimizationReport generate_report(const std::vector<double>& final_coeffs,
                                           double final_objective);
    
    /**
     * @brief Генерация текстового отчёта
     */
    std::string generate_text_report(const PostOptimizationReport& report);
    
    /**
     * @brief Адаптивная коррекция параметров при дисбалансе
     * @param report текущий отчёт
     * @param config конфигурация для модификации
     */
    void suggest_parameter_corrections(const PostOptimizationReport& report,
                                        ApproximationConfig& config);

private:
    const CompositePolynomial& param_;
    const OptimizationProblemData& data_;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OPTIMIZATION_POST_PROCESSOR_H
