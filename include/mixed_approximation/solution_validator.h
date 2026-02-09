#ifndef MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H
#define MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H

#include <string>
#include "types.h"
#include "composite_polynomial.h"
#include "optimization_problem_data.h"

namespace mixed_approx {

/**
 * @brief Валидатор решений после оптимизации
 * 
 * Реализует финальную валидацию согласно шагу 4.3.7:
 * - Численная корректность (NaN/Inf)
 * - Выполнение интерполяционных условий
 * - Безопасность относительно барьеров
 * - Физическая правдоподобность (отсутствие экстремальных осцилляций)
 */
class SolutionValidator {
public:
    // Параметры валидации
    double epsilon_safe;            // безопасное расстояние до барьеров (по умолчанию 1e-8)
    double interp_tolerance;        // допуск для интерполяции (по умолчанию 1e-10)
    double max_value_factor;        // множитель для проверки экстремальных значений (по умолчанию 100)
    int num_check_points;           // число контрольных точек для проверки осцилляций (по умолчанию 1000)
    
    // Конструктор
    SolutionValidator(double eps_safe = 1e-8, double interp_tol = 1e-10)
        : epsilon_safe(eps_safe), interp_tolerance(interp_tol)
        , max_value_factor(100.0), num_check_points(1000) {}
    
    /**
     * @brief Валидация решения
     * @param poly построенный полином F(x)
     * @param data данные задачи
     * @return результат валидации
     */
    ValidationResult validate(const Polynomial& poly, const OptimizationProblemData& data) const;
    
    /**
     * @brief Проверка интерполяционных условий
     */
    bool check_interpolation(const Polynomial& poly, const OptimizationProblemData& data,
                            double& max_error) const;
    
    /**
     * @brief Проверка безопасности барьеров
     */
    bool check_barrier_safety(const Polynomial& poly, const OptimizationProblemData& data,
                             double& min_distance) const;
    
    /**
     * @brief Проверка численной корректности
     */
    bool check_numerical_correctness(const Polynomial& poly, const OptimizationProblemData& data) const;
    
    /**
     * @brief Проверка физической правдоподобности (отсутствие экстремальных осцилляций)
     * @param poly полином
     * @param data данные задачи
     * @param max_value выходной параметр: максимальное значение полинома на интервале
     * @return true если значения в разумных пределах
     */
    bool check_physical_plausibility(const Polynomial& poly,
                                     const OptimizationProblemData& data,
                                     double& max_value) const;
    
    /**
     * @brief Применение проекционной коррекции при нарушении интерполяции
     * Пытается скорректировать коэффициенты полинома для точного выполнения интерполяционных условий
     * @param poly полином для коррекции (будет изменён)
     * @param data данные задачи
     * @return true если коррекция успешна
     */
    bool apply_projection_correction(Polynomial& poly, const OptimizationProblemData& data) const;
    
    /**
     * @brief Генерация отчёта о валидации
     * @param result результат валидации
     * @return строковое представление отчёта
     */
    std::string generate_report(const ValidationResult& result) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H
