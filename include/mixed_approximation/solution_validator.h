#ifndef MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H
#define MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H

#include <string>
#include "types.h"
#include "composite_polynomial.h"
#include "optimization_problem_data.h"
#include "functional.h"

namespace mixed_approx {

/**
 * @brief Коды статуса верификации решения
 */
enum class VerificationStatus {
    VERIFICATION_OK = 1,      // Все условия выполнены в допустимых пределах
    VERIFICATION_WARNING = 2,  // Незначительные нарушения, решение приемлемо с оговорками
    VERIFICATION_CRITICAL = 3, // Критические нарушения, требуется коррекция (шаг 6.2)
    VERIFICATION_FAILED = 4    // Фатальные ошибки, необходим перезапуск оптимизации
};

/**
 * @brief Коды качества решения
 */
enum class SolutionQuality {
    EXCELLENT = 0,        // quality >= 0.95 - готово к использованию
    GOOD = 1,             // 0.85 <= quality < 0.95 - рекомендуется визуальная проверка
    SATISFACTORY = 2,     // 0.70 <= quality < 0.85 - требует коррекции параметров
    UNACCEPTABLE = 3     // quality < 0.70 - обязательна коррекция шага 6.2
};

/**
 * @brief Расширенный результат верификации с диагностикой компонент функционала
 */
struct ExtendedVerificationResult {
    ValidationResult validation;           // Базовый результат валидации
    VerificationStatus status;              // Код статуса верификации
    SolutionQuality quality;                // Качество решения
    double quality_score;                  // Оценка качества [0, 1]
    FunctionalDiagnostics diagnostics;      // Диагностика компонент функционала
    std::vector<std::string> recommendations; // Рекомендации
    
    ExtendedVerificationResult()
        : status(VerificationStatus::VERIFICATION_OK)
        , quality(SolutionQuality::EXCELLENT)
        , quality_score(1.0) {}
};

/**
 * @brief Валидатор решений после оптимизации
 * 
 * Реализует комплексную верификацию решения согласно шагу 6.1:
 * - Шаг 6.1.2: Верификация интерполяционных условий с адаптивной точностью
 * - Шаг 6.1.3: Верификация безопасности относительно отталкивающих точек
 * - Шаг 6.1.4: Верификация численной стабильности решения
 * - Шаг 6.1.5: Анализ баланса компонент функционала
 * - Шаг 6.1.6: Формирование структурированного верификационного отчёта
 * - Шаг 6.1.7: Автоматическая классификация качества решения
 */
class SolutionValidator {
public:
    // ============== Параметры валидации ==============
    
    double epsilon_safe;            // Базовое безопасное расстояние до барьеров (по умолчанию 1e-8)
    double interp_tolerance;        // Базовый допуск для интерполяции (по умолчанию 1e-10)
    double max_value_factor;        // Множитель для проверки экстремальных значений (по умолчанию 100)
    int num_check_points;           // Число контрольных точек для проверки осцилляций (по умолчанию 1000)
    
    // ============== Конструкторы ==============
    
    SolutionValidator(double eps_safe = 1e-8, double interp_tol = 1e-10);
    
    // ============== Основные методы валидации ==============
    
    /**
     * @brief Валидация решения (базовый метод)
     * @param poly построенный полином F(x)
     * @param data данные задачи
     * @return результат валидации
     */
    ValidationResult validate(const Polynomial& poly, const OptimizationProblemData& data) const;
    
    /**
     * @brief Полная верификация решения с анализом компонент функционала
     * @param poly построенный полином F(x)
     * @param data данные задачи
     * @param func_diag диагностика функционала (выходной параметр)
     * @return расширенный результат верификации
     */
    ExtendedVerificationResult verify_full(const Polynomial& poly, 
                                         const OptimizationProblemData& data,
                                         FunctionalDiagnostics& func_diag) const;
    
    // ============== Методы проверки (шаг 6.1.2 - 6.1.4) ==============
    
    /**
     * @brief Шаг 6.1.2: Проверка интерполяционных условий с адаптивной точностью
     * @param poly полином
     * @param data данные задачи
     * @param max_error максимальная ошибка интерполяции (выходной параметр)
     * @return true если интерполяционные условия выполнены
     */
    bool check_interpolation(const Polynomial& poly, const OptimizationProblemData& data,
                             double& max_error) const;
    
    /**
     * @brief Шаг 6.1.3: Проверка безопасности барьеров
     * @param poly полином
     * @param data данные задачи
     * @param min_distance минимальное расстояние до запрещённых точек (выходной параметр)
     * @return true если барьеры безопасны
     */
    bool check_barrier_safety(const Polynomial& poly, const OptimizationProblemData& data,
                              double& min_distance) const;
    
    /**
     * @brief Шаг 6.1.4: Проверка численной стабильности решения
     * @param poly полином
     * @param data данные задачи
     * @return true если решение численно стабильно
     */
    bool check_numerical_stability(const Polynomial& poly, 
                                   const OptimizationProblemData& data) const;
    
    /**
     * @brief Проверка численной корректности (NaN/Inf)
     * @param poly полином
     * @param data данные задачи
     * @return true если значения корректны
     */
    bool check_numerical_correctness(const Polynomial& poly, 
                                     const OptimizationProblemData& data) const;
    
    /**
     * @brief Проверка физической правдоподобности (отсутствие экстремальных осцилляций)
     * @param poly полином
     * @param data данные задачи
     * @param max_value максимальное значение полинома на интервале (выходной параметр)
     * @return true если значения в разумных пределах
     */
    bool check_physical_plausibility(const Polynomial& poly,
                                     const OptimizationProblemData& data,
                                     double& max_value) const;
    
    // ============== Анализ баланса компонент (шаг 6.1.5) ==============
    
    /**
     * @brief Шаг 6.1.5: Анализ баланса компонент функционала
     * @param poly полином
     * @param data данные задачи
     * @param diagnostics структура для заполнения диагностикой
     * @return true если анализ выполнен успешно
     */
    bool analyze_functional_balance(const Polynomial& poly,
                                    const OptimizationProblemData& data,
                                    FunctionalDiagnostics& diagnostics) const;
    
    /**
     * @brief Классификация баланса компонент
     * @param share_approx доля аппроксимации (0-1)
     * @param share_repel доля отталкивания (0-1)
     * @param share_reg доля регуляризации (0-1)
     * @return строка с классификацией
     */
    std::string classify_balance(double share_approx, double share_repel, double share_reg) const;
    
    // ============== Оценка качества (шаг 6.1.7) ==============
    
    /**
     * @brief Шаг 6.1.7: Вычисление комплексной оценки качества решения
     * @param validation результат валидации
     * @param diagnostics диагностика функционала
     * @return оценка качества в диапазоне [0, 1]
     */
    double compute_quality_score(const ValidationResult& validation,
                                const FunctionalDiagnostics& diagnostics) const;
    
    /**
     * @brief Шаг 6.1.7: Классификация качества решения по шкале
     * @param quality_score оценка качества
     * @return уровень качества решения
     */
    SolutionQuality classify_quality(double quality_score) const;
    
    /**
     * @brief Генерация рекомендаций на основе классификации качества
     * @param quality уровень качества решения
     * @return вектор рекомендаций
     */
    std::vector<std::string> generate_quality_recommendations(SolutionQuality quality) const;
    
    // ============== Проекционная коррекция (опционально) ==============
    
    /**
     * @brief Применение проекционной коррекции при нарушении интерполяции
     * Пытается скорректировать коэффициенты полинома для точного выполнения интерполяционных условий
     * @param poly полином для коррекции (будет изменён)
     * @param data данные задачи
     * @return true если коррекция успешна
     */
    bool apply_projection_correction(Polynomial& poly, const OptimizationProblemData& data) const;
    
    // ============== Генерация отчётов (шаг 6.1.6) ==============
    
    /**
     * @brief Шаг 6.1.6: Генерация базового отчёта о валидации
     * @param result результат валидации
     * @return строковое представление отчёта
     */
    std::string generate_report(const ValidationResult& result) const;
    
    /**
     * @brief Шаг 6.1.6: Генерация расширенного отчёта с анализом функционала
     * @param result расширенный результат верификации
     * @return структурированный отчёт
     */
    std::string generate_extended_report(const ExtendedVerificationResult& result) const;
    
    /**
     * @brief Генерация краткой сводки о статусе верификации
     * @param status код статуса верификации
     * @return строка со сводкой
     */
    static std::string status_to_string(VerificationStatus status);
    
    /**
     * @brief Генерация краткой сводки о качестве решения
     * @param quality уровень качества
     * @return строка со сводкой
     */
    static std::string quality_to_string(SolutionQuality quality);
    
private:
    // ============== Вспомогательные методы ==============
    
    /**
     * @brief Вычисление адаптивного порога для интерполяции
     */
    double compute_adaptive_interpolation_threshold(const OptimizationProblemData& data) const;
    
    /**
     * @brief Вычисление адаптивного порога безопасности для барьеров
     */
    double compute_adaptive_safety_threshold(const OptimizationProblemData& data) const;
    
    /**
     * @brief Определение худшего интерполяционного узла
     */
    int find_worst_interpolation_node(const Polynomial& poly, 
                                       const OptimizationProblemData& data,
                                       double& max_error) const;
    
    /**
     * @brief Определение наиболее опасной отталкивающей точки
     */
    int find_most_dangerous_barrier(const Polynomial& poly,
                                    const OptimizationProblemData& data,
                                    double& min_distance) const;
    
    /**
     * @brief Вычисление числа экстремумов полинома на интервале
     */
    int count_extrema(const Polynomial& poly, const OptimizationProblemData& data) const;
    
    /**
     * @brief Вычисление нормированной кривизны
     */
    double compute_normalized_curvature(const Polynomial& poly,
                                        const OptimizationProblemData& data) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_SOLUTION_VALIDATOR_H
