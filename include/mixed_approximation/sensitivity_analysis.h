#ifndef MIXED_APPROXIMATION_SENSITIVITY_ANALYSIS_H
#define MIXED_APPROXIMATION_SENSITIVITY_ANALYSIS_H

#include "types.h"
#include "functional.h"
#include "solution_validator.h"
#include "composite_polynomial.h"
#include "objective_functor.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace mixed_approx {

// Forward declarations
class Optimizer;

// ============== Шаг 6.2: Структуры для анализа чувствительности ==============

/**
 * @brief Уровень чувствительности параметра
 */
enum class SensitivityLevel {
    LOW = 0,       // S < 0.2 - низкая чувствительность
    MODERATE = 1,  // 0.2 <= S < 1.0 - умеренная чувствительность
    HIGH = 2       // S >= 1.0 - высокая чувствительность
};

/**
 * @brief Тип критичности барьера
 */
enum class BarrierCriticality {
    NON_CRITICAL = 0,  // T_j <= 0.1 - некритичный
    MODERATE = 1,      // 0.1 < T_j <= 0.5 - умеренный
    CRITICAL = 2       // T_j > 0.5 - критический
};

/**
 * @brief Уровень устойчивости решения
 */
enum class StabilityLevel {
    HIGH = 0,      // CV_shape < 0.02
    MODERATE = 1,  // 0.02 <= CV_shape < 0.1
    LOW = 2        // CV_shape >= 0.1
};

/**
 * @brief Результат анализа чувствительности к одному параметру
 */
struct ParameterSensitivityResult {
    std::string parameter_name;     // Имя параметра
    double current_value;           // Текущее значение
    double sensitivity_coefficient; // Коэффициент чувствительности S
    SensitivityLevel level;        // Уровень чувствительности
    double recommended_min;         // Рекомендуемый минимум
    double recommended_max;         // Рекомендуемый максимум
    double optimal_value;           // Оптимальное значение (если определено)
    std::string recommendation;     // Текстовая рекомендация
    
    ParameterSensitivityResult()
        : current_value(0.0)
        , sensitivity_coefficient(0.0)
        , level(SensitivityLevel::LOW)
        , recommended_min(0.0)
        , recommended_max(0.0)
        , optimal_value(0.0) {}
};

/**
 * @brief Результат анализа чувствительности барьера
 */
struct BarrierSensitivityResult {
    int barrier_index;              // Индекс барьера
    double barrier_position;        // Позиция барьера y_j
    double current_weight;          // Текущий вес B_j
    BarrierCriticality criticality; // Критичность барьера
    double transfer_coefficient;   // Коэффициент передачи T_j
    double distance_change;         // Изменение расстояния при вариации веса
    double approximation_change;    // Изменение ошибки аппроксимации
    std::string recommendation;     // Рекомендация
    
    BarrierSensitivityResult()
        : barrier_index(-1)
        , barrier_position(0.0)
        , current_weight(0.0)
        , criticality(BarrierCriticality::NON_CRITICAL)
        , transfer_coefficient(0.0)
        , distance_change(0.0)
        , approximation_change(0.0) {}
};

/**
 * @brief Результат анализа кластера аппроксимирующих точек
 */
struct ClusterSensitivityResult {
    int cluster_id;                  // ID кластера
    std::vector<int> point_indices; // Индексы точек в кластере
    double locality_coefficient;    // Коэффициент локальности L_l
    double local_error;              // Локальная ошибка в кластере
    double global_distortion;       // Глобальное искажение формы
    std::string impact_description; // Описание влияния
    
    ClusterSensitivityResult()
        : cluster_id(-1)
        , locality_coefficient(0.0)
        , local_error(0.0)
        , global_distortion(0.0) {}
};

/**
 * @brief Результат стохастического анализа устойчивости
 */
struct StochasticStabilityResult {
    StabilityLevel stability_level;  // Уровень устойчивости
    double shape_variation_coef;     // Коэффициент вариации формы CV_shape
    double max_local_cv;             // Максимальный локальный CV
    double min_stability_margin;     // Минимальный запас устойчивости
    std::vector<std::pair<double, double>> unstable_intervals; // Нестабильные интервалы
    int sample_count;                 // Количество образцов M
    
    StochasticStabilityResult()
        : stability_level(StabilityLevel::HIGH)
        , shape_variation_coef(0.0)
        , max_local_cv(0.0)
        , min_stability_margin(0.0)
        , sample_count(0) {}
};

/**
 * @brief Элемент матрицы взаимной чувствительности
 */
struct SensitivityMatrixElement {
    std::string param_i;      // Имя первого параметра
    std::string param_j;      // Имя второго параметра
    double correlation;        // Корреляция C_ij
    bool strong_correlation;   // |C_ij| > 0.7
    
    SensitivityMatrixElement()
        : correlation(0.0)
        , strong_correlation(false) {}
};

/**
 * @brief Результат анализа компенсации параметров
 */
struct CompensationResult {
    std::string param_pair;        // Пара компенсирующих параметров
    double compensation_factor;    // Фактор компенсации
    bool requires_joint_tuning;    // Требуется совместная настройка
    
    CompensationResult()
        : compensation_factor(0.0)
        , requires_joint_tuning(false) {}
};

/**
 * @brief Тип проблемы, выявленной при анализе
 */
enum class ProblemType {
    NONE = 0,
    INSUFFICIENT_REGULARIZATION,  // Недостаточная регуляризация
    EXCESSIVE_BARRIER,           // Избыточно сильный барьер
    GLOBAL_INSTABILITY,          // Глобальная неустойчивость
    PARAMETER_CONFLICT,          // Конфликт параметров
    DATA_NOISE_SENSITIVITY       // Чувствительность к шуму в данных
};

/**
 * @brief Выявленная проблема с рекомендацией
 */
struct IdentifiedProblem {
    ProblemType type;             // Тип проблемы
    std::string description;     // Описание проблемы
    std::string recommendation;  // Рекомендация по решению
    double priority;             // Приоритет (выше = важнее)
    
    IdentifiedProblem()
        : type(ProblemType::NONE)
        , priority(0.0) {}
};

/**
 * @brief Результат полного анализа чувствительности
 */
struct SensitivityAnalysisResult {
    // Мета-информация
    std::string timestamp;              // Временная метка анализа
    std::string source_solution_info;   // Информация об исходном решении
    double original_objective;          // Исходное значение функционала
    
    // Параметрическая чувствительность
    ParameterSensitivityResult gamma_sensitivity;       // Чувствительность к gamma
    std::vector<BarrierSensitivityResult> barrier_sensitivities; // Чувствительности барьеров
    std::vector<ClusterSensitivityResult> cluster_sensitivities;  // Чувствительности кластеров
    
    // Устойчивость к возмущениям
    StochasticStabilityResult stochastic_stability;  // Стохастическая устойчивость
    
    // Матрица чувствительности
    std::vector<SensitivityMatrixElement> sensitivity_matrix;
    std::vector<CompensationResult> compensations;
    
    // Выявленные проблемы и рекомендации
    std::vector<IdentifiedProblem> problems;
    std::vector<std::string> prioritized_recommendations;
    
    // Итоговая оценка
    double overall_stability_score; // Оценка устойчивости [0, 100]
    std::string overall_assessment; // Общая оценка текстом
    
    // Конструктор
    SensitivityAnalysisResult()
        : original_objective(0.0)
        , overall_stability_score(100.0) {}
    
    /**
     * @brief Форматирование результата в отчёт
     */
    std::string format_report() const;
    
    /**
     * @brief Проверка, требуется ли коррекция
     */
    bool requires_correction() const {
        return overall_stability_score < 70.0 || !problems.empty();
    }
};

// ============== Шаг 6.2: Класс анализа чувствительности ==============

/**
 * @brief Класс для анализа чувствительности решения метода смешанной аппроксимации
 */
class SensitivityAnalyzer {
public:
    // ============== Параметры анализа ==============
    
    double gamma_min;                  // Минимальное значение gamma (по умолчанию 1e-8)
    double gamma_max_factor;          // Множитель для максимального gamma (по умолчанию 1e4)
    int gamma_trajectory_points;       // Число точек траектории gamma (по умолчанию 9)
    
    std::vector<double> barrier_variation_factors; // Факторы вариации весов барьеров
    int barrier_local_iterations;      // Число итераций для локальной оптимизации
    
    double cluster_distance_threshold; // Порог для кластеризации
    std::vector<double> cluster_beta_values; // Факторы вариации весов кластеров
    
    int stochastic_samples;            // Число образцов M (по умолчанию 50)
    double perturb_x_factor;           // Фактор возмущения координат delta_x (по умолчанию 0.01)
    double perturb_y_factor;          // Фактор возмущения значений delta_y (по умолчанию 0.05)
    int evaluation_points;             // Число точек для оценки вариации P (по умолчанию 200)
    
    bool use_warm_start;               // Использовать тёплый старт (по умолчанию true)
    bool parallel_analysis;            // Параллельный анализ (по умолчанию false)
    int analysis_level;                // Уровень анализа: 1 = быстрый, 2 = детальный
    
    // ============== Конструкторы ==============
    
    SensitivityAnalyzer();
    
    /**
     * @brief Конструктор с параметрами
     */
    explicit SensitivityAnalyzer(const ApproximationConfig& config);
    
    // ============== Основной метод анализа ==============
    
    /**
     * @brief Выполнить полный анализ чувствительности
     */
    SensitivityAnalysisResult analyze_full(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    // ============== Шаг 6.2.2: Параметрический анализ чувствительности ==============
    
    /**
     * @brief Шаг 6.2.2: Анализ чувствительности к параметру регуляризации gamma
     */
    ParameterSensitivityResult analyze_gamma_sensitivity(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    /**
     * @brief Шаг 6.2.2: Анализ чувствительности к весам отталкивания B_j
     */
    std::vector<BarrierSensitivityResult> analyze_barrier_sensitivity(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    /**
     * @brief Шаг 6.2.2: Анализ чувствительности к весам аппроксимации sigma_i
     */
    std::vector<ClusterSensitivityResult> analyze_cluster_sensitivity(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data);
    
    // ============== Шаг 6.2.3: Анализ чувствительности к возмущениям данных ==============
    
    /**
     * @brief Шаг 6.2.3: Стохастический анализ устойчивости
     */
    StochasticStabilityResult analyze_stochastic_stability(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    /**
     * @brief Шаг 6.2.3: Детерминированный анализ худших случаев
     */
    std::pair<double, double> analyze_worst_case(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    // ============== Шаг 6.2.4: Матрица взаимной чувствительности ==============
    
    /**
     * @brief Шаг 6.2.4: Построение матрицы взаимной чувствительности параметров
     */
    std::vector<SensitivityMatrixElement> build_sensitivity_matrix(
        const std::shared_ptr<Polynomial>& solution_poly,
        const OptimizationProblemData& data,
        const std::vector<double>& initial_coeffs);
    
    /**
     * @brief Шаг 6.2.4: Выявление компенсирующих пар параметров
     */
    std::vector<CompensationResult> detect_compensations(
        const std::vector<SensitivityMatrixElement>& matrix_elements);
    
    // ============== Шаг 6.2.5: Адаптивные рекомендации ==============
    
    /**
     * @brief Шаг 6.2.5: Классификация проблемы на основе анализа чувствительности
     */
    std::vector<IdentifiedProblem> classify_problems(
        const ParameterSensitivityResult& gamma_result,
        const std::vector<BarrierSensitivityResult>& barrier_results,
        const StochasticStabilityResult& stability_result);
    
    /**
     * @brief Шаг 6.2.5: Генерация персонализированных рекомендаций
     */
    std::vector<std::string> generate_recommendations(
        const std::vector<IdentifiedProblem>& problems,
        const ParameterSensitivityResult& gamma_result,
        const std::vector<BarrierSensitivityResult>& barrier_results);
    
    /**
     * @brief Шаг 6.2.5: Вычисление итоговой оценки устойчивости
     */
    double compute_overall_stability(
        const ParameterSensitivityResult& gamma_result,
        const std::vector<BarrierSensitivityResult>& barrier_results,
        const StochasticStabilityResult& stability_result);
    
    // ============== Вспомогательные методы ==============
    
    /**
     * @brief Генерация логарифмической сетки значений gamma
     */
    std::vector<double> generate_gamma_trajectory(double gamma_current) const;
    
    /**
     * @brief Построение кластеров аппроксимирующих точек
     */
    std::vector<std::vector<int>> build_clusters(const OptimizationProblemData& data) const;
    
    /**
     * @brief Вычисление метрик качества для заданного решения
     */
    struct QualityMetrics {
        double approx_error;    // E_approx = RMS(f(x_i) - F(x_i))
        double curvature_norm;  // C = sqrt(∫[F''(x)]²dx)
        double min_distance;    // D_min = min_j |F(y_j) - y_j^*|
        double total_functional; // J_total
    };
    
    QualityMetrics compute_quality_metrics(
        const Polynomial& poly,
        const OptimizationProblemData& data) const;
    
    /**
     * @brief Определение уровня чувствительности по коэффициенту
     */
    static SensitivityLevel get_sensitivity_level(double coefficient);
    
    /**
     * @brief Определение критичности барьера по коэффициенту передачи
     */
    static BarrierCriticality get_barrier_criticality(double transfer_coef);
    
    /**
     * @brief Определение уровня устойчивости по коэффициенту вариации
     */
    static StabilityLevel get_stability_level(double cv_shape);
    
private:
    // Конфигурация для создания функционала
    ApproximationConfig config_;
    
    /**
     * @brief Создать копию данных с изменённым gamma
     */
    OptimizationProblemData create_data_with_gamma(
        const OptimizationProblemData& original,
        double new_gamma) const;
    
    /**
     * @brief Создать копию данных с изменённым весом барьера
     */
    OptimizationProblemData create_data_with_barrier_weight(
        const OptimizationProblemData& original,
        int barrier_index,
        double new_weight) const;
    
    /**
     * @brief Создать возмущённую копию данных
     */
    OptimizationProblemData create_perturbed_data(
        const OptimizationProblemData& original,
        double perturb_x_factor,
        double perturb_y_factor,
        unsigned int seed) const;
    
    /**
     * @brief Выполнить оптимизацию с тёплым стартом
     */
    OptimizationResult optimize_with_warm_start(
        const OptimizationProblemData& data,
        const std::vector<double>& warm_start_coeffs,
        int max_iterations) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_SENSITIVITY_ANALYSIS_H
