#ifndef MIXED_APPROXIMATION_OVERFITTING_DETECTOR_H
#define MIXED_APPROXIMATION_OVERFITTING_DETECTOR_H

#include "types.h"
#include "composite_polynomial.h"
#include "optimization_problem_data.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace mixed_approx {

// Forward declarations
class CompositePolynomial;
class OptimizationProblemData;

/**
 * @brief Уровень риска переобучения
 */
enum class OverfittingRiskLevel {
    LOW = 0,       // risk_score < 0.3 — решение качественное
    MODERATE,     // 0.3 ≤ risk_score < 0.7 — умеренное переобучение
    HIGH          // risk_score ≥ 0.7 — сильное переобучение
};

/**
 * @brief Тип стратегии коррекции переобучения
 */
enum class CorrectionStrategy {
    NONE = 0,          // Коррекция не требуется
    REGULARIZATION,    // Усиление регуляризации (γ)
    DEGREE_REDUCTION,  // Снижение степени полинома
    WEIGHT_CORRECTION, // Коррекция весов данных
    MANUAL_REVIEW      // Требуется ручной просмотр
};

/**
 * @brief Результат вычисления метрики кривизны
 */
struct CurvatureMetricResult {
    double second_deriv_norm;           // ||F''||² = ∫(F''(x))² dx
    double expected_curvature_scale;    // Ожидаемый масштаб кривизны
    double normalized_curvature;        // κ = ||F''|| / scale_curvature_expected
    double threshold;                   // Порог для данной задачи
    bool is_over_threshold;             // Превышен ли порог
    
    CurvatureMetricResult()
        : second_deriv_norm(0.0), expected_curvature_scale(1.0)
        , normalized_curvature(0.0), threshold(10.0), is_over_threshold(false) {}
};

/**
 * @brief Результат анализа осцилляций
 */
struct OscillationMetricResult {
    int total_extrema;                  // Всего найдено экстремумов
    int extrema_in_empty_regions;       // Экстремумы в "пустых" областях
    double oscillation_score;            // Итоговый скор осцилляций
    std::vector<double> extremum_positions; // Позиции экстремумов
    std::vector<bool> is_suspicious;    // Флаг подозрительности для каждого
    
    OscillationMetricResult()
        : total_extrema(0), extrema_in_empty_regions(0)
        , oscillation_score(0.0) {}
};

/**
 * @brief Результат кросс-валидации
 */
struct CrossValidationMetricResult {
    double train_error;                 // Ошибка на обучающем множестве (RMS)
    double cv_error;                    // Ошибка кросс-валидации (RMS)
    double generalization_ratio;        // cv_error / train_error
    int num_folds;                      // Число фолдов
    std::vector<double> fold_errors;    // Ошибки по фолдам
    
    CrossValidationMetricResult()
        : train_error(0.0), cv_error(0.0)
        , generalization_ratio(1.0), num_folds(5) {}
};

/**
 * @brief Результат анализа чувствительности к шуму
 */
struct SensitivityMetricResult {
    double sensitivity_score;           // max_p(std_F[p]) / max_p|F(ξ_p)|
    double max_std;                     // Максимальное стандартное отклонение
    double max_value;                   // Максимальное значение полинома
    int perturbation_count;             // Число возмущённых копий
    
    SensitivityMetricResult()
        : sensitivity_score(0.0), max_std(0.0)
        , max_value(1.0), perturbation_count(10) {}
};

/**
 * @brief Результат диагностики переобучения
 */
struct OverfittingDiagnostics {
    // Индивидуальные метрики
    CurvatureMetricResult curvature;
    OscillationMetricResult oscillation;
    CrossValidationMetricResult cross_validation;
    SensitivityMetricResult sensitivity;
    
    // Комплексная оценка
    double risk_score;                  // Итоговый риск [0, 1]
    OverfittingRiskLevel risk_level;    // Уровень риска
    CorrectionStrategy recommended_strategy; // Рекомендуемая стратегия
    
    // Детали коррекции
    double suggested_gamma_multiplier;  // Множитель для γ (если применимо)
    int suggested_degree_reduction;     // Снижение степени (если применимо)
    std::vector<int> outlier_indices;   // Индексы потенциальных выбросов
    
    // Рекомендации
    std::vector<std::string> recommendations;
    
    OverfittingDiagnostics()
        : risk_score(0.0), risk_level(OverfittingRiskLevel::LOW)
        , recommended_strategy(CorrectionStrategy::NONE)
        , suggested_gamma_multiplier(1.0)
        , suggested_degree_reduction(0) {}
    
    /**
     * @brief Форматирование отчёта
     */
    std::string format_report() const;
    
    /**
     * @brief Проверка наличия проблем
     */
    bool has_problems() const { return risk_score >= 0.3; }
    bool has_critical_problems() const { return risk_score >= 0.7; }
};

/**
 * @brief Конфигурация детектора переобучения
 */
struct OverfittingDetectorConfig {
    // Пороги для метрик
    double curvature_threshold_base;        // Базовый порог для κ (10.0)
    double oscillation_threshold_low;       // Нижний порог осцилляций (0.5)
    double oscillation_threshold_high;      // Верхний порог осцилляций (2.0)
    double generalization_threshold_low;    // Нижний порог обобщения (1.5)
    double generalization_threshold_high;   // Верхний порог обобщения (3.0)
    double sensitivity_threshold_low;       // Нижний порог чувствительности (0.01)
    double sensitivity_threshold_high;      // Верхний порог чувствительности (0.1)
    
    // Веса для комплексной оценки
    double weight_curvature;            // w1 = 0.4
    double weight_oscillation;          // w2 = 0.3
    double weight_generalization;        // w3 = 0.2
    double weight_sensitivity;          // w4 = 0.1
    
    // Параметры анализа
    int num_grid_points;                // Число точек для сетки (1000)
    int num_extremum_refinements;       // Уточнений корня Ньютоном (3)
    double gap_ratio_threshold;          // Порог для "пустой области" (3.0)
    
    // Параметры кросс-валидации
    int max_cv_folds;                   // Макс число фолдов (5)
    int min_cv_folds;                   // Мин число фолдов (3)
    
    // Параметры анализа чувствительности
    int num_perturbations;              // Число возмущений (10)
    double perturb_x_scale;             // Масштаб возмущения x (1e-4)
    double perturb_y_scale;             // Масштаб возмущения y (1e-3)
    
    // Параметры коррекции
    double gamma_boost_factor;          // Коэффициент усиления γ (2.0)
    double gamma_max_multiplier;        // Макс множитель γ (100.0)
    double min_gamma;                   // Мин значение γ (1e-6)
    double degree_reduction_factor;     // Фактор снижения степени (0.2)
    int min_degree_from_constraints;    // Мин степень от числа ограничений
    
    // Режим работы
    bool enable_cross_validation;      // Включить кросс-валидацию
    bool enable_sensitivity_analysis;  // Включить анализ чувствительности
    bool enable_auto_correction;        // Включить авто-коррекцию
    bool assume_oscillating_data;       // Данные могут осциллировать (снижает вес осцилляций)
    
    OverfittingDetectorConfig()
        : curvature_threshold_base(10.0)
        , oscillation_threshold_low(0.5)
        , oscillation_threshold_high(2.0)
        , generalization_threshold_low(1.5)
        , generalization_threshold_high(3.0)
        , sensitivity_threshold_low(0.01)
        , sensitivity_threshold_high(0.1)
        , weight_curvature(0.4)
        , weight_oscillation(0.3)
        , weight_generalization(0.2)
        , weight_sensitivity(0.1)
        , num_grid_points(1000)
        , num_extremum_refinements(3)
        , gap_ratio_threshold(3.0)
        , max_cv_folds(5)
        , min_cv_folds(3)
        , num_perturbations(10)
        , perturb_x_scale(1e-4)
        , perturb_y_scale(1e-3)
        , gamma_boost_factor(2.0)
        , gamma_max_multiplier(100.0)
        , min_gamma(1e-6)
        , degree_reduction_factor(0.2)
        , min_degree_from_constraints(0)
        , enable_cross_validation(true)
        , enable_sensitivity_analysis(true)
        , enable_auto_correction(false)
        , assume_oscillating_data(false) {}
};

/**
 * @brief Результат коррекции переобучения
 */
struct OverfittingCorrectionResult {
    bool correction_applied;            // Была ли применена коррекция
    CorrectionStrategy strategy_used;   // Использованная стратегия
    double new_gamma;                   // Новый γ после коррекции
    int new_degree;                     // Новая степень полинома
    double risk_before;                 // Риск до коррекции
    double risk_after;                  // Риск после коррекции
    std::string message;                // Сообщение о результате
    std::vector<double> corrected_weights; // Скорректированные веса (если применимо)
    
    OverfittingCorrectionResult()
        : correction_applied(false)
        , strategy_used(CorrectionStrategy::NONE)
        , new_gamma(0.0), new_degree(0)
        , risk_before(0.0), risk_after(0.0) {}
};

/**
 * @brief Класс для диагностики переобучения в полиномиальной аппроксимации
 * 
 * Реализует шаг 5.3: проверка на переобучение с комплексными метриками
 * и адаптивными стратегиями коррекции.
 * 
 * Функциональность:
 * - Анализ нормы второй производной (κ)
 * - Обнаружение осцилляций через анализ экстремумов
 * - Кросс-валидация с исключением точек
 * - Анализ чувствительности к шуму
 * - Комплексная оценка риска переобучения
 * - Адаптивные стратегии коррекции
 */
class OverfittingDetector {
public:
    /**
     * @brief Конструктор
     * @param config конфигурация детектора
     */
    explicit OverfittingDetector(const OverfittingDetectorConfig& config = OverfittingDetectorConfig());
    
    /**
     * @brief Полная диагностика переобучения
     * @param poly построенный полином F(x)
     * @param data данные задачи оптимизации
     * @param gamma текущее значение параметра регуляризации
     * @return результат диагностики
     */
    OverfittingDiagnostics diagnose(const Polynomial& poly,
                                   const OptimizationProblemData& data,
                                   double gamma);
    
    /**
     * @brief Вычисление метрики кривизны
     */
    CurvatureMetricResult compute_curvature_metric(const Polynomial& poly,
                                                    const OptimizationProblemData& data,
                                                    double gamma);
    
    /**
     * @brief Вычисление метрики осцилляций
     */
    OscillationMetricResult compute_oscillation_metric(const Polynomial& poly,
                                                       const OptimizationProblemData& data);
    
    /**
     * @brief Вычисление метрики кросс-валидации
     */
    CrossValidationMetricResult compute_cross_validation_metric(
        const Polynomial& poly,
        const OptimizationProblemData& data,
        double gamma,
        std::function<Polynomial(const std::vector<double>&)> optimizer_callback = nullptr);
    
    /**
     * @brief Вычисление метрики чувствительности
     */
    SensitivityMetricResult compute_sensitivity_metric(const Polynomial& poly,
                                                        const OptimizationProblemData& data);
    
    /**
     * @brief Вычисление комплексного скрипта риска
     */
    double compute_risk_score(const OverfittingDiagnostics& diagnostics) const;
    
    /**
     * @brief Определение уровня риска
     */
    OverfittingRiskLevel assess_risk_level(double risk_score) const;
    
    /**
     * @brief Рекомендация стратегии коррекции
     */
    CorrectionStrategy recommend_correction_strategy(const OverfittingDiagnostics& diagnostics) const;
    
    /**
     * @brief Применение стратегии коррекции
     * @param diagnostics результат диагностики
     * @param current_gamma текущий γ
     * @param current_degree текущая степень полинома
     * @param weights текущие веса данных
     * @return результат коррекции
     */
    OverfittingCorrectionResult apply_correction(const OverfittingDiagnostics& diagnostics,
                                                  double current_gamma,
                                                  int current_degree,
                                                  std::vector<double> weights);
    
    /**
     * @brief Генерация рекомендаций
     */
    std::vector<std::string> generate_recommendations(const OverfittingDiagnostics& diagnostics) const;
    
    /**
     * @brief Адаптивный порог кривизны с учётом плотности данных
     */
    double compute_adaptive_curvature_threshold(const OptimizationProblemData& data) const;
    
    /**
     * @brief Обнаружение потенциальных выбросов
     */
    std::vector<int> detect_outliers(const Polynomial& poly,
                                      const OptimizationProblemData& data,
                                      double residual_threshold = 3.0);
    
    /**
     * @brief Получение конфигурации
     */
    const OverfittingDetectorConfig& get_config() const { return config_; }
    
    /**
     * @brief Установка callback для оптимизации (для кросс-валидации)
     */
    void set_optimization_callback(std::function<Polynomial(const std::vector<double>&)> callback) {
        optimizer_callback_ = callback;
    }
    
    // Методы для тестирования (made public for unit tests)
    
    /**
     * @brief Вычисление медианного расстояния между точками данных
     */
    double compute_median_spacing(const std::vector<double>& points) const;
    
    /**
     * @brief Нормировка метрики к диапазону [0, 1]
     */
    double normalize_metric(double value, double low, double high) const;
    
private:
    OverfittingDetectorConfig config_;
    std::function<Polynomial(const std::vector<double>&)> optimizer_callback_;
    
    /**
     * @brief Поиск корней производной на сетке с уточнением Ньютоном
     */
    std::vector<double> find_derivative_roots(const Polynomial& poly,
                                                double a, double b,
                                                int num_grid_points) const;
    
    /**
     * @brief Уточнение корня методом Ньютона
     */
    double refine_root_newton(const Polynomial& deriv,
                              double x_low, double x_high,
                              int max_iterations,
                              double tolerance) const;
    
    /**
     * @brief Определение типа экстремума
     */
    int classify_extremum(const Polynomial& poly, double x) const;
    
    /**
     * @brief Вычисление характерного масштаба кривизны
     */
    double compute_expected_curvature_scale(const OptimizationProblemData& data) const;
    
    /**
     * @brief Генерация возмущённых данных
     */
    std::vector<WeightedPoint> generate_perturbed_points(
        const std::vector<WeightedPoint>& points,
        double delta_x, double delta_y) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OVERFITTING_DETECTOR_H
