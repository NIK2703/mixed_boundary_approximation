#ifndef MIXED_APPROXIMATION_VARIABLE_NORMALIZER_H
#define MIXED_APPROXIMATION_VARIABLE_NORMALIZER_H

#include "mixed_approximation/types.h"
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <cmath>

namespace mixed_approx {

// Forward declarations
struct ApproximationConfig;
struct WeightedPoint;
struct RepulsionPoint;
struct InterpolationNode;
class Polynomial;

/**
 * @brief Статус нормализации
 */
enum class NormalizationStatus {
    SUCCESS,                    // Нормализация выполнена успешно
    DEGENERATE_X_RANGE,         // Вырожденный диапазон по оси X
    CONSTANT_Y_VALUES,          // Константные значения по оси Y
    LOGARITHMIC_APPLIED,        // Применено логарифмическое преобразование
    ADAPTIVE_SCALING_USED,      // Использована адаптивная стратегия масштабирования
    WARNING                     // Есть предупреждения, но нормализация выполнена
};

/**
 * @brief Стратегия нормализации для оси X
 */
enum class XNormalizationStrategy {
    LINEAR_TO_M1_1,    // Линейное преобразование в [-1, 1]
    LINEAR_TO_0_1,     // Линейное преобразование в [0, 1]
    ADAPTIVE           // Адаптивная стратегия
};

/**
 * @brief Стратегия нормализации для оси Y
 */
enum class YNormalizationStrategy {
    STANDARDIZE,       // Стандартизация (центрирование + масштабирование)
    LINEAR,           // Простое линейное преобразование
    LOG_LINEAR        // Логарифмическое + линейное преобразование
};

/**
 * @brief Параметры нормализации для одной оси
 */
struct AxisNormalizationParams {
    double center;         // Центр преобразования (среднее/медиана)
    double scale;          // Масштаб (разброс/стандартное отклонение)
    double shift;          // Сдвиг = -center / scale
    double t_scale;        // Масштабный множитель t_scale = 2 / x_range для [-1, 1] или 1 / x_range для [0, 1]
    
    // Границы исходного диапазона
    double original_min;
    double original_max;
    
    // Флаги состояния
    bool is_valid;
    bool uses_log;         // Применено логарифмическое преобразование
    double log_base;       // Основание логарифма (e или 10)
    
    AxisNormalizationParams()
        : center(0.0), scale(1.0), shift(0.0), t_scale(1.0)
        , original_min(0.0), original_max(1.0)
        , is_valid(true), uses_log(false), log_base(2.718281828459045) {}
};

/**
 * @brief Параметры полного нормализационного преобразования
 */
struct NormalizationParams {
    // Параметры для оси X (абсцисса)
    AxisNormalizationParams x_params;
    
    // Параметры для оси Y (ордината)
    AxisNormalizationParams y_params;
    
    // Границы интервала определения
    double interval_a;     // Левая граница в исходных координатах
    double interval_b;     // Правая граница в исходных координатах
    
    // Нормализованные границы
    double norm_a;         // t_a = normalize(a)
    double norm_b;         // t_b = normalize(b)
    
    // Коррекция параметра регуляризации
    double gamma_correction_factor;  // (x_range/2)^3
    
    // Информация о стратегиях
    XNormalizationStrategy x_strategy;
    YNormalizationStrategy y_strategy;
    
    // Статус и сообщения
    NormalizationStatus status;
    std::string message;
    std::vector<std::string> warnings;
    
    // Флаг готовности
    bool is_ready;
    
    NormalizationParams()
        : interval_a(0.0), interval_b(1.0)
        , norm_a(-1.0), norm_b(1.0)
        , gamma_correction_factor(1.0)
        , x_strategy(XNormalizationStrategy::LINEAR_TO_M1_1)
        , y_strategy(YNormalizationStrategy::STANDARDIZE)
        , status(NormalizationStatus::SUCCESS)
        , is_ready(false) {}
    
    /**
     * @brief Прямое преобразование X → t
     */
    double transform_x(double x) const {
        return x_params.t_scale * x + x_params.shift;
    }
    
    /**
     * @brief Обратное преобразование t → X
     */
    double inverse_transform_x(double t) const {
        return (t - x_params.shift) / x_params.t_scale;
    }
    
    /**
     * @brief Прямое преобразование Y → v
     */
    double transform_y(double y) const {
        return y_params.t_scale * y + y_params.shift;
    }
    
    /**
     * @brief Обратное преобразование v → Y
     */
    double inverse_transform_y(double v) const {
        return (v - y_params.shift) / y_params.t_scale;
    }
};

/**
 * @brief Результат нормализации данных
 */
struct NormalizationResult {
    bool success;
    NormalizationParams params;
    std::vector<double> approx_x_norm;      // Нормализованные аппроксимирующие x_i
    std::vector<double> approx_f_norm;      // Нормализованные значения f(x_i)
    std::vector<double> approx_weight_norm; // Скорректированные веса
    
    std::vector<double> repel_y_norm;       // Нормализованные отталкивающие y_j
    std::vector<double> repel_forbidden_norm; // Нормализованные запретные значения y_j^*
    // repel_weight НЕ нормализуется
    
    std::vector<double> interp_z_norm;       // Нормализованные интерполяционные z_e
    std::vector<double> interp_f_norm;       // Нормализованные значения f(z_e)
    
    double gamma_normalized;                 // Скорректированный параметр регуляризации
    
    std::string diagnostic_report;
    
    NormalizationResult()
        : success(false)
        , gamma_normalized(0.0) {}
};

/**
 * @brief Класс для масштабирования переменных (шаг 5.2)
 * 
 * Обеспечивает:
 * - Нормализацию входных данных к стандартным интервалам
 * - Обратное преобразование коэффициентов полинома
 * - Валидацию корректности преобразований
 * - Обработку крайних случаев
 */
class VariableNormalizer {
public:
    /**
     * @brief Конструктор с конфигурацией
     * @param config конфигурация задачи аппроксимации
     */
    explicit VariableNormalizer(const ApproximationConfig& config);
    
    /**
     * @brief Конструктор по умолчанию
     */
    VariableNormalizer();
    
    /**
     * @brief Деструктор
     */
    ~VariableNormalizer() = default;
    
    // Запрет копирования
    VariableNormalizer(const VariableNormalizer&) = delete;
    VariableNormalizer& operator=(const VariableNormalizer&) = delete;
    
    // Разрешение перемещения
    VariableNormalizer(VariableNormalizer&&) noexcept = default;
    VariableNormalizer& operator=(VariableNormalizer&&) noexcept = default;
    
    /**
     * @brief Выполнить нормализацию данных
     * @return результат нормализации
     */
    NormalizationResult normalize();
    
    /**
     * @brief Выполнить обратное преобразование коэффициентов полинома
     * @param normalized_coeffs коэффициенты в нормализованных координатах
     * @return коэффициенты в исходных координатах
     */
    std::vector<double> inverse_transform_coefficients(
        const std::vector<double>& normalized_coeffs) const;
    
    /**
     * @brief Обратное преобразование для мономиального базиса с использованием биномиальных коэффициентов
     * @param normalized_coeffs коэффициенты [g_n, g_{n-1}, ..., g_0] в нормализованных координатах
     * @return коэффициенты [a_n, a_{n-1}, ..., a_0] в исходных координатах
     */
    std::vector<double> inverse_transform_monomial(
        const std::vector<double>& normalized_coeffs) const;
    
    /**
     * @brief Прямое преобразование коэффициентов (для тестирования)
     * @param original_coeffs коэффициенты в исходных координатах
     * @return коэффициенты в нормализованных координатах
     */
    std::vector<double> forward_transform_coefficients(
        const std::vector<double>& original_coeffs) const;
    
    /**
     * @brief Получить параметры нормализации
     * @return константная ссылка на параметры
     */
    const NormalizationParams& get_params() const { return params_; }
    
    /**
     * @brief Получить статус нормализации
     * @return статус
     */
    NormalizationStatus get_status() const { return params_.status; }
    
    /**
     * @brief Проверить корректность нормализации (тест инвариантности)
     * @return true если все тесты пройдены
     */
    bool validate_normalization();
    
    /**
     * @brief Получить диагностический отчёт
     * @return строку с отчётом
     */
    std::string get_diagnostic_report() const;
    
    /**
     * @brief Проверка на вырожденные случаи
     * @return пара (is_degenerate, message)
     */
    std::pair<bool, std::string> check_degenerate_cases() const;
    
    /**
     * @brief Установить стратегию нормализации для оси X
     */
    void set_x_strategy(XNormalizationStrategy strategy) {
        x_strategy_ = strategy;
    }
    
    /**
     * @brief Установить стратегию нормализации для оси Y
     */
    void set_y_strategy(YNormalizationStrategy strategy) {
        y_strategy_ = strategy;
    }
    
    /**
     * @brief Установить пользовательские параметры (для тестирования)
     */
    void set_custom_params(const NormalizationParams& params) {
        params_ = params;
        params_.is_ready = true;
    }
    
private:
    // Исходные данные
    ApproximationConfig config_;
    
    // Параметры нормализации
    NormalizationParams params_;
    
    // Стратегии
    XNormalizationStrategy x_strategy_;
    YNormalizationStrategy y_strategy_;
    
    // Константы для порогов
    static constexpr double EPS_RANGE = 1e-12;
    static constexpr double EPS_SCALE = 1e-12;
    static constexpr double LOG_RATIO_THRESHOLD = 100.0;  // Порог для применения логарифма
    
    /**
     * @brief Вычислить параметры нормализации для оси X
     */
    void compute_x_params();
    
    /**
     * @brief Вычислить параметры нормализации для оси Y
     */
    void compute_y_params();
    
    /**
     * @brief Вычислить все значения для нормализации
     */
    void compute_normalized_values(NormalizationResult& result);
    
    /**
     * @brief Вычислить биномиальные коэффициенты C(n, k)
     */
    std::vector<std::vector<double>> compute_binomial_coefficients(int n) const;
    
    /**
     * @brief Вычислить степени массива
     */
    std::vector<double> compute_powers(double base, int max_power) const;
    
    /**
     * @brief Обработать вырожденные случаи
     */
    void handle_degenerate_cases(NormalizationResult& result);
    
    /**
     * @brief Сгенерировать диагностический отчёт
     */
    std::string generate_diagnostic_report() const;
    
    /**
     * @brief Проверить инвариантность интерполяционных условий
     */
    bool test_interpolation_invariance() const;
    
    /**
     * @brief Проверить инвариантность расстояний до барьеров
     */
    bool test_barrier_distance_invariance() const;
    
    /**
     * @brief Проверить точность обратного преобразования коэффициентов
     */
    bool test_coefficient_precision() const;
};

/**
 * @brief Вспомогательные функции для нормализации
 */
namespace NormalizationUtils {
    
    /**
     * @brief Вычислить медиану массива
     */
    double compute_median(std::vector<double> values);
    
    /**
     * @brief Вычислить робастное стандартное отклонение (MAD × 1.4826)
     */
    double compute_robust_std(const std::vector<double>& values);
    
    /**
     * @brief Проверить вырожденность диапазона
     */
    bool is_degenerate_range(double min_val, double max_val, double center);
    
    /**
     * @brief Проверить константность значений
     */
    bool is_constant_values(const std::vector<double>& values, double center);
    
    /**
     * @brief Применить логарифмическое преобразование (если применимо)
     */
    std::vector<double> apply_log_transform(const std::vector<double>& values);
    
    /**
     * @brief Проверить необходимость логарифмического преобразования
     */
    bool needs_log_transform(const std::vector<double>& values);
}

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_VARIABLE_NORMALIZER_H
