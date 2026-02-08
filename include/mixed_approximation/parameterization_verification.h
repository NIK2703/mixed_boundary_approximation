#ifndef MIXED_APPROXIMATION_PARAMETERIZATION_VERIFICATION_H
#define MIXED_APPROXIMATION_PARAMETERIZATION_VERIFICATION_H

#include "types.h"
#include "interpolation_basis.h"
#include "weight_multiplier.h"
#include "correction_polynomial.h"
#include "composite_polynomial.h"
#include <vector>
#include <string>
#include <sstream>

namespace mixed_approx {

/**
 * @brief Статус верификации
 */
enum class VerificationStatus {
    PASSED,       ///< Все тесты пройдены успешно
    WARNING,      ///< Есть предупреждения, но критических ошибок нет
    FAILED        ///< Критические ошибки, верификация не пройдена
};

/**
 * @brief Тип рекомендации по коррекции
 */
enum class RecommendationType {
    NONE,               ///< Рекомендаций нет
    CHANGE_BASIS,       ///< Изменить тип базиса (MONOMIAL -> CHEBYSHEV или наоборот)
    MERGE_NODES,        ///< Объединить близкие интерполяционные узлы
    REDUCE_DEGREE,      ///< Уменьшить степень полинома
    INCREASE_GAMMA,     ///< Увеличить коэффициент регуляризации
    USE_LONG_DOUBLE,    ///< Использовать арифметику повышенной точности
    REDUCE_TOLERANCE    ///< Ослабить допуск верификации
};

/**
 * @brief Структура для хранения информации об ошибке в одном узле
 */
struct NodeError {
    int node_index;              ///< Индекс узла
    double coordinate;           ///< Координата узла z_e
    double target_value;         ///< Ожидаемое значение f(z_e)
    double computed_value;       ///< Вычисленное значение F(z_e)
    double absolute_error;        ///< Абсолютная ошибка |F(z_e) - f(z_e)|
    double relative_error;       ///< Относительная ошибка
    double W_value;              ///< Значение W(z_e) (для диагностики)
    bool W_acceptable;           ///< Является ли W(z_e) достаточно близким к нулю
    
    NodeError()
        : node_index(-1), coordinate(0.0), target_value(0.0), computed_value(0.0),
          absolute_error(0.0), relative_error(0.0), W_value(0.0), W_acceptable(true) {}
};

/**
 * @brief Структура для хранения рекомендации
 */
struct Recommendation {
    RecommendationType type;
    std::string message;
    std::string rationale;
    
    Recommendation()
        : type(RecommendationType::NONE), message(""), rationale("") {}
    
    Recommendation(RecommendationType t, const std::string& msg, const std::string& rat)
        : type(t), message(msg), rationale(rat) {}
};

/**
 * @brief Результат теста интерполяции
 */
struct InterpolationTestResult {
    bool passed;                              ///< Пройден ли тест
    int total_nodes;                          ///< Общее число узлов
    int failed_nodes;                         ///< Число узлов с ошибкой
    double max_absolute_error;                 ///< Максимальная абсолютная ошибка
    double max_relative_error;                ///< Максимальная относительная ошибка
    double tolerance;                         ///< Использованный допуск
    std::vector<NodeError> node_errors;       ///< Детали ошибок по узлам
    std::vector<std::string> info_messages;   ///< Информационные сообщения
    
    InterpolationTestResult()
        : passed(false), total_nodes(0), failed_nodes(0),
          max_absolute_error(0.0), max_relative_error(0.0), tolerance(1e-10) {}
};

/**
 * @brief Результат теста полноты пространства решений
 */
struct CompletenessTestResult {
    bool passed;                              ///< Пройден ли тест
    int expected_rank;                        ///< Ожидаемый ранг (n_free)
    int actual_rank;                          ///< Фактический ранг
    double condition_number;                  ///< Число обусловленности
    double min_singular_value;                ///< Минимальное сингулярное значение
    double relative_min_sv;                   ///< Относительное мин. сингулярное значение (к max)
    std::vector<double> singular_values;      ///< Все сингулярные значения
    std::vector<std::string> warnings;        ///< Предупреждения
    std::vector<std::string> info_messages;   ///< Информационные сообщения
    
    CompletenessTestResult()
        : passed(false), expected_rank(0), actual_rank(0),
          condition_number(0.0), min_singular_value(0.0), relative_min_sv(0.0) {}
};

/**
 * @brief Результат теста численной устойчивости
 */
struct StabilityTestResult {
    bool passed;                              ///< Пройден ли тест
    double perturbation_sensitivity;           ///< Чувствительность к возмущениям
    double scale_balance_ratio;                ///< Баланс масштабов компонент
    double gradient_condition_number;          ///< Обусловленность градиента
    double max_component_scale;               ///< Макс. масштаб компонент
    double min_component_scale;               ///< Мин. масштаб компонент
    std::vector<std::string> warnings;        ///< Предупреждения
    std::vector<std::string> info_messages;   ///< Информационные сообщения
    
    StabilityTestResult()
        : passed(false), perturbation_sensitivity(0.0), scale_balance_ratio(0.0),
          gradient_condition_number(0.0), max_component_scale(0.0), min_component_scale(0.0) {}
};

/**
 * @brief Полный результат верификации параметризации
 */
struct ParameterizationVerification {
    // Общий статус
    VerificationStatus overall_status;
    
    // Параметры верификации
    int polynomial_degree;                    ///< Степень полинома n
    int num_constraints;                      ///< Число интерполяционных узлов m
    int num_free_params;                      ///< Число свободных параметров n_free = n - m + 1
    double interval_a;                        ///< Левая граница интервала
    double interval_b;                        ///< Правая граница интервала
    
    // Результаты тестов
    InterpolationTestResult interpolation_test;
    CompletenessTestResult completeness_test;
    StabilityTestResult stability_test;
    
    // Рекомендации
    std::vector<Recommendation> recommendations;
    
    // Диагностические сообщения
    std::vector<std::string> info_messages;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    ParameterizationVerification()
        : overall_status(VerificationStatus::PASSED),
          polynomial_degree(0), num_constraints(0), num_free_params(0),
          interval_a(0.0), interval_b(1.0) {}
    
    /**
     * @brief Проверка, есть ли предупреждения
     */
    bool has_warnings() const { return !warnings.empty(); }
    
    /**
     * @brief Проверка, есть ли ошибки
     */
    bool has_errors() const { return !errors.empty(); }
    
    /**
     * @brief Проверка, пройдена ли верификация
     */
    bool is_passed() const {
        return overall_status == VerificationStatus::PASSED ||
               overall_status == VerificationStatus::WARNING;
    }
    
    /**
     * @brief Форматирование результата в строку
     */
    std::string format(bool detailed = false) const;
};

/**
 * @brief Класс для верификации корректности параметризации
 * 
 * Реализует три теста:
 * 1. Тест на выполнение интерполяционных условий
 * 2. Тест на полноту пространства решений
 * 3. Тест на численную устойчивость
 */
class ParameterizationVerifier {
public:
    /**
     * @brief Конструктор с параметрами верификации
     * @param interp_tolerance Допуск для интерполяционных условий (по умолчанию 1e-10)
     * @param svd_tolerance Допуск для сингулярных значений (по умолчанию 1e-12)
     * @param condition_limit Максимальное допустимое число обусловленности (по умолчанию 1e8)
     * @param perturbation_scale Масштаб возмущения для теста устойчивости (по умолчанию 1e-8)
     */
    ParameterizationVerifier(
        double interp_tolerance = 1e-10,
        double svd_tolerance = 1e-12,
        double condition_limit = 1e8,
        double perturbation_scale = 1e-8);
    
    /**
     * @brief Выполнить полную верификацию параметризации
     * @param composite Композитный полином F(x) = P_int(x) + Q(x)*W(x)
     * @param interp_nodes Интерполяционные узлы (значения f(z_e))
     * @return Результат верификации
     */
    ParameterizationVerification verify(
        const CompositePolynomial& composite,
        const std::vector<InterpolationNode>& interp_nodes);
    
    /**
     * @brief Выполнить верификацию по компонентам
     * @param basis Базисный интерполяционный полином
     * @param W Весовой множитель
     * @param Q Корректирующий полином
     * @param interp_nodes Интерполяционные узлы
     * @return Результат верификации
     */
    ParameterizationVerification verify_components(
        const InterpolationBasis& basis,
        const WeightMultiplier& W,
        const CorrectionPolynomial& Q,
        const std::vector<InterpolationNode>& interp_nodes);
    
    /**
     * @brief Тест интерполяционных условий
     */
    InterpolationTestResult test_interpolation(
        const CompositePolynomial& composite,
        const std::vector<InterpolationNode>& interp_nodes);
    
    /**
     * @brief Тест полноты пространства решений
     */
    CompletenessTestResult test_completeness(
        const CorrectionPolynomial& Q,
        const WeightMultiplier& W,
        double interval_a,
        double interval_b);
    
    /**
     * @brief Тест численной устойчивости
     */
    StabilityTestResult test_stability(
        const CompositePolynomial& composite,
        const std::vector<InterpolationNode>& interp_nodes);
    
    /**
     * @brief Установить новые параметры верификации
     */
    void set_parameters(
        double interp_tolerance,
        double svd_tolerance,
        double condition_limit,
        double perturbation_scale);
    
private:
    // Параметры верификации
    double interp_tolerance_;
    double svd_tolerance_;
    double condition_limit_;
    double perturbation_scale_;
    
    /**
     * @brief Вычислить сингулярные значения матрицы (упрощённая версия)
     */
    std::vector<double> compute_singular_values(
        const std::vector<std::vector<double>>& matrix,
        double& condition_number);
    
    /**
     * @brief Диагностировать источник ошибки интерполяции
     */
    Recommendation diagnose_interpolation_error(
        const NodeError& error,
        double W_tolerance);
    
    /**
     * @brief Диагностировать причину плохой обусловленности
     */
    Recommendation diagnose_condition_issue(
        const CompletenessTestResult& result,
        BasisType current_basis);
    
    /**
     * @brief Сгенерировать узлы Чебышёва
     */
    std::vector<double> chebyshev_nodes(int n, double a, double b);
    
    /**
     * @brief Вычислить базисную функцию (публичная обёртка)
     */
    double compute_basis_function_public(const CorrectionPolynomial& Q, double x, int k) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_PARAMETERIZATION_VERIFICATION_H
