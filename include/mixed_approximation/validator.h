#ifndef MIXED_APPROXIMATION_VALIDATOR_H
#define MIXED_APPROXIMATION_VALIDATOR_H

#include "types.h"
#include <string>
#include <vector>
#include <sstream>

namespace mixed_approx {

/**
 * @brief Уровень серьёзности проблемы валидации
 */
enum class ValidationLevel {
    Error,     ///< Критическая ошибка, блокирующая выполнение
    Warning    ///< Предупреждение, не блокирующее, но требующее внимания
};

/**
 * @brief Структура для описания проблемы валидации
 */
struct ValidationIssue {
    ValidationLevel level;      ///< Уровень серьёзности
    std::string message;       ///< Описание проблемы
    std::string recommendation; ///< Рекомендация по исправлению
    
    ValidationIssue(ValidationLevel lvl, const std::string& msg, const std::string& rec = "")
        : level(lvl), message(msg), recommendation(rec) {}
};

/**
 * @brief Структура с результатами валидации
 */
struct ValidationReport {
    std::vector<ValidationIssue> errors;     ///< Критические ошибки
    std::vector<ValidationIssue> warnings;   ///< Предупреждения
    
    // Статистика
    size_t approx_points_count = 0;
    size_t repel_points_count = 0;
    size_t interp_nodes_count = 0;
    double min_approx_weight = 0.0;
    double max_approx_weight = 0.0;
    double min_repel_weight = 0.0;
    double max_repel_weight = 0.0;
    int polynomial_degree = 0;
    int free_parameters = 0;  // n - m (после учёта интерполяционных условий)
    
    bool has_errors() const { return !errors.empty(); }
    bool has_warnings() const { return !warnings.empty(); }
    
    std::string format(bool include_recommendations = true) const {
        std::ostringstream oss;
        
        if (errors.empty() && warnings.empty()) {
            return "Validation passed successfully.\n";
        }
        
        oss << "Validation Report:\n";
        oss << "=================\n\n";
        
        // Сводка
        oss << "Summary:\n";
        oss << "  - Approximation points: " << approx_points_count << "\n";
        oss << "  - Repel points: " << repel_points_count << "\n";
        oss << "  - Interpolation nodes: " << interp_nodes_count << "\n";
        oss << "  - Polynomial degree: " << polynomial_degree << "\n";
        oss << "  - Free parameters: " << free_parameters << "\n";
        if (approx_points_count > 0) {
            oss << "  - Approx weights: [" << min_approx_weight << ", " << max_approx_weight << "]\n";
        }
        if (repel_points_count > 0) {
            oss << "  - Repel weights: [" << min_repel_weight << ", " << max_repel_weight << "]\n";
        }
        oss << "\n";
        
        // Ошибки
        if (!errors.empty()) {
            oss << "Errors (" << errors.size() << "):\n";
            for (size_t i = 0; i < errors.size(); ++i) {
                oss << "  " << (i + 1) << ". " << errors[i].message;
                if (include_recommendations && !errors[i].recommendation.empty()) {
                    oss << "\n      Recommendation: " << errors[i].recommendation;
                }
                oss << "\n";
            }
            oss << "\n";
        }
        
        // Предупреждения
        if (!warnings.empty()) {
            oss << "Warnings (" << warnings.size() << "):\n";
            for (size_t i = 0; i < warnings.size(); ++i) {
                oss << "  " << (i + 1) << ". " << warnings[i].message;
                if (include_recommendations && !warnings[i].recommendation.empty()) {
                    oss << "\n      Recommendation: " << warnings[i].recommendation;
                }
                oss << "\n";
            }
            oss << "\n";
        }
        
        return oss.str();
    }
};

/**
 * @brief Класс для валидации входных данных метода смешанной аппроксимации
 */
class Validator {
public:
    /**
     * @brief Проверка корректности конфигурации (упрощённый интерфейс)
     * @param config конфигурация
     * @param strict_mode если true, то любые предупреждения считаются ошибками
     * @return пустая строка, если валидация прошла успешно, иначе сообщение
     */
    static std::string validate(const ApproximationConfig& config, bool strict_mode = false);
    
    /**
     * @brief Полная проверка с получением детального отчёта
     * @param config конфигурация
     * @param strict_mode если true, то предупреждения добавляются в errors
     * @return детальный отчёт о валидации
     */
    static ValidationReport validate_full(const ApproximationConfig& config, bool strict_mode = false);
    
    /**
     * @brief Проверка корректности интервала определения [a, b]
     * @param config конфигурация
     * @return пустая строка, если интервал корректен, иначе сообщение
     */
    static std::string check_interval(const ApproximationConfig& config);
    
    /**
     * @brief Проверка принадлежности всех точек интервалу [a, b]
     * @param config конфигурация
     * @return пустая строка, если все точки в интервале, иначе сообщение
     */
    static std::string check_points_in_interval(const ApproximationConfig& config);
    
    /**
     * @brief Проверка непересечения множеств точек
     * @param config конфигурация
     * @return пустая строка, если множества не пересекаются, иначе сообщение
     */
    static std::string check_disjoint_sets(const ApproximationConfig& config);
    
    /**
     * @brief Проверка положительности весов и корректности параметров
     * @param config конфигурация
     * @return пустая строка, если все веса корректны, иначе сообщение
     */
    static std::string check_positive_weights(const ApproximationConfig& config);
    
    /**
     * @brief Проверка количества интерполяционных узлов
     * @param config конфигурация
     * @return пустая строка, если m ≤ n+1, иначе сообщение
     */
    static std::string check_interpolation_nodes_count(const ApproximationConfig& config);
    
    /**
     * @brief Проверка уникальности интерполяционных узлов
     * @param config конфигурация
     * @return пустая строка, если все узлы различны, иначе сообщение
     */
    static std::string check_unique_interpolation_nodes(const ApproximationConfig& config);
    
    /**
     * @brief Проверка на пустые множества
     * @param config конфигурация
     * @return пустая строка, если нет проблем, иначе сообщение
     */
    static std::string check_nonempty_sets(const ApproximationConfig& config);
    
    /**
     * @brief Проверка на численные аномалии (очень большие/малые значения)
     * @param config конфигурация
     * @return пустая строка, если нет проблем, иначе сообщение
     */
    static std::string check_numerical_anomalies(const ApproximationConfig& config);
    
    /**
     * @brief Проверка конфликтов между отталкивающими точками и интерполяционными узлами по значениям
     * (шаг 1.2.4): если y_j^* близко к f(z_e) при близких x, это фатальный конфликт
     * @param config конфигурация
     * @return пустая строка, если нет конфликтов, иначе сообщение
     */
    static std::string check_repel_interp_value_conflict(const ApproximationConfig& config);
    
private:
    /**
     * @brief Проверка, содержится ли точка в векторе (с заданной точностью)
     */
    static bool contains_point(const std::vector<double>& points, double x, double tolerance = 1e-10);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_VALIDATOR_H
