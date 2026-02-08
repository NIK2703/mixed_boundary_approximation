#ifndef MIXED_APPROXIMATION_TYPES_H
#define MIXED_APPROXIMATION_TYPES_H

#include <vector>
#include <memory>
#include <string>

namespace mixed_approx {

/**
 * @brief Структура для представления точки с весом (аппроксимирующие точки)
 */
struct WeightedPoint {
    double x;           // координата точки
    double value;       // значение функции в точке f(x)
    double weight;      // вес σ_i
    
    WeightedPoint(double x, double value, double weight)
        : x(x), value(value), weight(weight) {}
};

/**
 * @brief Структура для интерполяционного узла
 */
struct InterpolationNode {
    double x;           // координата узла
    double value;       // значение функции в узле f(z)
    
    InterpolationNode(double x, double value)
        : x(x), value(value) {}
};

/**
 * @brief Структура для точки отталкивания
 * 
 * Отталкивающая точка характеризуется:
 * - координатой x (где применяется отталкивание)
 * - запрещённым значением y_forbidden (какого значения F(x) следует избегать)
 * - весом B_j (интенсивность барьера)
 */
struct RepulsionPoint {
    double x;           // абсцисса точки (координата на оси X)
    double y_forbidden; // запрещённое значение функции (ордината на оси Y)
    double weight;      // вес отталкивания B_j
    
    RepulsionPoint(double x, double y_forbidden, double weight)
        : x(x), y_forbidden(y_forbidden), weight(weight) {}
    
    // Упрощённый конструктор: отталкивание от нуля по умолчанию
    explicit RepulsionPoint(double x, double weight)
        : x(x), y_forbidden(0.0), weight(weight) {}
    
    // Конструктор-адаптер из WeightedPoint (для обратной совместимости)
    // ВАЖНО: value интерпретируется как y_forbidden, а не как f(y_j)
    RepulsionPoint(const WeightedPoint& wp)
        : x(wp.x), y_forbidden(wp.value), weight(wp.weight) {}
    
    // Преобразование в WeightedPoint (для обратной совместимости)
    operator WeightedPoint() const {
        return WeightedPoint(x, y_forbidden, weight);
    }
};

/**
 * @brief Конфигурация метода смешанной аппроксимации
 */
struct ApproximationConfig {
    // Степень полинома
    int polynomial_degree;
    
    // Интервал определения [a, b]
    double interval_start;
    double interval_end;
    
    // Параметры
    double gamma;                    // коэффициент регуляризации (γ ≥ 0)
    std::vector<WeightedPoint> approx_points;   // аппроксимирующие точки {x_i}
    std::vector<RepulsionPoint> repel_points;   // отталкивающие точки {y_j} с запрещёнными значениями y_j^*
    std::vector<InterpolationNode> interp_nodes; // интерполяционные узлы {z_e}
    
    // Параметры численной устойчивости
    double epsilon;                  // минимальный порог для знаменателя (по умолчанию 1e-8)
    double interpolation_tolerance;  // допуск для интерполяционных условий (по умолчанию 1e-10)
    
    ApproximationConfig()
        : polynomial_degree(0)
        , interval_start(0.0)
        , interval_end(1.0)
        , gamma(0.0)
        , epsilon(1e-8)
        , interpolation_tolerance(1e-10) {}
};

/**
 * @brief Результат оптимизации
 */
struct OptimizationResult {
    std::vector<double> coefficients;  // коэффициенты полинома [a_n, a_{n-1}, ..., a_0] (полные)
    double final_objective;            // конечное значение функционала
    int iterations;                    // количество итераций
    bool success;                      // флаг успешности оптимизации
    std::string message;               // сообщение о результате
    
    OptimizationResult()
        : final_objective(0.0)
        , iterations(0)
        , success(false) {}
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_TYPES_H
